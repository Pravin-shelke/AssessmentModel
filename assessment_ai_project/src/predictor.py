import numpy as np
import joblib
import logging
import os
import threading
import json
import datetime
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .config import Config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer

class AssessmentPredictor:
    def __init__(self):
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.models = {}         # Loaded models
        self.X_encoded = None    # Encoded training matrix (for KNN similar farms)
        self.df_raw = None       # Raw training rows (for displaying similar farm context)
        self._knn = None         # Lazy-built KNN index
        self._shap_explainers = {}  # Cache per indicator
        
    def run_training_pipeline(self):
        # 1. Load data — DataLoader.build_question_meta() runs automatically inside
        df = self.loader.load_data(Config.TRAIN_DATA_FILE)

        # 2. Propagate dynamically-derived question metadata to Config
        Config.set_question_meta(self.loader.question_meta)
        logging.info(f"Config.QUESTION_META set: {len(Config.QUESTION_META)} indicators")

        # 3. Features
        feature_cols = Config.FEATURE_COLUMNS.copy()
        df_encoded = self.engineer.fit_transform(df, feature_cols)
        feature_names = self.engineer.get_feature_names()

        # 4. Store encoded matrix and raw rows for KNN similar farms
        self.X_encoded = df_encoded[feature_names].values.astype(float)
        self.df_raw = df[Config.FEATURE_COLUMNS].copy().reset_index(drop=True)
        self._knn = None  # reset — will be rebuilt lazily

        # 5. Targets
        target_cols = [c for c in df.columns if c not in Config.FEATURE_COLUMNS and c != 'indicator']

        # 6. Train
        self.models = self.trainer.train(df_encoded, feature_names, target_cols)

        # 7. Save checkpoint
        self.save_checkpoint()
        
    def predict(self, input_dict):
        # Normalize key casing (defensive — api_server should already send lowercase)
        if 'Partner' in input_dict and 'partner' not in input_dict:
            input_dict['partner'] = input_dict.pop('Partner')

        features = self.engineer.transform(input_dict)
        X_input = np.array(features).reshape(1, -1)

        predictions = {}
        question_meta = Config.QUESTION_META  # always live — set by pipeline or checkpoint restore

        # First pass: include text/number questions as manual_entry (no model to run)
        for indicator, meta in question_meta.items():
            policy = self._get_policy(meta, indicator)
            if policy == 'manual':
                predictions[indicator] = {
                    "value": None,
                    "confidence": 0,
                    "policy": "manual_entry",
                    "ui_status": "red",
                    "source": "no_model",
                    "model_accuracy": 0,
                    "reason": f"Open {meta.get('type', 'text')} field — AI cannot predict free-form input",
                }

        # Second pass: run ML models for classifiable questions
        # NOTE: if a trained model exists for an indicator, always run it —
        # even if question_meta inferred type='number'/'text'. The trained model
        # with real accuracy takes precedence over the type heuristic.
        for indicator, packet in self.models.items():
            meta = question_meta.get(indicator, {})
            policy = self._get_policy(meta, indicator)
            
            # Check conditional
            if meta.get("conditional"):
                parent_val = predictions.get(meta["parent"], {}).get("value")
                if parent_val != meta.get("parent_value"):
                    continue

            try:
                model = packet['model']
                le = packet['encoder']
                
                y_pred = model.predict(X_input)[0]
                y_proba = model.predict_proba(X_input)[0]
                confidence = float(np.max(y_proba) * 100)
                val = le.inverse_transform([int(y_pred)])[0]
                
                result = {
                    "confidence": confidence,
                    "source": "xgboost_v2",
                    "model_accuracy": round(packet.get('accuracy', 0) * 100, 1),
                }

                # Middleware Logic: Apply Confidence Thresholds
                # NOTE: val is always kept — the predicted value is sent regardless of
                # confidence. The policy/ui_status tells the UI how to present it.
                if confidence >= Config.CONFIDENCE_AUTO_FILL_THRESHOLD:
                    result["policy"] = "auto_fill" # Green: auto-fill
                    result["ui_status"] = "green"
                elif confidence >= Config.CONFIDENCE_SUGGEST_THRESHOLD:
                    result["policy"] = "suggest" # Amber: show suggestion, user reviews
                    result["ui_status"] = "amber"
                else:
                    result["policy"] = "suggest_low" # Red: low confidence but still show value
                    result["ui_status"] = "red"
                
                if meta.get("type") == "checkbox":
                    result["suggestedOptions"] = [val] if val is not None else []
                else:
                    result["value"] = val
                    
                predictions[indicator] = result
            except Exception as e:
                logging.error(f"Prediction error {indicator}: {e}")
                
        return predictions

    def _get_policy(self, meta, indicator):
        q_type = meta.get("type")
        if q_type in ("text", "number"):     return "manual"
        if q_type == "checkbox":             return "suggest"
        if meta.get("conditional"):          return "suggest"
        return "auto"

    # ──────────────────────────────────────────────────────────────
    # 1. SHAP EXPLAINABILITY
    # ──────────────────────────────────────────────────────────────
    def explain_prediction(self, indicator, input_dict):
        """
        Returns SHAP-based feature contributions for one indicator.
        Shows which input fields most influenced the AI's answer.
        """
        import shap
        packet = self.models.get(indicator)
        if not packet:
            return {"error": f"No model trained for indicator: {indicator}"}

        features = self.engineer.transform(dict(input_dict))
        X_input = np.array(features).reshape(1, -1)
        feature_names = self.engineer.get_feature_names()

        model = packet['model']

        # Build/cache SHAP explainer per model
        if indicator not in self._shap_explainers:
            self._shap_explainers[indicator] = shap.TreeExplainer(model)
        explainer = self._shap_explainers[indicator]

        shap_values = explainer.shap_values(X_input)
        # For multi-class, use the predicted class's SHAP values
        le = packet['encoder']
        y_pred = int(model.predict(X_input)[0])
        if isinstance(shap_values, list):
            vals = shap_values[y_pred][0]
        else:
            vals = shap_values[0]

        predicted_value = le.inverse_transform([y_pred])[0]
        # Map encoded feature names back to human-readable names
        readable_names = [n.replace('_encoded', '').replace('_', ' ').title() for n in feature_names]
        contributions = sorted(
            [{"feature": readable_names[i], "raw_feature": feature_names[i],
              "impact": round(float(vals[i]), 4),
              "input_value": str(input_dict.get(feature_names[i].replace('_encoded', ''), ''))}
             for i in range(len(vals))],
            key=lambda x: abs(x["impact"]),
            reverse=True
        )
        top_drivers = [c for c in contributions if abs(c["impact"]) > 0.01][:5]
        # Build human-readable reason sentence
        if top_drivers:
            parts = [f"{d['feature']} ({d['input_value']})" for d in top_drivers[:3] if d['input_value']]
            reason = f"Predicted '{predicted_value}' mainly based on: " + ", ".join(parts)
        else:
            reason = f"Predicted '{predicted_value}' — inputs had similar influence"

        return {
            "indicator": indicator,
            "predicted_value": predicted_value,
            "reason": reason,
            "top_drivers": top_drivers,
            "all_contributions": contributions,
        }

    # ──────────────────────────────────────────────────────────────
    # 2. SIMILAR FARMS (KNN)
    # ──────────────────────────────────────────────────────────────
    def _build_knn_index(self):
        """Lazily build KNN index on encoded training data."""
        if self._knn is None and self.X_encoded is not None:
            n = min(20, len(self.X_encoded))
            self._knn = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', metric='euclidean')
            self._knn.fit(self.X_encoded)
            logging.info(f"KNN index built on {len(self.X_encoded)} training rows")

    def find_similar_farms(self, input_dict, n=5):
        """
        Returns top-N most similar farms from training data.
        Uses Euclidean distance on encoded feature space.
        """
        if self.X_encoded is None or self.df_raw is None:
            return {"error": "Training data not available — please retrain the model"}

        self._build_knn_index()
        features = self.engineer.transform(dict(input_dict))
        X_input = np.array(features).reshape(1, -1)

        distances, indices = self._knn.kneighbors(X_input, n_neighbors=min(n, len(self.X_encoded)))
        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.df_raw.iloc[idx].to_dict()
            # Compute similarity score (100% = identical, lower = more different)
            max_dist = distances[0][-1] if distances[0][-1] > 0 else 1.0
            similarity = round(max(0.0, (1 - dist / (max_dist + 1e-9)) * 100), 1)
            similar.append({
                "rank": len(similar) + 1,
                "similarity_pct": similarity,
                "country": row.get('country_code', ''),
                "crop": row.get('crop_name', ''),
                "partner": row.get('partner', ''),
                "subpartner": row.get('subpartner', ''),
                "irrigation": row.get('irrigation', ''),
                "hired_workers": str(row.get('hired_workers', '')),
                "area": str(row.get('area', '')),
                "planYear": str(row.get('planYear', '')),
            })
        return {"count": len(similar), "similar_farms": similar}

    # ──────────────────────────────────────────────────────────────
    # 3. SUSTAINABILITY SCORE
    # ──────────────────────────────────────────────────────────────
    def compute_sustainability_score(self, predictions):
        """
        Derives an estimated sustainability score from predictions.
        Score = weighted average of confidence on green+amber predictions,
        normalised to 0–100. Also returns a band label and section breakdown.
        """
        total_weight = 0.0
        score_sum = 0.0
        section_scores = {}

        for indicator, res in predictions.items():
            meta = Config.QUESTION_META.get(indicator, {})
            section = meta.get('section', 'General')
            ui_status = res.get('ui_status', 'red')
            conf = res.get('confidence', 0)
            model_acc = res.get('model_accuracy', 50)

            # Weight = model accuracy (trust of prediction)
            weight = model_acc / 100.0
            if ui_status == 'green':
                score = conf             # high confidence → full contribution
            elif ui_status == 'amber':
                score = conf * 0.6       # partial contribution
            else:
                score = 0.0              # red / manual → no positive contribution

            score_sum += score * weight
            total_weight += weight

            if section not in section_scores:
                section_scores[section] = {'sum': 0.0, 'weight': 0.0}
            section_scores[section]['sum'] += score * weight
            section_scores[section]['weight'] += weight

        overall = round(score_sum / total_weight, 1) if total_weight > 0 else 0.0

        if overall >= 80:
            band = "Excellent"
            band_color = "green"
        elif overall >= 60:
            band = "Good"
            band_color = "amber"
        elif overall >= 40:
            band = "Needs Improvement"
            band_color = "orange"
        else:
            band = "Poor"
            band_color = "red"

        section_breakdown = {
            sec: round(v['sum'] / v['weight'], 1) if v['weight'] > 0 else 0.0
            for sec, v in section_scores.items()
        }

        return {
            "estimated_score": overall,
            "band": band,
            "band_color": band_color,
            "range": f"{max(0, overall - 7):.0f}–{min(100, overall + 7):.0f}",
            "section_scores": section_breakdown,
            "note": "Estimated before assessment completion — actual score may vary",
        }

    # ──────────────────────────────────────────────────────────────
    # 4. YEAR-OVER-YEAR COMPARISON
    # ──────────────────────────────────────────────────────────────
    def compare_year_over_year(self, current_predictions, prev_plan_year, input_dict):
        """
        Compares current predictions against predictions for the previous year.
        Returns a per-indicator diff with change direction.
        """
        prev_input = dict(input_dict)
        prev_input['planYear'] = str(prev_plan_year)
        prev_predictions = self.predict(prev_input)

        changes = {}
        for indicator, curr in current_predictions.items():
            prev = prev_predictions.get(indicator, {})
            curr_val = curr.get('value') or str(curr.get('suggestedOptions', [''])[0])
            prev_val = prev.get('value') or str(prev.get('suggestedOptions', [''])[0])
            curr_conf = curr.get('confidence', 0)
            prev_conf = prev.get('confidence', 0)

            if curr_val != prev_val:
                direction = "improved" if curr_conf > prev_conf else "changed"
                changes[indicator] = {
                    "changed": True,
                    "direction": direction,
                    "previous_value": prev_val,
                    "current_value": curr_val,
                    "previous_confidence": round(prev_conf, 1),
                    "current_confidence": round(curr_conf, 1),
                    "confidence_delta": round(curr_conf - prev_conf, 1),
                }
            else:
                changes[indicator] = {
                    "changed": False,
                    "direction": "stable",
                    "previous_value": prev_val,
                    "current_value": curr_val,
                    "confidence_delta": round(curr_conf - prev_conf, 1),
                }

        changed_count = sum(1 for v in changes.values() if v['changed'])
        return {
            "compared_year": str(prev_plan_year),
            "total_indicators": len(changes),
            "changed_count": changed_count,
            "stable_count": len(changes) - changed_count,
            "changes": changes,
        }

    # ──────────────────────────────────────────────────────────────
    # 5. ACTIVE LEARNING — incorporate feedback into retraining
    # ──────────────────────────────────────────────────────────────
    def apply_feedback_to_training(self, min_corrections=10):
        """
        Reads feedback.jsonl and retrains models that have ≥ min_corrections.
        Returns dict of retrained indicators and their new accuracies.
        """
        feedback_file = os.path.join(Config.DATA_DIR, 'feedback.jsonl')
        if not os.path.exists(feedback_file):
            return {"retrained": [], "message": "No feedback file found"}

        # Group feedback by indicator
        corrections = {}
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    ind = entry.get('indicator')
                    if ind:
                        corrections.setdefault(ind, []).append(entry)
                except Exception:
                    continue

        retrained = []
        for indicator, entries in corrections.items():
            if len(entries) < min_corrections:
                continue
            if indicator not in self.models:
                continue
            try:
                packet = self.models[indicator]
                le = packet['encoder']
                # Build synthetic correction rows
                rows = []
                for e in entries:
                    inp = e.get('input', {})
                    actual = str(e.get('actual', ''))
                    if actual and inp:
                        row = {col: inp.get(col, '') for col in Config.FEATURE_COLUMNS}
                        row[indicator] = actual
                        rows.append(row)

                if not rows:
                    continue

                correction_df = pd.DataFrame(rows)
                # Re-encode with existing engineer
                feature_names = self.engineer.get_feature_names()
                corrected_encoded = self.engineer.fit_transform(correction_df, Config.FEATURE_COLUMNS.copy())

                # Only keep classes already known to the encoder
                valid_mask = corrected_encoded[indicator].astype(str).isin(le.classes_)
                corrected_encoded = corrected_encoded[valid_mask]
                if len(corrected_encoded) < 2:
                    continue

                X_c = corrected_encoded[feature_names].values
                y_c = le.transform(corrected_encoded[indicator].astype(str))

                import xgboost as xgb
                model = packet['model']
                model.fit(X_c, y_c, xgb_model=model.get_booster())  # warm-start on corrections
                packet['model'] = model
                self.models[indicator] = packet
                retrained.append(indicator)
                logging.info(f"Active learning: retrained {indicator} with {len(entries)} corrections")
            except Exception as e:
                logging.warning(f"Active learning failed for {indicator}: {e}")

        if retrained:
            self.save_checkpoint()

        return {"retrained": retrained, "corrections_used": {ind: len(corrections[ind]) for ind in retrained}}

    def save_checkpoint(self):
        model_summary = {
            indicator: {
                'accuracy': packet.get('accuracy'),
                'f1_score': packet.get('f1_score'),
                'num_classes': packet.get('num_classes'),
                'training_samples': packet.get('training_samples'),
                'classes': packet.get('classes', []),
            }
            for indicator, packet in self.models.items()
        }
        state = {
            'models': self.models,
            'engineer': self.engineer,
            'model_summary': model_summary,
            'question_meta': Config.QUESTION_META,
            'X_encoded': self.X_encoded,   # for KNN similar farms
            'df_raw': self.df_raw,         # for similar farm display
        }
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        joblib.dump(state, Config.MODEL_FILE)
        logging.info(f"Model checkpoint saved: {len(self.models)} models → {Config.MODEL_FILE}")
        avg_acc = sum(v['accuracy'] for v in model_summary.values() if v['accuracy']) / max(len(model_summary), 1)
        logging.info(f"Average model accuracy: {avg_acc:.3f}")

    def load_checkpoint(self):
        if not os.path.exists(Config.MODEL_FILE):
            raise FileNotFoundError(f"No checkpoint found at {Config.MODEL_FILE}")
        state = joblib.load(Config.MODEL_FILE)
        self.models = state['models']
        self.engineer = state['engineer']
        self.model_summary = state.get('model_summary', {})
        self.X_encoded = state.get('X_encoded')      # KNN data
        self.df_raw = state.get('df_raw')            # KNN display
        self._knn = None                              # rebuild lazily
        self._shap_explainers = {}                    # reset cache

        saved_meta = state.get('question_meta', {})
        Config.set_question_meta(saved_meta)
        logging.info(f"Checkpoint loaded: {len(self.models)} models | "
                     f"{len(saved_meta)} indicators in question_meta")

    def get_model_stats(self):
        """Returns per-indicator accuracy stats for the /health and /models endpoints."""
        summary = getattr(self, 'model_summary', {})
        if not summary:
            summary = {
                ind: {
                    'accuracy': p.get('accuracy'),
                    'f1_score': p.get('f1_score'),
                    'num_classes': p.get('num_classes'),
                    'training_samples': p.get('training_samples'),
                }
                for ind, p in self.models.items()
            }
        accuracies = [v['accuracy'] for v in summary.values() if v.get('accuracy') is not None]
        return {
            'total_models': len(self.models),
            'average_accuracy': round(sum(accuracies) / len(accuracies), 4) if accuracies else 0,
            'per_indicator': summary,
        }
