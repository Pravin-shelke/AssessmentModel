import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class AssessmentAIPredictor:
    """
    Professional AI model for assessment prediction.
    Loads data, trains XGBoost models, predicts answers, and reports coverage.
    """
    def __init__(self):
        """
        Initialize the AssessmentAIPredictor with model storage, encoders, and config.
        """
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        self.df = None
        self.feature_modes = {}
        self.min_training_samples = 10
        self.question_meta = {
            "BH-1": {"type": "radio", "options": ["Yes", "No", "In Progress"], "conditional": False},
            "BH-2": {"type": "radio", "options": ["Yes", "No"], "conditional": True, "parent": "BH-1", "parent_value": "Yes"},
            "CM-5": {"type": "text"},
            "WQ-3": {"type": "checkbox", "multi": True}
            # ...add more as needed for your demo
        }

    def load_data(self, csv_file):
        """
        Load assessment data from mock_test_data.csv (JSON-in-cell format) and build wide DataFrame.
        Aggregates all indicator answers per unique feature set, so each indicator column is populated across many rows.
        """
        import json
        try:
            logging.info(f"Loading data from {csv_file} (mock format)...")
            raw_df = pd.read_csv(csv_file, encoding='utf-8')
            logging.info(f"Loaded {len(raw_df)} records.")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

        self.feature_columns = [
            'country_code', 'crop_name', 'Partner', 'irrigation', 'hired_workers', 'area'
        ]

        # Explode each row's labelsAnswersMap into multiple rows, one per indicator
        records = []
        error_log_count = 0
        for idx, row in raw_df.iterrows():
            try:
                cell = row['labelsAnswersMap']
                if isinstance(cell, str):
                    label_list = json.loads(cell)
                else:
                    label_list = []
            except Exception as e:
                if error_log_count < 5:
                    logging.warning(f"Row {idx} JSON parse error: {e}\nRaw cell: {repr(row['labelsAnswersMap'])}")
                    error_log_count += 1
                continue
            for label in label_list:
                indicator = label.get('soaId')
                answer = label.get('answer')
                if indicator is not None:
                    rec = {col: row[col] for col in self.feature_columns}
                    rec['indicator'] = indicator
                    rec['answer'] = answer
                    records.append(rec)

        # Now pivot: rows = unique feature sets, columns = indicators, values = answer
        if not records:
            raise ValueError("No indicator records found after parsing labelsAnswersMap.")
        long_df = pd.DataFrame(records)
        wide_df = long_df.pivot_table(
            index=self.feature_columns,
            columns='indicator',
            values='answer',
            aggfunc=lambda x: x.iloc[0] if len(x) else None
        ).reset_index()
        # Flatten columns
        wide_df.columns.name = None
        self.df = wide_df
        self.target_columns = [col for col in self.df.columns if col not in self.feature_columns]
        logging.info(f"Found {len(self.target_columns)} indicator columns.")
        return self.df

    def prepare_training_data(self):
        """
        Prepare and encode features for model training.
        """
        logging.info("Preparing training data...")
        for col in self.feature_columns:
            if col == 'area':
                self.df['area'] = pd.to_numeric(self.df['area'], errors='coerce').fillna(0)
                continue
            le = LabelEncoder()
            values = self.df[col].astype(str).fillna('Unknown')
            classes = values.unique().tolist()
            if 'Unknown' not in classes:
                classes.append('Unknown')
            le.fit(classes)
            self.df[col] = values
            self.df[col + '_encoded'] = le.transform(values)
            self.label_encoders[col] = le
            self.feature_modes[col] = values.mode(dropna=True)[0]
        logging.info("Feature encoding complete.")

    def train_models(self):
        """
        Train an XGBoost model for each indicator with sufficient data.
        """
        logging.info("Training models...")
        X_cols = [col + '_encoded' if col + '_encoded' in self.df.columns else col for col in self.feature_columns]
        X = self.df[X_cols].values
        trained_count = 0
        skipped_count = 0
        for idx, target_col in enumerate(self.target_columns, 1):
            try:
                if self.df[target_col].isna().all():
                    skipped_count += 1
                    continue
                unique_vals = self.df[target_col].dropna().nunique()
                if unique_vals <= 1:
                    skipped_count += 1
                    continue
                valid_indices = self.df[target_col].notna()
                y = self.df.loc[valid_indices, target_col].astype(str)
                X_valid = X[valid_indices]
                if len(y) < self.min_training_samples:
                    skipped_count += 1
                    continue
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                num_classes = len(unique_vals) if isinstance(unique_vals, list) else unique_vals
                num_classes = len(le_target.classes_)
                
                if num_classes == 2:
                    objective = 'binary:logistic'
                    kwargs = {}
                else:
                    objective = 'multi:softmax'
                    kwargs = {'num_class': num_classes}
                
                model = xgb.XGBClassifier(
                    max_depth=3,
                    n_estimators=50,
                    learning_rate=0.1,
                    objective=objective,
                    random_state=42,
                    verbosity=0,
                    **kwargs
                )
                model.fit(X_valid, y_encoded)
                self.models[target_col] = {
                    'model': model,
                    'encoder': le_target,
                    'feature_cols': X_cols
                }
                trained_count += 1
                if trained_count % 20 == 0:
                    logging.info(f"Trained {trained_count} models...")
            except Exception as e:
                logging.warning(f"Skipped {target_col}: {e}")
                skipped_count += 1
        logging.info(f"Training complete. Trained: {trained_count}, Skipped: {skipped_count}")
        return trained_count

    def predict_assessment(self, country, crop, partner, irrigation, hired_workers, area):
        """
        Predict all assessment indicators based on 6 inputs.
        Returns a dict of predictions with value and confidence for each indicator.
        """
        logging.info(f"Predicting assessment for: {country}, {crop}, {partner}, {irrigation}, {hired_workers}, {area}")
        input_data = {
            'country_code': country,
            'crop_name': crop,
            'Partner': partner,
            'irrigation': irrigation,
            'hired_workers': hired_workers,
            'area': area
        }
        X_input = []
        for col in self.feature_columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    encoded_val = le.transform([input_data[col]])[0]
                except ValueError:
                    fallback = 'Unknown' if 'Unknown' in le.classes_ else self.feature_modes.get(col, le.classes_[0])
                    encoded_val = le.transform([fallback])[0]
                X_input.append(encoded_val)
            else:
                raw_val = input_data[col]
                if isinstance(raw_val, bool):
                    raw_val = int(raw_val)
                if raw_val is None:
                    raw_val = 0
                X_input.append(raw_val)
        X_input = np.array(X_input, dtype=float).reshape(1, -1)
        predictions = {}
        high_confidence_count = 0
        for target_col, model_data in self.models.items():
            meta = self.question_meta.get(target_col, {})
            # Conditional logic: skip if parent not satisfied
            if meta and meta.get("conditional"):
                parent = meta.get("parent")
                required_value = meta.get("parent_value")
                if parent and predictions.get(parent, {}).get("value") != required_value:
                    continue
            policy = self.ai_policy(target_col)
            if policy == "skip":
                continue
            try:
                model = model_data['model']
                encoder = model_data['encoder']
                y_pred = model.predict(X_input)[0]
                y_proba = model.predict_proba(X_input)[0]
                confidence = float(np.max(y_proba) * 100)
                predicted_value = encoder.inverse_transform([int(y_pred)])[0]
                # Relaxed: Always return predictions regardless of confidence
                # Checkbox handling
                if meta.get("type") == "checkbox":
                    predictions[target_col] = {
                        "suggestedOptions": [predicted_value],
                        "confidence": confidence,
                        "policy": policy,
                        "source": "xgboost"
                    }
                else:
                    predictions[target_col] = {
                        "value": predicted_value,
                        "confidence": confidence,
                        "policy": policy,
                        "source": "xgboost"
                    }
                if policy == "auto" and confidence >= 80:
                    high_confidence_count += 1
            except Exception as e:
                logging.warning(f"Prediction failed for {target_col}: {e}")
                predictions[target_col] = {
                    "value": "Unknown",
                    "confidence": 0.0,
                    "policy": policy,
                    "source": "xgboost"
                }
        logging.info(f"Predictions complete. Total: {len(predictions)}, High confidence (≥80%): {high_confidence_count}, Avg confidence: {np.mean([p['confidence'] for p in predictions.values()]):.1f}%")
        return predictions

    def save_models(self, filename='xgboost_balaji_models.pkl'):
        """
        Save all trained models to disk.
        """
        logging.info(f"Saving models to {filename}...")
        model_package = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        joblib.dump(model_package, filename)
        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)
        logging.info(f"Models saved! File size: {file_size:.1f} MB")

    def load_models(self, filename='xgboost_balaji_models.pkl'):
        """
        Load pre-trained models from disk.
        """
        logging.info(f"Loading models from {filename}...")
        model_package = joblib.load(filename)
        self.models = model_package['models']
        self.label_encoders = model_package['label_encoders']
        self.feature_columns = model_package['feature_columns']
        self.target_columns = model_package['target_columns']
        logging.info(f"Loaded {len(self.models)} trained models")

    def export_predictions_to_csv(self, predictions, output_file='predicted_assessment.csv'):
        """
        Export predictions to CSV format.
        """
        logging.info(f"Exporting predictions to {output_file}...")
        data = []
        for indicator, pred_data in predictions.items():
            val = pred_data.get('value')
            if val is None:
                val = pred_data.get('suggestedOptions')
            data.append({
                'Indicator': indicator,
                'Predicted_Value': val,
                'Confidence_%': f"{pred_data['confidence']:.1f}%"
            })
        df_export = pd.DataFrame(data)
        df_export.to_csv(output_file, index=False)
        logging.info(f"Predictions exported to {output_file}. Total indicators: {len(data)}")

    def report_coverage(self):
        """
        Print a summary of model coverage and skipped indicators.
        """
        if self.df is None or not self.target_columns:
            logging.warning("No data loaded for coverage report.")
            return
        covered = [col for col in self.target_columns if col in self.models]
        skipped = [col for col in self.target_columns if col not in self.models]
        logging.info(f"Model coverage: {len(covered)}/{len(self.target_columns)} indicators ({len(covered)/len(self.target_columns)*100:.1f}%)")
        if skipped:
            logging.info(f"Skipped indicators (insufficient data): {skipped}")

    def ai_policy(self, indicator):
        meta = self.question_meta.get(indicator, {})
        if meta.get("type") == "text":
            return "skip"
        if meta.get("type") == "checkbox":
            return "suggest"
        if meta.get("conditional"):
            return "suggest"
        if meta.get("type") == "radio" and len(meta.get("options", [])) == 2:
            return "suggest"
        return "auto"
