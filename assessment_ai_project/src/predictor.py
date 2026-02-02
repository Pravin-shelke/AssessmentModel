import numpy as np
import joblib
import logging
import os
from .config import Config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer

class AssessmentPredictor:
    def __init__(self):
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.models = {} # Loaded models
        
    def run_training_pipeline(self):
        # 1. Load
        df = self.loader.load_data(Config.TRAIN_DATA_FILE)
        
        # 2. Features
        feature_cols = Config.FEATURE_COLUMNS.copy()
        df_encoded = self.engineer.fit_transform(df, feature_cols)
        feature_names = self.engineer.get_feature_names()
        
        # 3. Targets
        target_cols = [c for c in df.columns if c not in Config.FEATURE_COLUMNS and c != 'indicator']
        
        # 4. Train
        self.models = self.trainer.train(df_encoded, feature_names, target_cols)
        
        # 5. Save
        self.save_checkpoint()
        
    def predict(self, input_dict):
        features = self.engineer.transform(input_dict)
        X_input = np.array(features).reshape(1, -1)
        
        predictions = {}
        for indicator, packet in self.models.items():
            meta = Config.QUESTION_META.get(indicator, {})
            policy = self._get_policy(meta, indicator)
            
            if policy == 'skip': continue
            
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
                    "source": "xgboost_v2"
                }

                # Middleware Logic: Apply Confidence Thresholds
                if confidence >= Config.CONFIDENCE_AUTO_FILL_THRESHOLD:
                    result["policy"] = "auto_fill" # Green: User skips
                    result["ui_status"] = "green"
                elif confidence >= Config.CONFIDENCE_SUGGEST_THRESHOLD:
                    result["policy"] = "suggest" # Amber: User reviews
                    result["ui_status"] = "amber"
                else:
                    result["policy"] = "manual_entry" # Red: User must enter
                    result["ui_status"] = "red"
                    val = None # Do not pre-fill if confidence is low

                # Force policy if config overrides (e.g. text always manual)
                if policy == 'skip':
                     result["policy"] = "manual_entry"
                     result["ui_status"] = "red"
                     val = None
                
                if meta.get("type") == "checkbox":
                    result["suggestedOptions"] = [val]
                else:
                    result["value"] = val
                    
                predictions[indicator] = result
            except Exception as e:
                logging.error(f"Prediction error {indicator}: {e}")
                
        return predictions

    def _get_policy(self, meta, indicator):
        if meta.get("type") == "text": return "skip"
        if meta.get("type") == "checkbox": return "suggest"
        if meta.get("conditional"): return "suggest"
        return "auto"

    def save_checkpoint(self):
        state = {
            'models': self.models,
            'engineer': self.engineer
        }
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        joblib.dump(state, Config.MODEL_FILE)
        logging.info(f"Model checkpoint saved to {Config.MODEL_FILE}")

    def load_checkpoint(self):
        if not os.path.exists(Config.MODEL_FILE):
             logging.error("No checkpoint found.")
             return
        state = joblib.load(Config.MODEL_FILE)
        self.models = state['models']
        self.engineer = state['engineer']
        logging.info("Checkpoint loaded.")
