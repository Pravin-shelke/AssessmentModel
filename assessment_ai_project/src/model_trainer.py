import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import logging
from .config import Config

class ModelTrainer:
    def __init__(self):
        self.models = {}
        
    def train(self, df, feature_names, target_columns):
        """
        Train models for valid target columns.
        """
        X = df[feature_names].values
        trained_count = 0
        
        for target_col in target_columns:
            if not self._is_valid_target(df, target_col):
                continue
                
            y_series = df.loc[df[target_col].notna(), target_col].astype(str)
            X_valid = X[df[target_col].notna()]
            
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y_series)
            
            # Determine Objective
            num_classes = len(le_target.classes_)
            params = Config.MODEL_PARAMS.copy()
            
            if num_classes == 2:
                params['objective'] = 'binary:logistic'
            else:
                params['objective'] = 'multi:softmax'
                params['num_class'] = num_classes
                
            try:
                model = xgb.XGBClassifier(**params)
                model.fit(X_valid, y_encoded)
                
                self.models[target_col] = {
                    'model': model,
                    'encoder': le_target,
                    'feature_names': feature_names # Save schema
                }
                trained_count += 1
            except Exception as e:
                logging.warning(f"Training failed for {target_col}: {e}")
                
        logging.info(f"Training complete. Models trained: {trained_count}")
        return self.models

    def _is_valid_target(self, df, col):
        if df[col].isna().all(): return False
        if df[col].dropna().nunique() <= 1: return False
        if df[col].count() < Config.MIN_TRAINING_SAMPLES: return False
        return True
