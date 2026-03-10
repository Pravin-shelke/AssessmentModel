import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
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
                # multi:softprob (not softmax) to get real probabilities for confidence
                params['objective'] = 'multi:softprob'
                params['num_class'] = num_classes

            try:
                # 80/20 train/test split — with fallbacks for rare-class indicators
                n_samples = len(X_valid)
                split_ok = False

                if n_samples >= 50:
                    try:
                        can_stratify = (
                            num_classes <= 20 and
                            min(pd.Series(y_encoded).value_counts()) >= 2
                        )
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_valid, y_encoded, test_size=0.2, random_state=42,
                            stratify=y_encoded if can_stratify else None
                        )
                        # Ensure all classes appear in training split (XGBoost requirement)
                        if len(set(y_train)) == num_classes:
                            split_ok = True
                        else:
                            logging.debug(f"{target_col}: not all classes in train split, falling back to full data")
                    except Exception:
                        pass

                if not split_ok:
                    # Use full data for both train and eval when split isn't feasible
                    X_train, X_test, y_train, y_test = X_valid, X_valid, y_encoded, y_encoded

                # ── Class imbalance handling ──────────────────────────────────
                # Binary: let XGBoost re-weight the minority class automatically.
                # Multi-class: compute per-sample weights so no external lib is needed.
                if num_classes == 2:
                    counts = np.bincount(y_train)
                    # scale_pos_weight = negative_count / positive_count
                    params['scale_pos_weight'] = float(counts[0]) / float(counts[1]) if counts[1] > 0 else 1.0
                    sample_weights = None
                else:
                    sample_weights = compute_sample_weight('balanced', y_train)

                # ── Hyperparameter tuning (optional, off by default) ──────────
                if Config.ENABLE_HYPERPARAMETER_TUNING and n_samples >= 500:
                    base = xgb.XGBClassifier(
                        random_state=42, verbosity=0, eval_metric='logloss',
                        objective=params['objective'],
                        **({'num_class': num_classes} if num_classes > 2 else {}),
                    )
                    n_splits_gs = min(3, min(pd.Series(y_train).value_counts()))
                    n_splits_gs = max(2, int(n_splits_gs))
                    cv_gs = StratifiedKFold(n_splits=n_splits_gs, shuffle=True, random_state=42)
                    grid = GridSearchCV(
                        base, Config.HYPERPARAM_GRID, cv=cv_gs,
                        scoring='accuracy', n_jobs=-1, refit=True
                    )
                    fit_kwargs = {'sample_weight': sample_weights} if sample_weights is not None else {}
                    grid.fit(X_train, y_train, **fit_kwargs)
                    model = grid.best_estimator_
                    logging.info(f"  [{target_col}] GridSearchCV best params: {grid.best_params_}")
                else:
                    model = xgb.XGBClassifier(**params)
                    fit_kwargs = {'sample_weight': sample_weights} if sample_weights is not None else {}
                    model.fit(X_train, y_train, **fit_kwargs)

                y_pred = model.predict(X_test)
                accuracy = float(accuracy_score(y_test, y_pred))
                avg_type = 'binary' if num_classes == 2 else 'weighted'
                f1 = float(f1_score(y_test, y_pred, average=avg_type, zero_division=0))
                precision = float(precision_score(y_test, y_pred, average=avg_type, zero_division=0))
                recall = float(recall_score(y_test, y_pred, average=avg_type, zero_division=0))

                # Per-class accuracy from confusion matrix
                cm = confusion_matrix(y_test, y_pred).tolist()

                # Cross-validation accuracy for more reliable estimate (when enough data)
                cv_accuracy = accuracy
                if n_samples >= 100 and len(set(y_encoded)) >= 2:
                    try:
                        n_splits = min(5, min(pd.Series(y_encoded).value_counts()))
                        n_splits = max(2, int(n_splits))
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(
                            xgb.XGBClassifier(**params), X_valid, y_encoded,
                            cv=cv, scoring='accuracy'
                        )
                        cv_accuracy = float(np.mean(cv_scores))
                    except Exception:
                        pass  # fall back to single-split accuracy

                self.models[target_col] = {
                    'model': model,
                    'encoder': le_target,
                    'feature_names': feature_names,
                    'accuracy': round(cv_accuracy, 4),   # cross-validated when possible
                    'holdout_accuracy': round(accuracy, 4),
                    'f1_score': round(f1, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'confusion_matrix': cm,
                    'num_classes': num_classes,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'classes': list(le_target.classes_),
                }
                trained_count += 1
                logging.info(
                    f"  [{target_col}] acc={accuracy:.3f} f1={f1:.3f} "
                    f"prec={precision:.3f} rec={recall:.3f} "
                    f"classes={num_classes} train_n={len(X_train)}"
                )
            except Exception as e:
                logging.warning(f"Training failed for {target_col}: {e}")
                
        logging.info(f"Training complete. Models trained: {trained_count}")
        return self.models

    def _is_valid_target(self, df, col):
        if df[col].isna().all(): return False
        if df[col].dropna().nunique() <= 1: return False
        if df[col].count() < Config.MIN_TRAINING_SAMPLES: return False
        return True
