import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    CONFIG_DIR = os.path.join(BASE_DIR, 'config')

    TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'mock_test_uncopied_data.csv')
    MODEL_FILE = os.path.join(MODELS_DIR, 'xgboost_models.pkl')
    PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predicted_assessment.csv')

    # Optional user-editable file for localized crop name overrides only.
    # e.g. { "Blé": "Wheat", "ARROZ": "Rice" }
    # Identity entries like { "Wheat": "Wheat" } are ignored automatically.
    # This file is NEVER auto-generated — it only contains human-added synonyms.
    CROP_SYNONYMS_FILE = os.path.join(CONFIG_DIR, 'crop_synonyms.json')

    # Model Hyperparameters — tuned for 158K row dataset
    MODEL_PARAMS = {
        'max_depth': 5,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0,
        'eval_metric': 'logloss',
    }

    # Minimum non-null samples required to train a model for a given indicator
    MIN_TRAINING_SAMPLES = 20

    CONFIDENCE_THRESHOLD = 80.0

    # Feature columns — must match CSV column names (case-sensitive)
    # planYear: year-over-year patterns significantly affect answers
    FEATURE_COLUMNS = [
        'country_code',
        'crop_name',
        'partner',
        'subpartner',      # regional sub-program — already in CSV, improves indicator accuracy
        'irrigation',
        'hired_workers',
        'area',
        'planYear',
    ]

    # Confidence thresholds for UI policy decision
    CONFIDENCE_AUTO_FILL_THRESHOLD = 70.0   # Green: auto-fill, user can skip
    CONFIDENCE_SUGGEST_THRESHOLD   = 40.0   # Amber: show suggestion, user reviews
    # Below CONFIDENCE_SUGGEST_THRESHOLD → Red: manual entry required

    # QUESTION_META is populated dynamically at runtime by AssessmentPredictor:
    #   - During training: DataLoader.build_question_meta() derives it from the CSV.
    #   - During inference: restored from the saved model checkpoint.
    # It is NEVER read from a static file. Data is the single source of truth.
    QUESTION_META = {}  # set by predictor pipeline

    @classmethod
    def get_question_meta(cls):
        """Returns the live QUESTION_META dict (always set by the pipeline)."""
        return cls.QUESTION_META

    @classmethod
    def set_question_meta(cls, meta: dict):
        """Called by AssessmentPredictor after training or loading a checkpoint."""
        cls.QUESTION_META = meta
        cls._cache_size = len(meta)  # for logging only
