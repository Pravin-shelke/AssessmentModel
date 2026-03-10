import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    CONFIG_DIR = os.path.join(BASE_DIR, 'config')

    TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'mock_test_uncopied_data.csv')
    MODEL_FILE = os.path.join(MODELS_DIR, 'xgboost_models.pkl')
    PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predicted_assessment.csv')  # kept for legacy export

    # SQLite database — replaces predicted_assessment.csv and feedback.jsonl
    # as the primary store for prediction history, feedback, and model versions.
    DB_FILE = os.path.join(DATA_DIR, 'assessment_ai.db')

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

    # Hyperparameter tuning — set True to enable GridSearchCV (slower, higher accuracy).
    # Recommended only when retraining with 500+ rows per indicator.
    ENABLE_HYPERPARAMETER_TUNING = False

    # Search space for GridSearchCV (used when ENABLE_HYPERPARAMETER_TUNING = True)
    HYPERPARAM_GRID = {
        'max_depth':     [3, 5, 7],
        'n_estimators':  [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample':     [0.7, 0.8, 1.0],
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
        'subpartner',      # regional sub-program — 97% null in current CSV but included for future data
        'irrigation',
        'hired_workers',
        'area',
        'planYear',
        # ── Future features ──────────────────────────────────────────────────────
        # Add columns here AND to your CSV / Plan API payload before enabling them.
        # DO NOT add columns that don't exist in the training CSV — they will add
        # only noise (all rows will become "Unknown"/0) and HURT model accuracy.
        #
        # 'season',               # Kharif / Rabi / Spring / Summer
        # 'soil_type',            # Clay / Loam / Sandy / Silt
        # 'farm_age',             # years since farm established (numeric)
        # 'certification_status', # Organic / Conventional / Transitioning
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

    @classmethod
    def get_versioned_model_path(cls):
        """Returns a timestamped versioned path and a version label for rollback support."""
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = os.path.join(cls.MODELS_DIR, f'v_{ts}')
        os.makedirs(version_dir, exist_ok=True)
        return os.path.join(version_dir, 'xgboost_models.pkl'), f'v_{ts}'

    @classmethod
    def list_model_versions(cls):
        """Returns all saved versioned checkpoints, newest first."""
        if not os.path.exists(cls.MODELS_DIR):
            return []
        versions = []
        for name in sorted(os.listdir(cls.MODELS_DIR), reverse=True):
            pkl_path = os.path.join(cls.MODELS_DIR, name, 'xgboost_models.pkl')
            if name.startswith('v_') and os.path.exists(pkl_path):
                versions.append({'version': name, 'path': pkl_path})
        return versions
