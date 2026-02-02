import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'mock_test_uncopied_data.csv')
    MODEL_FILE = os.path.join(MODELS_DIR, 'xgboost_models.pkl')
    PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predicted_assessment.csv')
    
    # Model Hyperparameters
    MODEL_PARAMS = {
        'max_depth': 4,           # Increased depth for better accuracy
        'n_estimators': 100,      # Increased trees
        'learning_rate': 0.05,    # Slower learning for robustness
        'random_state': 42,
        'verbosity': 0
    }
    
    MIN_TRAINING_SAMPLES = 5      # Lowered slightly for POC data
    CONFIDENCE_THRESHOLD = 80.0
    
    # Feature Configuration
    FEATURE_COLUMNS = [
        'country_code', 'crop_name', 'Partner', 'irrigation', 'hired_workers', 'area'
    ]

    # Confidence Thresholds
    CONFIDENCE_AUTO_FILL_THRESHOLD = 50.0
    CONFIDENCE_SUGGEST_THRESHOLD = 0.0  # Set to 0 to always return an answer during development
    
    # Question Metadata (This could be loaded from a JSON file in a real app)
    # Added 'section' for UI grouping logic
    QUESTION_META = {
        "BH-1": {"type": "radio", "options": ["Yes", "No", "In Progress"], "conditional": False, "section": "Biodiversity"},
        "BH-2": {"type": "radio", "options": ["Yes", "No"], "conditional": True, "parent": "BH-1", "parent_value": "Yes", "section": "Biodiversity"},
        "CM-5": {"type": "text", "section": "Crop Management"},
        "WQ-3": {"type": "checkbox", "multi": True, "section": "Water Quality"}
    }
