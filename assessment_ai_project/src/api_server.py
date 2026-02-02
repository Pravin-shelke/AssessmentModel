from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from .predictor import AssessmentPredictor
from .config import Config

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Initialize AI Predictor Global Instance
try:
    predictor = AssessmentPredictor()
    # Try loading checkpoint, otherwise need to train
    try:
        predictor.load_checkpoint()
        logger.info("Loaded pre-trained models.")
    except Exception:
        logger.info("No checkpoint found, training models on startup...")
        predictor.run_training_pipeline()
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Assessment AI Model",
        "models_loaded": predictor is not None and len(predictor.models) > 0
    })

@app.route('/api/v1/questions', methods=['GET'])
def get_questions():
    """Returns metadata about which questions are AI-supported"""
    questions = []
    for code, meta in Config.QUESTION_META.items():
        q_data = {
            "indicatorCode": code,
            "type": meta.get("type"),
            "aiAvailable": True,  # In this POC all in Config are supported
            "options": meta.get("options", [])
        }
        questions.append(q_data)
    
    return jsonify({
        "success": True,
        "total_questions": len(questions),
        "questions": questions
    })

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
        
    try:
        data = request.json
        required_fields = ['country', 'crop', 'partner']
        
        # Basic validation
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        # Map frontend keys to backend keys if necessary
        # Frontend: country, crop, partner, irrigation, hired_workers, area
        # Backend Expects: country_code, crop_name, Partner...
        
        input_data = {
            'country_code': data.get('country'),
            'crop_name': data.get('crop'),
            'Partner': data.get('partner'),
            'irrigation': data.get('irrigation'),
            'hired_workers': data.get('hired_workers'),
            'area': data.get('area')
        }
        
        predictions = predictor.predict(input_data)
        
        # Calculate stats for response
        confidences = [p['confidence'] for p in predictions.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        high_conf = len([c for c in confidences if c >= Config.CONFIDENCE_THRESHOLD])
        
        # Middleware: Group by status
        auto_filled_questions = []
        needs_review_questions = [] # Includes 'amber' and 'red'

        for indicator, res in predictions.items():
            meta = Config.QUESTION_META.get(indicator, {})
            item = {
                "indicator": indicator,
                "section": meta.get("section", "General"),
                "prediction": res
            }
            if res.get("ui_status") == "green":
                auto_filled_questions.append(item)
            else:
                needs_review_questions.append(item)

        return jsonify({
            "success": True,
            # Legacy Support
            "statistics": {
                "total_indicators": len(predictions),
                "high_confidence": high_conf,
                "average_confidence": avg_conf
            },
            # New Middleware Structure
            "summary": {
                "total_indicators": len(predictions),
                "auto_filled_count": len(auto_filled_questions),
                "needs_review_count": len(needs_review_questions),
                "saved_time_estimate": f"{len(auto_filled_questions) * 0.5} minutes"
            },
            "groups": {
                "auto_filled": auto_filled_questions,
                "needs_review": needs_review_questions
            },
            "predictions": predictions, # Legacy Support: Frontend expects this key
            "raw_predictions": predictions, # Kept for clarity in new integrations
            "metadata": data
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from mobile emulator/simulator
    app.run(host='0.0.0.0', port=5001, debug=True)
