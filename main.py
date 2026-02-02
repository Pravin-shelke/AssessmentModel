import logging
from assessment_predictor import AssessmentAIPredictor

def main():
    # Initialize predictor
    predictor = AssessmentAIPredictor()
    
    # Load data
    try:
        predictor.load_data('mock_test_data.csv')
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Prepare and train
    predictor.prepare_training_data()
    predictor.train_models()
    
    # Save models
    predictor.save_models()
    
    # Reload models to verify persistence
    predictor.load_models()
    
    # Test Prediction
    print("\n--- Test Prediction ---")
    inputs = {
        'country': 'IN',
        'crop': 'Wheat',
        'partner': 'PartnerA',
        'irrigation': True,
        'hired_workers': 5,
        'area': 10
    }
    predictions = predictor.predict_assessment(**inputs)
    
    # Export results
    predictor.export_predictions_to_csv(predictions)
    
    # Print sample output
    print(f"Predictions for {inputs}:")
    for k, v in list(predictions.items())[:5]: # Show first 5
        print(f"{k}: {v}")
    
    print("\nCheck 'predicted_assessment.csv' for full results.")
    predictor.report_coverage()

if __name__ == "__main__":
    main()
