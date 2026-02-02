import logging
import pandas as pd
from src.predictor import AssessmentPredictor
from src.config import Config

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("=== Assessment AI Project (Refactored) ===")
    
    app = AssessmentPredictor()
    
    # 1. Train
    print("\n[1] Starting Model Training...")
    app.run_training_pipeline()
    
    # 2. Inference
    print("\n[2] Running Test Prediction...")
    test_input = {
        'country_code': 'IN',
        'crop_name': 'Wheat',
        'partner': 'PartnerA',
        'irrigation': True,
        'hired_workers': 5,
        'area': 10
    }
    
    preds = app.predict(test_input)
    
    # 3. Export
    export_path = Config.PREDICTIONS_FILE
    print(f"\n[3] Exporting to {export_path}...")
    
    data = []
    for k, v in preds.items():
        val = v.get('value') or v.get('suggestedOptions')
        data.append({
            'Indicator': k,
            'Value': val,
            'Confidence': f"{v['confidence']:.1f}%",
            'Source': v['source']
        })
        print(f"  > {k}: {val} ({v['confidence']:.1f}%)")
        
    pd.DataFrame(data).to_csv(export_path, index=False)
    print("\nDone!")

if __name__ == "__main__":
    main()
