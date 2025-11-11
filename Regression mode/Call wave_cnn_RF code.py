# -*- coding: utf-8 -*-
"""
 Wave-CNN-RF Model Prediction
 Simplified version for Wave-CNN-RF model prediction
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Define paths
MODEL_PATH = r"xxxx\Wave_CNN_RF.pkl"
OUTPUT_PATH = r"xxxx\pred_results_wave_cnn_rf.csv"

# 2. Define feature columns (same as during training)
CONT_FEATURES = [
    'PGA', 'Building Floor Area', 'mag', 'Number of Storeys',
    'Rated Building Value', 'Vs30', 'r_rup'
]
CAT_FEATURES = [
    'Liquefaction', 'Wall and Roof Materials (Code)',
    'Wall and Roof Materials2 (Code)', 'Occupancy Type(code)',
    'Construction Category', 'ev_depth'
]

# 3. Input data (single sample)
input_data = {
    "PGA": 10,
    "Building Floor Area": 90,
    "mag": 5.36,
    "Number of Storeys": 1,
    "Rated Building Value": 103000,
    "Vs30": 293,
    "r_rup": 206,
    "Liquefaction": 0,
    "Wall and Roof Materials (Code)": 1,
    "Wall and Roof Materials2 (Code)": 9,
    "Occupancy Type(code)": 1,
    "Construction Category": 0,
    "ev_depth": 37,
}

# 4. Load model and make prediction
def predict_simple():
    try:
        # Load model
        print("📦 Loading Wave-CNN-RF model...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Wave-CNN-RF model loaded successfully")
        
        # Prepare input data
        df_input = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = set(CONT_FEATURES + CAT_FEATURES) - set(df_input.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        print("📄 Input data:")
        print(df_input[CONT_FEATURES + CAT_FEATURES])
        
        # Prediction - this is a regression model, only has predict method
        print("🎯 Making prediction...")
        predictions = model.predict(df_input)
        
        # Output results
        result_df = df_input.copy()
        result_df["pred_lr"] = predictions  # Regression prediction value, not classification probability
        
        print("\n🎯 Prediction result:")
        print(f"Predicted loss rate (lr): {predictions[0]:.6f}")
        print(f"Predicted loss rate percentage: {predictions[0]*100:.2f}%")
        print("\nComplete results:")
        print(result_df[['PGA', 'mag', 'r_rup', 'pred_lr']])
        
        # Save results
        result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"💾 Results saved to: {OUTPUT_PATH}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ An error occurred during prediction: {str(e)}")
        return None

# 5. Direct call
if __name__ == "__main__":
    print("=" * 50)
    print("Wave-CNN-RF Model Prediction")
    print("=" * 50)
    
    # Single prediction
    result = predict_simple()
    
    if result is not None:
        print("\n✅ Single prediction completed!")
        
        # Ask if batch prediction should be done
        user_input = input("\nDo you want to perform batch prediction? (Enter CSV file path or press Enter to skip): ").strip()
        if user_input:
            predict_batch(user_input)
    else:
        print("\n❌ Prediction failed!")
