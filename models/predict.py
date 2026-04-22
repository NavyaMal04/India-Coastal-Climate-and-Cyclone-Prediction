import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = 'models/model.pkl'
FEATURES_PATH = 'models/features.pkl'

def get_cyclone_probability(sst, wind_speed, pressure, rainfall):
    """
    Predicts the probability of a cyclone based on input meteorological features.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        # Fallback to simple heuristic if model not trained/found
        # This ensures the dashboard doesn't crash
        prob = 0
        if sst > 28: prob += 20
        if wind_speed > 30: prob += 40
        if pressure < 1005: prob += 30
        return min(prob, 95.0)

    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        
        # Prepare input data
        input_data = pd.DataFrame([[sst, wind_speed, pressure, rainfall]], columns=features)
        
        # Get probability for class 1 (Cyclone)
        # Some models might not have predict_proba if only one class was seen during training
        # but our training script now ensures two classes or handles it.
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_data)
            # If the model only saw class 0 during training, it might return a 1D array or 1-col array
            if probs.shape[1] > 1:
                prob = probs[0][1] * 100
            else:
                prob = 0.0 # Only class 0 was learned
        else:
            prob = float(model.predict(input_data)[0]) * 100
            
        return float(prob)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0

if __name__ == "__main__":
    # Test prediction
    test_prob = get_cyclone_probability(30.5, 55.0, 990.0, 150.0)
    print(f"Test Cyclone Probability: {test_prob:.2f}%")
