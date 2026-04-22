import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib
import os

def load_and_label_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check for required columns
    required_cols = ['sst', 'wind_speed', 'pressure', 'rainfall']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Handle NaNs (especially SST which we fixed in UI, but should fix in data too)
    df['sst'] = df['sst'].fillna(df['sst'].mean())
    df['rainfall'] = df['rainfall'].fillna(0)
    
    # Create Labels: Cyclone = 1 if (sst > 28 AND pressure < 1000 AND wind_speed > 45)
    df['target'] = ((df['sst'] > 28) & (df['pressure'] < 1000) & (df['wind_speed'] > 45)).astype(int)
    
    # Synthetic Augmentation for Demo (if no cyclones in real data)
    if df['target'].sum() == 0:
        print("Warning: No cyclones found in real data. Adding synthetic samples for model training...")
        synthetic_samples = [
            {'sst': 30.5, 'wind_speed': 55.0, 'pressure': 990.0, 'rainfall': 150.0, 'target': 1},
            {'sst': 29.8, 'wind_speed': 62.0, 'pressure': 985.0, 'rainfall': 200.0, 'target': 1},
            {'sst': 31.0, 'wind_speed': 48.0, 'pressure': 995.0, 'rainfall': 120.0, 'target': 1},
            {'sst': 28.5, 'wind_speed': 70.0, 'pressure': 975.0, 'rainfall': 300.0, 'target': 1}
        ]
        df = pd.concat([df, pd.DataFrame(synthetic_samples)], ignore_index=True)
    
    print(f"Class distribution: \n{df['target'].value_counts()}")
    return df

def train():
    data_path = 'data/processed/master_climate_data.csv'
    if not os.path.exists(data_path):
        print("Error: Processed data file not found.")
        return

    df = load_and_label_data(data_path)
    
    features = ['sst', 'wind_speed', 'pressure', 'rainfall']
    X = df[features]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_f1 = 0
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec}
        print(f"{name} Results - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
        
        # We prioritize Recall/Precision for safety-critical apps
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        if f1 >= best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with F1-score: {best_f1:.4f}")
    
    # Save best model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names for inference
    joblib.dump(features, os.path.join(model_dir, 'features.pkl'))

if __name__ == "__main__":
    train()
