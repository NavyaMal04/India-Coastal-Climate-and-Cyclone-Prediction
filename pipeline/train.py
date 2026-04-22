import pandas as pd
import numpy as np
import os
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUT_FILE = os.path.join(PROC_DIR, "labeled_training_data.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "cyclone_rf.pkl")
LOG_FILE = os.path.join(PROC_DIR, "preprocessing_log.txt")

os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def engineer_training_features(df):
    """Applies similar logic as pipeline/features.py for consistency"""
    logger.info("Engineering features for training...")
    
    # Thresholds
    df['sst_above_threshold'] = (df['sst'] > 26.5).astype(int)
    df['sst_danger_zone'] = (df['sst'] > 28.5).astype(int)
    
    # Regional Pressure Anomaly
    reg_means = df.groupby('region')['pressure'].transform('mean')
    df['pressure_anomaly'] = df['pressure'] - reg_means
    
    df['low_pressure_flag'] = (df['pressure'] < 1010).astype(int)
    df['high_wind_flag'] = (df['wind_speed'] > 5.0).astype(int)
    
    # Geography
    df['distance_from_equator'] = df['latitude'].abs()
    df['bay_of_bengal_flag'] = ((df['longitude'] > 80) & (df['latitude'] > 8)).astype(int)
    df['arabian_sea_flag'] = (df['longitude'] < 77).astype(int)
    
    return df

def train_model():
    logger.info("--- Starting ML Training Pipeline ---")
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}. Please run labeling.py first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Impute any remaining nulls (from NASA fetch misses)
    for col in ['sst', 'wind_speed', 'pressure', 'rainfall']:
        df[col] = df.groupby('region')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean()) # Global fallback
        
    # 2. Engineer Features
    df = engineer_training_features(df)
    
    # 3. Define Features and Target
    feature_cols = [
        'sst', 'wind_speed', 'pressure', 'rainfall',
        'sst_above_threshold', 'sst_danger_zone',
        'pressure_anomaly', 'low_pressure_flag',
        'high_wind_flag',
        'distance_from_equator', 'bay_of_bengal_flag', 'arabian_sea_flag'
    ]
    X = df[feature_cols]
    y = df['is_cyclone']
    
    # 4. Check Class Balance
    pos_ratio = y.mean()
    logger.info(f"Class imbalance: {pos_ratio:.2%} positive examples.")
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Apply SMOTE if needed and available
    if pos_ratio < 0.10 and SMOTE is not None:
        logger.info("Applying SMOTE oversampling to balance classes...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"New training set size: {len(X_train)}")
    elif pos_ratio < 0.10 and SMOTE is None:
        logger.warning("Positive class < 10% but 'imblearn' not installed. Skipping SMOTE.")
    
    # 7. Train Random Forest
    logger.info("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    # 8. Evaluate
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    logger.info("\n--- Model Evaluation ---")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # 9. Save Model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(rf, f)
    logger.info(f"Model saved to {MODEL_FILE}")
    
    # Feature Importance
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info(f"\nTop Features:\n{importances.head(10)}")

if __name__ == "__main__":
    train_model()
