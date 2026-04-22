import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime

# --- Configuration setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
INPUT_FILE = os.path.join(PROC_DIR, "master_climate_data.csv")
OUTPUT_FILE = os.path.join(PROC_DIR, "featured_climate_data.csv")
PARAMS_FILE = os.path.join(PROC_DIR, "normalization_params.json")
LOG_FILE = os.path.join(PROC_DIR, "preprocessing_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_features():
    logger.info("--- Starting Feature Engineering Pipeline ---")
    
    # Task 1 - Load and Inspect
    logger.info("Task 1: Loading input file and inspecting data...")
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    print("\n--- Task 1: Data Inspection ---")
    print(f"Shape of Dataframe: {df.shape}")
    print("\nNull count per column:")
    print(df.isnull().sum())
    
    print("\nMin, Max, Mean for target columns:")
    for col in ['rainfall', 'sst', 'wind_speed', 'pressure']:
        if col in df.columns:
            print(f"{col:<12} | Min: {df[col].min():.2f} | Max: {df[col].max():.2f} | Mean: {df[col].mean():.2f}")
            
    print("\nRow count per region:")
    print(df['region'].value_counts())
    
    # Task 2 - Handle Nulls
    logger.info("Task 2: Handling null values...")
    cols_to_fill = ['sst', 'wind_speed', 'pressure', 'rainfall'] # Added rainfall just in case
    
    fill_counts = {col: 0 for col in cols_to_fill}
    
    for col in cols_to_fill:
        if col in df.columns:
            null_mask_initial = df[col].isnull()
            fill_counts[col] = null_mask_initial.sum()
            
            # Fill with regional mean
            df[col] = df.groupby('region')[col].transform(lambda x: x.fillna(x.mean()))
            
            # Fallback to global mean
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
                
            logger.info(f"Filled {fill_counts[col]} nulls in {col}")
            
    print("\n--- Task 2: Post-Fill Null Check ---")
    nulls_after = df.isnull().sum()
    print(nulls_after)
    if nulls_after[['sst', 'wind_speed', 'pressure', 'rainfall']].sum() > 0:
        logger.error("Null values remain after filling!")
        
    # Task 3 - Engineer Features
    logger.info("Task 3: Engineering features...")
    
    # 3a. Cyclone formation threshold flags
    df['sst_above_threshold'] = (df['sst'] > 26.5).astype(int)
    df['sst_danger_zone'] = (df['sst'] > 28.5).astype(int)
    
    regional_pressure_mean = df.groupby('region')['pressure'].transform('mean')
    df['pressure_anomaly'] = df['pressure'] - regional_pressure_mean
    
    df['low_pressure_flag'] = (df['pressure'] < 1010).astype(int)
    df['high_wind_flag'] = (df['wind_speed'] > 5.0).astype(int)
    
    # 3b. Rainfall classification
    conditions = [
        (df['rainfall'] < 0.1),
        (df['rainfall'] >= 0.1) & (df['rainfall'] <= 0.3),
        (df['rainfall'] > 0.3) & (df['rainfall'] <= 0.6),
        (df['rainfall'] > 0.6)
    ]
    choices = [0, 1, 2, 3]
    df['rainfall_intensity'] = np.select(conditions, choices, default=0)
    df['heavy_rainfall_flag'] = (df['rainfall'] > 0.5).astype(int)
    
    # 3c. Composite risk score
    df['risk_score'] = 0
    df.loc[df['sst'] > 26.5, 'risk_score'] += 2
    df.loc[df['sst'] > 28.5, 'risk_score'] += 1
    df.loc[df['pressure'] < 1010, 'risk_score'] += 3
    df.loc[df['pressure_anomaly'] < -2, 'risk_score'] += 1
    df.loc[df['wind_speed'] > 5.0, 'risk_score'] += 2
    df.loc[df['rainfall'] > 0.5, 'risk_score'] += 1
    
    def calculate_risk_level(score):
        if score <= 2: return 'Low'
        elif score <= 5: return 'Moderate'
        else: return 'High'
    df['risk_level'] = df['risk_score'].apply(calculate_risk_level)
    
    # 3d. Cyclone probability score
    df['cyclone_probability'] = np.clip(df['risk_score'] / 10.0, 0.0, 1.0)
    
    # 3e. Geographic features
    df['distance_from_equator'] = df['latitude'].abs()
    df['bay_of_bengel_flag'] = ((df['longitude'] > 80) & (df['latitude'] > 8)).astype(int)
    df['arabian_sea_flag'] = (df['longitude'] < 77).astype(int)
    
    # 3f. Normalized features for ML input
    min_max_params = {}
    for col in ['sst', 'wind_speed', 'pressure', 'rainfall']:
        vmin = df[col].min()
        vmax = df[col].max()
        
        # Handle case where vmin == vmax
        denom = vmax - vmin if vmax > vmin else 1.0
        
        # Explicit col names requested: sst_norm, wind_norm, pressure_norm, rainfall_norm
        col_name = f"{col.replace('_speed', '')}_norm"
        df[col_name] = (df[col] - vmin) / denom
        
        min_max_params[col] = {"min": float(vmin), "max": float(vmax)}
        
    min_max_params["risk_score_max"] = 10
    min_max_params["created_at"] = datetime.now().isoformat()
    min_max_params["note"] = "Use identical min/max on live API data during inference"
    
    with open(PARAMS_FILE, 'w') as f:
        json.dump(min_max_params, f, indent=2)
        
    # Task 4 - Assemble Final Output DataFrame
    logger.info("Task 4: Assembling final dataframe...")
    final_cols = [
        'region', 'latitude', 'longitude',
        'rainfall', 'sst', 'wind_speed', 'pressure',
        'sst_above_threshold', 'sst_danger_zone',
        'pressure_anomaly', 'low_pressure_flag',
        'high_wind_flag',
        'rainfall_intensity', 'heavy_rainfall_flag',
        'risk_score', 'risk_level',
        'cyclone_probability',
        'distance_from_equator',
        'bay_of_bengel_flag', 'arabian_sea_flag',
        'sst_norm', 'wind_norm', 'pressure_norm', 'rainfall_norm'
    ]
    df = df[final_cols]
    
    logger.info(f"Saving to {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Task 5 - Validation Report
    print("\n--- Task 5: Validation Report ---")
    print(f"1. Shape of featured_climate_data.csv: {df.shape}")
    
    print("\n2. Null count per column:")
    print(df.isnull().sum())
    
    print("\n3. Risk level distribution:")
    risk_counts = df['risk_level'].value_counts()
    for level, count in risk_counts.items():
        print(f"   {level}: {count} ({(count/len(df))*100:.1f}%)")
        
    print("\n4. Rainfall intensity distribution (0, 1, 2, 3):")
    print(df['rainfall_intensity'].value_counts().sort_index())
    
    print("\n5. Cyclone probability:")
    cp = df['cyclone_probability']
    print(f"   Min: {cp.min():.2f} | Max: {cp.max():.2f} | Mean: {cp.mean():.2f} | Median: {cp.median():.2f}")
    
    print("\n6. SST:")
    sst = df['sst']
    print(f"   Mean: {sst.mean():.2f}")
    print(f"   % above 26.5: {(df['sst_above_threshold'].sum()/len(df))*100:.1f}%")
    print(f"   % above 28.5: {(df['sst_danger_zone'].sum()/len(df))*100:.1f}%")
    
    print("\n7. Water Body Context:")
    print(f"   Bay of Bengal rows: {df['bay_of_bengel_flag'].sum()}")
    print(f"   Arabian Sea rows: {df['arabian_sea_flag'].sum()}")
    
    print("\n8. Top 5 highest risk_score rows:")
    top_5 = df.nlargest(5, 'risk_score')
    for _, r in top_5.iterrows():
        print(f"   {r['region']:<15} | Lat: {r['latitude']:>5.1f} | Lon: {r['longitude']:>5.1f} | " +
              f"SST: {r['sst']:>5.2f} | Pres: {r['pressure']:>7.2f} | Wind: {r['wind_speed']:>4.2f} | " +
              f"Score: {r['risk_score']} | Level: {r['risk_level']}")
              
    logger.info("--- Feature Engineering Pipeline Complete ---")

if __name__ == '__main__':
    run_features()
