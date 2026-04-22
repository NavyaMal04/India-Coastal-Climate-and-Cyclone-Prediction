import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Sync with preprocess.py
REGIONS = {
    "Odisha Coast": {"lat": (19, 22), "lon": (84, 87)},
    "Andhra Pradesh": {"lat": (13, 19), "lon": (79, 82)},
    "Tamil Nadu": {"lat": (8, 13), "lon": (78, 80)},
    "Kerala": {"lat": (8, 12), "lon": (75, 77)},
    "Maharashtra": {"lat": (15, 20), "lon": (72, 74)},
    "Gujarat": {"lat": (20, 24), "lon": (68, 72)},
    "West Bengal": {"lat": (21, 23), "lon": (87, 89)}
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_climate_data.csv")

def calculate_cyclone_metrics(data):
    """
    Ensures all required prediction fields exist and are calculated if missing.
    Logic:
    - High risk weather = 70–95
    - Moderate = 40–69
    - Low = 10–39
    """
    region = data.get("region", "Unknown")
    
    # Regional SST Defaults
    SST_DEFAULTS = {
        "Odisha Coast": 28.5,
        "Andhra Pradesh": 29.1,
        "Tamil Nadu": 30.2,
        "Kerala": 29.4,
        "Gujarat": 27.8,
        "Maharashtra": 28.3,
        "West Bengal": 29.0
    }
    
    # Get SST with fallback to regional default then global default
    sst = data.get('sst')
    if pd.isna(sst) or sst is None:
        sst = SST_DEFAULTS.get(region, 28.0)
        
    wind = data.get('wind_speed', 15.0)
    pressure = data.get('pressure', 1010.0)
    
    # Handle NaNs for other fields too
    wind = 15.0 if pd.isna(wind) else wind
    pressure = 1010.0 if pd.isna(pressure) else pressure
    
    # Check if they already exist in data
    prob = data.get('cyclone_prob')
    risk = data.get('risk_level')
    
    if prob is None:
        # Generate probability based on weather conditions (Professional logic)
        prob = 20.0 # Base
        if sst > 30: prob += 30
        if pressure < 1000: prob += 30
        if wind > 40: prob += 15
        
        # Add slight randomness
        prob += np.random.uniform(-5, 5)
        prob = max(10, min(98, prob))
    
    if risk is None:
        if prob >= 70: risk = "High"
        elif prob >= 40: risk = "Moderate"
        else: risk = "Low"
        
    # Ensure all professional fields exist
    return {
        "region": data.get("region", "Unknown"),
        "sst": round(float(sst), 2),
        "wind_speed": round(float(wind), 2),
        "pressure": round(float(pressure), 2),
        "rainfall": round(float(data.get("rainfall", 0.0)), 2),
        "cyclone_prob": round(float(prob), 1),
        "risk_level": risk,
        "latitude": float(data.get("latitude", 0.0)),
        "longitude": float(data.get("longitude", 0.0))
    }

def get_latest_data(region_name):
    """
    Loads data for a specific region. Falls back to dummy data if missing.
    Ensures consistent schema and handles missing prediction columns.
    """
    data = None
    
    if os.path.exists(PROC_DATA_PATH):
        try:
            df = pd.read_csv(PROC_DATA_PATH)
            region_df = df[df['region'] == region_name]
            if not region_df.empty:
                raw_data = region_df.iloc[-1].to_dict()
                data = calculate_cyclone_metrics(raw_data)
        except Exception:
            pass
    
    if data is None:
        # Generate realistic dummy data for demo
        np.random.seed(hash(region_name) % 10**8)
        
        dummy_raw = {
            "region": region_name,
            "sst": 28.5 + np.random.uniform(-1, 2),
            "wind_speed": 15 + np.random.uniform(0, 10),
            "pressure": 1005 + np.random.uniform(-5, 5),
            "rainfall": np.random.uniform(0, 20),
            "latitude": (REGIONS[region_name]["lat"][0] + REGIONS[region_name]["lat"][1]) / 2,
            "longitude": (REGIONS[region_name]["lon"][0] + REGIONS[region_name]["lon"][1]) / 2
        }
        data = calculate_cyclone_metrics(dummy_raw)

    return data

def get_historical_trends(region_name, days=30):
    """
    Generates historical trend data for Plotly charts.
    """
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    np.random.seed(hash(region_name) % 10**8)
    
    data = {
        "datetime": dates,
        "sst": [28 + np.random.uniform(-1, 2) for _ in range(days)],
        "wind_speed": [10 + np.random.uniform(0, 15) for _ in range(days)],
        "pressure": [1000 + np.random.uniform(-5, 10) for _ in range(days)],
        "rainfall": [np.random.exponential(scale=5) for _ in range(days)]
    }
    
    return pd.DataFrame(data)

def get_all_regions_summary():
    """Returns a summary for all regions for map markers."""
    summary = []
    for region in REGIONS.keys():
        summary.append(get_latest_data(region))
    return summary
