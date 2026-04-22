import pandas as pd
import numpy as np
import requests
import os
import logging
from datetime import datetime

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INPUT_FILE = os.path.join(RAW_DIR, "historical_weather_2015_2024.csv")
IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
IBTRACS_FILE = os.path.join(RAW_DIR, "ibtracs_ni_basin.csv")
OUTPUT_FILE = os.path.join(PROC_DIR, "labeled_training_data.csv")
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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def download_ibtracs():
    if not os.path.exists(IBTRACS_FILE):
        logger.info(f"Downloading IBTrACS data from {IBTRACS_URL}...")
        response = requests.get(IBTRACS_URL)
        with open(IBTRACS_FILE, 'wb') as f:
            f.write(response.content)
        logger.info("Download complete.")
    else:
        logger.info("IBTrACS dataset already exists locally.")

def label_data():
    logger.info("--- Starting Data Labeling Pipeline ---")
    
    download_ibtracs()
    
    # Load historical weather
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}. Please run ingestion.py first.")
        return
    
    df_weather = pd.read_csv(INPUT_FILE)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather['year_month'] = df_weather['datetime'].dt.strftime('%Y-%m')
    
    # Load IBTrACS
    # CSV has a dummy row 1 with units, so we skip it or handle with low_memory=False
    df_ib = pd.read_csv(IBTRACS_FILE, low_memory=False, skiprows=[1])
    
    # Filter for Basin: NI and Season: 2015-2024
    df_ib = df_ib[df_ib['BASIN'] == 'NI'].copy()
    df_ib['ISO_TIME'] = pd.to_datetime(df_ib['ISO_TIME'])
    df_ib = df_ib[df_ib['ISO_TIME'].dt.year >= 2015]
    df_ib['year_month'] = df_ib['ISO_TIME'].dt.strftime('%Y-%m')
    
    # Clean Lat/Lon
    df_ib['LAT'] = pd.to_numeric(df_ib['LAT'], errors='coerce')
    df_ib['LON'] = pd.to_numeric(df_ib['LON'], errors='coerce')
    df_ib = df_ib.dropna(subset=['LAT', 'LON'])
    
    logger.info(f"Processing {len(df_weather)} weather records against {len(df_ib)} cyclone track points.")
    
    # Labeling Logic
    # Group IBTrACS by year_month to speed up lookups
    ib_groups = {name: group for name, group in df_ib.groupby('year_month')}
    
    is_cyclone_label = []
    
    for idx, row in df_weather.iterrows():
        ym = row['year_month']
        if ym not in ib_groups:
            is_cyclone_label.append(0)
            continue
            
        # Check distance to all cyclone points in that month
        group = ib_groups[ym]
        distances = haversine(row['latitude'], row['longitude'], group['LAT'].values, group['LON'].values)
        
        if np.any(distances < 200):
            is_cyclone_label.append(1)
        else:
            is_cyclone_label.append(0)
            
    df_weather['is_cyclone'] = is_cyclone_label
    
    # Summary
    pos_count = sum(is_cyclone_label)
    total = len(is_cyclone_label)
    logger.info(f"Labeling complete. Positive samples (cyclone hits): {pos_count}/{total} ({pos_count/total:.2%})")
    
    # Save output
    df_weather.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Labeled training data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    label_data()
