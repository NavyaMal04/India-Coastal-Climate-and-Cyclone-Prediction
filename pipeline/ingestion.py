import requests
import pandas as pd
import os
import logging
import time
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INPUT_FILE = os.path.join(PROC_DIR, "master_climate_data.csv")
OUTPUT_FILE = os.path.join(RAW_DIR, "historical_weather_2015_2024.csv")
LOG_FILE = os.path.join(PROC_DIR, "preprocessing_log.txt")

os.makedirs(RAW_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_nasa_power_monthly():
    logger.info("--- Starting Historical Data Ingestion (NASA POWER Monthly Resume Mode) ---")
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return
    
    master_df = pd.read_csv(INPUT_FILE)
    locations = master_df[['region', 'latitude', 'longitude']].drop_duplicates()
    
    # Check for existing data to resume
    processed_coords = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if not existing_df.empty:
                processed_coords = set(zip(existing_df['latitude'], existing_df['longitude']))
                logger.info(f"Resume mode: Found {len(processed_coords)} already processed coordinates.")
        except Exception as e:
            logger.warning(f"Could not read existing output file for resume: {e}")

    # Variables: TS (Skin Temp), WS10M (Wind), PS (Pressure), PRECTOTCORR (Precip)
    params_str = "TS,WS10M,PS,PRECTOTCORR"
    
    # Open file in append mode if resuming, else write mode
    mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    header = not os.path.exists(OUTPUT_FILE)

    for idx, loc in tqdm(locations.iterrows(), total=len(locations), desc="Fetching Locations"):
        lat, lon = loc['latitude'], loc['longitude']
        region = loc['region']
        
        if (lat, lon) in processed_coords:
            continue
            
        url = (
            f"https://power.larc.nasa.gov/api/temporal/monthly/point?"
            f"parameters={params_str}&community=AG&longitude={lon}&latitude={lat}"
            f"&start=2015&end=2024&format=JSON"
        )
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data_json = response.json()
                features = data_json['properties']['parameter']
                
                dates_keys = [k for k in features['TS'].keys() if not k.endswith('13') and k != 'ANN']
                loc_rows = []
                
                for d_key in dates_keys:
                    year = int(d_key[:4])
                    month = int(d_key[4:])
                    dt = pd.to_datetime(f"{year}-{month:02d}-01")
                    
                    loc_rows.append({
                        "datetime": dt,
                        "region": region,
                        "latitude": lat,
                        "longitude": lon,
                        "sst": features['TS'].get(d_key),
                        "pressure": features['PS'].get(d_key),
                        "rainfall": features['PRECTOTCORR'].get(d_key),
                        "wind_speed": features['WS10M'].get(d_key)
                    })
                
                if loc_rows:
                    batch_df = pd.DataFrame(loc_rows)
                    # Convert Pressure to hPa (NASA PS is kPa)
                    batch_df['pressure'] = batch_df['pressure'] * 10.0
                    
                    batch_df.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
                    header = False # Only write header once
                    mode = 'a' # Subsequent writes must append
                    
            else:
                logger.error(f"Failed to fetch for {lat}, {lon}: Status {response.status_code}")
                time.sleep(5) # Backoff on error
                
        except Exception as e:
            logger.error(f"Error fetching for {lat}, {lon}: {e}")
            
        time.sleep(0.5) # Throttling for NASA API

    logger.info(f"Ingestion process finished. Results in {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_nasa_power_monthly()
