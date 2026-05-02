import requests
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import time
from database.db import get_connection

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_READINGS_DIR = os.path.join(BASE_DIR, "data", "raw", "live_readings")
os.makedirs(LIVE_READINGS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

REGIONS = {
    "Andhra Pradesh":  {"lat": 15.9, "lon": 80.5},
    "Gujarat":         {"lat": 22.3, "lon": 69.7},
    "Kerala":          {"lat": 9.5,  "lon": 76.3},
    "Maharashtra":     {"lat": 17.5, "lon": 73.2},
    "Odisha Coast":    {"lat": 20.5, "lon": 85.8},
    "Tamil Nadu":      {"lat": 10.8, "lon": 79.8},
    "West Bengal":     {"lat": 21.6, "lon": 87.9}
}

def get_fallback_reading(region):
    """Retrieves the latest stored reading from DB as fallback"""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            SELECT rainfall, wind_speed, pressure, sst 
            FROM live_readings 
            WHERE region = ? 
            ORDER BY timestamp DESC LIMIT 1
        ''', (region,))
        row = c.fetchone()
        conn.close()
        if row:
            return dict(row)
    except Exception as e:
        logger.error(f"Fallback retrieval failed: {e}")
    
    return {"rainfall": 0.0, "wind_speed": 0.0, "pressure": 1010.0, "sst": 28.0}

def fetch_nasa_power(lat, lon):
    """Fetches climate data from NASA POWER API"""
    logger.info(f"Attempting NASA POWER fetch for {lat}, {lon}")
    try:
        # Get last 24 hours
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        url = (
            f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
            f"parameters=T2M,PRECTOTCORR,WS10M,PS&"
            f"community=RE&longitude={lon}&latitude={lat}&"
            f"start={start_date.strftime('%Y%m%d')}&"
            f"end={end_date.strftime('%Y%m%d')}&format=JSON"
        )
        
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get('properties', {}).get('parameter', {})
            
            # Extract latest non-null values
            # PS (Surface Pressure) is in kPa, convert to hPa (1 kPa = 10 hPa)
            # T2M (Temperature) is in C
            # PRECTOTCORR (Precipitation) is in mm/hr
            # WS10M (Wind Speed) is in m/s, convert to knots (1 m/s = 1.94384 knots)
            
            def get_latest(param_dict):
                vals = [v for v in param_dict.values() if v is not None and v != -999]
                return vals[-1] if vals else None

            temp = get_latest(features.get('T2M', {}))
            precip = sum([v for v in features.get('PRECTOTCORR', {}).values() if v is not None and v != -999])
            wind_ms = get_latest(features.get('WS10M', {}))
            pressure_kpa = get_latest(features.get('PS', {}))
            
            return {
                "sst": temp if temp else 28.0,
                "rainfall": precip if precip else 0.0,
                "wind_speed": wind_ms * 1.94384 if wind_ms else 0.0,
                "pressure": pressure_kpa * 10 if pressure_kpa else 1010.0,
                "source": "NASA-POWER"
            }
    except Exception as e:
        logger.error(f"NASA POWER API failed: {e}")
    return None

def fetch_live_data_for_point(lat, lon, region_name="Unknown"):
    """
    Main aggregator for live weather data. 
    Tries Open-Meteo, then NASA POWER, then fallback.
    """
    current_time = datetime.utcnow()
    
    # Defaults
    final_data = {
        "rainfall": 0.0,
        "wind_speed": 0.0,
        "pressure": 1010.0,
        "sst": 28.0,
        "source": "Fallback"
    }

    # 1. Try Open-Meteo Standard + Marine
    try:
        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=windspeed_10m,surface_pressure,precipitation,temperature_2m"
            f"&past_days=1&windspeed_unit=kn"
        )
        resp = requests.get(forecast_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get('hourly', {})
            if hourly:
                times = pd.to_datetime(hourly['time'])
                now_idx = (times <= current_time).sum() - 1
                if now_idx >= 0:
                    final_data["rainfall"] = sum([x for x in hourly.get('precipitation', [])[max(0, now_idx-23):now_idx+1] if x is not None])
                    final_data["wind_speed"] = hourly.get('windspeed_10m', [])[now_idx] or 0.0
                    final_data["pressure"] = hourly.get('surface_pressure', [])[now_idx] or 1010.0
                    final_data["sst"] = hourly.get('temperature_2m', [])[now_idx] or 28.0
                    final_data["source"] = "Open-Meteo"
        
        # Try for better SST from Marine API
        marine_url = (
            f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}"
            f"&hourly=sea_surface_temperature&past_days=1"
        )
        m_resp = requests.get(marine_url, timeout=10)
        if m_resp.status_code == 200:
            m_data = m_resp.json()
            m_hourly = m_data.get('hourly', {})
            if m_hourly:
                sst_vals = [v for v in m_hourly.get('sea_surface_temperature', []) if v is not None]
                if sst_vals:
                    final_data["sst"] = sst_vals[-1]
    except Exception as e:
        logger.warning(f"Open-Meteo primary fetch failed for {region_name}: {e}")

    # 2. If Open-Meteo failed or returned defaults, try NASA POWER
    if final_data["source"] == "Fallback":
        nasa_data = fetch_nasa_power(lat, lon)
        if nasa_data:
            final_data = nasa_data

    # 3. Final Fallback to DB if still failing
    if final_data["source"] == "Fallback":
        db_fallback = get_fallback_reading(region_name)
        final_data.update(db_fallback)
        final_data["source"] = "Database-Fallback"

    return final_data

def fetch_live_data():
    """Iterates through all regions and fetches live data"""
    logger.info("--- STARTING LIVE DATA INGESTION PIPELINE ---")
    
    current_time = datetime.utcnow()
    timestamp_str = current_time.strftime("%Y-%m-%d_%H")
    
    results = {}
    raw_results = []
    
    for region, coords in REGIONS.items():
        logger.info(f"Processing region: {region}")
        lat, lon = coords['lat'], coords['lon']
        
        data = fetch_live_data_for_point(lat, lon, region)
        
        region_data = {
            "region": region,
            "latitude": float(lat),
            "longitude": float(lon),
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "rainfall_mm": float(data['rainfall']),
            "wind_speed_knots": float(data['wind_speed']),
            "pressure_hpa": float(data['pressure']),
            "sst_celsius": float(data['sst']),
            "fetch_source": data['source']
        }
        
        results[region] = region_data
        raw_results.append(region_data)
        time.sleep(0.5) # Avoid rate limiting
        
    # Save to CSV for backup
    df = pd.DataFrame(raw_results)
    out_file = os.path.join(LIVE_READINGS_DIR, f"live_{timestamp_str}.csv")
    df.to_csv(out_file, index=False)
    
    logger.info(f"Ingestion complete. {len(results)} regions processed.")
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_live_data()
