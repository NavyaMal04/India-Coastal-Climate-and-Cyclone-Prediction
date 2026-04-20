import os
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
import h5py

# --- Configuration setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_FILE = os.path.join(PROC_DIR, "preprocessing_log.txt")

os.makedirs(PROC_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Regions Setup ---
REGIONS = {
    "Odisha Coast": {"lat": (19, 22), "lon": (84, 87)},
    "Andhra Pradesh": {"lat": (13, 19), "lon": (79, 82)},
    "Tamil Nadu": {"lat": (8, 13), "lon": (78, 80)},
    "Kerala": {"lat": (8, 12), "lon": (75, 77)},
    "Maharashtra": {"lat": (15, 20), "lon": (72, 74)},
    "Gujarat": {"lat": (20, 24), "lon": (68, 72)},
    "West Bengal": {"lat": (21, 23), "lon": (87, 89)}
}

def map_region(lat, lon):
    for name, bounds in REGIONS.items():
        if bounds["lat"][0] <= lat <= bounds["lat"][1] and bounds["lon"][0] <= lon <= bounds["lon"][1]:
            return name
    return "Other"

# --- 1. Load Dataset ---
def load_dataset(filepath):
    logger.info(f"Loading {os.path.basename(filepath)}...")
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.nc', '.nc4']:
        ds = xr.open_dataset(filepath)
        df = ds.to_dataframe().reset_index()
        logger.info(f"Loaded NetCDF. Shape: {df.shape}")
        
    elif ext in ['.hdf5', '.h5']:
        # Custom logic for IMERG or similar HDF5 grid files
        with h5py.File(filepath, 'r') as f:
            if 'Grid' in f.keys():
                lat = f['Grid']['lat'][:]
                lon = f['Grid']['lon'][:]
                precip = f['Grid']['precipitation'][0, :, :] 
                
                # Create meshgrid
                lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='ij')
                
                df = pd.DataFrame({
                    'longitude': lon_grid.flatten(),
                    'latitude': lat_grid.flatten(),
                    'precipitation': precip.flatten(),
                    'time': pd.to_datetime("2025-09-01 00:00:00") 
                })
                logger.info(f"Loaded HDF5. Shape: {df.shape}")
    elif ext == '.csv':
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df.get('time', pd.Timestamp.now()))
        logger.info(f"Loaded CSV. Shape: {df.shape}")
    else:
        logger.warning(f"Unknown format for {filepath}")
        return pd.DataFrame()

    # Find the time column dynamically
    time_col = next((c for c in df.columns if c.lower() in ['time', 'datetime', 'date']), None)
    
    if time_col:
        logger.info(f"Exact datetime column name: '{time_col}'")
        logger.info(f"Sample of 5 raw datetime values: \n{df[time_col].dropna().head(5).tolist()}")
        
        # Step B: Parse all datetime columns to pandas UTC datetime
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        
        # Attempt to infer frequency
        unique_dates = df[time_col].drop_duplicates().sort_values()
        if len(unique_dates) > 1:
            diffs = unique_dates.diff().dropna()
            logger.info(f"Inferred frequency based on gap: {diffs.value_counts().index[0]}")
        else:
            logger.info("Only 1 unique timestamp; frequency inference not possible, assuming snapshot.")
    else:
        logger.warning("No datetime column found in this dataset!")
        
    return df

# --- 2. Standardize Columns ---
def standardize_columns(df):
    col_map = {}
    
    for c in df.columns:
        cl = c.lower()
        if cl in ['time', 'datetime', 'date']:
            col_map[c] = 'datetime'
        elif cl in ['lat', 'latitude']:
            col_map[c] = 'latitude'
        elif cl in ['lon', 'longitude']:
            col_map[c] = 'longitude'
        elif cl in ['sst', 'sea_surface_temp']:
            col_map[c] = 'sst'
        elif cl in ['slp', 'pressure']:
            col_map[c] = 'pressure'
        elif cl in ['precipitation', 'precip', 'rainfall']:
            col_map[c] = 'rainfall'

    df = df.rename(columns=col_map)
    
    # Check. Have we u/v wind?
    if 'U10M' in df.columns and 'V10M' in df.columns:
        df['wind_speed'] = np.sqrt(df['U10M']**2 + df['V10M']**2)
        
    return df

# --- Filtering and Grid Regularization ---
def pre_filter_india(df):
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df[(df['latitude'] >= 5) & (df['latitude'] <= 28) & 
                (df['longitude'] >= 65) & (df['longitude'] <= 95)].copy()
        
        # Round lat/lon to 0.5 degrees for consistent multi-dataset joining
        df['latitude'] = (df['latitude'] * 2).round() / 2
        df['longitude'] = (df['longitude'] * 2).round() / 2
    return df

# --- 3. Handle outliers ---
def handle_outliers(df):
    # Fix for Problem 1: Detect Pascals and convert to hPa
    if 'pressure' in df.columns:
        p_median = df['pressure'].median()
        if p_median > 1100:
            logger.info(f"Pressure median is {p_median:.1f} (>1100). Converting Pa to hPa.")
            df['pressure'] = df['pressure'] / 100.0

    bounds = {
        'sst': (0, 35),
        'wind_speed': (0, 200),
        'pressure': (850, 1050),
        'rainfall': (0, 500)
    }
    for col, (vmin, vmax) in bounds.items():
        if col in df.columns:
            outliers = ((df[col] < vmin) | (df[col] > vmax)) & df[col].notna()
            outlier_count = outliers.sum()
            if outlier_count > 0:
                logger.warning(f"Clipping {outlier_count} outliers in {col}")
                df[col] = df[col].clip(lower=vmin, upper=vmax)
    return df

# --- Step A: Consolidate By Location ---
def consolidate_by_location(df):
    logger.info("Executing Option A: Forward-fill within region (multi-season feature set consolidation)")
    result_rows = []
    
    # Sort to ensure chronological order before taking iloc[-1]
    if 'datetime' in df.columns:
        df = df.sort_values(by='datetime')
        
    for (region, lat, lon), group in df.groupby(['region', 'latitude', 'longitude']):
        row = {
            'region': region,
            'latitude': lat,
            'longitude': lon,
        }
        # Take best available value for each variable
        for col in ['rainfall', 'sst', 'wind_speed', 'pressure']:
            if col in group.columns:
                non_null = group[col].dropna()
                row[col] = non_null.iloc[-1] if len(non_null) > 0 else None
            else:
                row[col] = None
        
        # Record which dates contributed data
        dates_used = {}
        for col in ['rainfall', 'sst', 'wind_speed', 'pressure']:
            if col in group.columns:
                non_null_rows = group[group[col].notna()]
                if len(non_null_rows) > 0:
                    dates_used[col] = str(non_null_rows['datetime'].iloc[-1])[:7]  # Get YYYY-MM
        
        row['data_period'] = str(dates_used)
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)


def run_pipeline():
    logger.info("Starting preprocessing pipeline...")
    
    all_dfs = []
    
    # Process each raw file
    for f in glob.glob(os.path.join(RAW_DIR, "*.*")):
        df = load_dataset(f)
        if df.empty: continue
        
        df = standardize_columns(df)
        df = pre_filter_india(df)
        df['region'] = df.apply(lambda row: map_region(row['latitude'], row['longitude']), axis=1)
        df = handle_outliers(df)
        
        agg_cols = [c for c in ['sst', 'wind_speed', 'pressure', 'rainfall'] if c in df.columns]
        if agg_cols:
            # Drop purely "Other" to keep merges clean for coastal India
            df = df[df['region'] != "Other"].copy()
            if df.empty:
                continue

            # First mean aggregate by exact spatial point to deduplicate
            df = df.groupby(['datetime', 'region', 'latitude', 'longitude'])[agg_cols].mean().reset_index()
            
            # Resample to Monthly Start independently to align time grids
            df = df.set_index('datetime').groupby(['region', 'latitude', 'longitude']).resample('MS').mean().reset_index()
            
            logger.info(f"After 'MS' resampling, dataset shape: {df.shape}, Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
            all_dfs.append(df)
            
    if not all_dfs:
        logger.error("No valid data processed. Exiting.")
        return
        
    # Merge Master DF on datetime + spatial coordinates
    logger.info("Merging datasets...")
    master_df = all_dfs[0]
    for i, nxt_df in enumerate(all_dfs[1:], start=1):
        master_df = pd.merge(master_df, nxt_df, 
                             on=['datetime', 'region', 'latitude', 'longitude'], 
                             how='outer')
                             
    nulls_after = len(master_df) - master_df.dropna(subset=['rainfall', 'sst', 'wind_speed', 'pressure'], how='any').shape[0]
    logger.info(f"Total rows after outer merge: {len(master_df)}. Rows with at least one NaN across targets: {nulls_after}")
                             
    # CONSOLIDATE BY LOCATION FIX
    master_df = consolidate_by_location(master_df)

    out_path = os.path.join(PROC_DIR, "master_climate_data.csv")
    master_df.to_csv(out_path, index=False)
    logger.info(f"Saved master file to {out_path}\n")

    # --- Validation Table ---
    print("\n" + "="*70)
    print("FINAL VALIDATION REPORT (POST-CONSOLIDATION)")
    print("="*70)
    
    cols_to_check = ['rainfall', 'sst', 'wind_speed', 'pressure']
    
    total_len = len(master_df)
    unique_locations = len(master_df[['region', 'latitude', 'longitude']].drop_duplicates())
    
    print(f"Total rows: {total_len} (Unique locations: {unique_locations})")
    if total_len == unique_locations:
        print("Success: Total rows equals number of unique lat/lon locations!")
    else:
        print("Warning: Duplicate locations found!")
        
    print("\n" + "-" * 88)
    print(f"{'Column':<18} | {'Non-null count':<15} | {'% complete':<11} | {'Min value':<10} | {'Max value':<10} | {'Mean':<10}")
    print("-" * 88)
    
    for col in cols_to_check:
        if col in master_df.columns:
            n_count = master_df[col].count()
            pct = (n_count / total_len) * 100 if total_len > 0 else 0
            vmin = master_df[col].min()
            vmax = master_df[col].max()
            vmean = master_df[col].mean()
            print(f"{col:<18} | {n_count:<15} | {pct:<10.1f}% | {vmin:<10.2f} | {vmax:<10.2f} | {vmean:<10.2f}")
        else:
            print(f"{col:<18} | {'MISSING':<15} | {'0.0':<10}% | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")

    print("\nRows per region:")
    print(master_df['region'].value_counts().to_string())

if __name__ == '__main__':
    run_pipeline()
