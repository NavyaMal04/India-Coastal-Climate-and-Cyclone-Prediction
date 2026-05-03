import os
import sys
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from database.db import (
        get_latest_predictions,
        get_latest_readings,
        get_active_alerts,
        get_latest_inference_log,
        get_region_history_enriched
    )
except ImportError:
    print("Error: Could not import database functions. Check project structure.")
    sys.exit(1)

app = FastAPI(title="COASTGUARD API")

# Enable CORS for the HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predictions")
async def read_predictions():
    return get_latest_predictions()

@app.get("/readings")
async def read_readings():
    return get_latest_readings()

@app.get("/alerts")
async def read_alerts():
    return get_active_alerts()

@app.get("/logs")
async def read_logs():
    return get_latest_inference_log()

@app.get("/history/{region}")
async def read_history(region: str, days: int = 30):
    data = get_region_history_enriched(region, days)
    if not data:
        return []
    return data

@app.post("/sync")
async def sync_data():
    try:
        scheduler_path = os.path.join(BASE_DIR, "pipeline", "scheduler.py")
        result = subprocess.run(
            ["python", scheduler_path, "--once"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
