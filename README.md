# COASTGUARD | India Coastal Climate & Cyclone Prediction

COASTGUARD is a premium, enterprise-grade SaaS platform designed for real-time surveillance and predictive intelligence of cyclonic threats along the Indian coastline. 

![Dashboard Preview](https://img.icons8.com/fluency/96/tsunami.png)

## 💎 Features

- **Operational Risk Center**: Real-time AI surveillance with a high-fidelity geospatial risk matrix.
- **Interactive Geospatial Intelligence**: Dual-mode interactive map (Survey/Satellite) powered by Leaflet.js with pulsing high-risk indicators.
- **Deep Analytics Hub**: 30-day historical correlation analysis of Pressure, Wind, and Sea Surface Temperature (SST).
- **Emergency Broadcast Center**: Automated alert grid with severity filtering and operational advisories.
- **ML Intelligence Engine**: Multi-source data ingestion (NASA POWER, Open-Meteo) feeding a predictive model for precise risk probability.

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3 (Premium Glassmorphism), JavaScript (Vanilla), Leaflet.js, Chart.js.
- **Backend**: Python 3.x, FastAPI, Uvicorn.
- **Database**: SQLite3 with Row-level tracking.
- **Data Pipeline**: Automated ingestion and ML inference engine.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js (for `npx live-server`, or any static web server)

### Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd India-Coastal-Climate-and-Cyclone-Prediction
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
1. **Start the FastAPI Backend**:
   ```bash
   python api.py
   ```
   The API will be available at `http://127.0.0.1:8000`.

2. **Launch the Dashboard**:
   ```bash
   cd dashboard_ui
   npx live-server
   ```
   Access the UI at `http://127.0.0.1:8080`.

## 📂 Project Structure

```text
├── api.py              # FastAPI Backend Server
├── dashboard_ui/       # Premium Web Interface
│   ├── index.html      # Main Dashboard Structure
│   ├── styles.css      # SaaS Design System
│   └── script.js       # Real-time Logic & Map Integration
├── pipeline/           # ML & Data Ingestion Pipeline
│   ├── ingestion.py    # Multi-source Data Fetching
│   ├── inference.py    # Risk Assessment Logic
│   └── train.py        # Model Training Scripts
├── database/           # SQLite Schema & DB Controllers
└── data/               # Processed & Raw Telemetry Storage
```

## 🛡️ License
Copyright © 2024 COASTGUARD Enterprise Operations. All rights reserved.
