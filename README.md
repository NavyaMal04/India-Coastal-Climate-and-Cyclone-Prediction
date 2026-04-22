# COASTGUARD

## India Coastal Climate Risk & Cyclone Prediction Dashboard

COASTGUARD is an AI-powered climate intelligence platform designed to monitor and predict cyclone risks across India's coastal regions using historical climate datasets, machine learning models, and interactive visual analytics.

It combines **Sea Surface Temperature (SST), Wind Speed, Pressure, and Rainfall** data to estimate cyclone formation probability and provide real-time coastal risk insights.

---

## 🚀 Project Highlights

✅ Coastal Climate Risk Monitoring
✅ Cyclone Probability Prediction using ML
✅ Interactive Streamlit Dashboard
✅ Regional Risk Mapping for Indian Coastlines
✅ Historical + Simulated + Future Live Data Support
✅ Professional Ocean-Themed UI
✅ RandomForest + XGBoost Models
✅ Alerts & Trend Analytics

---

## 📍 Covered Coastal Regions

* Odisha
* Andhra Pradesh
* Tamil Nadu
* Kerala
* Maharashtra
* Gujarat
* West Bengal

---

## 🧠 Machine Learning Core

COASTGUARD uses classification models to predict cyclone probability.

### Models Implemented:

* Random Forest Classifier
* XGBoost Classifier

### Input Features:

* Sea Surface Temperature (SST)
* Wind Speed
* Surface Pressure
* Rainfall

### Output:

* Cyclone Probability (%)
* Risk Level (Low / Moderate / High)

---

## 📊 Dashboard Features

### Main Dashboard Includes:

* 📈 Climate Trend Charts
* 🌪️ Cyclone Probability KPI
* 🛰️ Coastal Monitoring Map
* ⚠️ Risk Alert Panel
* 🌊 Ocean Temperature Analytics
* 📍 Region-wise Coastal Insights

---

## 🗂️ Project Structure

```bash
COASTGUARD/
│── app.py
│── requirements.txt
│── data/
│   ├── processed/
│   │   └── master_climate_data.csv
│
│── pipeline/
│   └── preprocess.py
│
│── utils/
│   └── data_manager.py
│
│── models/
│   ├── train_model.py
│   ├── predict.py
│   └── model.pkl
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/imannaswini/India-Coastal-Climate-and-Cyclone-Prediction.git
cd India-Coastal-Climate-and-Cyclone-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run app.py
```

---

## 📦 Dependencies

* Python 3.10+
* Streamlit
* Pandas
* NumPy
* Plotly
* Folium
* Scikit-learn
* XGBoost
* Joblib

---

## 🔮 Future Enhancements

* Live API Integration (Open-Meteo / NASA POWER)
* Real-time cyclone tracking
* Storm surge prediction
* Flood risk alerts
* Mobile responsive deployment
* PostgreSQL cloud storage

---

## 👨‍💻 My Contribution

This repository originally contained only preprocessing modules. I transformed it into a complete AI-powered analytics platform by adding:

* Full Streamlit dashboard
* Premium UI/UX redesign
* ML model training pipeline
* Prediction engine
* Dashboard integration
* Error handling & debugging
* Data schema management
* Final demo-ready system

---

## 📸 Preview

Premium Oceanic Dashboard with real-time climate intelligence.

---

## 📚 Technologies Used

Python • Streamlit • Plotly • Folium • Scikit-learn • XGBoost • Pandas • NumPy

---

## ⭐ If You Like This Project

Give this repository a star ⭐

---

## 📬 Contact

For collaboration, improvements, or discussions, feel free to connect.
