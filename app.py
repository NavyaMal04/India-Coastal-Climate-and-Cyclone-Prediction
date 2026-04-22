import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import folium
from folium import plugins
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
from utils.data_manager import get_latest_data, get_historical_trends, get_all_regions_summary, REGIONS
from models.predict import get_cyclone_probability
import time
from datetime import datetime
import base64
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="COASTGUARD | Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper to load local image as base64 ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

ASSETS_DIR = os.path.join(os.getcwd(), "assets")
HERO_BG_BASE64 = get_base64_of_bin_file(os.path.join(ASSETS_DIR, "light_hero.png"))
CYCLONE_ICON_BASE64 = get_base64_of_bin_file(os.path.join(ASSETS_DIR, "cyclone_icon.png"))

# --- Custom Styling: Light Ocean Theme ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    :root {{
        --primary: #0096FF;
        --aqua: #00D4FF;
        --deep-blue: #0077B6;
        --sky: #90E0EF;
        --background: #F0FBFF;
        --card: #FFFFFF;
        --navy: #0B1F3A;
        --gray-text: #5B6B7A;
        --success: #00C853;
        --danger: #FF5252;
    }}

    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
    }}
    
    .stApp {{
        background-color: var(--background);
        background-image: 
            radial-gradient(at 0% 0%, rgba(0, 212, 255, 0.05) 0, transparent 50%), 
            radial-gradient(at 100% 0%, rgba(0, 150, 255, 0.05) 0, transparent 50%);
    }}

    /* Sidebar Redesign - Vibrant Light Blue */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #E0F7FF 0%, #F0FBFF 100%) !important;
        border-right: 1px solid #BAE6FD !important;
    }}
    
    [data-testid="stSidebar"] .stMarkdown p {{
        color: var(--navy) !important;
        font-weight: 500;
    }}
    
    [data-testid="stSidebar"] h2 {{
        color: var(--primary) !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }}

    /* Global Headings */
    h1, h2, h3, .stHeader {{
        color: var(--navy) !important;
        font-weight: 800 !important;
    }}

    /* Hero Banner - Vibrant Ocean */
    .hero-banner {{
        background: linear-gradient(rgba(0, 119, 182, 0.2), rgba(0, 119, 182, 0.2)), url("data:image/png;base64,{HERO_BG_BASE64}");
        background-size: cover;
        background-position: center;
        height: 340px;
        border-radius: 32px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
        padding: 0 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 150, 255, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.8);
        position: relative;
        overflow: hidden;
    }}
    
    .hero-banner::after {{
        content: "";
        position: absolute;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: wave_anim 10s infinite linear;
    }}

    @keyframes wave_anim {{
        from {{ transform: translate(-50%, -50%) rotate(0deg); }}
        to {{ transform: translate(-50%, -50%) rotate(360deg); }}
    }}

    .hero-title {{
        font-size: 4rem !important;
        font-weight: 800 !important;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 0.5rem !important;
    }}

    /* KPI Cards - Vibrant Light */
    .metric-card {{
        background: var(--card);
        border: 1px solid rgba(0, 150, 255, 0.1);
        padding: 1.5rem;
        border-radius: 24px;
        box-shadow: 0 10px 25px rgba(0, 119, 182, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .metric-card:hover {{
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0, 150, 255, 0.12);
        border-color: var(--aqua);
    }}
    .metric-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 6px;
        background: linear-gradient(90deg, var(--primary), var(--aqua));
    }}
    
    .metric-label {{
        color: var(--gray-text);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .metric-value {{
        color: var(--navy);
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }}

    /* Status Pills */
    .status-pill {{
        background: rgba(0, 212, 255, 0.1);
        color: var(--deep-blue);
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(0, 212, 255, 0.2);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}

    /* Animations */
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .animate-fade {{ animation: fadeIn 0.8s ease-out forwards; }}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--background); }}
    ::-webkit-scrollbar-thumb {{ background: var(--sky); border-radius: 10px; }}
    </style>
    """, unsafe_allow_html=True)

# --- Components ---
def kpi_card(label, value, icon, trend_val, trend_up=True, subtext=""):
    trend_color = "#00C853" if trend_up else "#FF5252"
    trend_arrow = "↑" if trend_up else "↓"
    
    card_html = f"""
        <div class="metric-card animate-fade">
            <div>
                <div class="metric-label">
                    <span style="font-size: 1.2rem;">{icon}</span> {label}
                </div>
                <div class="metric-value">{value}</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="color: {trend_color}; font-weight: 700; font-size: 0.85rem;">
                    {trend_arrow} {trend_val}
                </div>
                <div style="font-size: 0.7rem; color: #5B6B7A; font-weight: 600;">{subtext}</div>
            </div>
        </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;" class="animate-fade">
            <img src="data:image/png;base64,{CYCLONE_ICON_BASE64}" width="70" style="filter: drop-shadow(0 5px 10px rgba(0, 150, 255, 0.2));">
            <h2 style="margin-top: 10px;">COASTGUARD</h2>
            <div style="font-size: 0.7rem; color: #5B6B7A; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">Ocean Climate Intelligence</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    selected_region = st.selectbox(
        "COASTAL SECTOR SELECTION",
        options=list(REGIONS.keys()),
        index=0
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📡 SYSTEM STATUS")
    st.markdown('<div class="status-pill"><span style="width: 8px; height: 8px; background: #00C853; border-radius: 50%;"></span> OCEAN SENSORS: ONLINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-top: 8px;" class="status-pill"><span style="width: 8px; height: 8px; background: #00C853; border-radius: 50%;"></span> SAT-INTELLIGENCE: ACTIVE</div>', unsafe_allow_html=True)
    
    if st_autorefresh:
        st_autorefresh(interval=30 * 1000, key="datarefresh")
    
    st.markdown("---")
    st.caption("v4.0 Vibrant Ocean | © 2026 ADMS")

# --- Main Logic ---
def main():
    # 1. Hero Banner
    st.markdown(f"""
        <div class="hero-banner">
            <h1 class="hero-title">COASTGUARD</h1>
            <p style="font-size: 1.2rem; font-weight: 500; opacity: 0.95;">India Coastal Climate Risk & Cyclone Prediction Intelligence</p>
            <div style="display: flex; gap: 15px; margin-top: 20px;">
                <div style="background: rgba(255,255,255,0.2); padding: 6px 16px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(255,255,255,0.5);">🛰️ SAT-ARRAY-V4</div>
                <div style="background: rgba(255,255,255,0.2); padding: 6px 16px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(255,255,255,0.5);">✨ NEURAL MODELING</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Data Fetching
    data = get_latest_data(selected_region)
    history = get_historical_trends(selected_region)

    # Sub-header
    c1, c2 = st.columns([1, 1])
    with c1:
        st.header(f"📍 {selected_region} Sector")
    with c2:
        st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-top: 25px;">
                <div style="background: white; border: 1px solid #E0F7FF; padding: 8px 16px; border-radius: 100px; font-size: 0.8rem; font-weight: 700; color: #0096FF; box-shadow: 0 4px 10px rgba(0, 150, 255, 0.05);">
                    ⏱️ SYNCED: {datetime.now().strftime('%H:%M:%S')} UTC
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 2. KPI Row
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        # Use real ML model for prediction
        prob = get_cyclone_probability(
            data.get('sst', 28.5),
            data.get('wind_speed', 0),
            data.get('pressure', 1013),
            data.get('rainfall', 0)
        )
        kpi_card("Cyclone Prob", f"{prob:.1f}%", "🌪️", "1.4%", trend_up=True, subtext="Neural Forecast")
    with k2:
        kpi_card("Risk Level", data.get('risk_level', 'LOW'), "🛡️", "STABLE", trend_up=True, subtext="Intelligence AI")
    with k3:
        wind = data.get('wind_speed', 0)
        kpi_card("Wind Velocity", f"{wind:.1f} km/h", "🌬️", "0.8", trend_up=False, subtext="Grid Sensors")
    with k4:
        sst = data.get('sst', 0)
        kpi_card("Ocean Temp", f"{sst:.1f}°C", "🌊", "0.3", trend_up=True, subtext="SST Satellite")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Analytics & Map
    left, right = st.columns([1.5, 1])
    
    with left:
        st.markdown("### 🌐 Intelligence Monitoring Map")
        m = folium.Map(location=[data.get('latitude', 20.0), data.get('longitude', 80.0)], zoom_start=7, tiles="cartodb positron")
        
        all_regions = get_all_regions_summary()
        for r in all_regions:
            risk = r.get('risk_level', 'Low')
            color = '#00C853' if risk == 'Low' else '#FF5252' if risk == 'High' else '#FF9800'
            folium.CircleMarker(
                location=[r.get('latitude', 20.0), r.get('longitude', 80.0)],
                radius=10, color=color, weight=2, fill=True, fill_color=color, fill_opacity=0.4,
                tooltip=f"{r.get('region')} ({risk} RISK)"
            ).add_to(m)
        folium_static(m, width=800, height=500)

    with right:
        st.markdown("### 📈 Neural Trend Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history['datetime'], y=history['sst'], name="SST", line=dict(color='#0096FF', width=4, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 150, 255, 0.05)'))
        fig.add_trace(go.Scatter(x=history['datetime'], y=history['wind_speed'], name="Wind", line=dict(color='#00D4FF', width=3, shape='spline'), yaxis="y2"))
        
        fig.update_layout(
            template="plotly_white", margin=dict(l=0,r=0,t=20,b=0), height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(showgrid=False, title="Temp (°C)"), 
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Wind (km/h)"),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Outfit")
        )
        st.plotly_chart(fig, use_container_width=True)

    # 4. Alerts
    st.markdown("<br>", unsafe_allow_html=True)
    if prob > 50:
        st.error(f"**CRITICAL ADVISORY:** High risk anomalies detected in {selected_region}. Sector command advised to initiate coastal clearance.")
    elif prob > 25:
        st.warning(f"**WATCH ISSUED:** Moderate instability observed. Monitoring storm cell development in {selected_region} quadrant.")
    else:
        st.success(f"**SYSTEM NORMAL:** No immediate threats detected for {selected_region}. Monitoring standard seasonal patterns.")

    # 5. Footer
    st.markdown(f"""
        <div style="text-align: center; margin-top: 4rem; padding: 2rem; color: #5B6B7A; font-weight: 600; border-top: 1px solid #E0F7FF;">
            COASTGUARD | India Coastal Intelligence Platform | © 2026 ADMS
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
