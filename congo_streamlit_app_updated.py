"""
PHOENIX DRC Dashboard — Congo (Katanga) Wildfire Monitoring
============================================================
Updated with Future Prediction Tab (calls Railway API)

Run locally:
    streamlit run congo_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import requests

# -----------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="PHOENIX DRC Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e65100;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-moderate { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("phoenix_climate_2020_2026.csv")
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") +                   pd.to_timedelta(df["DOY"] - 1, unit="D")
    return df

CLIMATE_DF = load_data()
LATEST_DATE = CLIMATE_DF["date"].max().date()

# Load shelters
@st.cache_data
def load_shelters():
    return pd.read_csv("drc_katanga_shelters_final.csv")

SHELTERS_DF = load_shelters()

# -----------------------------------------------------------------
# API Config (Future Prediction)
# -----------------------------------------------------------------
API_BASE_URL = "https://phoenix-drc-api-production.up.railway.app"

@st.cache_data(ttl=300)
def fetch_risk_map_future(target_date: str):
    try:
        resp = requests.get(f"{API_BASE_URL}/risk-map-future", params={"date": target_date}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return resp.status_code == 200
    except:
        return False

# -----------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/fire-element.png", width=80)
st.sidebar.title("🔥 PHOENIX DRC")
st.sidebar.markdown("**Wildfire Monitoring — Katanga Region**")
st.sidebar.markdown("---")

# Province filter
provinces = ["All Provinces", "Haut-Katanga", "Lualaba", "Tanganyika"]
selected_province = st.sidebar.selectbox("🗺️ Select Province", provinces)

# Date filter for historical data
selected_date = st.sidebar.date_input(
    "📅 Select Date",
    value=LATEST_DATE,
    min_value=CLIMATE_DF["date"].min().date(),
    max_value=LATEST_DATE,
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Data source: NASA POWER + NASA FIRMS\nModel: XGBoost Fire Risk Classifier")

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.markdown('<p class="main-header">🔥 PHOENIX DRC Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Wildfire Early Warning System — Democratic Republic of Congo</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "🗺️ Risk Map", 
    "🏠 Shelters",
    "🔮 Future Prediction"  # ← التاب الجديدة
])

# =================================================================
# TAB 1: Overview
# =================================================================
with tab1:
    st.header("📊 Historical Overview")

    day_data = CLIMATE_DF[CLIMATE_DF["date"] == pd.Timestamp(selected_date)]

    if day_data.empty:
        st.warning("No data available for selected date.")
    else:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_temp = day_data["T2M_MAX"].mean()
            st.metric("🌡️ Avg Max Temp", f"{avg_temp:.1f}°C")
        with col2:
            avg_hum = day_data["RH2M"].mean()
            st.metric("💧 Avg Humidity", f"{avg_hum:.1f}%")
        with col3:
            avg_wind = day_data["WS2M"].mean()
            st.metric("💨 Avg Wind Speed", f"{avg_wind:.1f} m/s")
        with col4:
            total_rain = day_data["PRECTOTCORR"].sum()
            st.metric("🌧️ Total Rain", f"{total_rain:.1f} mm")

        st.markdown("---")

        # Time series
        st.subheader("📈 Temperature Trend (2020-2026)")
        monthly = CLIMATE_DF.groupby(CLIMATE_DF["date"].dt.to_period("M")).agg({
            "T2M_MAX": "mean",
            "T2M_MIN": "mean",
            "RH2M": "mean",
        }).reset_index()
        monthly["date"] = monthly["date"].dt.to_timestamp()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["T2M_MAX"],
                                 mode="lines", name="Max Temp", line=dict(color="#e74c3c")))
        fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["T2M_MIN"],
                                 mode="lines", name="Min Temp", line=dict(color="#3498db")))
        fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°C)",
                          hovermode="x unified", height=400)
        st.plotly_chart(fig, use_container_width=True)

# =================================================================
# TAB 2: Risk Map
# =================================================================
with tab2:
    st.header("🗺️ Fire Risk Map")

    day_data = CLIMATE_DF[CLIMATE_DF["date"] == pd.Timestamp(selected_date)]

    if day_data.empty:
        st.warning("No data for selected date.")
    else:
        # Simple risk calculation (mock — replace with your actual model)
        day_data = day_data.copy()
        day_data["fire_risk_score"] = (
            (day_data["T2M_MAX"] - 20) / 30 * 0.4 +
            (100 - day_data["RH2M"]) / 100 * 0.3 +
            day_data["WS2M"] / 15 * 0.2 +
            (1 - day_data["PRECTOTCORR"] / 10).clip(0, 1) * 0.1
        ).clip(0, 1)

        day_data["risk_level"] = day_data["fire_risk_score"].apply(
            lambda x: "High" if x >= 0.7 else "Moderate" if x >= 0.4 else "Low"
        )

        # Map
        m = folium.Map(location=[-10.5, 27.5], zoom_start=7, tiles="CartoDB positron")
        colors = {"High": "red", "Moderate": "orange", "Low": "green"}

        for _, row in day_data.iterrows():
            folium.CircleMarker(
                location=[row["LAT"], row["LON"]],
                radius=4 + row["fire_risk_score"] * 8,
                color=colors.get(row["risk_level"], "gray"),
                fill=True,
                fill_opacity=0.6,
                popup=f"Risk: {row['risk_level']}<br>Prob: {row['fire_risk_score']:.1%}",
            ).add_to(m)

        st_folium(m, width="100%", height=600)

        # Risk stats
        st.subheader("📊 Risk Statistics")
        risk_counts = day_data["risk_level"].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 High Risk", risk_counts.get("High", 0))
        with col2:
            st.metric("🟡 Moderate", risk_counts.get("Moderate", 0))
        with col3:
            st.metric("🟢 Low Risk", risk_counts.get("Low", 0))

# =================================================================
# TAB 3: Shelters
# =================================================================
with tab3:
    st.header("🏠 Emergency Shelters")

    if selected_province != "All Provinces":
        filtered_shelters = SHELTERS_DF[SHELTERS_DF["province"] == selected_province]
    else:
        filtered_shelters = SHELTERS_DF

    st.metric("Total Shelters", len(filtered_shelters))

    # Map with shelters
    m = folium.Map(location=[-10.5, 27.5], zoom_start=7, tiles="CartoDB positron")

    for _, row in filtered_shelters.iterrows():
        icon_color = "blue" if row.get("is_shelter", False) else "gray"
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"{row['name']}<br>Capacity: {row.get('capacity', 'N/A')}",
            icon=folium.Icon(color=icon_color, icon="home"),
        ).add_to(m)

    st_folium(m, width="100%", height=500)

    st.subheader("📋 Shelter List")
    display_cols = ["name", "category", "province", "lat", "lon", "capacity"]
    available_cols = [c for c in display_cols if c in filtered_shelters.columns]
    st.dataframe(filtered_shelters[available_cols], use_container_width=True)

# =================================================================
# TAB 4: Future Prediction (NEW!)
# =================================================================
with tab4:
    st.header("🔮 Future Wildfire Prediction")
    st.markdown("Predict fire risk for future dates using historical climatology data.")

    # API Status
    api_online = check_api_health()
    if api_online:
        st.success("🟢 Forecast API Online — Predictions ready!")
    else:
        st.error("🔴 Forecast API Offline — Please check the API service.")
        st.stop()

    st.markdown("---")

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        future_date = st.date_input(
            "📅 Select Future Date",
            value=date.today() + timedelta(days=7),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=365),
            key="future_date_picker"
        )
    with col_ctrl2:
        st.info("💡 The model uses **Climatology** (historical averages for the same day of year) to predict future fire risk.")

    date_str = future_date.strftime("%Y-%m-%d")

    # Fetch predictions
    with st.spinner(f"🔍 Fetching predictions for {date_str}..."):
        future_data = fetch_risk_map_future(date_str)

    if future_data is None or len(future_data) == 0:
        st.error("Failed to fetch predictions. Please try again.")
        st.stop()

    df_future = pd.DataFrame(future_data)

    # Summary Metrics
    st.markdown("---")
    st.subheader("📊 Prediction Summary")

    high_risk = df_future[df_future["risk_level"] == "High"]
    moderate_risk = df_future[df_future["risk_level"] == "Moderate"]
    low_risk = df_future[df_future["risk_level"] == "Low"]
    avg_prob = df_future["fire_probability"].mean()
    max_prob = df_future["fire_probability"].max()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("🔴 High Risk", len(high_risk), help="Zones with >70% fire probability")
    with m2:
        st.metric("🟡 Moderate", len(moderate_risk), help="Zones with 40-70% probability")
    with m3:
        st.metric("🟢 Low Risk", len(low_risk), help="Zones with <40% probability")
    with m4:
        st.metric("📊 Avg Prob", f"{avg_prob:.1%}")
    with m5:
        st.metric("⚠️ Max Prob", f"{max_prob:.1%}")

    st.markdown("---")

    # Interactive Map
    st.subheader("🗺️ Predicted Risk Heatmap")

    m_future = folium.Map(
        location=[-10.5, 27.5],
        zoom_start=7,
        tiles="CartoDB dark_matter"
    )

    risk_colors = {"High": "#e74c3c", "Moderate": "#f39c12", "Low": "#27ae60"}

    for _, row in df_future.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5 + (row["fire_probability"] * 12),
            popup=folium.Popup(
                f"""
                <b>📍 Location:</b> {row['lat']:.3f}, {row['lon']:.3f}<br>
                <b>🔥 Probability:</b> {row['fire_probability']:.1%}<br>
                <b>⚠️ Risk Level:</b> <span style='color:{risk_colors.get(row['risk_level'])}'>{row['risk_level']}</span><br>
                <b>🌡️ Max Temp:</b> {row['t2m_max']:.1f}°C<br>
                <b>💧 Humidity:</b> {row['rh2m']:.1f}%
                """,
                max_width=250,
            ),
            color=risk_colors.get(row["risk_level"], "gray"),
            fill=True,
            fill_color=risk_colors.get(row["risk_level"], "gray"),
            fill_opacity=0.7,
            weight=2,
        ).add_to(m_future)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 9999; 
         background-color: rgba(0,0,0,0.8); padding: 12px; border-radius: 8px; 
         box-shadow: 2px 2px 10px rgba(0,0,0,0.5); color: white; font-size: 13px;">
    <b>🔥 Risk Level</b><br>
    <span style="color: #e74c3c;">●</span> High (>70%)<br>
    <span style="color: #f39c12;">●</span> Moderate (40-70%)<br>
    <span style="color: #27ae60;">●</span> Low (<40%)
    </div>
    """
    m_future.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m_future, width="100%", height=600, returned_objects=[])

    st.markdown("---")

    # Charts Row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📊 Risk Distribution")
        risk_counts = df_future["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        color_map = {"High": "#e74c3c", "Moderate": "#f39c12", "Low": "#27ae60"}
        fig_pie = px.pie(
            risk_counts,
            values="Count",
            names="Risk Level",
            color="Risk Level",
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont_size=14,
            marker=dict(line=dict(color="white", width=2))
        )
        fig_pie.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.subheader("🌡️ Temperature vs Fire Risk")
        fig_scatter = px.scatter(
            df_future,
            x="t2m_max",
            y="fire_probability",
            color="risk_level",
            color_discrete_map=color_map,
            labels={
                "t2m_max": "Max Temperature (°C)",
                "fire_probability": "Fire Probability",
                "risk_level": "Risk Level"
            },
            hover_data={"lat": ":.3f", "lon": ":.3f", "rh2m": ":.1f"},
            opacity=0.7,
        )
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # High Risk Alert Table
    st.subheader("🚨 High Risk Alert Zones")

    if len(high_risk) > 0:
        high_display = high_risk[["lat", "lon", "fire_probability", "t2m_max", "rh2m"]].copy()
        high_display["fire_probability"] = high_display["fire_probability"].apply(lambda x: f"{x:.1%}")
        high_display["t2m_max"] = high_display["t2m_max"].apply(lambda x: f"{x:.1f}°C")
        high_display["rh2m"] = high_display["rh2m"].apply(lambda x: f"{x:.1f}%")
        high_display.columns = ["Latitude", "Longitude", "Fire Probability", "Max Temp", "Humidity"]

        st.dataframe(
            high_display.sort_values("Fire Probability", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        # Download button
        csv = high_risk.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download High Risk Zones (CSV)",
            data=csv,
            file_name=f"high_risk_zones_{date_str}.csv",
            mime="text/csv",
        )
    else:
        st.success("✅ No high-risk zones predicted for this date!")

    st.markdown("---")

    # Footer
    st.caption(
        f"🔮 Predictions powered by PHOENIX Forecast API | "
        f"Date: {date_str} | "
        f"Method: Climatology (Historical Averages) | "
        f"API: {API_BASE_URL}"
    )

# -----------------------------------------------------------------
# Main Footer
# -----------------------------------------------------------------
st.markdown("---")
st.caption("🔥 PHOENIX DRC — Wildfire Early Warning System | Built with Streamlit")
