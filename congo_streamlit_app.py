"""
PHOENIX — Congo (Katanga) Wildfire Early Warning & Shelter Matching Dashboard
================================================================================
Updated with Future Prediction Tab (calls Railway API)

Run locally:
    pip install streamlit folium streamlit-folium xgboost pandas requests plotly
    streamlit run congo_streamlit_app.py

Files needed in the same folder:
    congo_predict.py, congo_fire_risk_model.json, risk_engine.py, air_quality.py,
    phoenix_climate_2020_2026.csv, drc_katanga_shelters_final.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2
import requests
import plotly.express as px
from datetime import date, timedelta

from congo_predict import predict_fire_risk
from risk_engine import get_alert
from air_quality import fetch_live_pm25

# -------------------------------------------------------------
# API Config (Future Prediction)
# -------------------------------------------------------------
API_BASE_URL = "https://phoenix-drc-api-production.up.railway.app"

@st.cache_data(ttl=300)
def fetch_risk_map_future(target_date: str):
    try:
        resp = requests.get(f"{API_BASE_URL}/risk-map-future", params={"date": target_date}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None

def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return resp.status_code == 200
    except:
        return False

# -------------------------------------------------------------
# Page setup
# -------------------------------------------------------------
st.set_page_config(page_title="PHOENIX — Congo Fire Early Warning", layout="wide")

COLOR_MAP = {"Low": "green", "Medium": "orange", "High": "red"}
FUTURE_COLOR_MAP = {"High": "#e74c3c", "Moderate": "#f39c12", "Low": "#27ae60"}

# -------------------------------------------------------------
# Load data (cached so it only loads once per session)
# -------------------------------------------------------------
@st.cache_data
def load_climate():
    df = pd.read_csv("phoenix_climate_2020_2026.csv")
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + \
                 pd.to_timedelta(df["DOY"] - 1, unit="D")
    return df

@st.cache_data
def load_shelters():
    df = pd.read_csv("drc_katanga_shelters_final.csv")
    df = df.rename(columns={"capacity_estimate": "capacity"})
    df["name"] = df["name"].fillna(df["category"] + " (unnamed)")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(25).astype(int)
    if "available" not in df.columns:
        df["available"] = df["capacity"]
    return df[["osm_id", "category", "name", "lat", "lon", "capacity", "available",
               "is_shelter", "province", "pm2_5", "us_aqi", "observation_time"]]

climate = load_climate()
shelters_all = load_shelters()

# -------------------------------------------------------------
# Sidebar: choose which facility types / provinces to show
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Locations to show on map")
CATEGORY_LABELS = {
    "school": "🏫 Schools (evacuee shelters)",
    "place_of_worship": "🕌 Places of worship (evacuee shelters)",
    "health_facility": "🏥 Health facilities (support only)",
}
selected_categories = [
    cat for cat, label in CATEGORY_LABELS.items()
    if st.sidebar.checkbox(label, value=(cat in ["school", "place_of_worship"]))
]

provinces = sorted(shelters_all["province"].dropna().unique())
selected_provinces = st.sidebar.multiselect("Provinces", provinces, default=provinces)

shelters = shelters_all[
    shelters_all["category"].isin(selected_categories) &
    shelters_all["province"].isin(selected_provinces)
].copy()
st.sidebar.caption(f"{len(shelters)} locations match filters out of {len(shelters_all)} total.")
st.sidebar.caption("Note: health facilities are support resources, "
                    "not counted as evacuee shelter capacity.")
max_markers = st.sidebar.slider("Max shelter markers drawn on map (performance)", 50, 2000, 300, step=50)
st.sidebar.caption("Lower this if the map feels slow to load. Largest-capacity locations are shown first; "
                    "matching/metrics below still use ALL selected locations, not just the ones drawn.")

# -------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------
st.sidebar.header("Controls")
available_dates = sorted(climate["date"].unique())
selected_date = st.sidebar.select_slider(
    "Select date",
    options=available_dates,
    value=available_dates[-1],
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Risk thresholds**\n\n"
    "- 🟢 Low: probability < 0.35\n"
    "- 🟡 Medium: 0.35 – 0.65\n"
    "- 🔴 High: probability ≥ 0.65"
)

# -------------------------------------------------------------
# Tabs
# -------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Current Monitoring", "🔮 Future Prediction"])

# =================================================================
# TAB 1: Current Monitoring (Original Dashboard)
# =================================================================
with tab1:
    st.title("🔥 PHOENIX — Congo (Katanga) Wildfire Early Warning & Shelter Matching")
    st.caption("Haut-Katanga, Lualaba & Tanganyika provinces (DRC)")

    # Run model on all grid cells for the selected date
    day_data = climate[climate["date"] == selected_date].copy()
    doy = pd.Timestamp(selected_date).dayofyear
    is_latest_available = pd.Timestamp(selected_date).normalize() == pd.Timestamp(available_dates[-1]).normalize()
    latest_date_str = pd.Timestamp(available_dates[-1]).strftime("%Y-%m-%d")

    results = []
    aqi_fetch_success_count = 0
    for _, row in day_data.iterrows():
        r = predict_fire_risk(
            lat=row["LAT"], lon=row["LON"], doy=doy,
            t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
            rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
        )
        entry = {**row.to_dict(), **r, "pm2_5": None, "health_level": None, "health_advice": None}
        if is_latest_available:
            pm25 = fetch_live_pm25(row["LAT"], row["LON"])
            if pm25 is not None:
                aqi_fetch_success_count += 1
                alert = get_alert(r["fire_probability"], pm25)
                entry.update({"pm2_5": pm25, "health_level": alert.health_level,
                               "health_advice": alert.health_advice})
        results.append(entry)
    res_df = pd.DataFrame(results)

    if is_latest_available:
        if aqi_fetch_success_count == 0:
            st.warning("🫁 **Air-quality data is temporarily unavailable** — could not reach the live "
                       "Open-Meteo service right now (this is a network hiccup, not a missing feature; "
                       "it will resume automatically once the connection is back). Fire-risk predictions "
                       "below are unaffected.")
        elif aqi_fetch_success_count < len(day_data):
            st.info(f"🫁 Live air-quality retrieved for {aqi_fetch_success_count}/{len(day_data)} zones "
                    f"(a few requests timed out — this is normal and will vary run to run).")
        else:
            st.caption(f"🫁 Air-quality readings below are live (fetched just now). Fire-risk inputs are from "
                       f"**{latest_date_str}** — the most recent weather data available.")
    else:
        st.caption(f"ℹ️ Air-quality/health warnings are only shown for the most recent available date "
                   f"(**{latest_date_str}**) — move the slider to the far right to see them.")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 High risk zones", int((res_df["risk_level"] == "High").sum()))
    col2.metric("🟡 Medium risk zones", int((res_df["risk_level"] == "Medium").sum()))
    col3.metric("🟢 Low risk zones", int((res_df["risk_level"] == "Low").sum()))
    shelter_only = shelters[shelters["is_shelter"]]
    col4.metric("Shelters with capacity", int((shelter_only["available"] > 0).sum()))

    # Map
    st.subheader("Live Risk Map")
    center_lat = res_df["LAT"].mean()
    center_lon = res_df["LON"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    for _, row in res_df.iterrows():
        health_html = ""
        health_level_raw = row.get("health_level")
        if pd.notna(health_level_raw):
            health_level = str(health_level_raw).strip().title()
            pm25_raw = row.get("pm2_5")
            if pd.notna(pm25_raw):
                try:
                    pm25 = float(pm25_raw)
                    health_advice = row.get("health_advice", "")
                    if pd.isna(health_advice):
                        health_advice = ""
                    health_advice = str(health_advice)
                    health_html = (
                        f"<br><b>🫁 Air quality:</b> {health_level} "
                        f"(PM2.5: {pm25:.0f} µg/m³)<br>"
                        f"{health_advice}"
                    )
                except (ValueError, TypeError):
                    health_html = f"<br><b>🫁 Air quality:</b> {health_level}"
            else:
                health_html = f"<br><b>🫁 Air quality:</b> {health_level}"

        risk_color = COLOR_MAP.get(row["risk_level"], "gray")
        folium.CircleMarker(
            location=[row["LAT"], row["LON"]],
            radius=14,
            color=risk_color,
            fill=True,
            fill_color=risk_color,
            fill_opacity=0.7,
            weight=2,
            popup=folium.Popup(
                f"<b>{row['risk_level']} risk</b><br>"
                f"Fire probability: {row['fire_probability'] * 100:.1f}%<br>"
                f"Max Temp: {row['T2M_MAX']:.1f}°C | "
                f"Humidity: {row['RH2M']:.1f}%"
                f"{health_html}",
                max_width=260,
            ),
            tooltip=f"{row['risk_level']} risk",
        ).add_to(m)

    shelter_cluster = MarkerCluster(name="Shelters & Resources").add_to(m)
    shelters_to_draw = shelters.sort_values("capacity", ascending=False).head(max_markers)

    for _, s in shelters_to_draw.iterrows():
        if s["is_shelter"]:
            color = "blue"
            kind = "Shelter"
        else:
            color = "darkcyan"
            kind = "Support resource (not for housing evacuees)"

        aqi_html = ""
        pm25_shelter = s.get("pm2_5")
        us_aqi_shelter = s.get("us_aqi")
        if pd.notna(pm25_shelter):
            if pd.notna(us_aqi_shelter):
                aqi_html = f"<br><b>🫁 Current AQI:</b> {float(us_aqi_shelter):.0f} (PM2.5: {float(pm25_shelter):.1f} µg/m³)"
            else:
                aqi_html = f"<br><b>🫁 PM2.5:</b> {float(pm25_shelter):.1f} µg/m³"
            observation_time = s.get("observation_time")
            if pd.notna(observation_time):
                aqi_html += f"<br><small>as of {observation_time}</small>"

        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(
                f"<b>{s['name']}</b><br>"
                f"{kind}<br>"
                f"{s['province']}<br>"
                f"Est. capacity: {s['available']}/{s['capacity']}"
                f"{aqi_html}",
                max_width=260,
            ),
            tooltip=s["name"],
        ).add_to(shelter_cluster)

    if len(shelters) > max_markers:
        st.caption(
            f"Showing the {max_markers} largest-capacity locations out of {len(shelters)} matching your filters."
        )

    st_folium(m, width=1100, height=550, returned_objects=[])

# =================================================================
# TAB 2: Future Prediction (NEW!)
# =================================================================
with tab2:
    st.title("🔮 Future Wildfire Prediction")
    st.caption("Predict fire risk for future dates using historical climatology data.")

    api_online = check_api_health()
    if api_online:
        st.success("🟢 Forecast API Online — Predictions ready!")
    else:
        st.error("🔴 Forecast API Offline — Please check the API service.")
        st.stop()

    st.markdown("---")

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
        zoom_start=6,
        tiles="CartoDB dark_matter"
    )

    for _, row in df_future.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5 + (row["fire_probability"] * 12),
            popup=folium.Popup(
                f"""
                <b>📍 Location:</b> {row['lat']:.3f}, {row['lon']:.3f}<br>
                <b>🔥 Probability:</b> {row['fire_probability']:.1%}<br>
                <b>⚠️ Risk Level:</b> {row['risk_level']}<br>
                <b>🌡️ Max Temp:</b> {row['t2m_max']:.1f}°C<br>
                <b>💧 Humidity:</b> {row['rh2m']:.1f}%
                """,
                max_width=250,
            ),
            color=FUTURE_COLOR_MAP.get(row["risk_level"], "gray"),
            fill=True,
            fill_color=FUTURE_COLOR_MAP.get(row["risk_level"], "gray"),
            fill_opacity=0.7,
            weight=2,
        ).add_to(m_future)

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

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📊 Risk Distribution")
        risk_counts = df_future["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_pie = px.pie(
            risk_counts, values="Count", names="Risk Level",
            color="Risk Level", color_discrete_map=FUTURE_COLOR_MAP,
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=14)
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.subheader("🌡️ Temperature vs Fire Risk")
        fig_scatter = px.scatter(
            df_future, x="t2m_max", y="fire_probability",
            color="risk_level", color_discrete_map=FUTURE_COLOR_MAP,
            labels={"t2m_max": "Max Temperature (°C)", "fire_probability": "Fire Probability"},
            hover_data={"lat": ":.3f", "lon": ":.3f", "rh2m": ":.1f"},
            opacity=0.7,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # High Risk Table
    st.subheader("🚨 High Risk Alert Zones")
    if len(high_risk) > 0:
        high_display = high_risk[["lat", "lon", "fire_probability", "t2m_max", "rh2m"]].copy()
        high_display["fire_probability"] = high_display["fire_probability"].apply(lambda x: f"{x:.1%}")
        high_display["t2m_max"] = high_display["t2m_max"].apply(lambda x: f"{x:.1f}°C")
        high_display["rh2m"] = high_display["rh2m"].apply(lambda x: f"{x:.1f}%")
        high_display.columns = ["Latitude", "Longitude", "Fire Probability", "Max Temp", "Humidity"]
        st.dataframe(high_display.sort_values("Fire Probability", ascending=False), use_container_width=True, hide_index=True)

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
    st.caption(
        f"🔮 Predictions powered by PHOENIX Forecast API | "
        f"Date: {date_str} | Method: Climatology | API: {API_BASE_URL}"
    )

# Footer
st.markdown("---")
st.caption("🔥 PHOENIX DRC — Wildfire Early Warning System | Built with Streamlit")
