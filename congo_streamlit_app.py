"""
PHOENIX — Congo (Katanga) Wildfire Early Warning & Shelter Matching Dashboard
================================================================================
Updated with Future Prediction Tab (calls Railway API)

PERFORMANCE: batched PM2.5, cached predictions, vectorized GeoJson risk zones.

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
import requests
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2
import plotly.express as px
from datetime import date, timedelta

from congo_predict import predict_fire_risk
from risk_engine import get_alert

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
    except Exception:
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
    # NASA POWER has a ~3-5 day processing lag; the most recent unprocessed
    # days come back as the fill value -999 instead of real numbers. Drop
    # those rows entirely so broken -999 readings never reach the map/table
    # (replacing with NaN and keeping the row was letting bad/incomplete
    # temperature values leak into the "latest date" view).
    weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
    bad = (df[weather_cols] < -900).any(axis=1)
    if bad.any():
        df = df[~bad].copy()
    df = df.dropna(subset=["YEAR", "DOY"])  # a few rows have genuinely missing YEAR/DOY
    df["YEAR"] = df["YEAR"].astype(int)
    df["DOY"] = df["DOY"].astype(int)
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

# Latest date with FULL grid coverage, not just any row's max date —
# staggered NASA POWER processing across the region can otherwise leave a
# "latest date" that only covers part of the area (and shows wrong/partial
# temperature numbers for the rest).
_total_cells = climate[["LAT", "LON"]].drop_duplicates().shape[0]
_counts_per_date = climate.groupby("date").size()
_full_coverage_dates = _counts_per_date[_counts_per_date == _total_cells]
LATEST_FULL_COVERAGE_DATE = (_full_coverage_dates.index.max() if not _full_coverage_dates.empty
                              else climate["date"].max())
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
    value=LATEST_FULL_COVERAGE_DATE,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
)
st.sidebar.caption(f"Defaulting to {pd.Timestamp(LATEST_FULL_COVERAGE_DATE).strftime('%Y-%m-%d')} — "
                    f"the most recent date with complete data for all grid cells. "
                    f"Some later dates may only have partial coverage.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Risk thresholds**\n\n"
    "- 🟢 Low: probability < 0.35\n"
    "- 🟡 Medium: 0.35 – 0.65\n"
    "- 🔴 High: probability ≥ 0.65"
)

# -------------------------------------------------------------
# PERFORMANCE FIX: batched air-quality fetch
# -------------------------------------------------------------
def fetch_pm25_batch(lat_lon_pairs, chunk_size=30):
    AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    results = {}
    for i in range(0, len(lat_lon_pairs), chunk_size):
        chunk = lat_lon_pairs[i:i + chunk_size]
        lats = ",".join(str(round(p[0], 4)) for p in chunk)
        lons = ",".join(str(round(p[1], 4)) for p in chunk)
        try:
            resp = requests.get(
                AQ_URL,
                params={"latitude": lats, "longitude": lons,
                        "current": "pm2_5,us_aqi", "timezone": "auto"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue
        data_list = data if isinstance(data, list) else [data]
        for point, res in zip(chunk, data_list):
            current = res.get("current", {})
            if current.get("pm2_5") is not None:
                results[point] = {
                    "pm2_5": current.get("pm2_5"),
                    "us_aqi": current.get("us_aqi"),
                }
    return results

# -------------------------------------------------------------
# PERFORMANCE FIX: cache predictions per date
# -------------------------------------------------------------
@st.cache_data(ttl=900)
def compute_results_for_date(day_data_records, doy, is_latest):
    pm25_lookup = {}
    if is_latest:
        pairs = [(r["LAT"], r["LON"]) for r in day_data_records]
        pm25_lookup = fetch_pm25_batch(pairs)

    weather_keys = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
    results = []
    success_count = 0
    skipped_missing = 0
    for row in day_data_records:
        if any(pd.isna(row.get(k)) for k in weather_keys):
            skipped_missing += 1
            entry = {**row, "risk_level": "No data", "fire_probability": None,
                      "pm2_5": None, "health_level": None, "health_advice": None}
            results.append(entry)
            continue

        r = predict_fire_risk(
            lat=row["LAT"], lon=row["LON"], doy=doy,
            t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
            rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
        )
        entry = {**row, **r, "pm2_5": None, "health_level": None, "health_advice": None}
        aq = pm25_lookup.get((row["LAT"], row["LON"]))
        if aq:
            success_count += 1
            alert = get_alert(r["fire_probability"], aq["pm2_5"])
            entry.update({"pm2_5": aq["pm2_5"], "health_level": alert.health_level,
                           "health_advice": alert.health_advice})
        results.append(entry)

    return pd.DataFrame(results), success_count, skipped_missing

# -------------------------------------------------------------
# Tabs
# -------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Current Monitoring", "🔮 Future Prediction"])

# =================================================================
# TAB 1: Current Monitoring (Original Optimized Dashboard)
# =================================================================
with tab1:
    st.title("🔥 PHOENIX — Congo (Katanga) Wildfire Early Warning & Shelter Matching")
    st.caption("Haut-Katanga, Lualaba & Tanganyika provinces (DRC) — AI for All Hackathon")

    day_data = climate[climate["date"] == selected_date].copy()
    doy = pd.Timestamp(selected_date).dayofyear
    is_latest_available = pd.Timestamp(selected_date).normalize() == pd.Timestamp(LATEST_FULL_COVERAGE_DATE).normalize()
    latest_date_str = pd.Timestamp(LATEST_FULL_COVERAGE_DATE).strftime("%Y-%m-%d")

    res_df, aqi_fetch_success_count, _ = compute_results_for_date(
        day_data.to_dict("records"), doy, is_latest_available
    )

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
            st.caption(f"🫁 Air-quality readings below are live (fetched just now, cached for 15 min). "
                       f"Fire-risk inputs are from **{latest_date_str}** — the most recent weather data available.")
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

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron"
    )

    risk_features = []
    for _, row in res_df.iterrows():
        pm25_val = row.get("pm2_5")
        health_level_val = row.get("health_level")
        health_advice_val = row.get("health_advice")
        risk_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["LON"], row["LAT"]]},
            "properties": {
                "risk_level": row["risk_level"],
                "fire_probability_pct": round(row["fire_probability"] * 100, 1) if pd.notna(row["fire_probability"]) else None,
                "t2m_max": round(row["T2M_MAX"], 1),
                "rh2m": round(row["RH2M"], 1),
                "health_level": health_level_val if pd.notna(health_level_val) else "N/A",
                "pm2_5": round(float(pm25_val), 0) if pd.notna(pm25_val) else None,
                "health_advice": health_advice_val if pd.notna(health_advice_val) else "",
            },
        })

    def risk_style_function(feature):
        color = COLOR_MAP.get(feature["properties"]["risk_level"], "gray")
        return {"fillColor": color, "color": color, "weight": 2, "fillOpacity": 0.7}

    folium.GeoJson(
        {"type": "FeatureCollection", "features": risk_features},
        name="Fire risk zones",
        marker=folium.CircleMarker(radius=14, fill=True),
        style_function=risk_style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["risk_level", "fire_probability_pct", "t2m_max", "rh2m", "health_level", "pm2_5"],
            aliases=["Risk:", "Fire probability (%):", "Max temp (°C):", "Humidity (%):",
                     "Air quality:", "PM2.5 (µg/m³):"],
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["risk_level", "fire_probability_pct", "t2m_max", "rh2m", "health_level", "pm2_5", "health_advice"],
            aliases=["Risk:", "Fire probability (%):", "Max temp (°C):", "Humidity (%):",
                     "Air quality:", "PM2.5 (µg/m³):", "Advice:"],
            max_width=260,
        ),
    ).add_to(m)

    heat_data = [[row["LAT"], row["LON"], row["fire_probability"]] for _, row in res_df.iterrows() if pd.notna(row["fire_probability"])]
    heat_fg = folium.FeatureGroup(name="Risk density (heatmap)", show=False)
    HeatMap(heat_data, radius=18, blur=22, max_zoom=8).add_to(heat_fg)
    heat_fg.add_to(m)

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
                aqi_html = (
                    f"<br><b>🫁 Current AQI:</b> "
                    f"{float(us_aqi_shelter):.0f} "
                    f"(PM2.5: {float(pm25_shelter):.1f} µg/m³)"
                )
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

    folium.LayerControl(position="topleft", collapsed=False).add_to(m)

    if len(shelters) > max_markers:
        st.caption(
            f"Showing the {max_markers} largest-capacity locations "
            f"out of {len(shelters)} matching your filters."
        )

    st_folium(m, use_container_width=True, height=550, returned_objects=[])

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
