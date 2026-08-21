"""
PHOENIX — Congo (Katanga) Wildfire Early Warning & Shelter Matching Dashboard
================================================================================
Same structure as the Algeria streamlit_app.py, pointed at the Congo model
and shelters database.

Run locally:
    pip install streamlit folium streamlit-folium xgboost pandas requests
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

from congo_predict import predict_fire_risk
from risk_engine import get_alert
from air_quality import fetch_live_pm25

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="PHOENIX — Congo Fire Early Warning", layout="wide")
st.title("🔥 PHOENIX — Congo (Katanga) Wildfire Early Warning & Shelter Matching")
st.caption("Haut-Katanga, Lualaba & Tanganyika provinces (DRC) — AI for All Hackathon")

COLOR_MAP = {"Low": "green", "Medium": "orange", "High": "red"}

# ---------------------------------------------------------------
# Load data (cached so it only loads once per session)
# ---------------------------------------------------------------
@st.cache_data
def load_climate():
    df = pd.read_csv("phoenix_climate_2020_2026.csv")
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + \
                 pd.to_timedelta(df["DOY"] - 1, unit="D")
    return df

@st.cache_data
def load_shelters():
    # Real facility data: GRID3 health facilities + OSM/GRID3 schools & places
    # of worship, across Haut-Katanga, Lualaba, and Tanganyika.
    df = pd.read_csv("drc_katanga_shelters_final.csv")
    df = df.rename(columns={"capacity_estimate": "capacity"})
    df["name"] = df["name"].fillna(df["category"] + " (unnamed)")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(25).astype(int)
    if "available" not in df.columns:
        df["available"] = df["capacity"]
    # Schools & places of worship are real usable buildings (is_shelter=True).
    # Health facilities are support resources only (is_shelter=False) —
    # same rule as the Algeria prototype.
    return df[["osm_id", "category", "name", "lat", "lon", "capacity", "available",
               "is_shelter", "province", "pm2_5", "us_aqi", "observation_time"]]

climate = load_climate()
shelters_all = load_shelters()

# ---------------------------------------------------------------
# Sidebar: choose which facility types / provinces to show
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Run model on all grid cells for the selected date
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔴 High risk zones", int((res_df["risk_level"] == "High").sum()))
col2.metric("🟡 Medium risk zones", int((res_df["risk_level"] == "Medium").sum()))
col3.metric("🟢 Low risk zones", int((res_df["risk_level"] == "Low").sum()))
shelter_only = shelters[shelters["is_shelter"]]
col4.metric("Shelters with capacity", int((shelter_only["available"] > 0).sum()))

# ---------------------------------------------------------------
# Map
# ---------------------------------------------------------------
st.subheader("Live Risk Map")

center_lat = res_df["LAT"].mean()
center_lon = res_df["LON"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB positron"
)

# ---------------------------------------------------------------
# Add wildfire risk zones
# ---------------------------------------------------------------
for _, row in res_df.iterrows():

    health_html = ""

    # Safely handle health level / PM2.5 values
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
                health_html = (
                    f"<br><b>🫁 Air quality:</b> {health_level}"
                )
        else:
            health_html = (
                f"<br><b>🫁 Air quality:</b> {health_level}"
            )

    risk_color = COLOR_MAP.get(
        row["risk_level"],
        "gray"
    )

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


# ---------------------------------------------------------------
# Add shelters and support resources
# ---------------------------------------------------------------
shelter_cluster = MarkerCluster(
    name="Shelters & Resources"
).add_to(m)

shelters_to_draw = (
    shelters
    .sort_values("capacity", ascending=False)
    .head(max_markers)
)

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
            aqi_html = (
                f"<br><b>🫁 PM2.5:</b> "
                f"{float(pm25_shelter):.1f} µg/m³"
            )

        observation_time = s.get("observation_time")

        if pd.notna(observation_time):
            aqi_html += (
                f"<br><small>as of {observation_time}</small>"
            )

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
            f"Est. capacity: "
            f"{s['available']}/{s['capacity']}"
            f"{aqi_html}",
            max_width=260,
        ),
        tooltip=s["name"],
    ).add_to(shelter_cluster)


# ---------------------------------------------------------------
# Map marker information
# ---------------------------------------------------------------
if len(shelters) > max_markers:
    st.caption(
        f"Showing the {max_markers} largest-capacity locations "
        f"out of {len(shelters)} matching your filters "
        f"(adjust the slider in the sidebar to show more). "
        f"All {len(shelters)} are still used in the "
        f"shelter-matching table below."
    )

# ---------------------------------------------------------------
# Display map
# ---------------------------------------------------------------
st_folium(
    m,
    width=1100,
    height=550,
    returned_objects=[]
)
