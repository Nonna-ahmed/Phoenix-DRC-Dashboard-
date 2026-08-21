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
center_lat, center_lon = res_df["LAT"].mean(), res_df["LON"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

for _, row in res_df.iterrows():
    health_html = ""
    if row["health_level"]:
        health_html = f"<br><b>🫁 Air quality:</b> {row['health_level'].title()} (PM2.5: {row['pm2_5']:.0f} µg/m³)<br>{row['health_advice']}"
    folium.CircleMarker(
        location=[row["LAT"], row["LON"]],
        radius=14,
        color=COLOR_MAP[row["risk_level"]],
        fill=True,
        fill_color=COLOR_MAP[row["risk_level"]],
        fill_opacity=0.7,
        weight=2,
        popup=folium.Popup(
            f"<b>{row['risk_level']} risk</b><br>"
            f"Fire probability: {row['fire_probability']*100:.1f}%<br>"
            f"Max Temp: {row['T2M_MAX']:.1f}°C | Humidity: {row['RH2M']:.1f}%"
            f"{health_html}",
            max_width=260,
        ),
        tooltip=f"{row['risk_level']} risk",
    ).add_to(m)

shelter_cluster = MarkerCluster(name="Shelters & Resources").add_to(m)
shelters_to_draw = shelters.sort_values("capacity", ascending=False).head(max_markers)
for _, s in shelters_to_draw.iterrows():
    if s["is_shelter"]:
        color, kind = "blue", "Shelter"
    else:
        color, kind = "darkcyan", "Support resource (not for housing evacuees)"
    aqi_html = ""
    if pd.notna(s.get("pm2_5")):
        aqi_html = f"<br><b>🫁 Current AQI:</b> {s['us_aqi']:.0f} (PM2.5: {s['pm2_5']:.1f} µg/m³)<br><small>as of {s['observation_time']}</small>"
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        weight=1,
        popup=folium.Popup(
            f"<b>{s['name']}</b><br>{kind}<br>{s['province']}<br>"
            f"Est. capacity: {s['available']}/{s['capacity']}{aqi_html}",
            max_width=260,
        ),
        tooltip=s["name"],
    ).add_to(shelter_cluster)

if len(shelters) > max_markers:
    st.caption(f"Showing the {max_markers} largest-capacity locations out of {len(shelters)} matching your filters "
               f"(adjust the slider in the sidebar to show more). All {len(shelters)} are still used in the "
               f"shelter-matching table below.")

st_folium(m, width=1100, height=550, returned_objects=[])

# ---------------------------------------------------------------
# High-risk zones -> nearest shelter matching
# ---------------------------------------------------------------
def haversine_km_vec(lat1, lon1, lat_arr, lon_arr):
    """Vectorized haversine distance (km) from one point to an array of points."""
    R = 6371
    lat1r, lon1r = radians(lat1), radians(lon1)
    lat2r, lon2r = np.radians(lat_arr), np.radians(lon_arr)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

st.subheader("⚠️ High-Risk Zones → Nearest Available Shelter")
high_risk = res_df[res_df["risk_level"] == "High"].copy()

if high_risk.empty:
    st.success("No high-risk zones detected for this date.")
else:
    matches = []
    avail = shelters[(shelters["is_shelter"]) & (shelters["available"] > 0)].copy()
    for _, zone in high_risk.iterrows():
        if avail.empty:
            matches.append({"Zone (lat,lon)": f"{zone['LAT']}, {zone['LON']}",
                             "Fire probability": f"{zone['fire_probability']*100:.1f}%",
                             "Nearest shelter": "⚠️ No capacity available nearby"})
            continue
        dists = haversine_km_vec(zone["LAT"], zone["LON"], avail["lat"].values, avail["lon"].values)
        nearest_idx = dists.argmin()
        nearest = avail.iloc[nearest_idx]
        matches.append({
            "Zone (lat,lon)": f"{zone['LAT']}, {zone['LON']}",
            "Fire probability": f"{zone['fire_probability']*100:.1f}%",
            "Nearest shelter": nearest["name"],
            "Province": nearest["province"],
            "Distance (km)": f"{dists[nearest_idx]:.1f}",
            "Shelter capacity": f"{nearest['available']}/{nearest['capacity']}",
        })
    st.dataframe(pd.DataFrame(matches), use_container_width=True)
    if (pd.DataFrame(matches).get("Distance (km)", pd.Series(dtype=float)).astype(str).str.replace(".","",1).str.isdigit()).any():
        long_dist = pd.DataFrame(matches)
        if "Distance (km)" in long_dist.columns:
            long_dist["_d"] = pd.to_numeric(long_dist["Distance (km)"], errors="coerce")
            if (long_dist["_d"] > 30).any():
                st.caption("⚠️ Some matched shelters are 30+ km away — shelter coverage in Lualaba/Tanganyika "
                           "is sparser than Haut-Katanga; this is a genuine data-coverage gap, not a bug.")

# ---------------------------------------------------------------
# Simulated SMS alert log
# ---------------------------------------------------------------
st.subheader("📱 Simulated SMS/USSD Alerts")
if not high_risk.empty:
    for _, zone in high_risk.iterrows():
        health_line = f" {zone['health_advice']}" if zone.get("health_advice") else ""
        st.code(
            f"[ALERT] High wildfire risk near ({zone['LAT']}, {zone['LON']}). "
            f"Probability: {zone['fire_probability']*100:.0f}%. "
            f"Move livestock/valuables and check nearest shelter now.{health_line}",
            language=None,
        )
else:
    st.info("No alerts to send for this date.")

st.caption("Data sources: NASA FIRMS (fire history) · NASA POWER (climate) · GRID3/OSM (facilities) · "
           "Model: XGBoost, threshold-tuned")
