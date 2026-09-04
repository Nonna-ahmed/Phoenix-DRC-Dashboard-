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
import time
import json
import io
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
# Nearest-facility helper (vectorized haversine — fast even for
# many risk zones x many shelters/hospitals)
# -------------------------------------------------------------
def nearest_facility(zone_lats, zone_lons, cand_df):
    """For each (zone_lat, zone_lon), find the nearest row in cand_df
    (which must have 'lat'/'lon' columns). Returns a DataFrame with the
    matched candidate's info plus a 'distance_km' column, aligned by
    position to zone_lats/zone_lons. If cand_df is empty, all fields
    come back as None/NaN."""
    if cand_df.empty:
        return pd.DataFrame({
            "name": [None] * len(zone_lats),
            "lat": [None] * len(zone_lats),
            "lon": [None] * len(zone_lats),
            "available": [None] * len(zone_lats),
            "capacity": [None] * len(zone_lats),
            "province": [None] * len(zone_lats),
            "distance_km": [np.nan] * len(zone_lats),
        })

    R = 6371.0  # Earth radius, km
    zlat = np.radians(np.asarray(zone_lats, dtype=float))[:, None]
    zlon = np.radians(np.asarray(zone_lons, dtype=float))[:, None]
    clat = np.radians(cand_df["lat"].to_numpy(dtype=float))[None, :]
    clon = np.radians(cand_df["lon"].to_numpy(dtype=float))[None, :]

    dlat = clat - zlat
    dlon = clon - zlon
    a = np.sin(dlat / 2) ** 2 + np.cos(zlat) * np.cos(clat) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    nearest_idx = np.argmin(dist, axis=1)
    nearest_dist = dist[np.arange(len(zone_lats)), nearest_idx]

    matched = cand_df.iloc[nearest_idx].reset_index(drop=True)
    matched["distance_km"] = nearest_dist
    return matched


def fetch_route(start_lat, start_lon, end_lat, end_lon, profile="foot"):
    """Fetches a real road/path route between two points via OSRM's free
    public demo server (no API key needed). profile: 'foot' (walking) or
    'driving'. Returns (route_latlon_list, distance_km, duration_min), or
    (None, None, None) on any failure — the public OSRM demo server is
    best-effort/rate-limited, not guaranteed for heavy production use, so
    callers should fall back to the straight-line distance already shown
    elsewhere if this fails rather than blocking on it."""
    url = (f"https://router.project-osrm.org/route/v1/{profile}/"
           f"{start_lon},{start_lat};{end_lon},{end_lat}"
           f"?overview=full&geometries=geojson")
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None, None, None
        route = data["routes"][0]
        coords = route["geometry"]["coordinates"]  # [[lon, lat], ...]
        latlon = [[c[1], c[0]] for c in coords]
        return latlon, route["distance"] / 1000, route["duration"] / 60
    except Exception:
        return None, None, None


MAJOR_TOWNS = {
    "Lubumbashi (Haut-Katanga)": (-11.6609, 27.4794),
    "Kolwezi (Lualaba)": (-10.7167, 25.4667),
    "Kalemie (Tanganyika)": (-5.9475, 29.1947),
}


def render_nearest_shelters(shelters_all: pd.DataFrame, ref_lat: float, ref_lon: float,
                             ref_label: str, n: int = 8):
    """Render a ranked card grid of the nearest shelters to a reference point,
    styled like the 'Nearest Shelters' panel — distance, category, and an
    AQI badge colored by health level."""
    st.subheader("🏥 Nearest Shelters")
    st.caption(f"Ranked by distance from {ref_label} · live sample")

    pool = shelters_all[shelters_all["is_shelter"] | (shelters_all["category"] == "health_facility")].copy()
    if pool.empty:
        st.info("No shelter data available.")
        return

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    pool["distance_km"] = pool.apply(lambda r: haversine(ref_lat, ref_lon, r["lat"], r["lon"]), axis=1)
    nearest = pool.nsmallest(n, "distance_km")

    AQI_COLOR = {"low": "#2e7d32", "medium": "#e65100", "high": "#c62828"}
    cols_per_row = 3
    rows = [nearest.iloc[i:i + cols_per_row] for i in range(0, len(nearest), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, (_, s) in zip(cols, row.iterrows()):
            with col:
                aqi_val = s.get("us_aqi")
                pm25_val = s.get("pm2_5")
                aqi_badge = ""
                if pd.notna(aqi_val):
                    try:
                        from risk_engine import health_risk_level  # reuses existing thresholds
                        level = health_risk_level(pm25_val) if pd.notna(pm25_val) else "unknown"
                    except ImportError:
                        level = "unknown"
                    color = AQI_COLOR.get(level, "#616161")
                    aqi_badge = (f'<span style="border:1px solid {color};color:{color};'
                                 f'border-radius:12px;padding:2px 8px;font-size:12px;">'
                                 f'AQI {aqi_val:.0f} · {level.title()}</span>')
                st.markdown(
                    f"""
                    <div style="border:1px solid #e0e0e0;border-radius:10px;padding:12px;margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;">
                            <b>{s['name']}</b>
                            <span style="color:#e65100;">{s['distance_km']:.1f} km</span>
                        </div>
                        <div style="color:#757575;font-size:13px;">{s['category'].replace('_',' ').title()} · {s.get('province','')}</div>
                        <div style="margin-top:6px;">{aqi_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def send_sms_from_dashboard(recipients: list, message: str) -> dict:
    """Sends a real SMS via Africa's Talking using credentials from
    st.secrets. Returns the API response, or an error dict."""
    import africastalking

    username = st.secrets.get("AT_USERNAME")
    api_key = st.secrets.get("AT_API_KEY")
    if not username or not api_key:
        return {"error": "AT_USERNAME / AT_API_KEY not set in Streamlit secrets."}

    africastalking.initialize(username, api_key)
    sms = africastalking.SMS
    try:
        return sms.send(message, recipients)
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------------
# Bilingual alert text (English / French)
# -------------------------------------------------------------
# USSD code assigned by Africa's Talking's Sandbox for this app's USSD
# channel (pointed at {API_BASE_URL}/ussd). Production (real telecom
# delivery, not just the Sandbox simulator) requires applying separately
# with DRC's regulator — ARPTC — for a dedicated/shared code there.
USSD_SERVICE_CODE = "*384*99838#"

_ALERT_TEMPLATES = {
    "en": "[PHOENIX ALERT] High wildfire risk detected. {count} zone(s) currently High risk "
          "as of {date}. Move livestock/valuables now. No smartphone? Dial {ussd} from any "
          "phone for shelter info — no internet needed.",
    "fr": "[ALERTE PHOENIX] Risque élevé d'incendie détecté. {count} zone(s) actuellement à "
          "haut risque au {date}. Déplacez le bétail/les biens de valeur maintenant. Pas de "
          "smartphone ? Composez {ussd} depuis n'importe quel téléphone pour les infos abris "
          "— sans internet.",
    "sw": "[TAHADHARI YA PHOENIX] Hatari kubwa ya moto imegunduliwa. Maeneo {count} yana hatari "
          "kubwa kufikia {date}. Hamisha mifugo/vitu vya thamani sasa. Huna simu janja? Piga "
          "{ussd} kutoka simu yoyote kwa taarifa za makazi — hauitaji intaneti.",
}

_LANG_LABELS = {"en": "English", "fr": "Français", "sw": "Kiswahili"}


def build_alert_message(count: int, date_str: str, lang: str) -> str:
    """Builds the alert text in English, French, Swahili, or all three
    (lang: 'en', 'fr', 'sw', 'all')."""
    if lang == "all":
        return "\n---\n".join(
            _ALERT_TEMPLATES[l].format(count=count, date=date_str, ussd=USSD_SERVICE_CODE)
            for l in ("en", "fr", "sw")
        )
    return _ALERT_TEMPLATES[lang].format(count=count, date=date_str, ussd=USSD_SERVICE_CODE)


_ZONE_ALERT_TEMPLATES = {
    "en": "[PHOENIX ALERT] High wildfire risk at ({lat}, {lon}). Risk: {risk} ({prob}). "
          "{aqi} Nearest shelter: {shelter} ({dist}). Move now. Dial {ussd} for more info "
          "— no internet needed.",
    "fr": "[ALERTE PHOENIX] Risque élevé d'incendie à ({lat}, {lon}). Risque : {risk} ({prob}). "
          "{aqi} Abri le plus proche : {shelter} ({dist}). Déplacez-vous maintenant. Composez "
          "{ussd} pour plus d'infos — sans internet.",
    "sw": "[TAHADHARI YA PHOENIX] Hatari kubwa ya moto karibu na ({lat}, {lon}). Hatari: {risk} "
          "({prob}). {aqi} Makazi ya karibu: {shelter} ({dist}). Hamia sasa. Piga {ussd} kwa "
          "maelezo zaidi — hauitaji intaneti.",
}


def build_zone_alert_message(zone: dict, lang: str) -> str:
    """Builds an alert message for ONE specific zone (the one the user clicked
    on the map), including its risk level, air quality, and nearest shelter —
    instead of a dashboard-wide zone count."""
    def render(single_lang: str) -> str:
        prob = f"{zone['fire_probability_pct']}%" if zone.get("fire_probability_pct") is not None else "N/A"
        if zone.get("pm2_5") is not None:
            aqi_texts = {
                "en": f"Air quality: {zone['pm2_5']:.0f} µg/m³ ({zone.get('health_level') or 'N/A'}).",
                "fr": f"Qualité de l'air : {zone['pm2_5']:.0f} µg/m³ ({zone.get('health_level') or 'N/A'}).",
                "sw": f"Ubora wa hewa: {zone['pm2_5']:.0f} µg/m³ ({zone.get('health_level') or 'N/A'}).",
            }
        else:
            aqi_texts = {
                "en": "Air quality: no live data.",
                "fr": "Qualité de l'air : pas de données en direct.",
                "sw": "Ubora wa hewa: hakuna data ya sasa.",
            }
        not_found = {"en": "Not found", "fr": "Introuvable", "sw": "Haipatikani"}
        shelter = zone.get("shelter_name") or not_found[single_lang]
        dist = f"{zone['shelter_dist']:.1f} km" if zone.get("shelter_dist") is not None else "?"
        return _ZONE_ALERT_TEMPLATES[single_lang].format(
            lat=round(zone["lat"], 3), lon=round(zone["lon"], 3),
            risk=zone.get("risk_level") or "N/A", prob=prob, aqi=aqi_texts[single_lang],
            shelter=shelter, dist=dist, ussd=USSD_SERVICE_CODE,
        )

    if lang == "all":
        return "\n---\n".join(render(l) for l in ("en", "fr", "sw"))
    return render(lang)


def render_language_picker(key: str) -> str:
    """Renders a language selector and returns 'en' / 'fr' / 'sw' / 'all'."""
    choice = st.radio(
        "Alert language / Langue de l'alerte / Lugha ya tahadhari",
        ["English", "Français", "Kiswahili", "All / Tous / Zote"],
        horizontal=True,
        key=key,
    )
    return {"English": "en", "Français": "fr", "Kiswahili": "sw", "All / Tous / Zote": "all"}[choice]


def send_voice_call_from_dashboard(recipients: list, message: str, lang: str = "en") -> dict:
    """Places a real outbound voice call via Africa's Talking's Voice REST API.
    Africa's Talking calls back {API_BASE_URL}/voice once the call connects to
    fetch what to actually say — the message + language are passed through as
    clientState so the backend (congo_api.py) knows what to speak.

    NOTE: Africa's Talking's Voice API does NOT accept sandbox credentials at
    all (unlike SMS, which works fine in sandbox) — it requires a real, live
    production app username + API key. That's also why there's no
    "voice.sandbox.africastalking.com" — it doesn't exist. Set
    AT_VOICE_USERNAME / AT_VOICE_API_KEY in Streamlit secrets to your
    production app's credentials; if those aren't set, this falls back to
    AT_USERNAME / AT_API_KEY (which will fail if those are still "sandbox")."""
    import json

    username = st.secrets.get("AT_VOICE_USERNAME") or st.secrets.get("AT_USERNAME")
    api_key = st.secrets.get("AT_VOICE_API_KEY") or st.secrets.get("AT_API_KEY")
    caller_id = st.secrets.get("AT_VOICE_CALLER_ID")  # virtual number Africa's Talking issues for Voice
    if not username or not api_key or not caller_id:
        return {"error": "AT_VOICE_USERNAME / AT_VOICE_API_KEY / AT_VOICE_CALLER_ID not set in Streamlit secrets "
                          "(or AT_USERNAME / AT_API_KEY as a fallback)."}
    if username == "sandbox":
        return {"error": "Africa's Talking Voice doesn't accept sandbox credentials. Create a live production "
                          "app in the Africa's Talking dashboard and set AT_VOICE_USERNAME / AT_VOICE_API_KEY "
                          "in Streamlit secrets to that app's username/API key."}

    url = "https://voice.africastalking.com/call"
    try:
        resp = requests.post(
            url,
            headers={"apiKey": api_key, "Accept": "application/json"},
            data={
                "username": username,
                "from": caller_id,
                "to": ",".join(recipients),
                "clientState": json.dumps({"message": message, "lang": lang}),
            },
            timeout=20,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def render_alert_dispatch_section(zone_count: int, ref_date_str: str, key_prefix: str, zone: dict = None):
    """Renders the full 'Send Real Alert' block: language picker, phone input,
    SMS button, Voice call button, and a USSD info card. Shared by Tab 1 and
    Tab 2. If `zone` is given (a specific clicked risk zone with lat/lon/
    risk_level/fire_probability_pct/pm2_5/health_level/shelter_name/
    shelter_dist), the message targets that exact spot instead of the
    dashboard-wide zone count."""
    if zone:
        st.subheader("📱☎️ Send Alert for Selected Zone")
        st.caption(f"Targeting the zone at ({zone['lat']:.3f}, {zone['lon']:.3f}) — "
                   f"{zone.get('risk_level') or 'N/A'} risk.")
    else:
        st.subheader("📱☎️ Send Real Alert")

    lang = render_language_picker(key=f"{key_prefix}_lang")
    recipient_input = st.text_input("Phone number (e.g. +243800000001)", key=f"{key_prefix}_recipient")

    if zone:
        message = build_zone_alert_message(zone, lang)
    else:
        message = build_alert_message(zone_count, ref_date_str, lang)

    col_sms, col_voice = st.columns(2)
    with col_sms:
        if st.button("🚨 Send SMS Now", type="primary", key=f"{key_prefix}_sms_btn"):
            if not recipient_input.startswith("+"):
                st.error("Enter the number in international format, e.g. +243800000001")
            else:
                with st.spinner("Sending SMS..."):
                    sms_result = send_sms_from_dashboard([recipient_input], message)
                if "error" in sms_result:
                    st.error(f"Failed: {sms_result['error']}")
                else:
                    st.success(f"Sent! Response: {sms_result}")
    with col_voice:
        if st.button("☎️ Call Now", key=f"{key_prefix}_voice_btn"):
            if not recipient_input.startswith("+"):
                st.error("Enter the number in international format, e.g. +243800000001")
            else:
                voice_lang = "en" if lang == "all" else lang
                voice_message = (build_zone_alert_message(zone, voice_lang) if zone
                                  else build_alert_message(zone_count, ref_date_str, voice_lang))
                with st.spinner("Placing call..."):
                    voice_result = send_voice_call_from_dashboard(
                        [recipient_input], voice_message, voice_lang
                    )
                if "error" in voice_result:
                    st.error(f"Failed: {voice_result['error']}")
                else:
                    st.success(f"Call placed! Response: {voice_result}")

    st.caption(
        "⚠️ SMS sends for free in the Sandbox. **Voice calls require a live production Africa's "
        "Talking app** (sandbox credentials aren't accepted for Voice at all) — set "
        "`AT_VOICE_USERNAME` / `AT_VOICE_API_KEY` / `AT_VOICE_CALLER_ID` in Streamlit secrets, and "
        "test with your own number first since every call is billed for real."
    )

    st.markdown(
        f"""
        <div style="border:1px solid #90caf9;background:#e3f2fd;border-radius:10px;padding:12px;margin-top:10px;">
            <b>📟 No smartphone? No internet? No problem.</b><br>
            <span style="font-size:14px;">
            Dial <b>{USSD_SERVICE_CODE}</b> from any basic phone to check the current fire risk and
            nearest shelter — works on any network, no app or data connection needed, in English or French.<br>
            <i>Composez {USSD_SERVICE_CODE} depuis n'importe quel téléphone pour vérifier le risque
            d'incendie et l'abri le plus proche — fonctionne sans internet ni application.</i>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_climate():
    df = pd.read_csv("phoenix_climate_2020_2026.csv")
    # NASA POWER has a ~3-5 day processing lag; the most recent unprocessed
    # days come back as the fill value -999 instead of real numbers. Mark
    # those as NaN (don't drop the row) so the date itself still counts as
    # "available" — compute_results_for_date() reports "No data" for the
    # specific points that are missing, instead of us silently rolling the
    # default date back to an older, fully-clean one.
    df = df.dropna(subset=["YEAR", "DOY"])  # a few rows have genuinely missing YEAR/DOY
    df["YEAR"] = df["YEAR"].astype(int)
    df["DOY"] = df["DOY"].astype(int)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + \
                 pd.to_timedelta(df["DOY"] - 1, unit="D")
    weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
    df[weather_cols] = df[weather_cols].where(df[weather_cols] >= -900)  # -999 -> NaN
    if "WD2M" in df.columns:  # wind direction — only present for dates fetched after this was added
        df["WD2M"] = df["WD2M"].where(df["WD2M"] >= -900)
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

# Default to the latest date where EVERY grid cell has complete data — no
# NaN/"No data" on the default view. If the most recent day(s) still have
# gaps from NASA POWER's processing lag, this rolls back to the most recent
# fully-clean date instead. The person can still manually pick a newer,
# partially-populated date with the slider below if they want to.
_weather_cols_check = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
_total_cells = climate[["LAT", "LON"]].drop_duplicates().shape[0]
_complete_counts = (
    climate.dropna(subset=_weather_cols_check)
    .groupby("date").size()
)
_full_coverage_dates = _complete_counts[_complete_counts == _total_cells]
LATEST_AVAILABLE_DATE = (_full_coverage_dates.index.max() if not _full_coverage_dates.empty
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
    value=LATEST_AVAILABLE_DATE,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
)
st.sidebar.caption(f"Defaulting to {pd.Timestamp(LATEST_AVAILABLE_DATE).strftime('%Y-%m-%d')} — "
                    f"the most recent date with complete data for every grid cell. Newer dates may "
                    f"exist but can still show 'No data' for some points if NASA POWER hasn't "
                    f"finished processing them yet.")
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
# NASA FIRMS — confirmed active fire detections (satellite, NOT prediction)
# -------------------------------------------------------------
# Bounding box covering Haut-Katanga, Lualaba & Tanganyika: west,south,east,north
FIRMS_BBOX = "24,-13,31,-4"
FIRMS_SOURCE = "VIIRS_NOAA20_NRT"  # near-real-time VIIRS/NOAA-20 detections
FIRMS_DAY_RANGE = 2  # last 2 days of detections


@st.cache_data(ttl=1800)  # FIRMS refreshes every few hours; 30 min cache is plenty
def fetch_active_fires():
    """Fetches CONFIRMED active fire detections from NASA FIRMS (satellite
    hotspots — actually observed, not predicted) for the Katanga bounding
    box. Requires a free MAP_KEY (see https://firms.modaps.eosdis.nasa.gov/api/map_key/)
    in Streamlit secrets as FIRMS_MAP_KEY. Returns an empty DataFrame (not
    an error) if the key is missing or the request fails, so this never
    blocks the rest of the dashboard from working."""
    map_key = st.secrets.get("FIRMS_MAP_KEY")
    if not map_key:
        return pd.DataFrame(), "no_key"

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{FIRMS_SOURCE}/{FIRMS_BBOX}/{FIRMS_DAY_RANGE}"
    try:
        df = pd.read_csv(url, timeout=15)
    except Exception:
        return pd.DataFrame(), "fetch_failed"

    if df.empty or "latitude" not in df.columns:
        return pd.DataFrame(), "empty"
    return df, "ok"


@st.cache_data(ttl=180)  # short cache — citizen reports should feel near-live
def fetch_fire_reports():
    """Fetches recent citizen-submitted fire reports from the API
    (crowd-sourced via the USSD 'Report a fire' menu). Returns an empty
    DataFrame (not an error) on any failure, so this never blocks the rest
    of the dashboard."""
    try:
        resp = requests.get(f"{API_BASE_URL}/fire-reports", params={"hours": 72}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame(), "fetch_failed"
    if not data:
        return pd.DataFrame(), "empty"
    return pd.DataFrame(data), "ok"


def fetch_population_estimate(lat: float, lon: float, buffer_deg: float = 0.25,
                               year: int = 2020, max_wait_seconds: int = 20):
    """Estimates the population within a small square area around (lat, lon)
    using WorldPop's free public stats API (no key needed, up to 1000 calls/
    day). This is an ASYNC API — it returns a task ID immediately, then we
    poll for the result. Only called on-demand (a button click on ONE
    selected zone), never automatically for every zone, since polling can
    take several seconds. Returns (population_count, None) on success, or
    (None, error_message) on failure/timeout."""
    half = buffer_deg / 2
    polygon = {
        "type": "Feature", "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon - half, lat - half], [lon + half, lat - half],
                [lon + half, lat + half], [lon - half, lat + half],
                [lon - half, lat - half],
            ]],
        },
    }
    try:
        resp = requests.get(
            "https://api.worldpop.org/v1/services/stats",
            params={"dataset": "wpgppop", "year": year, "geojson": json.dumps(polygon)},
            timeout=10,
        )
        resp.raise_for_status()
        task_id = resp.json().get("taskid")
        if not task_id:
            return None, "WorldPop didn't return a task ID."

        for _ in range(max_wait_seconds):
            time.sleep(1)
            poll = requests.get(f"https://api.worldpop.org/v1/tasks/{task_id}", timeout=10)
            poll.raise_for_status()
            pdata = poll.json()
            status = pdata.get("status")
            if status == "finished":
                total = pdata.get("data", {}).get("total_population")
                return total, None
            if status == "failed":
                return None, "WorldPop couldn't process this area."
        return None, "Timed out waiting for WorldPop (try again — it's sometimes just slow)."
    except Exception as e:
        return None, f"Request failed: {e}"


def build_area_report_pdf(zone: dict, shelters_all: pd.DataFrame, fires_df: pd.DataFrame,
                           pop_estimate=None) -> bytes:
    """Builds a one-page offline-friendly PDF report for a selected zone —
    risk level, air quality, nearest shelter details, nearby confirmed
    fires, and the USSD code — for a field team to take somewhere without
    internet access. Returns the PDF as bytes, ready for st.download_button."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBig", parent=styles["Title"], textColor=colors.HexColor("#c0392b"))
    story = []

    story.append(Paragraph("🔥 PHOENIX — Wildfire Area Report", title_style))
    story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} (local time)",
                            styles["Normal"]))
    story.append(Spacer(1, 0.6 * cm))

    risk_color = {"High": colors.HexColor("#c0392b"), "Medium": colors.HexColor("#e67e22"),
                  "Low": colors.HexColor("#27ae60")}.get(zone.get("risk_level"), colors.grey)
    risk_style = ParagraphStyle("Risk", parent=styles["Heading2"], textColor=risk_color)
    story.append(Paragraph(f"Risk Level: {zone.get('risk_level', 'N/A')}", risk_style))

    prob = f"{zone['fire_probability_pct']}%" if zone.get("fire_probability_pct") is not None else "N/A"
    aqi = (f"{zone['pm2_5']:.0f} µg/m³ ({zone.get('health_level', 'N/A')})"
           if zone.get("pm2_5") is not None else "No live data")

    info_rows = [
        ["Location (lat, lon)", f"{zone['lat']:.4f}, {zone['lon']:.4f}"],
        ["Fire probability", prob],
        ["Air quality (PM2.5)", aqi],
    ]
    if pop_estimate is not None:
        info_rows.append(["Estimated population nearby (~28×28 km)", f"{pop_estimate:,.0f} people"])

    info_table = Table(info_rows, colWidths=[7 * cm, 9 * cm])
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("Nearest Shelter", styles["Heading2"]))
    if zone.get("shelter_name"):
        shelter_match = shelters_all[shelters_all["name"] == zone["shelter_name"]]
        cap_txt = "N/A"
        if not shelter_match.empty:
            s = shelter_match.iloc[0]
            cap_txt = f"{int(s['available'])} / {int(s['capacity'])} spots open"
        shelter_rows = [
            ["Name", zone["shelter_name"]],
            ["Distance (straight-line)", f"{zone['shelter_dist']:.1f} km"],
            ["Capacity", cap_txt],
        ]
        shelter_table = Table(shelter_rows, colWidths=[7 * cm, 9 * cm])
        shelter_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ]))
        story.append(shelter_table)
    else:
        story.append(Paragraph("No nearby shelter found.", styles["Normal"]))
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("Nearby Confirmed Fires (NASA FIRMS, last 2 days)", styles["Heading2"]))
    if fires_df is not None and not fires_df.empty:
        nearby = fires_df.copy()
        nearby["dist_km"] = nearby.apply(
            lambda f: 111 * ((f["latitude"] - zone["lat"]) ** 2 + (f["longitude"] - zone["lon"]) ** 2) ** 0.5,
            axis=1,
        )
        nearby = nearby[nearby["dist_km"] <= 25].sort_values("dist_km")
        if not nearby.empty:
            fire_rows = [["Date/Time (UTC)", "Distance", "Confidence"]]
            for _, f in nearby.head(10).iterrows():
                fire_rows.append([f"{f.get('acq_date', '')} {f.get('acq_time', '')}",
                                   f"{f['dist_km']:.1f} km", str(f.get("confidence", "n/a"))])
            fire_table = Table(fire_rows, colWidths=[6 * cm, 4 * cm, 6 * cm])
            fire_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ]))
            story.append(fire_table)
        else:
            story.append(Paragraph("None within 25 km.", styles["Normal"]))
    else:
        story.append(Paragraph("No confirmed-fire data available right now.", styles["Normal"]))
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("No Internet? No Smartphone?", styles["Heading2"]))
    story.append(Paragraph(
        f"Dial <b>{USSD_SERVICE_CODE}</b> from any basic phone to check current fire risk and the "
        f"nearest shelter — works on any network, no app or data connection needed, in English, "
        f"French, or Swahili.", styles["Normal"],
    ))

    doc.build(story)
    return buffer.getvalue()


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
    is_latest_available = pd.Timestamp(selected_date).normalize() == pd.Timestamp(LATEST_AVAILABLE_DATE).normalize()
    latest_date_str = pd.Timestamp(LATEST_AVAILABLE_DATE).strftime("%Y-%m-%d")

    # Open-Meteo's air-quality API only ever returns RIGHT NOW's live reading
    # — it has no historical record. We still show it for any date within
    # about the last year so it's usable while browsing recent history, but
    # it's always today's live reading, never the actual historical air
    # quality for an older selected date — the label below makes that clear
    # instead of implying it's specific to the selected date.
    days_from_latest = (pd.Timestamp(LATEST_AVAILABLE_DATE).normalize()
                         - pd.Timestamp(selected_date).normalize()).days
    show_live_aq = 0 <= days_from_latest <= 365

    res_df, aqi_fetch_success_count, _ = compute_results_for_date(
        day_data.to_dict("records"), doy, show_live_aq
    )

    if show_live_aq:
        if aqi_fetch_success_count == 0:
            st.warning("🫁 **Air-quality data is temporarily unavailable** — could not reach the live "
                       "Open-Meteo service right now (this is a network hiccup, not a missing feature; "
                       "it will resume automatically once the connection is back). Fire-risk predictions "
                       "below are unaffected.")
        elif aqi_fetch_success_count < len(day_data):
            st.info(f"🫁 Live air-quality retrieved for {aqi_fetch_success_count}/{len(day_data)} zones "
                    f"(a few requests timed out — this is normal and will vary run to run).")
        else:
            st.caption("🫁 Air-quality readings below are **live, right now** (fetched just now, cached "
                       "for 15 min) — not historical for the selected date, since Open-Meteo only "
                       f"provides current conditions. Fire-risk inputs are from **{latest_date_str}**.")
    else:
        st.caption(f"ℹ️ Air-quality is only shown for dates within the last year of "
                   f"**{latest_date_str}** — move the slider closer to the right to see it.")

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
        tiles="OpenStreetMap"
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

    # ---------------------------------------------------------------
    # Wind direction arrows — only available for dates fetched after WD2M
    # was added to update_climate_data.py; older historical dates won't
    # have this column, so the layer is simply empty (not an error) then.
    # ---------------------------------------------------------------
    if "WD2M" in day_data.columns and day_data["WD2M"].notna().any():
        wind_fg = folium.FeatureGroup(name="🧭 Wind direction", show=False)
        for _, wrow in day_data.dropna(subset=["WD2M"]).iterrows():
            # WD2M is the direction wind blows FROM (meteorological convention);
            # rotate +180° so the arrow visually points where the wind is
            # blowing TO — i.e. the direction fire would likely spread.
            heading = (float(wrow["WD2M"]) + 180) % 360
            arrow_html = (
                f'<div style="transform: rotate({heading}deg); font-size: 22px; '
                f'color: #01579b; text-shadow: 0 0 3px white;">➤</div>'
            )
            folium.Marker(
                location=[wrow["LAT"], wrow["LON"]],
                icon=folium.DivIcon(html=arrow_html, icon_size=(24, 24), icon_anchor=(12, 12)),
                tooltip=f"Wind from {wrow['WD2M']:.0f}° · likely spread direction shown",
            ).add_to(wind_fg)
        wind_fg.add_to(m)

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

    # ---------------------------------------------------------------
    # NASA FIRMS — confirmed active fires (satellite-observed, not predicted)
    # ---------------------------------------------------------------
    fires_df, fires_status = fetch_active_fires()
    if fires_status == "ok" and not fires_df.empty:
        fires_fg = folium.FeatureGroup(name="🔥 Confirmed fires (NASA FIRMS, live)", show=True)
        for _, f in fires_df.iterrows():
            confidence = f.get("confidence", "n/a")
            folium.Marker(
                location=[f["latitude"], f["longitude"]],
                icon=folium.Icon(color="orange", icon="fire", prefix="fa"),
                tooltip=f"Confirmed fire · {f.get('acq_date', '')} {f.get('acq_time', '')}",
                popup=folium.Popup(
                    f"<b>🔥 Confirmed active fire</b><br>"
                    f"Satellite-detected (NASA FIRMS)<br>"
                    f"Date: {f.get('acq_date', 'n/a')} {f.get('acq_time', 'n/a')} UTC<br>"
                    f"Satellite: {f.get('satellite', 'n/a')}<br>"
                    f"Confidence: {confidence}<br>"
                    f"Fire radiative power: {f.get('frp', 'n/a')} MW",
                    max_width=260,
                ),
            ).add_to(fires_fg)
        fires_fg.add_to(m)

    # ---------------------------------------------------------------
    # Citizen-reported fires (crowd-sourced via USSD "Report a fire")
    # ---------------------------------------------------------------
    reports_df, reports_status = fetch_fire_reports()
    if reports_status == "ok" and not reports_df.empty:
        reports_fg = folium.FeatureGroup(name="📢 Citizen fire reports (last 72h)", show=True)
        for _, rep in reports_df.iterrows():
            folium.Marker(
                location=[rep["lat"], rep["lon"]],
                icon=folium.Icon(color="darkred", icon="bullhorn", prefix="fa"),
                tooltip=f"Citizen report · {rep.get('province', '')}",
                popup=folium.Popup(
                    f"<b>📢 Citizen-reported fire</b><br>"
                    f"Province: {rep.get('province', 'n/a')}<br>"
                    f"Reported: {rep.get('reported_at_utc', 'n/a')} UTC<br>"
                    f"<small>Approximate location (province reference point — USSD "
                    f"callers have no GPS), not a precise pin.</small>",
                    max_width=260,
                ),
            ).add_to(reports_fg)
        reports_fg.add_to(m)

    # Route from the selected zone (if any) to its nearest shelter. Uses the
    # PREVIOUS click's selection from session_state — this run's own click
    # (if any) is only known after st_folium returns further below, so the
    # click handler triggers an immediate rerun to make the route appear
    # right away instead of one interaction later.
    _sel = st.session_state.get("tab1_selected_zone")
    if _sel and _sel.get("shelter_lat") is not None:
        _route_mode = st.session_state.get("tab1_route_mode", "foot")
        _route_latlon, _route_dist_km, _route_dur_min = fetch_route(
            _sel["lat"], _sel["lon"], _sel["shelter_lat"], _sel["shelter_lon"], profile=_route_mode
        )
        if _route_latlon:
            _mode_label = "walking" if _route_mode == "foot" else "driving"
            folium.PolyLine(
                _route_latlon, color="#1976d2", weight=5, opacity=0.85,
                tooltip=f"{_route_dist_km:.1f} km · ~{_route_dur_min:.0f} min ({_mode_label})",
            ).add_to(m)

            # OSRM snaps the start/end to the nearest road IT knows about —
            # in remote areas with incomplete OpenStreetMap road coverage,
            # that snapped point can be a real gap away from the actual
            # zone/shelter location. Close that visual gap with a dashed
            # line so it's clear which part is a real mapped road vs. an
            # off-road estimate, instead of the line just looking "cut off".
            def _gap_km(lat1, lon1, lat2, lon2):
                dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
                a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
                return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))

            _route_start, _route_end = _route_latlon[0], _route_latlon[-1]
            _start_gap = _gap_km(_route_start[0], _route_start[1], _sel["lat"], _sel["lon"])
            _end_gap = _gap_km(_route_end[0], _route_end[1], _sel["shelter_lat"], _sel["shelter_lon"])
            if _start_gap > 0.05:
                folium.PolyLine(
                    [[_sel["lat"], _sel["lon"]], _route_start],
                    color="#1976d2", weight=3, opacity=0.6, dash_array="6,8",
                    tooltip=f"Off-road (~{_start_gap:.1f} km) — no mapped road here, straight-line estimate",
                ).add_to(m)
            if _end_gap > 0.05:
                folium.PolyLine(
                    [_route_end, [_sel["shelter_lat"], _sel["shelter_lon"]]],
                    color="#1976d2", weight=3, opacity=0.6, dash_array="6,8",
                    tooltip=f"Off-road (~{_end_gap:.1f} km) — no mapped road here, straight-line estimate",
                ).add_to(m)

            folium.Marker(
                [_sel["lat"], _sel["lon"]],
                icon=folium.Icon(color="red", icon="fire", prefix="fa"),
                tooltip="Selected zone",
            ).add_to(m)
            folium.Marker(
                [_sel["shelter_lat"], _sel["shelter_lon"]],
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
                tooltip=_sel.get("shelter_name") or "Nearest shelter",
            ).add_to(m)
            st.session_state["tab1_route_info"] = {
                "distance_km": _route_dist_km, "duration_min": _route_dur_min, "mode": _route_mode,
            }
        else:
            st.session_state["tab1_route_info"] = None

    folium.LayerControl(position="topleft", collapsed=False).add_to(m)

    if len(shelters) > max_markers:
        st.caption(
            f"Showing the {max_markers} largest-capacity locations "
            f"out of {len(shelters)} matching your filters."
        )

    map_data = st_folium(
        m, use_container_width=True, height=550,
        returned_objects=["last_active_drawing"],
    )

    if fires_status == "no_key":
        st.caption("🔥 **Confirmed fires (NASA FIRMS)** layer needs a free API key — add `FIRMS_MAP_KEY` "
                   "to Streamlit secrets (get one at https://firms.modaps.eosdis.nasa.gov/api/map_key/) "
                   "to see satellite-confirmed fires alongside the predicted risk zones.")
    elif fires_status == "fetch_failed":
        st.caption("🔥 Couldn't reach NASA FIRMS just now (network hiccup) — confirmed-fire markers are "
                   "temporarily unavailable. Predicted risk zones above are unaffected.")

    # ---------------------------------------------------------------
    # Zone selection (click a risk zone on the map above)
    # ---------------------------------------------------------------
    if "tab1_selected_zone" not in st.session_state:
        st.session_state.tab1_selected_zone = None
    if "tab1_route_mode" not in st.session_state:
        st.session_state.tab1_route_mode = "foot"
    if "tab1_last_click_key" not in st.session_state:
        st.session_state.tab1_last_click_key = None

    clicked = map_data.get("last_active_drawing") if map_data else None
    if clicked and clicked.get("geometry", {}).get("type") == "Point":
        lon_c, lat_c = clicked["geometry"]["coordinates"]
        click_key = (round(lat_c, 6), round(lon_c, 6))
        props = clicked.get("properties", {})
        # Only act on risk-zone features (ignore shelter markers), and only
        # on a NEW click — st_folium keeps returning the same last click on
        # every rerun, and without this guard the st.rerun() below (which
        # makes the route appear immediately) would loop forever.
        if props.get("risk_level") is not None and click_key != st.session_state.tab1_last_click_key:
            st.session_state.tab1_last_click_key = click_key
            shelter_pool = shelters_all[shelters_all["is_shelter"]]
            nearest = nearest_facility([lat_c], [lon_c], shelter_pool)
            shelter_name = nearest.iloc[0]["name"] if not nearest.empty else None
            shelter_dist = nearest.iloc[0]["distance_km"] if not nearest.empty else None
            shelter_lat = float(nearest.iloc[0]["lat"]) if not nearest.empty else None
            shelter_lon = float(nearest.iloc[0]["lon"]) if not nearest.empty else None
            st.session_state.tab1_selected_zone = {
                "lat": lat_c, "lon": lon_c,
                "risk_level": props.get("risk_level"),
                "fire_probability_pct": props.get("fire_probability_pct"),
                "pm2_5": props.get("pm2_5"),
                "health_level": props.get("health_level"),
                "shelter_name": shelter_name,
                "shelter_dist": shelter_dist,
                "shelter_lat": shelter_lat,
                "shelter_lon": shelter_lon,
            }
            st.rerun()  # redraw the map now, with the route included

    zone = st.session_state.tab1_selected_zone
    if zone:
        st.markdown("### 📍 Selected Zone")
        zc1, zc2, zc3 = st.columns(3)
        zc1.metric("Risk Level", zone["risk_level"] or "N/A")
        zc2.metric("Fire Probability",
                   f"{zone['fire_probability_pct']}%" if zone["fire_probability_pct"] is not None else "N/A")
        aqi_label = (f"{zone['pm2_5']:.0f} µg/m³ ({zone['health_level']})"
                     if zone["pm2_5"] is not None else "No live data")
        zc3.metric("Air Quality (PM2.5)", aqi_label)
        if zone["shelter_name"]:
            st.caption(f"🏠 Nearest shelter: **{zone['shelter_name']}** ({zone['shelter_dist']:.1f} km away, straight-line)")

            mode_label = st.radio("Route by", ["🚶 Walking", "🚗 Driving"], horizontal=True, key="tab1_route_mode_radio")
            new_mode = "foot" if "Walking" in mode_label else "driving"
            if new_mode != st.session_state.tab1_route_mode:
                st.session_state.tab1_route_mode = new_mode
                st.rerun()

            route_info = st.session_state.get("tab1_route_info")
            if route_info:
                mode_txt = "walking" if route_info["mode"] == "foot" else "driving"
                st.success(f"🛣️ Road route: **{route_info['distance_km']:.1f} km**, "
                           f"~**{route_info['duration_min']:.0f} min** ({mode_txt}) — shown on the map above.")
            else:
                st.caption("⚠️ Road route unavailable right now (routing service may be busy) — "
                           "showing straight-line distance only.")
        else:
            st.caption("🏠 No nearby shelter found.")

        st.markdown("**👥 Affected population**")
        if st.button("Estimate population in this area (WorldPop)", key="tab1_pop_btn"):
            with st.spinner("Querying WorldPop (can take up to ~20 seconds)..."):
                pop_count, pop_error = fetch_population_estimate(zone["lat"], zone["lon"])
            if pop_error:
                st.warning(f"Couldn't get a population estimate: {pop_error}")
                st.session_state.tab1_pop_estimate = None
            else:
                st.session_state.tab1_pop_estimate = pop_count
                st.info(f"👥 Approximately **{pop_count:,.0f} people** live within ~28×28 km around this zone "
                        f"(WorldPop 2020 estimate — the most recent stable, complete dataset available).")

        st.markdown("**📄 Offline report**")
        pdf_bytes = build_area_report_pdf(
            zone, shelters_all, fires_df if fires_status == "ok" else pd.DataFrame(),
            pop_estimate=st.session_state.get("tab1_pop_estimate"),
        )
        st.download_button(
            "📄 Download area report (PDF)", data=pdf_bytes,
            file_name=f"phoenix_area_report_{zone['lat']:.3f}_{zone['lon']:.3f}.pdf",
            mime="application/pdf", key="tab1_pdf_download",
        )
        st.caption("For field teams heading somewhere without internet access — risk, air quality, "
                   "nearest shelter, and nearby confirmed fires, all on one printable page.")

        if st.button("✕ Clear selection", key="tab1_clear_zone"):
            st.session_state.tab1_selected_zone = None
            st.session_state.tab1_route_info = None
            st.session_state.tab1_pop_estimate = None
            st.rerun()
    else:
        st.caption("💡 Click a risk zone on the map to select it, see the route to its nearest shelter, "
                   "and send a targeted alert for that exact spot.")

    # ---------------------------------------------------------------
    # Nearest shelter & hospital table (for at-risk zones)
    # ---------------------------------------------------------------
    st.subheader("🏥 Nearest Shelter & Hospital for At-Risk Zones")

    at_risk = res_df[res_df["risk_level"].isin(["High", "Medium"])].copy()

    if at_risk.empty:
        st.info("No Medium/High risk zones for the selected date — nothing to match to shelters right now.")
    else:
        shelters_pool = shelters_all[shelters_all["is_shelter"]]
        hospitals_pool = shelters_all[shelters_all["category"] == "health_facility"]

        nearest_shelter = nearest_facility(at_risk["LAT"], at_risk["LON"], shelters_pool)
        nearest_hospital = nearest_facility(at_risk["LAT"], at_risk["LON"], hospitals_pool)

        table = pd.DataFrame({
            "Risk": at_risk["risk_level"].values,
            "Fire Probability": (at_risk["fire_probability"] * 100).round(1).astype(str).values,
            "Zone Lat": at_risk["LAT"].round(3).values,
            "Zone Lon": at_risk["LON"].round(3).values,
            "Nearest Shelter": nearest_shelter["name"].values,
            "Shelter Dist (km)": nearest_shelter["distance_km"].round(1).values,
            "Shelter Avail/Cap": [
                f"{a}/{c}" if pd.notna(a) else "—"
                for a, c in zip(nearest_shelter["available"], nearest_shelter["capacity"])
            ],
            "Nearest Hospital": nearest_hospital["name"].values,
            "Hospital Dist (km)": nearest_hospital["distance_km"].round(1).values,
        })
        table["Fire Probability"] = table["Fire Probability"] + "%"
        table = table.sort_values(["Risk", "Shelter Dist (km)"], ascending=[True, True])

        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Shows Medium/High risk zones only, matched to the closest evacuee shelter "
            "(school or place of worship) and closest health facility by straight-line distance. "
            "Distances are approximate (haversine), not driving distance."
        )

    st.markdown("---")

    # ---------------------------------------------------------------
    # Update shelter availability (for shelter staff)
    # ---------------------------------------------------------------
    with st.expander("🏠 Update Shelter Availability (for shelter staff)"):
        st.caption("If you manage a shelter, update how many spots are currently open so evacuees "
                   "aren't directed to a shelter that's already full.")
        shelter_options = shelters_all[shelters_all["is_shelter"]].sort_values("name")
        shelter_choice = st.selectbox(
            "Shelter", shelter_options["name"],
            key="shelter_update_choice",
        )
        chosen = shelter_options[shelter_options["name"] == shelter_choice].iloc[0]
        new_available = st.number_input(
            "Currently available spots", min_value=0, max_value=int(chosen["capacity"]),
            value=int(chosen["available"]), key="shelter_update_value",
        )
        if st.button("Update availability", key="shelter_update_btn"):
            try:
                resp = requests.patch(
                    f"{API_BASE_URL}/shelters/{chosen['osm_id']}/availability",
                    json={"available": int(new_available)}, timeout=10,
                )
                resp.raise_for_status()
                st.success(f"✅ Updated {shelter_choice}: {new_available}/{int(chosen['capacity'])} spots available.")
                st.cache_data.clear()  # so the map/table reflect the change on next load
            except Exception as e:
                st.error(f"Failed to update: {e}")
        st.caption("⚠️ Updates apply to the live server for now — they're overwritten by the next "
                   "scheduled data refresh unless also saved back to the CSV in the repo.")

    st.markdown("---")

    # ---------------------------------------------------------------
    # Was the forecast right? — spot-check predictions against confirmed fires
    # ---------------------------------------------------------------
    st.subheader("📊 Was the Forecast Right?")
    st.caption("Compares NASA FIRMS confirmed fire detections (last 2 days) against what PHOENIX "
               "predicted for the nearest grid cell in the 1-3 days before each fire was observed.")

    if fires_status != "ok" or fires_df.empty:
        st.info("No confirmed fires in the last 2 days to validate against yet.")
    else:
        _grid_points = climate[["LAT", "LON"]].drop_duplicates().reset_index(drop=True)
        _validation_rows = []
        for _, _fire in fires_df.iterrows():
            f_lat, f_lon = _fire["latitude"], _fire["longitude"]
            try:
                acq_date = pd.to_datetime(_fire["acq_date"])
            except Exception:
                continue
            _dists = ((_grid_points["LAT"] - f_lat) ** 2 + (_grid_points["LON"] - f_lon) ** 2) ** 0.5
            g_lat, g_lon = _grid_points.loc[_dists.idxmin(), ["LAT", "LON"]]

            was_flagged, checked_any = False, False
            for days_before in (1, 2, 3):
                check_date = acq_date - pd.Timedelta(days=days_before)
                row = climate[(climate["LAT"] == g_lat) & (climate["LON"] == g_lon)
                               & (climate["date"] == check_date)]
                if row.empty or pd.isna(row.iloc[0]["T2M_MAX"]):
                    continue
                row = row.iloc[0]
                checked_any = True
                pred = predict_fire_risk(
                    lat=g_lat, lon=g_lon, doy=check_date.dayofyear,
                    t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
                    rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
                )
                if pred["risk_level"] in ("High", "Medium"):
                    was_flagged = True
                    break

            _validation_rows.append({
                "Fire date": _fire.get("acq_date"),
                "Lat": round(f_lat, 3), "Lon": round(f_lon, 3),
                "Nearest grid cell": f"{g_lat:.2f}, {g_lon:.2f}",
                "Flagged High/Medium in prior 1-3 days?": (
                    "✅ Yes" if was_flagged else ("❓ No prior data" if not checked_any else "❌ No")
                ),
            })

        if _validation_rows:
            val_df = pd.DataFrame(_validation_rows)
            checked_count = (val_df["Flagged High/Medium in prior 1-3 days?"] != "❓ No prior data").sum()
            flagged_count = (val_df["Flagged High/Medium in prior 1-3 days?"] == "✅ Yes").sum()
            if checked_count > 0:
                st.metric("Prediction hit rate",
                          f"{flagged_count}/{checked_count} confirmed fires were in a zone flagged "
                          f"High/Medium risk in the prior 1-3 days")
            st.dataframe(val_df, use_container_width=True, hide_index=True)
            st.caption("This is a simple spot-check, not a rigorous accuracy metric — it only covers "
                       "the last 2 days of confirmed fires, and doesn't account for zones the model "
                       "correctly flagged where no fire happened to be detected by satellite (true "
                       "negatives aren't visible here).")
        else:
            st.info("Not enough overlapping data yet to validate.")

    st.markdown("---")

    # ---------------------------------------------------------------
    # Seasonal fire-risk analysis
    # ---------------------------------------------------------------
    st.subheader("📈 Seasonal Fire Risk Analysis")
    st.caption("Typical fire-risk probability by month, based on the historical average weather for "
               "each month across all years in the dataset (region-wide average location). Useful for "
               "advance planning ahead of the fire season — no extra data source needed, this reuses "
               "the climate data already collected.")

    _monthly = climate.copy()
    _monthly["month"] = _monthly["date"].dt.month
    _monthly_avg = _monthly.groupby("month")[["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]].mean().dropna()

    if not _monthly_avg.empty:
        _month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        _region_lat, _region_lon = climate["LAT"].mean(), climate["LON"].mean()
        _season_rows = []
        for month, row in _monthly_avg.iterrows():
            representative_doy = pd.Timestamp(2023, int(month), 15).dayofyear  # any non-leap year, day 15
            pred = predict_fire_risk(
                lat=_region_lat, lon=_region_lon, doy=representative_doy,
                t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"], rh2m=row["RH2M"],
                ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
            )
            _season_rows.append({
                "Month": _month_names[int(month) - 1], "month_num": int(month),
                "Typical Fire Probability (%)": round(pred["fire_probability"] * 100, 1),
            })
        _season_df = pd.DataFrame(_season_rows).sort_values("month_num")

        fig = px.bar(
            _season_df, x="Month", y="Typical Fire Probability (%)",
            title="Typical Fire Risk by Month (region-wide average, all years)",
            color="Typical Fire Probability (%)",
            color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        peak_month = _season_df.loc[_season_df["Typical Fire Probability (%)"].idxmax(), "Month"]
        st.caption(f"📌 Historically, **{peak_month}** shows the highest typical fire risk in this "
                   f"region — useful for planning resources ahead of the season.")
    else:
        st.info("Not enough historical data yet to compute a seasonal pattern.")

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
        tiles="OpenStreetMap"
    )

    future_features = []
    for _, row in df_future.iterrows():
        future_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {
                "risk_level": row["risk_level"],
                "fire_probability_pct": f"{row['fire_probability']:.1%}",
                "t2m_max": round(row["t2m_max"], 1),
                "rh2m": round(row["rh2m"], 1),
            },
        })

    def future_style_function(feature):
        color = FUTURE_COLOR_MAP.get(feature["properties"]["risk_level"], "gray")
        return {"fillColor": color, "color": color, "weight": 2, "fillOpacity": 0.7}

    folium.GeoJson(
        {"type": "FeatureCollection", "features": future_features},
        name="Predicted risk zones",
        marker=folium.CircleMarker(radius=12, fill=True),
        style_function=future_style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["risk_level", "fire_probability_pct", "t2m_max", "rh2m"],
            aliases=["Risk:", "Fire probability:", "Max temp (°C):", "Humidity (%):"],
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["risk_level", "fire_probability_pct", "t2m_max", "rh2m"],
            aliases=["⚠️ Risk Level:", "🔥 Probability:", "🌡️ Max Temp:", "💧 Humidity:"],
            max_width=250,
        ),
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

    # Route from the selected zone (if any) to its nearest shelter — same
    # approach as Tab 1: uses the PREVIOUS click's selection from
    # session_state, since this run's own click is only known after
    # st_folium returns below; the click handler triggers an immediate
    # rerun so the route appears right away.
    _sel2 = st.session_state.get("tab2_selected_zone")
    if _sel2 and _sel2.get("shelter_lat") is not None:
        _route_mode2 = st.session_state.get("tab2_route_mode", "foot")
        _route_latlon2, _route_dist_km2, _route_dur_min2 = fetch_route(
            _sel2["lat"], _sel2["lon"], _sel2["shelter_lat"], _sel2["shelter_lon"], profile=_route_mode2
        )
        if _route_latlon2:
            _mode_label2 = "walking" if _route_mode2 == "foot" else "driving"
            folium.PolyLine(
                _route_latlon2, color="#1976d2", weight=5, opacity=0.85,
                tooltip=f"{_route_dist_km2:.1f} km · ~{_route_dur_min2:.0f} min ({_mode_label2})",
            ).add_to(m_future)

            def _gap_km2(lat1, lon1, lat2, lon2):
                dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
                a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
                return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))

            _rstart2, _rend2 = _route_latlon2[0], _route_latlon2[-1]
            _start_gap2 = _gap_km2(_rstart2[0], _rstart2[1], _sel2["lat"], _sel2["lon"])
            _end_gap2 = _gap_km2(_rend2[0], _rend2[1], _sel2["shelter_lat"], _sel2["shelter_lon"])
            if _start_gap2 > 0.05:
                folium.PolyLine(
                    [[_sel2["lat"], _sel2["lon"]], _rstart2],
                    color="#1976d2", weight=3, opacity=0.6, dash_array="6,8",
                    tooltip=f"Off-road (~{_start_gap2:.1f} km) — no mapped road here, straight-line estimate",
                ).add_to(m_future)
            if _end_gap2 > 0.05:
                folium.PolyLine(
                    [_rend2, [_sel2["shelter_lat"], _sel2["shelter_lon"]]],
                    color="#1976d2", weight=3, opacity=0.6, dash_array="6,8",
                    tooltip=f"Off-road (~{_end_gap2:.1f} km) — no mapped road here, straight-line estimate",
                ).add_to(m_future)

            folium.Marker(
                [_sel2["lat"], _sel2["lon"]],
                icon=folium.Icon(color="red", icon="fire", prefix="fa"),
                tooltip="Selected zone",
            ).add_to(m_future)
            folium.Marker(
                [_sel2["shelter_lat"], _sel2["shelter_lon"]],
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
                tooltip=_sel2.get("shelter_name") or "Nearest shelter",
            ).add_to(m_future)
            st.session_state["tab2_route_info"] = {
                "distance_km": _route_dist_km2, "duration_min": _route_dur_min2, "mode": _route_mode2,
            }
        else:
            st.session_state["tab2_route_info"] = None

    map_data_future = st_folium(
        m_future, width="100%", height=600,
        returned_objects=["last_active_drawing"],
    )

    # ---------------------------------------------------------------
    # Zone selection (click a predicted risk zone on the map above)
    # ---------------------------------------------------------------
    if "tab2_selected_zone" not in st.session_state:
        st.session_state.tab2_selected_zone = None
    if "tab2_route_mode" not in st.session_state:
        st.session_state.tab2_route_mode = "foot"
    if "tab2_last_click_key" not in st.session_state:
        st.session_state.tab2_last_click_key = None

    clicked2 = map_data_future.get("last_active_drawing") if map_data_future else None
    if clicked2 and clicked2.get("geometry", {}).get("type") == "Point":
        lon_c2, lat_c2 = clicked2["geometry"]["coordinates"]
        click_key2 = (round(lat_c2, 6), round(lon_c2, 6))
        props2 = clicked2.get("properties", {})
        if props2.get("risk_level") is not None and click_key2 != st.session_state.tab2_last_click_key:
            st.session_state.tab2_last_click_key = click_key2
            shelter_pool2 = shelters_all[shelters_all["is_shelter"]]
            nearest2 = nearest_facility([lat_c2], [lon_c2], shelter_pool2)
            shelter_name2 = nearest2.iloc[0]["name"] if not nearest2.empty else None
            shelter_dist2 = nearest2.iloc[0]["distance_km"] if not nearest2.empty else None
            shelter_lat2 = float(nearest2.iloc[0]["lat"]) if not nearest2.empty else None
            shelter_lon2 = float(nearest2.iloc[0]["lon"]) if not nearest2.empty else None
            st.session_state.tab2_selected_zone = {
                "lat": lat_c2, "lon": lon_c2,
                "risk_level": props2.get("risk_level"),
                "fire_probability_pct": props2.get("fire_probability_pct"),
                "shelter_name": shelter_name2,
                "shelter_dist": shelter_dist2,
                "shelter_lat": shelter_lat2,
                "shelter_lon": shelter_lon2,
            }
            st.rerun()

    zone2 = st.session_state.tab2_selected_zone
    if zone2:
        st.markdown("### 📍 Selected Zone (Forecast)")
        zc1b, zc2b = st.columns(2)
        zc1b.metric("Risk Level", zone2["risk_level"] or "N/A")
        zc2b.metric("Fire Probability", zone2["fire_probability_pct"] or "N/A")
        if zone2["shelter_name"]:
            st.caption(f"🏠 Nearest shelter: **{zone2['shelter_name']}** ({zone2['shelter_dist']:.1f} km away, straight-line)")

            mode_label2 = st.radio("Route by", ["🚶 Walking", "🚗 Driving"], horizontal=True, key="tab2_route_mode_radio")
            new_mode2 = "foot" if "Walking" in mode_label2 else "driving"
            if new_mode2 != st.session_state.tab2_route_mode:
                st.session_state.tab2_route_mode = new_mode2
                st.rerun()

            route_info2 = st.session_state.get("tab2_route_info")
            if route_info2:
                mode_txt2 = "walking" if route_info2["mode"] == "foot" else "driving"
                st.success(f"🛣️ Road route: **{route_info2['distance_km']:.1f} km**, "
                           f"~**{route_info2['duration_min']:.0f} min** ({mode_txt2}) — shown on the map above.")
            else:
                st.caption("⚠️ Road route unavailable right now (routing service may be busy) — "
                           "showing straight-line distance only.")
        else:
            st.caption("🏠 No nearby shelter found.")
        if st.button("✕ Clear selection", key="tab2_clear_zone"):
            st.session_state.tab2_selected_zone = None
            st.session_state.tab2_route_info = None
            st.rerun()
    else:
        st.caption("💡 Click a predicted risk zone on the map to see the route to its nearest shelter.")

    st.markdown("---")

    town_choice = st.selectbox("Reference town", list(MAJOR_TOWNS.keys()), key="future_ref_town")
    ref_lat, ref_lon = MAJOR_TOWNS[town_choice]
    render_nearest_shelters(shelters_all, ref_lat, ref_lon, town_choice)

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

    # ---------------------------------------------------------------
    # Real alert dispatch: SMS + Voice + USSD info (Africa's Talking)
    # ---------------------------------------------------------------
    render_alert_dispatch_section(len(high_risk), date_str, key_prefix="tab2")

    st.markdown("---")
    st.caption(
        f"🔮 Predictions powered by PHOENIX Forecast API | "
        f"Date: {date_str} | Method: Climatology | API: {API_BASE_URL}"
    )

# Footer
st.markdown("---")
st.caption("🔥 PHOENIX DRC — Wildfire Early Warning System | Built with Streamlit")
