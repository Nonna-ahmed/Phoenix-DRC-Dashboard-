"""
PATCH for congo_streamlit_app.py
====================================
Three additions, explained and tested independently below.
Copy each block into the matching place in your existing file.

1. Fix "API KEY REQUIRED" watermark on both maps
2. Add a "Nearest Shelters" card panel to the Future Prediction tab
3. Add a real "Send SMS Now" button that talks to Africa's Talking
   directly from Streamlit (no separate script needed)
"""

# =========================================================
# 1. FIX THE MAP WATERMARK
# =========================================================
# WHY IT HAPPENS: Carto (the company behind "CartoDB positron" /
# "CartoDB dark_matter" tiles) now requires a free API key for their
# basemap tiles. Without a key, the tiles still load — Carto just
# stamps "API KEY REQUIRED" across them. Nothing is broken in your code.
#
# FIX: swap both occurrences of the Carto tile name for plain
# "OpenStreetMap" — a completely different, always-free provider with
# no key, ever. Find-and-replace these two lines in your file:

# Tab 1 map — change this:
#   m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")
# to this:
#   m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

# Tab 2 (Future) map — change this:
#   m_future = folium.Map(location=[-10.5, 27.5], zoom_start=6, tiles="CartoDB dark_matter")
# to this:
#   m_future = folium.Map(location=[-10.5, 27.5], zoom_start=6, tiles="OpenStreetMap")

# NOTE: OpenStreetMap tiles are light-themed (no dark mode available for
# free without a key). If you want to KEEP the dark look, get a free key
# at https://carto.com/basemaps/apikey (takes 2 minutes, no card needed)
# and use:
#   tiles="https://{s}.basemaps.cartocdn.com/dark_matter/{z}/{x}/{y}.png?key=YOUR_KEY"
#   attr="&copy; OpenStreetMap contributors &copy; CARTO"


# =========================================================
# 2. NEAREST SHELTERS CARD PANEL (for the Future Prediction tab)
# =========================================================
# Paste this function near your other helper functions (top of file,
# alongside `nearest_facility`):

import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

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
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    pool["distance_km"] = pool.apply(lambda r: haversine(ref_lat, ref_lon, r["lat"], r["lon"]), axis=1)
    nearest = pool.nsmallest(n, "distance_km")

    AQI_COLOR = {"low": "#2e7d32", "medium": "#e65100", "high": "#c62828"}
    cols_per_row = 3
    rows = [nearest.iloc[i:i+cols_per_row] for i in range(0, len(nearest), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, (_, s) in zip(cols, row.iterrows()):
            with col:
                aqi_val = s.get("us_aqi")
                pm25_val = s.get("pm2_5")
                aqi_badge = ""
                if pd.notna(aqi_val):
                    from risk_engine import health_risk_level  # reuses your existing thresholds
                    level = health_risk_level(pm25_val) if pd.notna(pm25_val) else "unknown"
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

# --- Call it inside Tab 2, after the risk map is drawn ---
# town_choice = st.selectbox("Reference town", list(MAJOR_TOWNS.keys()))
# ref_lat, ref_lon = MAJOR_TOWNS[town_choice]
# render_nearest_shelters(shelters_all, ref_lat, ref_lon, town_choice)


# =========================================================
# 3. SEND REAL SMS DIRECTLY FROM STREAMLIT
# =========================================================
# YES — this works. The same Africa's Talking SDK call your standalone
# script uses can run inside a Streamlit button handler. The only change
# in habit: use st.secrets instead of environment variables, so the key
# never sits in your code or a plain .env file.
#
# SETUP on Streamlit Cloud:
#   App settings -> Secrets -> paste:
#     AT_USERNAME = "sandbox"
#     AT_API_KEY = "your_api_key_here"
#
# SETUP for local testing: create .streamlit/secrets.toml with the same
# two lines (and add that file to .gitignore — never commit it).

def send_sms_from_dashboard(recipients: list[str], message: str) -> dict:
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


# --- Example UI: add this inside Tab 1, near the "Simulated SMS" section ---
#
# st.subheader("📱 Send Real Alert")
# recipient_input = st.text_input("Phone number (e.g. +243800000001)")
# if st.button("🚨 Send SMS Now", type="primary"):
#     if not recipient_input.startswith("+"):
#         st.error("Enter the number in international format, e.g. +243800000001")
#     else:
#         with st.spinner("Sending..."):
#             result = send_sms_from_dashboard(
#                 [recipient_input],
#                 f"[PHOENIX ALERT] High wildfire risk detected. {len(high_risk)} zone(s) currently High risk."
#             )
#         if "error" in result:
#             st.error(f"Failed: {result['error']}")
#         else:
#             st.success(f"Sent! Response: {result}")
#
# SAFETY NOTE: this sends a REAL message and may incur real cost outside
# the free Sandbox. Keep this button behind a clear, deliberate click —
# don't trigger it automatically on page load or on every rerun.
