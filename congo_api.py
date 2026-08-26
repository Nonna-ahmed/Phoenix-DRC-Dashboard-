"""
Congo (Katanga) Backend API — FastAPI
=========================================
Same structure as the Algeria api.py, pointed at the Congo model and
shelters database. Reuses risk_engine.py and air_quality.py as-is
(both are region-agnostic).

Run locally:
    pip install -r requirements.txt
    uvicorn congo_api:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs.

Endpoints:
    GET  /health
    POST /predict
    GET  /risk-map?date=YYYY-MM-DD
    GET  /shelters
    GET  /shelters/nearest
    GET  /alerts?date=YYYY-MM-DD
    POST /ussd    (Africa's Talking USSD callback — bilingual EN/FR menu,
                   works from any basic phone, no internet/app needed)
    POST /voice   (Africa's Talking Voice callback — speaks the alert text
                   passed in clientState when the dashboard places a call)
"""

from datetime import date as date_type
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, List
import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from congo_predict import predict_fire_risk
from risk_engine import get_alert
from air_quality import fetch_live_pm25

# -----------------------------------------------------------------
# App & data loading
# -----------------------------------------------------------------
app = FastAPI(
    title="PHOENIX API — Congo (Katanga)",
    description="Wildfire early-warning & shelter-matching API for Haut-Katanga, Lualaba & Tanganyika (DRC)",
    version="1.0.0",
)

CLIMATE_DF = pd.read_csv("phoenix_climate_2020_2026.csv")
CLIMATE_DF = CLIMATE_DF.dropna(subset=["YEAR", "DOY"])  # a few rows have genuinely missing YEAR/DOY
CLIMATE_DF["YEAR"] = CLIMATE_DF["YEAR"].astype(int)
CLIMATE_DF["DOY"] = CLIMATE_DF["DOY"].astype(int)
CLIMATE_DF["date"] = pd.to_datetime(CLIMATE_DF["YEAR"].astype(str), format="%Y") + \
                      pd.to_timedelta(CLIMATE_DF["DOY"] - 1, unit="D")
# NASA POWER has a ~3-5 day processing lag; unprocessed recent days come back
# as the fill value -999 instead of real numbers. Mark those as NaN (don't
# drop the row) so the date itself still counts as "available" — /risk-map
# reports "No data" for the specific points that are missing, instead of
# silently rolling the whole default date back to an older, fully-clean one.
_weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
CLIMATE_DF[_weather_cols] = CLIMATE_DF[_weather_cols].where(CLIMATE_DF[_weather_cols] >= -900)

# True latest date in the data (not the latest date with full grid coverage —
# individual missing points are handled per-row in /risk-map instead).
LATEST_AVAILABLE_DATE = CLIMATE_DF["date"].max().normalize()

SHELTERS_DF = pd.read_csv("drc_katanga_shelters_final.csv")
SHELTERS_DF = SHELTERS_DF.rename(columns={"capacity_estimate": "capacity"})
if "available" not in SHELTERS_DF.columns:
    SHELTERS_DF["available"] = SHELTERS_DF["capacity"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# Reference point per province, used by the USSD menu to give a quick risk +
# nearest-shelter summary without needing the caller's GPS location (basic
# phones on USSD have none). Same towns used as reference points in the
# Streamlit dashboard's "Nearest Shelters" panel.
PROVINCE_REF_POINTS = {
    "Haut-Katanga": (-11.6609, 27.4794),   # Lubumbashi
    "Lualaba": (-10.7167, 25.4667),        # Kolwezi
    "Tanganyika": (-5.9475, 29.1947),      # Kalemie
}


# -----------------------------------------------------------------
# Request / response schemas
# -----------------------------------------------------------------
class PredictRequest(BaseModel):
    lat: float = Field(..., json_schema_extra={"example": -9.9})
    lon: float = Field(..., json_schema_extra={"example": 27.5})
    doy: int = Field(..., ge=1, le=366)
    t2m_max: float
    t2m_min: float
    rh2m: float
    ws2m: float
    prectotcorr: float


class PredictResponse(BaseModel):
    fire_probability: float
    risk_level: str


class ShelterOut(BaseModel):
    osm_id: str
    category: str
    name: str
    lat: float
    lon: float
    capacity: int
    available: int
    is_shelter: bool
    province: Optional[str] = None
    pm2_5: Optional[float] = None
    us_aqi: Optional[float] = None
    observation_time: Optional[str] = None


class NearestShelterResponse(BaseModel):
    name: str
    lat: float
    lon: float
    distance_km: float
    capacity: int
    available: int


class AlertOut(BaseModel):
    lat: float
    lon: float
    fire_probability: float
    risk_level: str
    nearest_shelter: Optional[str]
    nearest_shelter_distance_km: Optional[float]
    message: str


# -----------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "PHOENIX API — Congo (Katanga)", "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return predict_fire_risk(
        lat=req.lat, lon=req.lon, doy=req.doy,
        t2m_max=req.t2m_max, t2m_min=req.t2m_min,
        rh2m=req.rh2m, ws2m=req.ws2m, prectotcorr=req.prectotcorr,
    )


@app.get("/risk-map", response_model=List[dict])
def risk_map(date: date_type = Query(..., description="Date to evaluate, e.g. 2026-08-12")):
    """Fire risk for every grid cell. Health/AQI fields only populate for the
    LATEST available weather date (live data source), same rule as Algeria."""
    day_data = CLIMATE_DF[CLIMATE_DF["date"] == pd.Timestamp(date)]
    if day_data.empty:
        raise HTTPException(status_code=404, detail="No climate data available for this date.")

    is_latest_available = pd.Timestamp(date).normalize() == LATEST_AVAILABLE_DATE
    doy = pd.Timestamp(date).dayofyear
    results = []
    for _, row in day_data.iterrows():
        if any(pd.isna(row[c]) for c in _weather_cols):
            results.append({
                "lat": row["LAT"], "lon": row["LON"],
                "risk_level": "No data", "fire_probability": None,
                "pm2_5": None, "health_level": None, "health_advice": None,
            })
            continue
        r = predict_fire_risk(
            lat=row["LAT"], lon=row["LON"], doy=doy,
            t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
            rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
        )
        entry = {"lat": row["LAT"], "lon": row["LON"], **r,
                  "pm2_5": None, "health_level": None, "health_advice": None}
        if is_latest_available:
            pm25 = fetch_live_pm25(row["LAT"], row["LON"])
            if pm25 is not None:
                alert = get_alert(r["fire_probability"], pm25)
                entry.update({"pm2_5": pm25, "health_level": alert.health_level,
                               "health_advice": alert.health_advice})
        results.append(entry)
    return results


@app.get("/shelters", response_model=List[ShelterOut])
def list_shelters(
    category: Optional[str] = Query(None, description="school, place_of_worship, or health_facility"),
    province: Optional[str] = Query(None, description="Haut-Katanga, Lualaba, or Tanganyika"),
    only_shelters: bool = Query(False, description="If true, exclude health facilities (support-only)"),
):
    df = SHELTERS_DF
    if category:
        df = df[df["category"] == category]
    if province:
        df = df[df["province"] == province]
    if only_shelters:
        df = df[df["is_shelter"]]
    cols = ["osm_id", "category", "name", "lat", "lon", "capacity", "available", "is_shelter",
            "province", "pm2_5", "us_aqi", "observation_time"]
    return df[cols].to_dict(orient="records")


@app.get("/shelters/nearest", response_model=NearestShelterResponse)
def nearest_shelter(
    lat: float = Query(..., json_schema_extra={"example": -9.9}),
    lon: float = Query(..., json_schema_extra={"example": 27.5}),
    only_shelters: bool = Query(True, description="Restrict to real shelters (exclude health facilities)"),
):
    df = SHELTERS_DF[SHELTERS_DF["available"] > 0].copy()
    if only_shelters:
        df = df[df["is_shelter"]]
    if df.empty:
        raise HTTPException(status_code=404, detail="No available shelters found.")

    df["distance_km"] = df.apply(lambda s: haversine_km(lat, lon, s["lat"], s["lon"]), axis=1)
    nearest = df.loc[df["distance_km"].idxmin()]
    return {
        "name": nearest["name"], "lat": nearest["lat"], "lon": nearest["lon"],
        "distance_km": round(nearest["distance_km"], 2),
        "capacity": int(nearest["capacity"]), "available": int(nearest["available"]),
    }


@app.get("/alerts", response_model=List[AlertOut])
def alerts(date: date_type = Query(..., description="Date to evaluate, e.g. 2026-08-12")):
    zones = risk_map(date)
    high_risk = [z for z in zones if z["risk_level"] == "High"]

    shelters = SHELTERS_DF[(SHELTERS_DF["is_shelter"]) & (SHELTERS_DF["available"] > 0)].copy()

    out = []
    for z in high_risk:
        nearest_name, nearest_dist = None, None
        if not shelters.empty:
            shelters["distance_km"] = shelters.apply(
                lambda s: haversine_km(z["lat"], z["lon"], s["lat"], s["lon"]), axis=1
            )
            nearest = shelters.loc[shelters["distance_km"].idxmin()]
            nearest_name, nearest_dist = nearest["name"], round(nearest["distance_km"], 2)

        health_note = f" {z['health_advice']}" if z.get("health_advice") else ""
        out.append({
            "lat": z["lat"], "lon": z["lon"],
            "fire_probability": z["fire_probability"], "risk_level": z["risk_level"],
            "nearest_shelter": nearest_name, "nearest_shelter_distance_km": nearest_dist,
            "message": (
                f"[ALERT] High wildfire risk near ({z['lat']}, {z['lon']}). "
                f"Probability: {z['fire_probability']*100:.0f}%. "
                + (f"Nearest shelter: {nearest_name} ({nearest_dist} km)." if nearest_name else "No nearby shelter capacity found.")
                + health_note
            ),
        })
    return out

# -----------------------------------------------------------------
# USSD & Voice — Africa's Talking webhooks
# -----------------------------------------------------------------
def _ussd_summary(lat: float, lon: float, lang: str) -> str:
    """Current high-risk zone count + nearest available shelter, in the
    requested language. Reuses /risk-map for the latest date so USSD/Voice
    always match what the dashboard shows."""
    zones = risk_map(LATEST_AVAILABLE_DATE.date())
    high_count = sum(1 for z in zones if z["risk_level"] == "High")

    shelters = SHELTERS_DF[(SHELTERS_DF["is_shelter"]) & (SHELTERS_DF["available"] > 0)].copy()
    nearest_name, nearest_dist = None, None
    if not shelters.empty:
        shelters["distance_km"] = shelters.apply(
            lambda s: haversine_km(lat, lon, s["lat"], s["lon"]), axis=1
        )
        nearest = shelters.loc[shelters["distance_km"].idxmin()]
        nearest_name, nearest_dist = nearest["name"], round(nearest["distance_km"], 1)

    date_str = LATEST_AVAILABLE_DATE.strftime("%Y-%m-%d")
    if lang == "fr":
        shelter_line = (f"Abri le plus proche: {nearest_name} ({nearest_dist} km)"
                         if nearest_name else "Aucun abri disponible trouve.")
        return (f"PHOENIX - Alerte Incendie ({date_str})\n"
                f"Zones a haut risque: {high_count}\n{shelter_line}\nRestez en securite.")
    shelter_line = (f"Nearest shelter: {nearest_name} ({nearest_dist} km)"
                     if nearest_name else "No available shelter found.")
    return (f"PHOENIX - Fire Alert ({date_str})\n"
            f"High-risk zones: {high_count}\n{shelter_line}\nStay safe.")


@app.post("/ussd")
async def ussd(request: Request):
    """Africa's Talking USSD callback. Register a USSD channel in the
    Africa's Talking console pointed at this URL — anyone can then dial the
    assigned code from ANY phone (no smartphone, app, or internet needed) to
    check current fire risk and the nearest shelter, in English or French.

    Protocol: Africa's Talking POSTs form-encoded sessionId / serviceCode /
    phoneNumber / text. `text` accumulates the caller's choices separated by
    '*' as the session progresses (e.g. '', '1', '1*2'). The response must
    start with 'CON ' to keep the session open and show another menu, or
    'END ' to send a final message and hang up."""
    form = await request.form()
    text = form.get("text", "")
    steps = text.split("*") if text else []

    if text == "":
        response = "CON Welcome to PHOENIX Fire Alert\nBienvenue a PHOENIX\n1. English\n2. Francais"
    elif len(steps) == 1:
        lang = "en" if steps[0] == "1" else "fr"
        response = ("CON Choose your province:\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika"
                    if lang == "en" else
                    "CON Choisissez votre province:\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika")
    elif len(steps) == 2:
        lang = "en" if steps[0] == "1" else "fr"
        province = {"1": "Haut-Katanga", "2": "Lualaba", "3": "Tanganyika"}.get(steps[1])
        if not province:
            response = "END Invalid choice. / Choix invalide."
        else:
            ref_lat, ref_lon = PROVINCE_REF_POINTS[province]
            try:
                summary = _ussd_summary(ref_lat, ref_lon, lang)
            except Exception:
                summary = ("No data available right now. Try again later." if lang == "en"
                           else "Aucune donnee disponible. Reessayez plus tard.")
            response = f"END {summary}"
    else:
        response = "END Session error. Please try again. / Erreur de session. Reessayez."

    return PlainTextResponse(content=response, media_type="text/plain")


@app.post("/voice")
async def voice(request: Request):
    """Africa's Talking Voice callback. Fires when an outbound call placed
    from the dashboard connects. Reads the message + language passed via
    clientState (set when the call was initiated) and responds with Voice
    XML telling Africa's Talking what to say aloud."""
    form = await request.form()
    client_state = form.get("clientState", "")

    message, lang = "PHOENIX fire alert.", "en"
    if client_state:
        try:
            state = json.loads(client_state)
            message = state.get("message", message)
            lang = state.get("lang", lang)
        except (json.JSONDecodeError, TypeError):
            pass

    safe_message = (message.replace("&", "&amp;").replace("<", "&lt;")
                            .replace(">", "&gt;").replace('"', "&quot;"))

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="woman" playBeep="false">{safe_message}</Say>
</Response>"""
    return Response(content=xml, media_type="application/xml")
