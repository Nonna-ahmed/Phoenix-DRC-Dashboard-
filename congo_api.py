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
"""

from datetime import date as date_type
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
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
CLIMATE_DF["date"] = pd.to_datetime(CLIMATE_DF["YEAR"].astype(str), format="%Y") + \
                      pd.to_timedelta(CLIMATE_DF["DOY"] - 1, unit="D")
LATEST_AVAILABLE_DATE = CLIMATE_DF["date"].max().normalize()

SHELTERS_DF = pd.read_csv("drc_katanga_shelters_final.csv")
SHELTERS_DF = SHELTERS_DF.rename(columns={"capacity_estimate": "capacity"})
if "available" not in SHELTERS_DF.columns:
    SHELTERS_DF["available"] = SHELTERS_DF["capacity"]

# ── FIX #1: is_shelter column ───────────────────────────────────
if "is_shelter" not in SHELTERS_DF.columns:
    if "capacity_source" in SHELTERS_DF.columns:
        SHELTERS_DF["is_shelter"] = SHELTERS_DF["capacity_source"] != "default_estimate_non_shelter"
    else:
        SHELTERS_DF["is_shelter"] = True

# ── FIX #2: name column (no nulls) ──────────────────────────────
if "name" not in SHELTERS_DF.columns:
    SHELTERS_DF["name"] = None
SHELTERS_DF["name"] = SHELTERS_DF["name"].fillna(SHELTERS_DF["category"].astype(str) + " (unnamed)")

# ── FIX #3: province column ─────────────────────────────────────
if "province" not in SHELTERS_DF.columns:
    SHELTERS_DF["province"] = None

# ── FIX #4: AQI columns (no live feed yet) ──────────────────────
for col in ["pm2_5", "us_aqi", "observation_time"]:
    if col not in SHELTERS_DF.columns:
        SHELTERS_DF[col] = None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


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
    df = SHELTERS_DF.copy()
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
