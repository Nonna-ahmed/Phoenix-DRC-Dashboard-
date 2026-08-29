"""
Congo API — PHOENIX Backend (Katanga, DRC)
=============================================
Single FastAPI service for the PHOENIX wildfire early-warning & shelter-
matching system. This file merges what used to be two separate services:

  - congo_api.py           -> current-conditions monitoring, shelters, alerts
  - congo_forecast_api.py  -> future prediction using climatology

...into one app, so there is only one backend to deploy and one base URL
for both the dashboard and Africa's Talking (USSD/Voice) to call.

Run locally:
    pip install -r requirements.txt
    uvicorn congo_api:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs.

Endpoints
---------
Current monitoring (historical / latest recorded weather):
    GET  /health
    POST /predict
    GET  /risk-map?date=YYYY-MM-DD
    GET  /shelters
    GET  /shelters/nearest
    GET  /alerts?date=YYYY-MM-DD

Future forecast (climatology — historical average for the same day-of-year):
    POST /predict-future           single point, future date
    GET  /risk-map-future          full grid, future date
    POST /predict-forecast-live    optional live weather forecast (OpenWeatherMap),
                                    falls back to climatology if no API key

Africa's Talking webhooks:
    POST /ussd    USSD callback — bilingual EN/FR menu, works from any basic
                  phone with no internet/app. Reports the FORECAST (not the
                  latest historical reading) for today, so it always reflects
                  a prediction rather than old recorded data.
    POST /voice   Voice callback — speaks the alert text passed in
                  clientState when the dashboard places a call.
"""

from datetime import date as date_type, timedelta
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, List, Dict
import json
import os

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from risk_engine import get_alert
from air_quality import fetch_live_pm25

# -----------------------------------------------------------------
# Predictor — use congo_predict if available, fall back to a simple
# rule-based score otherwise so the API never fails to start.
# -----------------------------------------------------------------
try:
    from congo_predict import predict_fire_risk
    HAS_CONGO_PREDICT = True
except ImportError:
    HAS_CONGO_PREDICT = False
    predict_fire_risk = None


def dummy_predict_fire_risk(**kwargs) -> dict:
    """Simple rule-based fallback used only if congo_predict.py is missing."""
    t_max = kwargs.get("t2m_max", 30)
    rh = kwargs.get("rh2m", 50)
    ws = kwargs.get("ws2m", 3)
    prectot = kwargs.get("prectotcorr", 0)

    score = 0.0
    score += max(0, (t_max - 20) / 30) * 0.35      # temperature contribution
    score += max(0, (100 - rh) / 100) * 0.30       # humidity contribution
    score += min(ws / 15, 1.0) * 0.20              # wind contribution
    score += (1 - min(prectot / 10, 1.0)) * 0.15   # rain contribution (inverse)

    prob = min(max(score, 0.0), 1.0)
    if prob >= 0.7:
        level = "High"
    elif prob >= 0.4:
        level = "Moderate"
    else:
        level = "Low"
    return {"fire_probability": round(prob, 4), "risk_level": level}


def call_predict(**kwargs) -> dict:
    """Single entry point for fire-risk prediction used across every
    endpoint — uses congo_predict when available, otherwise the fallback."""
    if HAS_CONGO_PREDICT and predict_fire_risk is not None:
        return predict_fire_risk(**kwargs)
    return dummy_predict_fire_risk(**kwargs)


# -----------------------------------------------------------------
# App
# -----------------------------------------------------------------
app = FastAPI(
    title="Congo API — PHOENIX (Katanga)",
    description="Wildfire early-warning, forecast & shelter-matching API for "
                "Haut-Katanga, Lualaba & Tanganyika (DRC)",
    version="2.0.0",
)

# -----------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------
CLIMATE_CSV = "phoenix_climate_2020_2026.csv"
if not os.path.exists(CLIMATE_CSV):
    raise FileNotFoundError(
        f"Climate file not found: {CLIMATE_CSV}. Place it next to this script."
    )

CLIMATE_DF = pd.read_csv(CLIMATE_CSV)
CLIMATE_DF = CLIMATE_DF.dropna(subset=["YEAR", "DOY"])  # a few rows have genuinely missing YEAR/DOY
CLIMATE_DF["YEAR"] = CLIMATE_DF["YEAR"].astype(int)
CLIMATE_DF["DOY"] = CLIMATE_DF["DOY"].astype(int)
CLIMATE_DF["date"] = pd.to_datetime(CLIMATE_DF["YEAR"].astype(str), format="%Y") + \
                      pd.to_timedelta(CLIMATE_DF["DOY"] - 1, unit="D")

# NASA POWER has a ~3-5 day processing lag; unprocessed recent days come back
# as the fill value -999 instead of real numbers. Mark those as NaN (don't
# drop the row) so the date itself still counts as "available" — /risk-map
# reports "No data" for the specific points that are missing, and the
# climatology engine's averages simply ignore NaN automatically.
_weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
CLIMATE_DF[_weather_cols] = CLIMATE_DF[_weather_cols].where(CLIMATE_DF[_weather_cols] >= -900)

# Latest date with FULL grid coverage — no NaN for any point. Newer dates
# may exist in the data but can still have partial "No data" from NASA
# POWER's processing lag; /risk-map still reports those individually, but
# LATEST_AVAILABLE_DATE (used as the "current" reference date by /alerts,
# USSD forecast fallback, etc.) is the most recent FULLY clean date.
_total_cells = CLIMATE_DF[["LAT", "LON"]].drop_duplicates().shape[0]
_complete_counts = CLIMATE_DF.dropna(subset=_weather_cols).groupby("date").size()
_full_coverage_dates = _complete_counts[_complete_counts == _total_cells]
LATEST_AVAILABLE_DATE = (_full_coverage_dates.index.max() if not _full_coverage_dates.empty
                          else CLIMATE_DF["date"].max()).normalize()

SHELTERS_DF = pd.read_csv("drc_katanga_shelters_final.csv")
SHELTERS_DF = SHELTERS_DF.rename(columns={"capacity_estimate": "capacity"})
if "available" not in SHELTERS_DF.columns:
    SHELTERS_DF["available"] = SHELTERS_DF["capacity"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# Reference point per province, used by the USSD menu to give a quick
# forecast + nearest-shelter summary without needing the caller's GPS
# location (basic phones on USSD have none). Same towns used as reference
# points in the Streamlit dashboard's "Nearest Shelters" panel.
PROVINCE_REF_POINTS = {
    "Haut-Katanga": (-11.6609, 27.4794),   # Lubumbashi
    "Lualaba": (-10.7167, 25.4667),        # Kolwezi
    "Tanganyika": (-5.9475, 29.1947),      # Kalemie
}


# -----------------------------------------------------------------
# Climatology engine — historical average for a given day-of-year,
# used for every FUTURE date prediction (no live weather data exists
# for dates that haven't happened yet).
# -----------------------------------------------------------------
class ClimatologyEngine:
    """Computes the historical average weather for each (LAT, LON) grid
    cell, for a given day-of-year, across every year available in the data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["doy"] = self.df["date"].dt.dayofyear

    def get_climatology_for_doy(self, doy: int) -> pd.DataFrame:
        same_doy = self.df[self.df["doy"] == doy]
        if same_doy.empty:
            raise ValueError(f"No historical data for day-of-year={doy}")
        grouped = same_doy.groupby(["LAT", "LON"]).agg({
            "T2M_MAX": "mean", "T2M_MIN": "mean", "RH2M": "mean",
            "WS2M": "mean", "PRECTOTCORR": "mean",
        }).reset_index()
        grouped["DOY"] = doy
        return grouped

    def get_climatology_for_date(self, target_date: date_type) -> pd.DataFrame:
        doy = target_date.timetuple().tm_yday
        return self.get_climatology_for_doy(doy)

    def get_point_climatology(self, lat: float, lon: float, target_date: date_type) -> Dict:
        doy = target_date.timetuple().tm_yday
        same_doy = self.df[self.df["doy"] == doy]
        if same_doy.empty:
            raise ValueError(f"No historical data for day-of-year={doy}")

        same_doy = same_doy.copy()
        same_doy["dist"] = np.sqrt((same_doy["LAT"] - lat) ** 2 + (same_doy["LON"] - lon) ** 2)
        nearest = same_doy.loc[same_doy["dist"].idxmin()]
        point_data = same_doy[(same_doy["LAT"] == nearest["LAT"]) & (same_doy["LON"] == nearest["LON"])]
        if point_data.empty:
            raise ValueError(f"No data for point ({lat}, {lon})")

        return {
            "lat": float(nearest["LAT"]), "lon": float(nearest["LON"]), "doy": doy,
            "t2m_max": float(point_data["T2M_MAX"].mean()),
            "t2m_min": float(point_data["T2M_MIN"].mean()),
            "rh2m": float(point_data["RH2M"].mean()),
            "ws2m": float(point_data["WS2M"].mean()),
            "prectotcorr": float(point_data["PRECTOTCORR"].mean()),
            "historical_years": int(point_data["YEAR"].nunique()),
        }


CLIM_ENGINE = ClimatologyEngine(CLIMATE_DF)


# -----------------------------------------------------------------
# Request / response schemas — current monitoring
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
# Request / response schemas — future forecast
# -----------------------------------------------------------------
class PredictFutureRequest(BaseModel):
    lat: float = Field(..., json_schema_extra={"example": -9.9})
    lon: float = Field(..., json_schema_extra={"example": 27.5})
    date: str = Field(..., description="Future date YYYY-MM-DD, e.g. 2026-09-15")


class PredictFutureResponse(BaseModel):
    lat: float
    lon: float
    date: str
    doy: int
    t2m_max: float
    t2m_min: float
    rh2m: float
    ws2m: float
    prectotcorr: float
    fire_probability: float
    risk_level: str
    method: str = "climatology"
    historical_years: int


class RiskMapFutureResponse(BaseModel):
    lat: float
    lon: float
    fire_probability: float
    risk_level: str
    t2m_max: float
    rh2m: float


class ForecastLiveRequest(BaseModel):
    lat: float = Field(..., json_schema_extra={"example": -9.9})
    lon: float = Field(..., json_schema_extra={"example": 27.5})
    date: str = Field(..., description="Future date YYYY-MM-DD")
    api_key: Optional[str] = Field(None, description="OpenWeatherMap API key (optional)")


class ForecastLiveResponse(BaseModel):
    lat: float
    lon: float
    date: str
    fire_probability: float
    risk_level: str
    weather_source: str
    temp_max: float
    humidity: float
    wind_speed: float
    rain_probability: Optional[float] = None


# ===================================================================
# Current-monitoring endpoints
# ===================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Congo API — PHOENIX (Katanga)",
        "version": "2.0.0",
        "predictor": "congo_predict" if HAS_CONGO_PREDICT else "dummy_fallback",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return call_predict(
        lat=req.lat, lon=req.lon, doy=req.doy,
        t2m_max=req.t2m_max, t2m_min=req.t2m_min,
        rh2m=req.rh2m, ws2m=req.ws2m, prectotcorr=req.prectotcorr,
    )


@app.get("/risk-map", response_model=List[dict])
def risk_map(date: date_type = Query(..., description="Date to evaluate, e.g. 2026-08-12")):
    """Fire risk for every grid cell using RECORDED weather. Health/AQI
    fields populate for any date within the last year of LATEST_AVAILABLE_DATE
    — Open-Meteo only ever returns the CURRENT live reading, never historical
    air quality for the selected date, so this is always "right now", just
    made available while browsing recent history rather than only on the
    single most-recent date."""
    day_data = CLIMATE_DF[CLIMATE_DF["date"] == pd.Timestamp(date)]
    if day_data.empty:
        raise HTTPException(status_code=404, detail="No climate data available for this date.")

    days_from_latest = (LATEST_AVAILABLE_DATE - pd.Timestamp(date).normalize()).days
    show_live_aq = 0 <= days_from_latest <= 365
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
        r = call_predict(
            lat=row["LAT"], lon=row["LON"], doy=doy,
            t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
            rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
        )
        entry = {"lat": row["LAT"], "lon": row["LON"], **r,
                  "pm2_5": None, "health_level": None, "health_advice": None}
        if show_live_aq:
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


# ===================================================================
# Future-forecast endpoints (climatology)
# ===================================================================
@app.post("/predict-future", response_model=PredictFutureResponse)
def predict_future(req: PredictFutureRequest):
    """Predicts fire risk at a single point for a FUTURE date, using the
    historical climatology average for that day-of-year."""
    try:
        target_date = date_type.fromisoformat(req.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    try:
        clim = CLIM_ENGINE.get_point_climatology(req.lat, req.lon, target_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = call_predict(
        lat=clim["lat"], lon=clim["lon"], doy=clim["doy"],
        t2m_max=clim["t2m_max"], t2m_min=clim["t2m_min"],
        rh2m=clim["rh2m"], ws2m=clim["ws2m"], prectotcorr=clim["prectotcorr"],
    )

    return PredictFutureResponse(
        lat=clim["lat"], lon=clim["lon"], date=req.date, doy=clim["doy"],
        t2m_max=round(clim["t2m_max"], 2), t2m_min=round(clim["t2m_min"], 2),
        rh2m=round(clim["rh2m"], 2), ws2m=round(clim["ws2m"], 2),
        prectotcorr=round(clim["prectotcorr"], 4),
        fire_probability=result["fire_probability"], risk_level=result["risk_level"],
        method="climatology", historical_years=clim["historical_years"],
    )


@app.get("/risk-map-future", response_model=List[RiskMapFutureResponse])
def risk_map_future(date: date_type = Query(..., description="Future date to predict, e.g. 2026-09-15")):
    """Full-grid fire risk forecast for a FUTURE date, using climatology."""
    try:
        clim_df = CLIM_ENGINE.get_climatology_for_date(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    results = []
    for _, row in clim_df.iterrows():
        r = call_predict(
            lat=row["LAT"], lon=row["LON"], doy=row["DOY"],
            t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
            rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
        )
        results.append(RiskMapFutureResponse(
            lat=row["LAT"], lon=row["LON"],
            fire_probability=r["fire_probability"], risk_level=r["risk_level"],
            t2m_max=round(row["T2M_MAX"], 2), rh2m=round(row["RH2M"], 2),
        ))
    return results


@app.post("/predict-forecast-live", response_model=ForecastLiveResponse)
def predict_forecast_live(req: ForecastLiveRequest):
    """Predicts using a REAL weather forecast (OpenWeatherMap) if an API key
    is supplied; falls back to climatology otherwise."""
    try:
        target_date = date_type.fromisoformat(req.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    weather = None
    if req.api_key:
        try:
            import requests
            url = (
                f"https://api.openweathermap.org/data/2.5/forecast"
                f"?lat={req.lat}&lon={req.lon}&appid={req.api_key}&units=metric"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                target_ts = pd.Timestamp(target_date)
                best, best_diff = None, timedelta(days=999)
                for item in data.get("list", []):
                    item_dt = pd.Timestamp(item["dt"], unit="s")
                    diff = abs(item_dt - target_ts)
                    if diff < best_diff:
                        best_diff, best = diff, item
                if best:
                    main = best["main"]
                    wind = best.get("wind", {})
                    rain = best.get("rain", {})
                    weather = {
                        "t2m_max": main.get("temp_max", main.get("temp", 30)),
                        "t2m_min": main.get("temp_min", main.get("temp", 20)),
                        "rh2m": main.get("humidity", 50),
                        "ws2m": wind.get("speed", 3),
                        "prectotcorr": rain.get("3h", 0) if rain else 0,
                        "source": "openweathermap",
                        "rain_prob": best.get("pop", None),
                    }
        except Exception:
            weather = None

    if weather is None:
        try:
            clim = CLIM_ENGINE.get_point_climatology(req.lat, req.lon, target_date)
            weather = {
                "t2m_max": clim["t2m_max"], "t2m_min": clim["t2m_min"],
                "rh2m": clim["rh2m"], "ws2m": clim["ws2m"],
                "prectotcorr": clim["prectotcorr"],
                "source": "climatology_fallback", "rain_prob": None,
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    doy = target_date.timetuple().tm_yday
    result = call_predict(
        lat=req.lat, lon=req.lon, doy=doy,
        t2m_max=weather["t2m_max"], t2m_min=weather["t2m_min"],
        rh2m=weather["rh2m"], ws2m=weather["ws2m"], prectotcorr=weather["prectotcorr"],
    )

    return ForecastLiveResponse(
        lat=req.lat, lon=req.lon, date=req.date,
        fire_probability=result["fire_probability"], risk_level=result["risk_level"],
        weather_source=weather["source"],
        temp_max=round(weather["t2m_max"], 2), humidity=round(weather["rh2m"], 2),
        wind_speed=round(weather["ws2m"], 2),
        rain_probability=round(weather["rain_prob"], 2) if weather["rain_prob"] is not None else None,
    )


# ===================================================================
# USSD & Voice — Africa's Talking webhooks
# ===================================================================
def _ussd_forecast_summary(lat: float, lon: float, lang: str) -> str:
    """High-risk zone count FORECAST for today (via climatology) + nearest
    available shelter, in the requested language. Uses the climatology
    engine for today's date rather than the historical CSV's latest recorded
    reading, so USSD/Voice always reflect a forecast, not old recorded data."""
    target_date = date_type.today()
    high_count = 0
    try:
        clim_df = CLIM_ENGINE.get_climatology_for_date(target_date)
        for _, row in clim_df.iterrows():
            r = call_predict(
                lat=row["LAT"], lon=row["LON"], doy=row["DOY"],
                t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
                rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
            )
            if r["risk_level"] == "High":
                high_count += 1
    except ValueError:
        pass  # no historical data for this day-of-year — report 0 and continue

    shelters = SHELTERS_DF[(SHELTERS_DF["is_shelter"]) & (SHELTERS_DF["available"] > 0)].copy()
    nearest_name, nearest_dist = None, None
    if not shelters.empty:
        shelters["distance_km"] = shelters.apply(
            lambda s: haversine_km(lat, lon, s["lat"], s["lon"]), axis=1
        )
        nearest = shelters.loc[shelters["distance_km"].idxmin()]
        nearest_name, nearest_dist = nearest["name"], round(nearest["distance_km"], 1)

    date_str = target_date.strftime("%Y-%m-%d")
    if lang == "fr":
        shelter_line = (f"Abri le plus proche: {nearest_name} ({nearest_dist} km)"
                         if nearest_name else "Aucun abri disponible trouve.")
        return (f"PHOENIX - Prevision Incendie ({date_str})\n"
                f"Zones a haut risque (prevues): {high_count}\n{shelter_line}\nRestez en securite.")
    shelter_line = (f"Nearest shelter: {nearest_name} ({nearest_dist} km)"
                     if nearest_name else "No available shelter found.")
    return (f"PHOENIX - Fire Forecast ({date_str})\n"
            f"High-risk zones (forecast): {high_count}\n{shelter_line}\nStay safe.")


@app.post("/ussd")
async def ussd(request: Request):
    """Africa's Talking USSD callback. Register a USSD channel in the
    Africa's Talking console pointed at this URL — anyone can then dial the
    assigned code from ANY phone (no smartphone, app, or internet needed) to
    check the fire risk FORECAST and the nearest shelter, in English or
    French.

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
                summary = _ussd_forecast_summary(ref_lat, ref_lon, lang)
            except Exception:
                summary = ("No forecast available right now. Try again later." if lang == "en"
                           else "Aucune prevision disponible. Reessayez plus tard.")
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
