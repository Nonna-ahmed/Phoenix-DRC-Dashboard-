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
    GET  /fwi?lat=..&lon=..            Canadian Fire Weather Index (cross-check)

Crowd-sourced reports & shelter management:
    POST /fire-reports             citizen-submitted fire sighting
    GET  /fire-reports?hours=72    recent reports, newest first
    POST /assistance-requests      evacuation-assistance request (elderly/disabled)
    GET  /assistance-requests?hours=72
    PATCH /shelters/{osm_id}/availability   update open spots at a shelter

Dashboard visit tracking & admin stats:
    POST /track-visit
    GET  /stats

Future forecast (climatology — historical average for the same day-of-year):
    POST /predict-future           single point, future date
    GET  /risk-map-future          full grid, future date
    POST /predict-forecast-live    optional live weather forecast (OpenWeatherMap),
                                    falls back to climatology if no API key

Africa's Talking webhooks:
    POST /ussd    USSD callback — English/French/Swahili menu, works from any
                  phone with no internet/app. Menu: 1) check the fire risk
                  FORECAST + nearest shelter, 2) report a fire you saw
                  (crowd-sourced, saved via /fire-reports, rate-limited to
                  1 report per phone number per hour to reduce spam), or
                  3) request evacuation help for an elderly/disabled person.
    POST /voice   Voice callback — speaks the alert text passed in
                  clientState when the dashboard places a call.
"""

from datetime import date as date_type, timedelta
from math import radians, sin, cos, sqrt, atan2, exp, log
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
# Accessibility info for elderly/disabled evacuees — genuinely unknown
# until shelter staff report it via PATCH /shelters/{osm_id}/availability,
# never invented or assumed.
for _col in ("wheelchair_accessible", "ground_floor", "medical_staff_onsite"):
    if _col not in SHELTERS_DF.columns:
        SHELTERS_DF[_col] = None

# -----------------------------------------------------------------
# Citizen fire reports (crowd-sourced via USSD) — stored as a local CSV.
# NOTE: Railway's filesystem is EPHEMERAL — this file persists across
# requests on the SAME running instance, but is wiped on every redeploy or
# restart. Fine for a demo/hackathon; for real production use, swap this
# for a proper database (e.g. a small Postgres add-on) or a Google Sheet.
# -----------------------------------------------------------------
FIRE_REPORTS_CSV = "citizen_fire_reports.csv"
_FIRE_REPORT_COLUMNS = ["report_id", "province", "lat", "lon", "phone_number", "reported_at_utc"]
FIRE_REPORT_COOLDOWN_MINUTES = 60  # basic anti-spam: one report per phone number per hour


class FireReportCooldownError(Exception):
    """Raised when the same phone number tries to report again too soon —
    a simple guard against fake/spam reports flooding the map."""
    def __init__(self, minutes_remaining: float):
        self.minutes_remaining = minutes_remaining
        super().__init__(f"Please wait {minutes_remaining:.0f} more minute(s) before reporting again.")


def _load_fire_reports() -> pd.DataFrame:
    if os.path.exists(FIRE_REPORTS_CSV):
        return pd.read_csv(FIRE_REPORTS_CSV)
    return pd.DataFrame(columns=_FIRE_REPORT_COLUMNS)


def _save_fire_report(province: str, lat: float, lon: float, phone_number: str) -> int:
    df = _load_fire_reports()

    # Anti-spam: block a new report from the same phone number within the
    # cooldown window. Only enforced when we actually have a phone number
    # (USSD always provides one; direct API calls might not).
    if phone_number and not df.empty:
        same_caller = df[df["phone_number"].astype(str) == str(phone_number)]
        if not same_caller.empty:
            last_report_time = pd.to_datetime(same_caller["reported_at_utc"]).max()
            elapsed = pd.Timestamp.utcnow().tz_localize(None) - last_report_time
            remaining = FIRE_REPORT_COOLDOWN_MINUTES - elapsed.total_seconds() / 60
            if remaining > 0:
                raise FireReportCooldownError(remaining)

    report_id = int(df["report_id"].max()) + 1 if not df.empty else 1
    new_row = pd.DataFrame([{
        "report_id": report_id, "province": province, "lat": lat, "lon": lon,
        "phone_number": phone_number,
        "reported_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FIRE_REPORTS_CSV, index=False)
    return report_id


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# -----------------------------------------------------------------
# Evacuation assistance requests — for elderly or disabled people (or
# their family) who need help evacuating, not just a fire sighting.
# Same ephemeral-storage caveat as fire reports.
# -----------------------------------------------------------------
ASSISTANCE_REQUESTS_CSV = "assistance_requests.csv"
_ASSISTANCE_COLUMNS = ["request_id", "province", "lat", "lon", "phone_number", "requested_at_utc"]
ASSISTANCE_COOLDOWN_MINUTES = 10  # short — a genuine urgent need shouldn't be blocked for long


def _load_assistance_requests() -> pd.DataFrame:
    if os.path.exists(ASSISTANCE_REQUESTS_CSV):
        return pd.read_csv(ASSISTANCE_REQUESTS_CSV)
    return pd.DataFrame(columns=_ASSISTANCE_COLUMNS)


def _save_assistance_request(province: str, lat: float, lon: float, phone_number: str) -> int:
    df = _load_assistance_requests()
    if phone_number and not df.empty:
        same_caller = df[df["phone_number"].astype(str) == str(phone_number)]
        if not same_caller.empty:
            last_time = pd.to_datetime(same_caller["requested_at_utc"]).max()
            elapsed = pd.Timestamp.utcnow().tz_localize(None) - last_time
            remaining = ASSISTANCE_COOLDOWN_MINUTES - elapsed.total_seconds() / 60
            if remaining > 0:
                raise FireReportCooldownError(remaining)  # same cooldown mechanism, reused

    request_id = int(df["request_id"].max()) + 1 if not df.empty else 1
    new_row = pd.DataFrame([{
        "request_id": request_id, "province": province, "lat": lat, "lon": lon,
        "phone_number": phone_number,
        "requested_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ASSISTANCE_REQUESTS_CSV, index=False)
    return request_id


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
# Canadian Fire Weather Index (FWI) System — Van Wagner & Pickett (1985/87)
# Independent, internationally-used fire-danger standard (used by Canada,
# and adapted elsewhere), run alongside the ML model as a cross-check.
#
# No external library is used for this — every equation below is the
# standard Van Wagner (1987) implementation written directly in pure
# Python (math.exp/log/sqrt from the standard library only). If /fwi
# 404s on a deployed server, it's because this route/function block
# isn't present in the deployed copy of this file yet — not a missing
# pip package.
# -----------------------------------------------------------------
# APPROXIMATIONS made for this near-equatorial region (Katanga, DRC):
#   - The FWI System's day-length factors (Le for DMC, Lf for DC) are
#     normally looked up per calendar month from a table tuned for Canadian
#     latitudes, where day length varies a lot across the year. Near the
#     equator, day length is close to constant (~11.5-12.5h) year-round, so
#     month-specific factors barely matter — we use fixed near-equatorial
#     constants instead of Canada's table.
#   - "Noon temperature" is approximated using the daily T2M_MAX, since
#     NASA POWER provides daily max/min rather than hourly readings — a
#     common substitution when only daily data is available.
#   - Wind speed is converted from NASA POWER's m/s to the km/h the FWI
#     System's equations expect.
_FWI_LE_EQUATOR = 9.0   # DMC effective day-length factor, near-equatorial
_FWI_LF_EQUATOR = 1.4   # DC day-length factor, near-equatorial

_FWI_STARTUP = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}  # standard Van Wagner (1987) spring startup values


def _fwi_danger_class(fwi_value: float) -> str:
    """Approximate danger classification — commonly cited FWI bins, not an
    exact match to any single jurisdiction's calibrated thresholds."""
    if fwi_value < 5:
        return "Low"
    if fwi_value < 10:
        return "Moderate"
    if fwi_value < 20:
        return "High"
    if fwi_value < 30:
        return "Very High"
    return "Extreme"


def compute_fwi(lat: float, lon: float):
    """Runs the full Canadian FWI System recursively over EVERY day of
    weather on record for the nearest grid cell (each day's fuel moisture
    codes depend on the previous day's — this is inherent to the FWI
    System, not something we can skip), and returns the final day's
    component values. Returns None if there's no usable weather data for
    that location."""
    grid_points = CLIMATE_DF[["LAT", "LON"]].drop_duplicates()
    if grid_points.empty:
        return None
    dists = ((grid_points["LAT"] - lat) ** 2 + (grid_points["LON"] - lon) ** 2) ** 0.5
    g_lat, g_lon = grid_points.loc[dists.idxmin(), ["LAT", "LON"]]

    series = CLIMATE_DF[(CLIMATE_DF["LAT"] == g_lat) & (CLIMATE_DF["LON"] == g_lon)].sort_values("date")
    series = series.dropna(subset=["T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR"])
    if series.empty:
        return None

    ffmc, dmc, dc = _FWI_STARTUP["ffmc"], _FWI_STARTUP["dmc"], _FWI_STARTUP["dc"]
    last_date, last_wind_kmh = None, 0.0

    for _, row in series.iterrows():
        T = float(row["T2M_MAX"])
        RH = min(max(float(row["RH2M"]), 0.0), 100.0)
        W = float(row["WS2M"]) * 3.6  # m/s -> km/h
        H = max(float(row["PRECTOTCORR"]), 0.0)

        # --- FFMC (Fine Fuel Moisture Code) ---
        mo = 147.2 * (101 - ffmc) / (59.5 + ffmc)
        if H > 0.5:
            rf = H - 0.5
            mr = mo + 42.5 * rf * exp(-100 / (251 - mo)) * (1 - exp(-6.93 / rf))
            if mo > 150:
                mr += 0.0015 * (mo - 150) ** 2 * sqrt(rf)
            mo = min(mr, 250)
        Ed = (0.942 * RH ** 0.679 + 11 * exp((RH - 100) / 10)
              + 0.18 * (21.1 - T) * (1 - exp(-0.115 * RH)))
        if mo > Ed:
            ko = 0.424 * (1 - (RH / 100) ** 1.7) + 0.0694 * sqrt(W) * (1 - (RH / 100) ** 8)
            kd = ko * 0.581 * exp(0.0365 * T)
            m = Ed + (mo - Ed) * 10 ** (-kd)
        else:
            Ew = (0.618 * RH ** 0.753 + 10 * exp((RH - 100) / 10)
                  + 0.18 * (21.1 - T) * (1 - exp(-0.115 * RH)))
            if mo < Ew:
                k1 = 0.424 * (1 - ((100 - RH) / 100) ** 1.7) + 0.0694 * sqrt(W) * (1 - ((100 - RH) / 100) ** 8)
                kw = k1 * 0.581 * exp(0.0365 * T)
                m = Ew - (Ew - mo) * 10 ** (-kw)
            else:
                m = mo
        ffmc = min(max(59.5 * (250 - m) / (147.2 + m), 0.0), 101.0)

        # --- DMC (Duff Moisture Code) ---
        Tc = max(T, -1.1)
        if H > 1.5:
            re = 0.92 * H - 1.27
            mo_dmc = 20 + exp(5.6348 - dmc / 43.43)
            if dmc <= 33:
                b = 100 / (0.5 + 0.3 * dmc)
            elif dmc <= 65:
                b = 14 - 1.3 * log(dmc)
            else:
                b = 6.2 * log(dmc) - 17.2
            mr_dmc = mo_dmc + 1000 * re / (48.77 + b * re)
            # Guard: mr_dmc must exceed 20 for log(mr_dmc - 20) to be valid.
            # In rare edge cases (e.g. very small re/b combinations) this can
            # dip to <= 20; clamp it just above 20 instead of letting a
            # ValueError ("math domain error") crash the whole endpoint.
            mr_dmc = max(mr_dmc, 20.0001)
            dmc = max(244.72 - 43.43 * log(mr_dmc - 20), 0.0)
        k = 1.894 * (Tc + 1.1) * (100 - RH) * _FWI_LE_EQUATOR * 1e-6
        dmc = dmc + 100 * k

        # --- DC (Drought Code) ---
        Tc2 = max(T, -2.8)
        if H > 2.8:
            rd = 0.83 * H - 1.27
            Qo = 800 * exp(-dc / 400)
            Qr = Qo + 3.937 * rd
            # Guard: Qr must stay positive for log(800 / Qr) to be valid.
            Qr = max(Qr, 0.0001)
            dc = max(400 * log(800 / Qr), 0.0)
        V = max(0.36 * (Tc2 + 2.8) + _FWI_LF_EQUATOR, 0.0)
        dc = dc + 0.5 * V

        last_date, last_wind_kmh = row["date"], W

    # --- ISI, BUI, FWI (final day only) ---
    m_final = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    fF = 91.9 * exp(-0.1386 * m_final) * (1 + m_final ** 5.31 / 4.93e7)
    fW = exp(0.05039 * last_wind_kmh)
    isi = 0.208 * fW * fF

    denom = dmc + 0.4 * dc
    bui = 0.8 * dmc * dc / denom if denom > 0 and dmc <= 0.4 * dc else (
        dmc - (1 - 0.8 * dc / denom) * (0.92 + (0.0114 * dmc) ** 1.7) if denom > 0 else 0.0
    )
    bui = max(bui, 0.0)

    fD = 0.626 * bui ** 0.809 + 2 if bui <= 80 else 1000 / (25 + 108.64 * exp(-0.023 * bui))
    B = 0.1 * isi * fD
    # Guard: log(B) is only valid for B > 0; the B > 1 branch already avoids
    # calling log on a value <= 1, so no extra clamp needed for that path,
    # but keep B non-negative defensively.
    B = max(B, 0.0)
    fwi_value = exp(2.72 * (0.434 * log(B)) ** 0.647) if B > 1 else B

    return {
        "lat": float(g_lat), "lon": float(g_lon), "as_of_date": str(last_date.date()),
        "ffmc": round(ffmc, 1), "dmc": round(dmc, 1), "dc": round(dc, 1),
        "isi": round(isi, 1), "bui": round(bui, 1), "fwi": round(fwi_value, 1),
        "danger_class": _fwi_danger_class(fwi_value),
    }


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
    # Accessibility info — only set once shelter staff enter it via
    # PATCH /shelters/{osm_id}/availability; None means "not yet known",
    # never assumed or invented.
    wheelchair_accessible: Optional[bool] = None
    ground_floor: Optional[bool] = None
    medical_staff_onsite: Optional[bool] = None


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
# Request / response schemas — citizen fire reports & shelter updates
# -----------------------------------------------------------------
class FireReportRequest(BaseModel):
    province: str = Field(..., description="Haut-Katanga, Lualaba, or Tanganyika")
    lat: float = Field(..., json_schema_extra={"example": -11.66})
    lon: float = Field(..., json_schema_extra={"example": 27.48})
    phone_number: Optional[str] = Field(None, description="Reporter's phone number, if available")


class FireReportOut(BaseModel):
    report_id: int
    province: str
    lat: float
    lon: float
    phone_number: Optional[str] = None
    reported_at_utc: str


class AssistanceRequestIn(BaseModel):
    province: str = Field(..., description="Haut-Katanga, Lualaba, or Tanganyika")
    lat: float = Field(..., json_schema_extra={"example": -11.66})
    lon: float = Field(..., json_schema_extra={"example": 27.48})
    phone_number: Optional[str] = Field(None, description="Requester's phone number, if available")


class AssistanceRequestOut(BaseModel):
    request_id: int
    province: str
    lat: float
    lon: float
    phone_number: Optional[str] = None
    requested_at_utc: str


class ShelterAvailabilityRequest(BaseModel):
    available: int = Field(..., ge=0, description="Current number of open spots")
    wheelchair_accessible: Optional[bool] = Field(None, description="Set only if you know for certain")
    ground_floor: Optional[bool] = Field(None, description="Set only if you know for certain")
    medical_staff_onsite: Optional[bool] = Field(None, description="Set only if you know for certain")


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
            "province", "pm2_5", "us_aqi", "observation_time",
            "wheelchair_accessible", "ground_floor", "medical_staff_onsite"]
    return df[cols].where(pd.notna(df[cols]), None).to_dict(orient="records")


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
# Citizen fire reports (crowd-sourced) & shelter availability updates
# ===================================================================
@app.post("/fire-reports", response_model=FireReportOut)
def submit_fire_report(req: FireReportRequest):
    """Records a citizen-submitted fire sighting (from USSD or the API
    directly). Helps cover the ~3-5 day gap in NASA POWER's own processing
    lag with real-time, on-the-ground reports. Rate-limited to one report
    per phone number per hour to reduce fake/spam reports."""
    if req.province not in PROVINCE_REF_POINTS:
        raise HTTPException(status_code=400, detail=f"province must be one of {list(PROVINCE_REF_POINTS)}")
    try:
        report_id = _save_fire_report(req.province, req.lat, req.lon, req.phone_number)
    except FireReportCooldownError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return {
        "report_id": report_id, "province": req.province, "lat": req.lat, "lon": req.lon,
        "phone_number": req.phone_number,
        "reported_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/fire-reports", response_model=List[FireReportOut])
def list_fire_reports(hours: int = Query(72, description="Only reports from the last N hours")):
    """Lists recent citizen fire reports, newest first. Defaults to the
    last 72 hours so old reports don't linger on the map forever."""
    df = _load_fire_reports()
    if df.empty:
        return []
    df["reported_at_utc"] = pd.to_datetime(df["reported_at_utc"])
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=hours)
    df = df[df["reported_at_utc"] >= cutoff].sort_values("reported_at_utc", ascending=False)
    df["reported_at_utc"] = df["reported_at_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.to_dict(orient="records")


@app.post("/assistance-requests", response_model=AssistanceRequestOut)
def submit_assistance_request(req: AssistanceRequestIn):
    """Records a request for evacuation assistance — for elderly or
    disabled people (or someone calling on their behalf) who need help
    physically evacuating, not just a fire sighting. Surfaced separately
    from fire reports so responders can prioritize accordingly."""
    if req.province not in PROVINCE_REF_POINTS:
        raise HTTPException(status_code=400, detail=f"province must be one of {list(PROVINCE_REF_POINTS)}")
    try:
        request_id = _save_assistance_request(req.province, req.lat, req.lon, req.phone_number)
    except FireReportCooldownError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return {
        "request_id": request_id, "province": req.province, "lat": req.lat, "lon": req.lon,
        "phone_number": req.phone_number,
        "requested_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/assistance-requests", response_model=List[AssistanceRequestOut])
def list_assistance_requests(hours: int = Query(72, description="Only requests from the last N hours")):
    """Lists recent evacuation-assistance requests, newest first."""
    df = _load_assistance_requests()
    if df.empty:
        return []
    df["requested_at_utc"] = pd.to_datetime(df["requested_at_utc"])
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=hours)
    df = df[df["requested_at_utc"] >= cutoff].sort_values("requested_at_utc", ascending=False)
    df["requested_at_utc"] = df["requested_at_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.to_dict(orient="records")


@app.patch("/shelters/{osm_id}/availability")
def update_shelter_availability(osm_id: str, req: ShelterAvailabilityRequest):
    """Lets shelter staff update how many spots are currently open, and
    optionally report accessibility info (wheelchair access, ground floor,
    medical staff on-site) for elderly/disabled evacuees — only set fields
    the caller actually knows; anything left out stays as it was (never
    reset to "no" by omission). Changes persist for the life of this
    running instance (see the ephemeral-storage note on FIRE_REPORTS_CSV
    above — same caveat applies here)."""
    global SHELTERS_DF
    match = SHELTERS_DF["osm_id"].astype(str) == str(osm_id)
    if not match.any():
        raise HTTPException(status_code=404, detail=f"No shelter with osm_id={osm_id}")
    SHELTERS_DF.loc[match, "available"] = req.available
    for field in ("wheelchair_accessible", "ground_floor", "medical_staff_onsite"):
        value = getattr(req, field)
        if value is not None:
            SHELTERS_DF.loc[match, field] = value
    SHELTERS_DF.to_csv("drc_katanga_shelters_final.csv", index=False)
    updated = SHELTERS_DF.loc[match].iloc[0]
    return {
        "osm_id": osm_id, "name": updated["name"],
        "available": int(updated["available"]), "capacity": int(updated["capacity"]),
        "wheelchair_accessible": (None if pd.isna(updated["wheelchair_accessible"])
                                   else bool(updated["wheelchair_accessible"])),
        "ground_floor": None if pd.isna(updated["ground_floor"]) else bool(updated["ground_floor"]),
        "medical_staff_onsite": (None if pd.isna(updated["medical_staff_onsite"])
                                  else bool(updated["medical_staff_onsite"])),
    }


@app.get("/fwi")
def get_fwi(
    lat: float = Query(..., json_schema_extra={"example": -11.66}),
    lon: float = Query(..., json_schema_extra={"example": 27.48}),
):
    """Computes the real Canadian Fire Weather Index (FFMC/DMC/DC/ISI/BUI/
    FWI) for the nearest grid cell — an independent, internationally-used
    fire-danger standard, run alongside (not replacing) the ML model, as a
    cross-check. See compute_fwi()'s docstring for the approximations made
    for this near-equatorial region."""
    result = compute_fwi(lat, lon)
    if result is None:
        raise HTTPException(status_code=404, detail="No weather data available for this location.")
    return result


# -----------------------------------------------------------------
# Dashboard visit tracking & admin stats — same ephemeral-storage caveat
# as fire reports and shelter updates (resets on redeploy).
# -----------------------------------------------------------------
VISITS_LOG_CSV = "dashboard_visits.csv"


@app.post("/track-visit")
def track_visit():
    """Increments a simple visit counter. The dashboard calls this once per
    browser session (not per interaction), giving the admin view a rough
    usage signal."""
    df = pd.read_csv(VISITS_LOG_CSV) if os.path.exists(VISITS_LOG_CSV) else pd.DataFrame(columns=["timestamp_utc"])
    new_row = pd.DataFrame([{"timestamp_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(VISITS_LOG_CSV, index=False)
    return {"status": "ok", "total_visits": len(df)}


@app.get("/stats")
def get_stats():
    """Aggregate usage/activity numbers for the admin dashboard: visits,
    citizen fire reports, evacuation-assistance requests, and shelter
    capacity — all real, measured figures (not simulated), though visit
    tracking only covers time since the last redeploy (ephemeral storage)."""
    visits_df = pd.read_csv(VISITS_LOG_CSV) if os.path.exists(VISITS_LOG_CSV) else pd.DataFrame()
    reports_df = _load_fire_reports()
    assistance_df = _load_assistance_requests()
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7)

    visits_last_7d = 0
    if not visits_df.empty:
        visits_df["timestamp_utc"] = pd.to_datetime(visits_df["timestamp_utc"])
        visits_last_7d = int((visits_df["timestamp_utc"] >= cutoff).sum())

    reports_last_7d = 0
    if not reports_df.empty:
        reports_df["reported_at_utc"] = pd.to_datetime(reports_df["reported_at_utc"])
        reports_last_7d = int((reports_df["reported_at_utc"] >= cutoff).sum())

    assistance_last_7d = 0
    if not assistance_df.empty:
        assistance_df["requested_at_utc"] = pd.to_datetime(assistance_df["requested_at_utc"])
        assistance_last_7d = int((assistance_df["requested_at_utc"] >= cutoff).sum())

    return {
        "total_visits": len(visits_df), "visits_last_7_days": visits_last_7d,
        "total_fire_reports": len(reports_df), "fire_reports_last_7_days": reports_last_7d,
        "total_assistance_requests": len(assistance_df),
        "assistance_requests_last_7_days": assistance_last_7d,
        "total_shelters": int(len(SHELTERS_DF)),
        "total_shelter_capacity": int(SHELTERS_DF["capacity"].sum()),
        "total_shelter_available": int(SHELTERS_DF["available"].sum()),
    }


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
    if lang == "sw":
        shelter_line = (f"Makazi ya karibu: {nearest_name} ({nearest_dist} km)"
                         if nearest_name else "Hakuna makazi yanayopatikana.")
        return (f"PHOENIX - Utabiri wa Moto ({date_str})\n"
                f"Maeneo hatari zaidi (utabiri): {high_count}\n{shelter_line}\nKaa salama.")
    shelter_line = (f"Nearest shelter: {nearest_name} ({nearest_dist} km)"
                     if nearest_name else "No available shelter found.")
    return (f"PHOENIX - Fire Forecast ({date_str})\n"
            f"High-risk zones (forecast): {high_count}\n{shelter_line}\nStay safe.")


_USSD_LANG_MAP = {"1": "en", "2": "fr", "3": "sw"}

_USSD_TEXT = {
    "main_menu": {
        "en": "CON What would you like to do?\n1. Check fire risk & shelter\n2. Report a fire you saw\n"
              "3. Request evacuation help (elderly/disabled)",
        "fr": "CON Que voulez-vous faire ?\n1. Verifier le risque et l'abri\n2. Signaler un incendie\n"
              "3. Demander de l'aide pour evacuer (personnes agees/handicapees)",
        "sw": "CON Ungependa kufanya nini?\n1. Angalia hatari ya moto na makazi\n2. Ripoti moto ulioona\n"
              "3. Omba msaada wa uhamishaji (wazee/walemavu)",
    },
    "province_check": {
        "en": "CON Choose your province:\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
        "fr": "CON Choisissez votre province:\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
        "sw": "CON Chagua mkoa wako:\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
    },
    "province_report": {
        "en": "CON Which province is the fire in?\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
        "fr": "CON Dans quelle province est l'incendie ?\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
        "sw": "CON Moto uko mkoa gani?\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
    },
    "province_assistance": {
        "en": "CON Which province do you need help in?\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
        "fr": "CON Dans quelle province avez-vous besoin d'aide ?\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
        "sw": "CON Unahitaji msaada mkoa gani?\n1. Haut-Katanga\n2. Lualaba\n3. Tanganyika",
    },
    "invalid": {
        "en": "END Invalid choice.", "fr": "END Choix invalide.", "sw": "END Chaguo batili.",
    },
    "no_forecast": {
        "en": "No forecast available right now. Try again later.",
        "fr": "Aucune prevision disponible. Reessayez plus tard.",
        "sw": "Hakuna utabiri unaopatikana sasa. Jaribu tena baadaye.",
    },
    "report_thanks": {
        "en": "END Thank you! Your report (#{id}) has been recorded for {province}. Stay safe.",
        "fr": "END Merci ! Votre signalement (#{id}) a ete enregistre pour {province}. Restez en securite.",
        "sw": "END Asante! Ripoti yako (#{id}) imesajiliwa kwa {province}. Kaa salama.",
    },
    "report_cooldown": {
        "en": "END You already reported recently — thank you. Please wait a bit before reporting again.",
        "fr": "END Vous avez deja signale recemment — merci. Veuillez patienter avant de signaler a nouveau.",
        "sw": "END Tayari umeripoti hivi karibuni — asante. Tafadhali subiri kabla ya kuripoti tena.",
    },
    "report_failed": {
        "en": "END Could not save your report right now. Please try again later.",
        "fr": "END Impossible d'enregistrer votre signalement. Reessayez plus tard.",
        "sw": "END Imeshindikana kuhifadhi ripoti yako. Jaribu tena baadaye.",
    },
    "assistance_thanks": {
        "en": "END Help request (#{id}) recorded for {province}. A responder will try to reach you. Stay safe.",
        "fr": "END Demande d'aide (#{id}) enregistree pour {province}. Un intervenant essaiera de vous "
              "joindre. Restez en securite.",
        "sw": "END Ombi la msaada (#{id}) limesajiliwa kwa {province}. Mwokozi atajaribu kuwafikia. "
              "Kaa salama.",
    },
    "assistance_cooldown": {
        "en": "END A help request was already sent recently — it's been recorded. Please wait a few "
              "minutes before requesting again.",
        "fr": "END Une demande d'aide a deja ete envoyee recemment — elle a ete enregistree. Veuillez "
              "patienter quelques minutes avant de redemander.",
        "sw": "END Ombi la msaada tayari limetumwa hivi karibuni — limesajiliwa. Tafadhali subiri "
              "dakika chache kabla ya kuomba tena.",
    },
    "assistance_failed": {
        "en": "END Could not save your help request right now. Please try again, or ask someone nearby "
              "for help.",
        "fr": "END Impossible d'enregistrer votre demande d'aide. Reessayez, ou demandez de l'aide a "
              "quelqu'un a proximite.",
        "sw": "END Imeshindikana kuhifadhi ombi lako la msaada. Jaribu tena, au omba msaada kwa mtu "
              "aliye karibu.",
    },
    "session_error": {
        "en": "END Session error. Please try again.",
        "fr": "END Erreur de session. Reessayez.",
        "sw": "END Hitilafu ya kikao. Tafadhali jaribu tena.",
    },
}


@app.post("/ussd")
async def ussd(request: Request):
    """Africa's Talking USSD callback. Register a USSD channel in the
    Africa's Talking console pointed at this URL — anyone can then dial the
    assigned code from ANY phone (no smartphone, app, or internet needed) to
    check the fire risk FORECAST and nearest shelter, OR report a fire they
    saw, in English, French, or Swahili.

    Protocol: Africa's Talking POSTs form-encoded sessionId / serviceCode /
    phoneNumber / text. `text` accumulates the caller's choices separated by
    '*' as the session progresses (e.g. '', '1', '1*2', '1*2*1'). The
    response must start with 'CON ' to keep the session open and show
    another menu, or 'END ' to send a final message and hang up."""
    form = await request.form()
    text = form.get("text", "")
    phone_number = form.get("phoneNumber", "")
    steps = text.split("*") if text else []

    if text == "":
        response = "CON Welcome to PHOENIX Fire Alert\nBienvenue a PHOENIX\nKaribu PHOENIX\n1. English\n2. Francais\n3. Kiswahili"

    elif len(steps) == 1:
        lang = _USSD_LANG_MAP.get(steps[0], "en")
        response = _USSD_TEXT["main_menu"][lang]

    elif len(steps) == 2:
        lang = _USSD_LANG_MAP.get(steps[0], "en")
        action = steps[1]
        if action not in ("1", "2", "3"):
            response = _USSD_TEXT["invalid"][lang]
        elif action == "1":
            response = _USSD_TEXT["province_check"][lang]
        elif action == "2":
            response = _USSD_TEXT["province_report"][lang]
        else:
            response = _USSD_TEXT["province_assistance"][lang]

    elif len(steps) == 3:
        lang = _USSD_LANG_MAP.get(steps[0], "en")
        action = steps[1]
        province = {"1": "Haut-Katanga", "2": "Lualaba", "3": "Tanganyika"}.get(steps[2])
        if not province:
            response = _USSD_TEXT["invalid"][lang]
        elif action == "1":
            ref_lat, ref_lon = PROVINCE_REF_POINTS[province]
            try:
                summary = _ussd_forecast_summary(ref_lat, ref_lon, lang)
            except Exception:
                summary = _USSD_TEXT["no_forecast"][lang]
            response = f"END {summary}"
        elif action == "2":
            # Crowd-sourced report — no GPS on USSD, so we log it at the
            # province's reference point. Good enough for "something is
            # happening in this province, worth a look" — not a precise pin.
            ref_lat, ref_lon = PROVINCE_REF_POINTS[province]
            try:
                report_id = _save_fire_report(province, ref_lat, ref_lon, phone_number)
                response = _USSD_TEXT["report_thanks"][lang].format(id=report_id, province=province)
            except FireReportCooldownError:
                response = _USSD_TEXT["report_cooldown"][lang]
            except Exception:
                response = _USSD_TEXT["report_failed"][lang]
        else:
            # Evacuation assistance request — same province-level location
            # limitation as fire reports (no GPS on USSD).
            ref_lat, ref_lon = PROVINCE_REF_POINTS[province]
            try:
                request_id = _save_assistance_request(province, ref_lat, ref_lon, phone_number)
                response = _USSD_TEXT["assistance_thanks"][lang].format(id=request_id, province=province)
            except FireReportCooldownError:
                response = _USSD_TEXT["assistance_cooldown"][lang]
            except Exception:
                response = _USSD_TEXT["assistance_failed"][lang]

    else:
        lang = _USSD_LANG_MAP.get(steps[0], "en") if steps else "en"
        response = _USSD_TEXT["session_error"][lang]

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
