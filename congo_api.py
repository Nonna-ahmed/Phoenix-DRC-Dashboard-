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
import regions as region_config

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


def call_predict(region: str = region_config.DEFAULT_REGION, **kwargs) -> dict:
    """Single entry point for fire-risk prediction used across every
    endpoint. Routes by region:
      - "congo" (or any region flagged use_congo_predict) -> the existing
        congo_predict.py, completely unchanged, so Congo's already-tested
        behavior never shifts just because other regions were added.
      - everything else -> the generic XGBoost predictor in regions.py,
        which works for any region sharing the same 8-feature schema
        (confirmed true for Algeria's model).
      - if congo_predict.py itself is missing, falls back to the dummy
        rule-based score exactly as before (region-independent safety net)."""
    cfg = region_config.get_region(region)
    if cfg.get("use_congo_predict", region == "congo"):
        if HAS_CONGO_PREDICT and predict_fire_risk is not None:
            return predict_fire_risk(**kwargs)
        return dummy_predict_fire_risk(**kwargs)
    return region_config.predict_fire_risk_generic(region, **kwargs)


# -----------------------------------------------------------------
# App
# -----------------------------------------------------------------
app = FastAPI(
    title="Congo API — PHOENIX (multi-region)",
    description="Wildfire early-warning, forecast & shelter-matching API — "
                "currently covering Congo (Katanga) and northern Algeria. "
                "Every endpoint takes an optional ?region=congo|algeria "
                "query param (defaults to congo for backward compatibility).",
    version="3.0.0",
)

# -----------------------------------------------------------------
# Data loading — PER REGION, cached on first use.
#
# Each region gets its own climate DataFrame, "latest available date",
# shelters DataFrame, and ClimatologyEngine, keyed by region id. Nothing
# is loaded at import time anymore (unlike the old single-region version)
# — a region's files are only read from disk the first time that region
# is actually requested, so adding a region to regions.py never risks
# breaking startup for regions whose files exist and work fine.
# -----------------------------------------------------------------
_weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]

_CLIMATE_CACHE: dict = {}
_LATEST_DATE_CACHE: dict = {}
_SHELTERS_CACHE: dict = {}
_CLIM_ENGINE_CACHE: dict = {}


def _load_climate_for_region(region: str) -> pd.DataFrame:
    cfg = region_config.get_region(region)
    csv_path = cfg["climate_csv"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Climate file not found for region '{region}': {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["YEAR", "DOY"])  # a few rows have genuinely missing YEAR/DOY
    df["YEAR"] = df["YEAR"].astype(int)
    df["DOY"] = df["DOY"].astype(int)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + \
                 pd.to_timedelta(df["DOY"] - 1, unit="D")

    # NASA POWER has a ~3-5 day processing lag; unprocessed recent days come
    # back as the fill value -999 instead of real numbers. Mark those as NaN
    # (don't drop the row) so the date itself still counts as "available" —
    # /risk-map reports "No data" for the specific points that are missing.
    df[_weather_cols] = df[_weather_cols].where(df[_weather_cols] >= -900)

    # Latest date with FULL grid coverage for this region.
    total_cells = df[["LAT", "LON"]].drop_duplicates().shape[0]
    complete_counts = df.dropna(subset=_weather_cols).groupby("date").size()
    full_coverage_dates = complete_counts[complete_counts == total_cells]
    latest = (full_coverage_dates.index.max() if not full_coverage_dates.empty
              else df["date"].max()).normalize()

    _CLIMATE_CACHE[region] = df
    _LATEST_DATE_CACHE[region] = latest
    return df


def get_climate_df(region: str) -> pd.DataFrame:
    if region not in _CLIMATE_CACHE:
        _load_climate_for_region(region)
    return _CLIMATE_CACHE[region]


def get_latest_available_date(region: str) -> pd.Timestamp:
    if region not in _LATEST_DATE_CACHE:
        _load_climate_for_region(region)
    return _LATEST_DATE_CACHE[region]


def get_shelters_df(region: str) -> pd.DataFrame:
    if region not in _SHELTERS_CACHE:
        cfg = region_config.get_region(region)
        df = pd.read_csv(cfg["shelters_csv"])
        # No-op for files that already use "capacity" (e.g. Algeria's
        # already-prepped file) — rename() silently ignores columns that
        # aren't present, so this stays safe for every region.
        df = df.rename(columns={"capacity_estimate": "capacity"})
        if "available" not in df.columns:
            df["available"] = df["capacity"]
        # Accessibility info for elderly/disabled evacuees — genuinely
        # unknown until shelter staff report it via
        # PATCH /shelters/{osm_id}/availability, never invented or assumed.
        for col in ("wheelchair_accessible", "ground_floor", "medical_staff_onsite"):
            if col not in df.columns:
                df[col] = None
        _SHELTERS_CACHE[region] = df
    return _SHELTERS_CACHE[region]


def set_shelters_df(region: str, df: pd.DataFrame):
    """Used by PATCH /shelters/{osm_id}/availability to write back the
    updated DataFrame into the per-region cache."""
    _SHELTERS_CACHE[region] = df


def get_clim_engine(region: str) -> "ClimatologyEngine":
    if region not in _CLIM_ENGINE_CACHE:
        _CLIM_ENGINE_CACHE[region] = ClimatologyEngine(get_climate_df(region))
    return _CLIM_ENGINE_CACHE[region]

# -----------------------------------------------------------------
# Citizen fire reports (crowd-sourced via USSD) — stored as a local CSV.
# NOTE: Railway's filesystem is EPHEMERAL — this file persists across
# requests on the SAME running instance, but is wiped on every redeploy or
# restart. Fine for a demo/hackathon; for real production use, swap this
# for a proper database (e.g. a small Postgres add-on) or a Google Sheet.
# -----------------------------------------------------------------
FIRE_REPORTS_CSV = "citizen_fire_reports.csv"
_FIRE_REPORT_COLUMNS = ["report_id", "region", "province", "lat", "lon", "phone_number", "reported_at_utc"]
FIRE_REPORT_COOLDOWN_MINUTES = 60  # basic anti-spam: one report per phone number per hour


class FireReportCooldownError(Exception):
    """Raised when the same phone number tries to report again too soon —
    a simple guard against fake/spam reports flooding the map."""
    def __init__(self, minutes_remaining: float):
        self.minutes_remaining = minutes_remaining
        super().__init__(f"Please wait {minutes_remaining:.0f} more minute(s) before reporting again.")


def _load_fire_reports() -> pd.DataFrame:
    if os.path.exists(FIRE_REPORTS_CSV):
        df = pd.read_csv(FIRE_REPORTS_CSV)
        # Reports saved before multi-region support won't have a "region"
        # column — treat those as "congo" (this API's original single
        # region) rather than dropping/breaking on old data.
        if "region" not in df.columns:
            df["region"] = region_config.DEFAULT_REGION
        return df
    return pd.DataFrame(columns=_FIRE_REPORT_COLUMNS)


def _save_fire_report(region: str, province: str, lat: float, lon: float, phone_number: str) -> int:
    df = _load_fire_reports()

    # Anti-spam: block a new report from the same phone number within the
    # cooldown window. Only enforced when we actually have a phone number
    # (USSD always provides one; direct API calls might not). Scoped to the
    # SAME region too — a phone number legitimately reporting once in Congo
    # and once in Algeria isn't spam.
    if phone_number and not df.empty:
        same_caller = df[(df["phone_number"].astype(str) == str(phone_number)) & (df["region"] == region)]
        if not same_caller.empty:
            last_report_time = pd.to_datetime(same_caller["reported_at_utc"]).max()
            elapsed = pd.Timestamp.utcnow().tz_localize(None) - last_report_time
            remaining = FIRE_REPORT_COOLDOWN_MINUTES - elapsed.total_seconds() / 60
            if remaining > 0:
                raise FireReportCooldownError(remaining)

    report_id = int(df["report_id"].max()) + 1 if not df.empty else 1
    new_row = pd.DataFrame([{
        "report_id": report_id, "region": region, "province": province, "lat": lat, "lon": lon,
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
_ASSISTANCE_COLUMNS = ["request_id", "region", "province", "lat", "lon", "phone_number", "requested_at_utc"]
ASSISTANCE_COOLDOWN_MINUTES = 10  # short — a genuine urgent need shouldn't be blocked for long


def _load_assistance_requests() -> pd.DataFrame:
    if os.path.exists(ASSISTANCE_REQUESTS_CSV):
        df = pd.read_csv(ASSISTANCE_REQUESTS_CSV)
        if "region" not in df.columns:
            df["region"] = region_config.DEFAULT_REGION
        return df
    return pd.DataFrame(columns=_ASSISTANCE_COLUMNS)


def _save_assistance_request(region: str, province: str, lat: float, lon: float, phone_number: str) -> int:
    df = _load_assistance_requests()
    if phone_number and not df.empty:
        same_caller = df[(df["phone_number"].astype(str) == str(phone_number)) & (df["region"] == region)]
        if not same_caller.empty:
            last_time = pd.to_datetime(same_caller["requested_at_utc"]).max()
            elapsed = pd.Timestamp.utcnow().tz_localize(None) - last_time
            remaining = ASSISTANCE_COOLDOWN_MINUTES - elapsed.total_seconds() / 60
            if remaining > 0:
                raise FireReportCooldownError(remaining)  # same cooldown mechanism, reused

    request_id = int(df["request_id"].max()) + 1 if not df.empty else 1
    new_row = pd.DataFrame([{
        "request_id": request_id, "region": region, "province": province, "lat": lat, "lon": lon,
        "phone_number": phone_number,
        "requested_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ASSISTANCE_REQUESTS_CSV, index=False)
    return request_id


# Reference points per province, used by the USSD menu to give a quick
# forecast + nearest-shelter summary without needing the caller's GPS
# location (basic phones on USSD have none). Now sourced per-region from
# regions.py (region_config.get_region(region)["province_ref_points"])
# instead of a single hardcoded Congo dict, so USSD works for any region.


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


# Per-region ClimatologyEngine instances are created lazily by
# get_clim_engine(region), defined earlier alongside the other per-region
# caches — no single global engine anymore.


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
#
# DAY-LENGTH ADJUSTMENT — GENERALIZED FOR ANY LATITUDE:
#   The DMC and DC codes both depend on a day-length adjustment factor
#   (Le for DMC, Lf for DC) that normally varies by calendar month AND by
#   latitude — day length swings a lot across the year at high latitudes
#   (Canada), barely at all near the equator. An earlier version of this
#   function used a single fixed near-equatorial constant, which was a
#   reasonable shortcut for Congo/Katanga alone but wrong for any region
#   far from the equator (e.g. northern Algeria at ~36N, where day length
#   genuinely varies by season).
#
#   day_length_factors() below replaces that with the full latitude-banded
#   monthly lookup tables, exactly as documented in Lawson & Armitage
#   (2008), "Weather Guide for the Canadian Forest Fire Danger Rating
#   System" (Natural Resources Canada), and implemented identically in the
#   official `cffdrs` R package used by Canadian fire agencies. This makes
#   the SAME compute_fwi() correct for every region — equatorial, temperate,
#   Southern Hemisphere — with no per-region constant to hand-tune when a
#   new region gets added later.
#
# OTHER APPROXIMATIONS (unchanged, and not latitude-specific):
#   - "Noon temperature" is approximated using the daily T2M_MAX, since
#     NASA POWER provides daily max/min rather than hourly readings — a
#     common substitution when only daily data is available.
#   - Wind speed is converted from NASA POWER's m/s to the km/h the FWI
#     System's equations expect.
# -----------------------------------------------------------------
_DMC_LE_BY_MONTH = {
    # lat >= 30N — the original Canadian standard table (Van Wagner 1987)
    "north_temperate": [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0],
    # 10N <= lat < 30N
    "north_subtropic": [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],
    # -30N <= lat < -10N (i.e. 10S-30S)
    "south_subtropic": [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],
    # lat < -30 (south of 30S)
    "south_temperate": [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8],
}
_DC_LF_BY_MONTH = {
    "north": [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],  # lat > 20N
    "south": [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8],  # lat < -20 (south of 20S)
}
_DMC_LE_EQUATOR = 9.0   # -10 <= lat <= 10: day length barely varies near the equator, constant all year
_DC_LF_EQUATOR = 1.4    # -20 <= lat <= 20: same reasoning, wider band for DC specifically


def day_length_factors(lat: float, month: int):
    """Returns (Le, Lf) — the DMC and DC day-length adjustment factors for
    this latitude and calendar month (1-12) — using the standard
    latitude-banded tables described above instead of a single fixed
    constant. Works correctly for any latitude, either hemisphere."""
    m = month - 1  # 0-indexed into the monthly tables

    if lat >= 30:
        le = _DMC_LE_BY_MONTH["north_temperate"][m]
    elif lat >= 10:
        le = _DMC_LE_BY_MONTH["north_subtropic"][m]
    elif lat >= -10:
        le = _DMC_LE_EQUATOR
    elif lat >= -30:
        le = _DMC_LE_BY_MONTH["south_subtropic"][m]
    else:
        le = _DMC_LE_BY_MONTH["south_temperate"][m]

    if lat > 20:
        lf = _DC_LF_BY_MONTH["north"][m]
    elif lat < -20:
        lf = _DC_LF_BY_MONTH["south"][m]
    else:
        lf = _DC_LF_EQUATOR

    return le, lf


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


def compute_fwi(region: str, lat: float, lon: float):
    """Runs the full Canadian FWI System recursively over EVERY day of
    weather on record for the nearest grid cell IN THE GIVEN REGION (each
    day's fuel moisture codes depend on the previous day's — this is
    inherent to the FWI System, not something we can skip), and returns
    the final day's component values. Returns None if there's no usable
    weather data for that location.

    Day-length factors (Le for DMC, Lf for DC) are looked up per day from
    day_length_factors(), keyed by this grid cell's actual latitude and
    each day's calendar month — so this same function is correct whether
    the nearest grid cell sits near the equator (Katanga) or at temperate
    latitudes (northern Algeria), with no region-specific constant."""
    climate_df = get_climate_df(region)
    grid_points = climate_df[["LAT", "LON"]].drop_duplicates()
    if grid_points.empty:
        return None
    dists = ((grid_points["LAT"] - lat) ** 2 + (grid_points["LON"] - lon) ** 2) ** 0.5
    g_lat, g_lon = grid_points.loc[dists.idxmin(), ["LAT", "LON"]]

    series = climate_df[(climate_df["LAT"] == g_lat) & (climate_df["LON"] == g_lon)].sort_values("date")
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
        # Day-length factors for THIS grid cell's latitude and THIS day's
        # calendar month — varies day to day for latitudes far from the
        # equator (e.g. northern Algeria), constant year-round near the
        # equator (e.g. Katanga) — both handled correctly by the same call.
        le, lf = day_length_factors(float(g_lat), row["date"].month)

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
        k = 1.894 * (Tc + 1.1) * (100 - RH) * le * 1e-6
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
        V = max(0.36 * (Tc2 + 2.8) + lf, 0.0)
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
    region: str = Field(region_config.DEFAULT_REGION, description="congo or algeria")
    province: str = Field(..., description="Province/wilaya name — must match the chosen region's list")
    lat: float = Field(..., json_schema_extra={"example": -11.66})
    lon: float = Field(..., json_schema_extra={"example": 27.48})
    phone_number: Optional[str] = Field(None, description="Reporter's phone number, if available")


class FireReportOut(BaseModel):
    report_id: int
    region: str
    province: str
    lat: float
    lon: float
    phone_number: Optional[str] = None
    reported_at_utc: str


class AssistanceRequestIn(BaseModel):
    region: str = Field(region_config.DEFAULT_REGION, description="congo or algeria")
    province: str = Field(..., description="Province/wilaya name — must match the chosen region's list")
    lat: float = Field(..., json_schema_extra={"example": -11.66})
    lon: float = Field(..., json_schema_extra={"example": 27.48})
    phone_number: Optional[str] = Field(None, description="Requester's phone number, if available")


class AssistanceRequestOut(BaseModel):
    request_id: int
    region: str
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
    region: str = Field(region_config.DEFAULT_REGION, description="congo or algeria")
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
    region: str = Field(region_config.DEFAULT_REGION, description="congo or algeria")
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
        "service": "Congo API — PHOENIX (multi-region)",
        "version": "3.0.0",
        "regions": list(region_config.REGIONS.keys()),
        "predictor": "congo_predict (congo) + generic XGBoost (other regions)" if HAS_CONGO_PREDICT else "dummy_fallback",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria")):
    return call_predict(
        region=region, lat=req.lat, lon=req.lon, doy=req.doy,
        t2m_max=req.t2m_max, t2m_min=req.t2m_min,
        rh2m=req.rh2m, ws2m=req.ws2m, prectotcorr=req.prectotcorr,
    )


@app.get("/risk-map", response_model=List[dict])
def risk_map(
    date: date_type = Query(..., description="Date to evaluate, e.g. 2026-08-12"),
    region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria"),
):
    """Fire risk for every grid cell using RECORDED weather. Health/AQI
    fields populate for any date within the last year of this region's
    latest available date — Open-Meteo only ever returns the CURRENT live
    reading, never historical air quality for the selected date, so this
    is always "right now", just made available while browsing recent
    history rather than only on the single most-recent date."""
    climate_df = get_climate_df(region)
    latest_available_date = get_latest_available_date(region)
    day_data = climate_df[climate_df["date"] == pd.Timestamp(date)]
    if day_data.empty:
        raise HTTPException(status_code=404, detail="No climate data available for this date.")

    days_from_latest = (latest_available_date - pd.Timestamp(date).normalize()).days
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
            region=region, lat=row["LAT"], lon=row["LON"], doy=doy,
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
    category: Optional[str] = Query(None, description="school, place_of_worship, health_facility, "
                                                        "fire_station, or emergency_shelter"),
    province: Optional[str] = Query(None, description="Province/wilaya name — depends on region"),
    only_shelters: bool = Query(False, description="If true, exclude support-only facilities"),
    region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria"),
):
    df = get_shelters_df(region)
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
    only_shelters: bool = Query(True, description="Restrict to real shelters (exclude support-only facilities)"),
    region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria"),
):
    df = get_shelters_df(region)
    df = df[df["available"] > 0].copy()
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
def alerts(
    date: date_type = Query(..., description="Date to evaluate, e.g. 2026-08-12"),
    region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria"),
):
    zones = risk_map(date, region)
    high_risk = [z for z in zones if z["risk_level"] == "High"]

    shelters_df = get_shelters_df(region)
    shelters = shelters_df[(shelters_df["is_shelter"]) & (shelters_df["available"] > 0)].copy()

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
    per phone number per hour (per region) to reduce fake/spam reports."""
    region_cfg = region_config.get_region(req.region)  # raises ValueError -> caught by FastAPI as 500;
    # kept simple since an invalid region is a programmer error, not a user one
    if req.province not in region_cfg["province_ref_points"]:
        raise HTTPException(status_code=400,
                             detail=f"province must be one of {list(region_cfg['province_ref_points'])} for region '{req.region}'")
    try:
        report_id = _save_fire_report(req.region, req.province, req.lat, req.lon, req.phone_number)
    except FireReportCooldownError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return {
        "report_id": report_id, "region": req.region, "province": req.province,
        "lat": req.lat, "lon": req.lon, "phone_number": req.phone_number,
        "reported_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/fire-reports", response_model=List[FireReportOut])
def list_fire_reports(
    hours: int = Query(72, description="Only reports from the last N hours"),
    region: Optional[str] = Query(None, description="Filter to one region; omit for all regions"),
):
    """Lists recent citizen fire reports, newest first. Defaults to the
    last 72 hours so old reports don't linger on the map forever."""
    df = _load_fire_reports()
    if df.empty:
        return []
    if region:
        df = df[df["region"] == region]
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
    region_cfg = region_config.get_region(req.region)
    if req.province not in region_cfg["province_ref_points"]:
        raise HTTPException(status_code=400,
                             detail=f"province must be one of {list(region_cfg['province_ref_points'])} for region '{req.region}'")
    try:
        request_id = _save_assistance_request(req.region, req.province, req.lat, req.lon, req.phone_number)
    except FireReportCooldownError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return {
        "request_id": request_id, "region": req.region, "province": req.province,
        "lat": req.lat, "lon": req.lon, "phone_number": req.phone_number,
        "requested_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/assistance-requests", response_model=List[AssistanceRequestOut])
def list_assistance_requests(
    hours: int = Query(72, description="Only requests from the last N hours"),
    region: Optional[str] = Query(None, description="Filter to one region; omit for all regions"),
):
    """Lists recent evacuation-assistance requests, newest first."""
    df = _load_assistance_requests()
    if df.empty:
        return []
    if region:
        df = df[df["region"] == region]
    df["requested_at_utc"] = pd.to_datetime(df["requested_at_utc"])
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=hours)
    df = df[df["requested_at_utc"] >= cutoff].sort_values("requested_at_utc", ascending=False)
    df["requested_at_utc"] = df["requested_at_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.to_dict(orient="records")


@app.patch("/shelters/{osm_id}/availability")
def update_shelter_availability(
    osm_id: str, req: ShelterAvailabilityRequest,
    region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria"),
):
    """Lets shelter staff update how many spots are currently open, and
    optionally report accessibility info (wheelchair access, ground floor,
    medical staff on-site) for elderly/disabled evacuees — only set fields
    the caller actually knows; anything left out stays as it was (never
    reset to "no" by omission). Changes persist for the life of this
    running instance (see the ephemeral-storage note on FIRE_REPORTS_CSV
    above — same caveat applies here)."""
    shelters_df = get_shelters_df(region)
    match = shelters_df["osm_id"].astype(str) == str(osm_id)
    if not match.any():
        raise HTTPException(status_code=404, detail=f"No shelter with osm_id={osm_id} in region '{region}'")
    shelters_df.loc[match, "available"] = req.available
    for field in ("wheelchair_accessible", "ground_floor", "medical_staff_onsite"):
        value = getattr(req, field)
        if value is not None:
            shelters_df.loc[match, field] = value
    set_shelters_df(region, shelters_df)
    shelters_df.to_csv(region_config.get_region(region)["shelters_csv"], index=False)
    updated = shelters_df.loc[match].iloc[0]
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
    region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria"),
):
    """Computes the real Canadian Fire Weather Index (FFMC/DMC/DC/ISI/BUI/
    FWI) for the nearest grid cell in the given region — an independent,
    internationally-used fire-danger standard, run alongside (not
    replacing) the ML model, as a cross-check. Day-length factors adapt
    automatically to the region's latitude — see compute_fwi()'s
    docstring."""
    result = compute_fwi(region, lat, lon)
    if result is None:
        raise HTTPException(status_code=404, detail="No weather data available for this location.")
    return result


# -----------------------------------------------------------------
# Dashboard visit tracking & admin stats — same ephemeral-storage caveat
# as fire reports and shelter updates (resets on redeploy).
# -----------------------------------------------------------------
VISITS_LOG_CSV = "dashboard_visits.csv"


@app.post("/track-visit")
def track_visit(region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria")):
    """Increments a simple visit counter. The dashboard calls this once per
    browser session (not per interaction), giving the admin view a rough
    usage signal, broken down by which region the visitor was looking at."""
    df = (pd.read_csv(VISITS_LOG_CSV) if os.path.exists(VISITS_LOG_CSV)
          else pd.DataFrame(columns=["timestamp_utc", "region"]))
    if "region" not in df.columns:
        df["region"] = region_config.DEFAULT_REGION  # back-fill old rows from before regions existed
    new_row = pd.DataFrame([{"timestamp_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                              "region": region}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(VISITS_LOG_CSV, index=False)
    return {"status": "ok", "total_visits": len(df)}


@app.get("/stats")
def get_stats(region: Optional[str] = Query(None, description="Filter to one region; omit for all regions combined")):
    """Aggregate usage/activity numbers for the admin dashboard: visits,
    citizen fire reports, evacuation-assistance requests, and shelter
    capacity — all real, measured figures (not simulated), though visit
    tracking only covers time since the last redeploy (ephemeral storage).
    Pass ?region=congo or ?region=algeria to scope to one region, or omit
    for every region combined."""
    visits_df = pd.read_csv(VISITS_LOG_CSV) if os.path.exists(VISITS_LOG_CSV) else pd.DataFrame()
    reports_df = _load_fire_reports()
    assistance_df = _load_assistance_requests()
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7)

    if region:
        if not visits_df.empty and "region" in visits_df.columns:
            visits_df = visits_df[visits_df["region"] == region]
        if not reports_df.empty:
            reports_df = reports_df[reports_df["region"] == region]
        if not assistance_df.empty:
            assistance_df = assistance_df[assistance_df["region"] == region]
        shelters_df = get_shelters_df(region)
    else:
        # Combined across every region that has ever been loaded this
        # process — good enough for a demo admin view; a persistent store
        # would let this reflect regions not yet touched this run too.
        shelters_frames = [get_shelters_df(rid) for rid in region_config.REGIONS]
        shelters_df = pd.concat(shelters_frames, ignore_index=True)

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
        "total_shelters": int(len(shelters_df)),
        "total_shelter_capacity": int(shelters_df["capacity"].sum()),
        "total_shelter_available": int(shelters_df["available"].sum()),
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
        clim = get_clim_engine(req.region).get_point_climatology(req.lat, req.lon, target_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = call_predict(
        region=req.region, lat=clim["lat"], lon=clim["lon"], doy=clim["doy"],
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
def risk_map_future(
    date: date_type = Query(..., description="Future date to predict, e.g. 2026-09-15"),
    region: str = Query(region_config.DEFAULT_REGION, description="congo or algeria"),
):
    """Full-grid fire risk forecast for a FUTURE date, using climatology."""
    try:
        clim_df = get_clim_engine(region).get_climatology_for_date(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    results = []
    for _, row in clim_df.iterrows():
        r = call_predict(
            region=region, lat=row["LAT"], lon=row["LON"], doy=row["DOY"],
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
            clim = get_clim_engine(req.region).get_point_climatology(req.lat, req.lon, target_date)
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
        region=req.region, lat=req.lat, lon=req.lon, doy=doy,
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
def _ussd_forecast_summary(region: str, lat: float, lon: float, lang: str) -> str:
    """High-risk zone count FORECAST for today (via climatology) + nearest
    available shelter, in the requested language. Uses the climatology
    engine for today's date rather than the historical CSV's latest recorded
    reading, so USSD/Voice always reflect a forecast, not old recorded data."""
    target_date = date_type.today()
    high_count = 0
    try:
        clim_df = get_clim_engine(region).get_climatology_for_date(target_date)
        for _, row in clim_df.iterrows():
            r = call_predict(
                region=region, lat=row["LAT"], lon=row["LON"], doy=row["DOY"],
                t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
                rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
            )
            if r["risk_level"] == "High":
                high_count += 1
    except ValueError:
        pass  # no historical data for this day-of-year — report 0 and continue

    shelters_df = get_shelters_df(region)
    shelters = shelters_df[(shelters_df["is_shelter"]) & (shelters_df["available"] > 0)].copy()
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
    if lang == "ar":
        shelter_line = (f"أقرب ملجأ: {nearest_name} ({nearest_dist} كم)"
                         if nearest_name else "لم يتم العثور على ملجأ متاح.")
        return (f"فينيكس - توقعات الحرائق ({date_str})\n"
                f"مناطق عالية الخطورة (متوقعة): {high_count}\n{shelter_line}\nابق آمنًا.")
    shelter_line = (f"Nearest shelter: {nearest_name} ({nearest_dist} km)"
                     if nearest_name else "No available shelter found.")
    return (f"PHOENIX - Fire Forecast ({date_str})\n"
            f"High-risk zones (forecast): {high_count}\n{shelter_line}\nStay safe.")


# Display name (in its own script) for each language code that ANY region
# might offer — a region only shows the subset listed in its own
# regions.py "languages" entry, so adding a region with a new language
# just means adding one entry here plus the corresponding _USSD_TEXT
# strings below.
_LANGUAGE_LABELS = {"en": "English", "fr": "Francais", "sw": "Kiswahili", "ar": "العربية"}


def _build_language_menu(region_cfg: dict) -> str:
    """CON menu text offering only the languages this region actually
    supports, numbered 1..N in the order regions.py lists them."""
    lines = ["CON Choose your language / Choisissez / Chagua lugha:"]
    for i, code in enumerate(region_cfg["languages"], start=1):
        lines.append(f"{i}. {_LANGUAGE_LABELS.get(code, code)}")
    return "\n".join(lines)


def _resolve_language_choice(region_cfg: dict, choice: str) -> Optional[str]:
    """Maps the digit the caller pressed back to a language code, scoped
    to THIS region's language list (so pressing '3' means Swahili for
    Congo but Arabic for Algeria — same digit, different meaning, exactly
    matching what that region's menu just displayed)."""
    langs = region_cfg["languages"]
    idx = int(choice) - 1 if choice.isdigit() else -1
    if 0 <= idx < len(langs):
        return langs[idx]
    return None


def _build_region_menu() -> str:
    lines = ["CON Welcome to PHOENIX Fire Alert\nBienvenue a PHOENIX\nمرحبا بكم في فينيكس\n"
             "Choose your country / Choisissez / اختر بلدك:"]
    for i, (rid, cfg) in enumerate(region_config.REGIONS.items(), start=1):
        lines.append(f"{i}. {cfg['flag']} {cfg['label']}")
    return "\n".join(lines)


def _resolve_region_choice(choice: str) -> Optional[str]:
    region_ids = list(region_config.REGIONS.keys())
    idx = int(choice) - 1 if choice.isdigit() else -1
    if 0 <= idx < len(region_ids):
        return region_ids[idx]
    return None


def _build_province_menu(region_cfg: dict, prompt: dict) -> Dict[str, str]:
    """Builds the CON menu text (per language) for choosing a province,
    numbered 1..N from THIS region's actual province list — replaces the
    old hardcoded 3-item Congo-only menu, since Algeria's prep script
    assigned 13 wilayas, not 3."""
    province_names = list(region_cfg["province_ref_points"].keys())
    menu_by_lang = {}
    for lang, header in prompt.items():
        lines = [f"CON {header}"]
        for i, name in enumerate(province_names, start=1):
            lines.append(f"{i}. {name}")
        menu_by_lang[lang] = "\n".join(lines)
    return menu_by_lang


def _resolve_province_choice(region_cfg: dict, choice: str) -> Optional[str]:
    province_names = list(region_cfg["province_ref_points"].keys())
    idx = int(choice) - 1 if choice.isdigit() else -1
    if 0 <= idx < len(province_names):
        return province_names[idx]
    return None


_USSD_TEXT = {
    "main_menu": {
        "en": "CON What would you like to do?\n1. Check fire risk & shelter\n2. Report a fire you saw\n"
              "3. Request evacuation help (elderly/disabled)",
        "fr": "CON Que voulez-vous faire ?\n1. Verifier le risque et l'abri\n2. Signaler un incendie\n"
              "3. Demander de l'aide pour evacuer (personnes agees/handicapees)",
        "sw": "CON Ungependa kufanya nini?\n1. Angalia hatari ya moto na makazi\n2. Ripoti moto ulioona\n"
              "3. Omba msaada wa uhamishaji (wazee/walemavu)",
        "ar": "CON ماذا تريد أن تفعل؟\n1. التحقق من خطر الحريق والملجأ\n2. الإبلاغ عن حريق شاهدته\n"
              "3. طلب مساعدة الإخلاء (كبار السن/ذوو الإعاقة)",
    },
    "province_check_prompt": {
        "en": "Choose your province:", "fr": "Choisissez votre province:",
        "sw": "Chagua mkoa wako:", "ar": "اختر منطقتك:",
    },
    "province_report_prompt": {
        "en": "Which province is the fire in?", "fr": "Dans quelle province est l'incendie ?",
        "sw": "Moto uko mkoa gani?", "ar": "في أي منطقة يوجد الحريق؟",
    },
    "province_assistance_prompt": {
        "en": "Which province do you need help in?",
        "fr": "Dans quelle province avez-vous besoin d'aide ?",
        "sw": "Unahitaji msaada mkoa gani?", "ar": "في أي منطقة تحتاج المساعدة؟",
    },
    "invalid": {
        "en": "END Invalid choice.", "fr": "END Choix invalide.", "sw": "END Chaguo batili.",
        "ar": "END اختيار غير صالح.",
    },
    "no_forecast": {
        "en": "No forecast available right now. Try again later.",
        "fr": "Aucune prevision disponible. Reessayez plus tard.",
        "sw": "Hakuna utabiri unaopatikana sasa. Jaribu tena baadaye.",
        "ar": "لا توجد توقعات متاحة الآن. حاول مرة أخرى لاحقًا.",
    },
    "report_thanks": {
        "en": "END Thank you! Your report (#{id}) has been recorded for {province}. Stay safe.",
        "fr": "END Merci ! Votre signalement (#{id}) a ete enregistre pour {province}. Restez en securite.",
        "sw": "END Asante! Ripoti yako (#{id}) imesajiliwa kwa {province}. Kaa salama.",
        "ar": "END شكرًا لك! تم تسجيل بلاغك (#{id}) لمنطقة {province}. ابق آمنًا.",
    },
    "report_cooldown": {
        "en": "END You already reported recently — thank you. Please wait a bit before reporting again.",
        "fr": "END Vous avez deja signale recemment — merci. Veuillez patienter avant de signaler a nouveau.",
        "sw": "END Tayari umeripoti hivi karibuni — asante. Tafadhali subiri kabla ya kuripoti tena.",
        "ar": "END لقد أبلغت مؤخرًا بالفعل — شكرًا لك. يرجى الانتظار قليلاً قبل الإبلاغ مرة أخرى.",
    },
    "report_failed": {
        "en": "END Could not save your report right now. Please try again later.",
        "fr": "END Impossible d'enregistrer votre signalement. Reessayez plus tard.",
        "sw": "END Imeshindikana kuhifadhi ripoti yako. Jaribu tena baadaye.",
        "ar": "END تعذر حفظ بلاغك الآن. يرجى المحاولة مرة أخرى لاحقًا.",
    },
    "assistance_thanks": {
        "en": "END Help request (#{id}) recorded for {province}. A responder will try to reach you. Stay safe.",
        "fr": "END Demande d'aide (#{id}) enregistree pour {province}. Un intervenant essaiera de vous "
              "joindre. Restez en securite.",
        "sw": "END Ombi la msaada (#{id}) limesajiliwa kwa {province}. Mwokozi atajaribu kuwafikia. "
              "Kaa salama.",
        "ar": "END تم تسجيل طلب المساعدة (#{id}) لمنطقة {province}. سيحاول أحد المستجيبين الوصول إليك. "
              "ابق آمنًا.",
    },
    "assistance_cooldown": {
        "en": "END A help request was already sent recently — it's been recorded. Please wait a few "
              "minutes before requesting again.",
        "fr": "END Une demande d'aide a deja ete envoyee recemment — elle a ete enregistree. Veuillez "
              "patienter quelques minutes avant de redemander.",
        "sw": "END Ombi la msaada tayari limetumwa hivi karibuni — limesajiliwa. Tafadhali subiri "
              "dakika chache kabla ya kuomba tena.",
        "ar": "END تم إرسال طلب مساعدة مؤخرًا بالفعل — وتم تسجيله. يرجى الانتظار بضع دقائق قبل الطلب مرة "
              "أخرى.",
    },
    "assistance_failed": {
        "en": "END Could not save your help request right now. Please try again, or ask someone nearby "
              "for help.",
        "fr": "END Impossible d'enregistrer votre demande d'aide. Reessayez, ou demandez de l'aide a "
              "quelqu'un a proximite.",
        "sw": "END Imeshindikana kuhifadhi ombi lako la msaada. Jaribu tena, au omba msaada kwa mtu "
              "aliye karibu.",
        "ar": "END تعذر حفظ طلب المساعدة الآن. يرجى المحاولة مرة أخرى، أو طلب المساعدة من شخص قريب منك.",
    },
    "session_error": {
        "en": "END Session error. Please try again.",
        "fr": "END Erreur de session. Reessayez.",
        "sw": "END Hitilafu ya kikao. Tafadhali jaribu tena.",
        "ar": "END خطأ في الجلسة. يرجى المحاولة مرة أخرى.",
    },
}


@app.post("/ussd")
async def ussd(request: Request):
    """Africa's Talking USSD callback. Register a USSD channel in the
    Africa's Talking console pointed at this URL — anyone can then dial the
    assigned code from ANY phone (no smartphone, app, or internet needed) to
    check the fire risk FORECAST and nearest shelter, OR report a fire they
    saw, OR request evacuation help — in whichever languages the chosen
    region supports.

    Every deployed region normally gets its OWN USSD service code from
    Africa's Talking (a code is tied to one country's telecom routing), so
    in practice a caller in Congo and a caller in Algeria would dial
    different numbers that both point at this SAME endpoint — the region
    picker below exists mainly for testing multiple regions against one
    shared USSD code/simulator, and degrades gracefully to "just Congo"
    for any region without its own registered code yet (see
    regions.py's "ussd_code": None for Algeria).

    Protocol: Africa's Talking POSTs form-encoded sessionId / serviceCode /
    phoneNumber / text. `text` accumulates the caller's choices separated by
    '*' as the session progresses (e.g. '', '1', '1*2', '1*2*1'). The
    response must start with 'CON ' to keep the session open and show
    another menu, or 'END ' to send a final message and hang up.

    Menu depth: [0] region -> [1] language -> [2] action -> [3] province."""
    form = await request.form()
    text = form.get("text", "")
    phone_number = form.get("phoneNumber", "")
    steps = text.split("*") if text else []

    # Step 0: no input yet -> show the region picker.
    if text == "":
        response = _build_region_menu()

    # Step 1: region chosen -> show that region's language menu.
    elif len(steps) == 1:
        region = _resolve_region_choice(steps[0])
        if region is None:
            response = _USSD_TEXT["invalid"]["en"]  # no region selected yet, default to English for this one message
        else:
            response = _build_language_menu(region_config.get_region(region))

    # Step 2: region + language chosen -> show the main action menu.
    elif len(steps) == 2:
        region = _resolve_region_choice(steps[0])
        if region is None:
            response = _USSD_TEXT["invalid"]["en"]
        else:
            cfg = region_config.get_region(region)
            lang = _resolve_language_choice(cfg, steps[1])
            if lang is None:
                response = _USSD_TEXT["invalid"]["en"]
            else:
                response = _USSD_TEXT["main_menu"][lang]

    # Step 3: region + language + action chosen -> show the province menu
    # for that specific action (check / report / assistance).
    elif len(steps) == 3:
        region = _resolve_region_choice(steps[0])
        if region is None:
            response = _USSD_TEXT["invalid"]["en"]
        else:
            cfg = region_config.get_region(region)
            lang = _resolve_language_choice(cfg, steps[1])
            action = steps[2]
            if lang is None or action not in ("1", "2", "3"):
                response = _USSD_TEXT["invalid"][lang or "en"]
            else:
                prompt_key = {"1": "province_check_prompt", "2": "province_report_prompt",
                              "3": "province_assistance_prompt"}[action]
                menu_by_lang = _build_province_menu(cfg, _USSD_TEXT[prompt_key])
                response = menu_by_lang[lang]

    # Step 4: everything chosen, including province -> take the action.
    elif len(steps) == 4:
        region = _resolve_region_choice(steps[0])
        if region is None:
            response = _USSD_TEXT["invalid"]["en"]
        else:
            cfg = region_config.get_region(region)
            lang = _resolve_language_choice(cfg, steps[1])
            action = steps[2]
            province = _resolve_province_choice(cfg, steps[3])
            if lang is None or province is None:
                response = _USSD_TEXT["invalid"][lang or "en"]
            elif action == "1":
                ref_lat, ref_lon = cfg["province_ref_points"][province]
                try:
                    summary = _ussd_forecast_summary(region, ref_lat, ref_lon, lang)
                except Exception:
                    summary = _USSD_TEXT["no_forecast"][lang]
                response = f"END {summary}"
            elif action == "2":
                # Crowd-sourced report — no GPS on USSD, so we log it at the
                # province's reference point. Good enough for "something is
                # happening in this province, worth a look" — not a precise pin.
                ref_lat, ref_lon = cfg["province_ref_points"][province]
                try:
                    report_id = _save_fire_report(region, province, ref_lat, ref_lon, phone_number)
                    response = _USSD_TEXT["report_thanks"][lang].format(id=report_id, province=province)
                except FireReportCooldownError:
                    response = _USSD_TEXT["report_cooldown"][lang]
                except Exception:
                    response = _USSD_TEXT["report_failed"][lang]
            else:
                # Evacuation assistance request — same province-level location
                # limitation as fire reports (no GPS on USSD).
                ref_lat, ref_lon = cfg["province_ref_points"][province]
                try:
                    request_id = _save_assistance_request(region, province, ref_lat, ref_lon, phone_number)
                    response = _USSD_TEXT["assistance_thanks"][lang].format(id=request_id, province=province)
                except FireReportCooldownError:
                    response = _USSD_TEXT["assistance_cooldown"][lang]
                except Exception:
                    response = _USSD_TEXT["assistance_failed"][lang]

    else:
        response = _USSD_TEXT["session_error"]["en"]

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
