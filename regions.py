"""
regions.py
==========
Single source of truth for per-region configuration (Congo + Algeria, and
any region added later). Both congo_api.py (FastAPI) and
congo_streamlit_app.py (Streamlit) import from this module — adding a
third region later means editing REGIONS here, not the app code.

GENERIC PREDICTOR
-----------------
The Congo model uses the original 8-feature schema and is still served by
congo_predict.py (region flag ``use_congo_predict: True``) — UNCHANGED.

The Algeria model was retrained and now has a DIFFERENT schema (13 features:
LAT, LON, doy_sin, doy_cos, month, PRECTOTCORR, RH2M, T2M_MAX, T2M_MIN,
WS2M, temp_range, temp_avg_7d, rain_sum_30d). Its model JSON is a single
file that carries its own metadata under the "phoenix_meta" key:
    - features          -> exact column order the model expects
    - risk_thresholds   -> tuned Medium / High cut-offs for THIS model

So predict_fire_risk_generic() no longer hard-codes a feature list or
thresholds: it reads both from the model file. Two of the features
(temp_avg_7d, rain_sum_30d) are rolling windows — they need the previous
days' weather, so they are either passed in by the caller or computed here
from the region's climate CSV.
"""

import json
from datetime import date as _date, datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

# -------------------------------------------------------------------
# Default (shared) risk-level thresholds — used by Congo and by any
# region whose model file does NOT carry its own "phoenix_meta".
# The Algeria model DOES carry tuned thresholds, and those take priority
# (see get_thresholds). Its probabilities are inflated by class weighting
# (scale_pos_weight), so the shared 0.35 / 0.65 cut-offs do not fit it.
# -------------------------------------------------------------------
RISK_THRESHOLDS = {"low_max": 0.35, "medium_max": 0.65}


def get_thresholds(region_id: Optional[str] = None) -> dict:
    """{'low_max', 'medium_max'} for a region: Low < low_max <= Medium <
    medium_max <= High. Tuned per-model values when the model file has them,
    otherwise the shared defaults. Use this for the dashboard sidebar text too,
    so the UI always shows the thresholds actually being applied."""
    if region_id and region_id in REGIONS and not REGIONS[region_id].get("use_congo_predict"):
        try:
            meta = _load_bundle(region_id)[1]
            thr = meta.get("risk_thresholds")
            if thr:
                return {"low_max": float(thr["medium"]), "medium_max": float(thr["high"])}
        except Exception:
            pass  # model file missing/unreadable -> fall back to shared defaults
    return dict(RISK_THRESHOLDS)


def classify_risk(prob: float, region_id: Optional[str] = None) -> str:
    """Probability -> 'Low' / 'Medium' / 'High'. region_id is optional, so
    existing calls like classify_risk(p) behave exactly as before."""
    thr = get_thresholds(region_id)
    if prob >= thr["medium_max"]:
        return "High"
    if prob >= thr["low_max"]:
        return "Medium"
    return "Low"


# -------------------------------------------------------------------
# Region registry
# -------------------------------------------------------------------
REGIONS: Dict[str, dict] = {
    "congo": {
        "id": "congo",
        "label": "Congo — Katanga (DRC)",
        "flag": "🇨🇩",
        "climate_csv": "phoenix_climate_2020_2026.csv",
        "shelters_csv": "drc_katanga_shelters_final.csv",
        "model_json": "congo_fire_risk_model.json",
        "use_congo_predict": True,  # use the existing congo_predict.py, not the generic XGBoost path
        "firms_bbox": "24,-13,31,-4",  # west,south,east,north
        "map_center": (-9.9, 27.5),
        "map_zoom": 6,
        "province_ref_points": {
            "Haut-Katanga": (-11.6609, 27.4794),   # Lubumbashi
            "Lualaba": (-10.7167, 25.4667),        # Kolwezi
            "Tanganyika": (-5.9475, 29.1947),      # Kalemie
        },
        "languages": ["en", "fr", "sw"],
        "ussd_code": "*384*99838#",
    },
    "algeria": {
        "id": "algeria",
        "label": "Algeria — North-East",
        "flag": "🇩🇿",
        "climate_csv": "north_algeria_climate_final.csv",
        "shelters_csv": "north_algeria_shelters_final.csv",
        "model_json": "north_algeria_fire_risk_model.json",
        "use_congo_predict": False,  # route through the generic XGBoost predictor below
        "firms_bbox": "4,35.5,9,37.5",
        "map_center": (36.5, 6.5),
        "map_zoom": 7,
        "province_ref_points": {
            "Sétif": (36.1898, 5.4108),
            "Constantine": (36.3650, 6.6147),
            "Annaba": (36.9000, 7.7667),
            "Béjaïa": (36.7509, 5.0567),
            "Jijel": (36.8190, 5.7663),
            "Guelma": (36.4620, 7.4260),
            "Skikda": (36.8761, 6.9094),
            "Bordj Bou Arréridj": (36.0740, 4.7610),
            "Mila": (36.4503, 6.2646),
            "Souk Ahras": (36.2864, 7.9511),
            "El Tarf": (36.7672, 8.3138),
            "Oum El Bouaghi": (35.8753, 7.1135),
            "Tizi Ouzou": (36.7169, 4.0497),
        },
        "languages": ["en", "fr", "ar"],
        "ussd_code": "*384*96678#",
    },
}

DEFAULT_REGION = "congo"


def get_region(region_id: str) -> dict:
    if region_id not in REGIONS:
        raise ValueError(f"Unknown region '{region_id}'. Valid options: {list(REGIONS)}")
    return REGIONS[region_id]


def region_choices() -> list:
    """[(region_id, 'flag label'), ...] — handy for building a selectbox."""
    return [(rid, f"{cfg['flag']} {cfg['label']}") for rid, cfg in REGIONS.items()]


# -------------------------------------------------------------------
# Generic XGBoost predictor — used for any region that isn't Congo
# -------------------------------------------------------------------
# Only used if a model file has no "phoenix_meta" (should not happen for
# the current Algeria model, whose file lists its own features).
_FALLBACK_FEATURES = [
    "LAT", "LON", "doy_sin", "doy_cos", "month", "PRECTOTCORR", "RH2M",
    "T2M_MAX", "T2M_MIN", "WS2M", "temp_range", "temp_avg_7d", "rain_sum_30d",
]

_BUNDLE_CACHE: Dict[str, Tuple["xgb.Booster", dict]] = {}
_CLIMATE_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def _load_bundle(region_id: str):
    """Loads (and caches) (booster, meta) for a region. The model file is
    read from disk once per region per running process."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is not installed — required for any region other than "
                           "'congo' (which uses congo_predict.py instead).")
    if region_id not in _BUNDLE_CACHE:
        path = get_region(region_id)["model_json"]
        booster = xgb.Booster()
        booster.load_model(path)
        with open(path, encoding="utf-8") as f:
            meta = json.load(f).get("phoenix_meta", {})
        _BUNDLE_CACHE[region_id] = (booster, meta)
    return _BUNDLE_CACHE[region_id]


def load_model(region_id: str):
    """Kept for backward compatibility: returns just the Booster."""
    return _load_bundle(region_id)[0]


def _clean(x) -> float:
    """None / NASA POWER fill values (-999) -> NaN, so a placeholder is never
    mistaken for a real measurement (XGBoost treats NaN as 'missing')."""
    if x is None:
        return np.nan
    x = float(x)
    return np.nan if x <= -900 else x


def _to_date(d) -> Optional[_date]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, _date):
        return d
    return pd.to_datetime(d).date()


def _climate_history(region_id: str) -> Optional[pd.DataFrame]:
    """Loads the region's climate CSV once (date rebuilt from YEAR/DOY, -999 ->
    NaN — the same cleaning used when the model was trained)."""
    if region_id not in _CLIMATE_CACHE:
        try:
            df = pd.read_csv(get_region(region_id)["climate_csv"],
                             usecols=["LAT", "LON", "YEAR", "DOY", "T2M_MAX", "PRECTOTCORR"])
            for c in ("T2M_MAX", "PRECTOTCORR"):
                df.loc[df[c] <= -900, c] = np.nan
            df["date"] = pd.to_datetime(df["YEAR"].astype(int).astype(str), format="%Y") + \
                pd.to_timedelta(df["DOY"].astype(int) - 1, unit="D")
            _CLIMATE_CACHE[region_id] = df
        except (FileNotFoundError, ValueError, KeyError):
            _CLIMATE_CACHE[region_id] = None
    return _CLIMATE_CACHE[region_id]


def _rolling_features(region_id: str, lat: float, lon: float, target: _date,
                      t2m_max: float, prectotcorr: float) -> Tuple[float, float, bool]:
    """temp_avg_7d / rain_sum_30d exactly as built for training: 7-day mean of
    T2M_MAX and 30-day sum of PRECTOTCORR, INCLUDING the target day. The target
    day uses the caller's values; the previous 6 / 29 days come from the
    climate CSV of the nearest grid cell. Returns (temp_avg_7d, rain_sum_30d,
    history_found). Note: NASA POWER lags ~3-5 days, so the most recent days may
    be missing -> the rain sum can be a slight underestimate right at 'today'."""
    df = _climate_history(region_id)
    hist_t, hist_r = [], []
    if df is not None and len(df):
        cells = df[["LAT", "LON"]].drop_duplicates()
        i = ((cells["LAT"] - lat) ** 2 + (cells["LON"] - lon) ** 2).idxmin()
        cell = df[(df["LAT"] == cells.loc[i, "LAT"]) & (df["LON"] == cells.loc[i, "LON"])]
        t0 = pd.Timestamp(target)
        w7 = cell[(cell["date"] >= t0 - pd.Timedelta(days=6)) & (cell["date"] < t0)]
        w30 = cell[(cell["date"] >= t0 - pd.Timedelta(days=29)) & (cell["date"] < t0)]
        hist_t = w7["T2M_MAX"].dropna().tolist()
        hist_r = w30["PRECTOTCORR"].dropna().tolist()
    found = len(hist_t) > 0 or len(hist_r) > 0
    temps = [v for v in hist_t + [t2m_max] if not np.isnan(v)]
    rains = [v for v in hist_r + [prectotcorr] if not np.isnan(v)]
    temp7 = float(np.mean(temps)) if temps else np.nan
    rain30 = float(np.sum(rains)) if rains else np.nan
    return temp7, rain30, found


def predict_fire_risk_generic(region_id: str, lat: float, lon: float, doy: int,
                              t2m_max: float, t2m_min: float, rh2m: float,
                              ws2m: float, prectotcorr: float, *,
                              date=None, temp_avg_7d: Optional[float] = None,
                              rain_sum_30d: Optional[float] = None) -> dict:
    """Region-agnostic prediction — same kwargs-in, dict-out shape as
    congo_predict.predict_fire_risk(), so callers don't need to care which
    path a given region takes. Existing calls still work unchanged; the new
    keyword-only arguments are optional:

      date          date of the prediction (str / date). Used for `month` and to
                    find the previous days in the climate CSV. Default: today.
      temp_avg_7d,
      rain_sum_30d  pass them if you already have the rolling values; otherwise
                    they're computed from the region's climate CSV.

    Result also includes `history_available`: False means the rolling features
    could not be built from history (only today's values were used), so treat
    that prediction as lower-confidence."""
    booster, meta = _load_bundle(region_id)
    features = meta.get("features", _FALLBACK_FEATURES)

    t2m_max, t2m_min, rh2m, ws2m, prectotcorr = map(_clean, (t2m_max, t2m_min, rh2m, ws2m, prectotcorr))

    target = _to_date(date)
    if target is None:
        target = _date(_date.today().year, 1, 1) + timedelta(days=int(doy) - 1)

    history_available = True
    t7, r30 = _clean(temp_avg_7d), _clean(rain_sum_30d)
    if np.isnan(t7) or np.isnan(r30):
        t7_h, r30_h, history_available = _rolling_features(
            region_id, lat, lon, target, t2m_max, prectotcorr)
        t7 = t7_h if np.isnan(t7) else t7
        r30 = r30_h if np.isnan(r30) else r30

    row = {
        "LAT": lat, "LON": lon,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "month": target.month,
        "PRECTOTCORR": prectotcorr, "RH2M": rh2m,
        "T2M_MAX": t2m_max, "T2M_MIN": t2m_min, "WS2M": ws2m,
        "temp_range": t2m_max - t2m_min,
        "temp_avg_7d": t7, "rain_sum_30d": r30,
    }
    dmat = xgb.DMatrix(pd.DataFrame([row])[features], feature_names=features)
    prob = float(booster.predict(dmat)[0])
    return {"fire_probability": round(prob, 4),
            "risk_level": classify_risk(prob, region_id),
            "history_available": history_available}
