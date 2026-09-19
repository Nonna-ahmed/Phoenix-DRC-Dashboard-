"""
regions.py
==========
Single source of truth for per-region configuration (Congo + Algeria, and
any region added later). Both congo_api.py (FastAPI) and
congo_streamlit_app.py (Streamlit) import from this module — adding a
third region later means editing REGIONS here, not the app code.

GENERIC PREDICTOR
-----------------
The Congo and Algeria fire-risk models were confirmed to share the exact
same 8-feature schema (LAT, LON, DOY, T2M_MAX, T2M_MIN, RH2M, WS2M,
PRECTOTCORR) and the same binary:logistic XGBoost objective. This module
provides a region-agnostic predictor (predict_fire_risk_generic) that
loads either model as a plain xgboost.Booster and applies ONE shared
risk-level threshold — the same thresholds already shown to users in the
dashboard sidebar (Low < 0.35, Medium 0.35-0.65, High >= 0.65).

congo_api.py keeps using the existing congo_predict.py for the "congo"
region specifically (unchanged behavior for the model that's already been
tested), and only routes to predict_fire_risk_generic() for every other
region. If congo_predict.py does anything beyond raw XGBoost inference
(extra calibration, different thresholds), that stays exactly as it was —
this module only affects newly-added regions.
"""

from typing import Dict, Tuple

import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

# -------------------------------------------------------------------
# Shared risk-level thresholds — must match the text already shown in
# the dashboard sidebar ("🟢 Low: probability < 0.35", etc.) so a region
# using the generic predictor reports risk levels consistently with what
# the UI tells the user to expect.
# -------------------------------------------------------------------
RISK_THRESHOLDS = {"low_max": 0.35, "medium_max": 0.65}


def classify_risk(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["medium_max"]:
        return "High"
    if prob >= RISK_THRESHOLDS["low_max"]:
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
_MODEL_CACHE: Dict[str, "xgb.Booster"] = {}

_FEATURE_ORDER = ["LAT", "LON", "DOY", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]


def load_model(region_id: str):
    """Loads (and caches) the XGBoost booster for a region. Cached so the
    model file is only read from disk once per region per running process."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is not installed — required for any region other than "
                            "'congo' (which uses congo_predict.py instead).")
    if region_id not in _MODEL_CACHE:
        cfg = get_region(region_id)
        booster = xgb.Booster()
        booster.load_model(cfg["model_json"])
        _MODEL_CACHE[region_id] = booster
    return _MODEL_CACHE[region_id]


def predict_fire_risk_generic(region_id: str, lat: float, lon: float, doy: int,
                               t2m_max: float, t2m_min: float, rh2m: float,
                               ws2m: float, prectotcorr: float) -> dict:
    """Region-agnostic prediction — same **kwargs-in, dict-out shape as
    congo_predict.predict_fire_risk(), so callers don't need to care which
    path a given region takes."""
    booster = load_model(region_id)
    row = pd.DataFrame([{
        "LAT": lat, "LON": lon, "DOY": doy,
        "T2M_MAX": t2m_max, "T2M_MIN": t2m_min,
        "RH2M": rh2m, "WS2M": ws2m, "PRECTOTCORR": prectotcorr,
    }])
    dmat = xgb.DMatrix(row, feature_names=_FEATURE_ORDER)
    prob = float(booster.predict(dmat)[0])
    return {"fire_probability": round(prob, 4), "risk_level": classify_risk(prob)}
