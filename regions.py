"""
regions.py
==========
Single source of truth for per-region configuration (Congo + Algeria, and
any region added later). Both congo_api.py (FastAPI) and
congo_streamlit_app.py (Streamlit) import from this module — adding a
third region later means editing REGIONS here, not the app code.

GENERIC PREDICTOR
-----------------
Every region other than Congo shares the same 8-feature schema (LAT, LON,
DOY, T2M_MAX, T2M_MIN, RH2M, WS2M, PRECTOTCORR) and the same shared
risk-level thresholds (Low < 0.35, Medium 0.35-0.65, High >= 0.65 — the
same thresholds already shown to users in the dashboard sidebar).

The model FILE FORMAT is picked per-region by file extension, so a region
can use whichever library actually trained its model:
  - "*.json"   -> loaded as an xgboost.Booster (Booster.load_model)
  - "*.joblib" -> loaded as a plain scikit-learn estimator (joblib.load),
                  called via .predict_proba(...)[:, 1]
Algeria's model is currently a HistGradientBoostingClassifier (.joblib) —
xgboost wasn't available in the environment it was trained in. Swapping
in an XGBoost .json model later for the same region needs no code change
here, only regions.py's "model_path" config updated to the new file.

congo_api.py keeps using the existing congo_predict.py for the "congo"
region specifically (unchanged behavior for the model that's already been
tested), and only routes to predict_fire_risk_generic() for every other
region. If congo_predict.py does anything beyond raw model inference
(extra calibration, different thresholds), that stays exactly as it was —
this module only affects newly-added regions.
"""

from typing import Dict

import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    joblib = None

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
        "model_path": "congo_fire_risk_model.json",
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
        "model_path": "north_algeria_fire_risk_model.joblib",
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
# Generic predictor — used for any region that isn't Congo. Supports
# both an xgboost.Booster (.json) and a plain scikit-learn estimator
# (.joblib), picked by the model file's extension — see module docstring.
# -------------------------------------------------------------------
_MODEL_CACHE: Dict[str, object] = {}
_MODEL_KIND: Dict[str, str] = {}  # region_id -> "xgboost" | "sklearn"

_FEATURE_ORDER = ["LAT", "LON", "DOY", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]


def load_model(region_id: str):
    """Loads (and caches) the model for a region, in whichever format its
    file actually is. Cached so the model file is only read from disk
    once per region per running process."""
    if region_id not in _MODEL_CACHE:
        cfg = get_region(region_id)
        path = cfg["model_path"]
        if path.endswith(".joblib"):
            if not HAS_JOBLIB:
                raise RuntimeError("joblib is not installed — required to load "
                                    f"'{path}' for region '{region_id}'.")
            _MODEL_CACHE[region_id] = joblib.load(path)
            _MODEL_KIND[region_id] = "sklearn"
        else:
            if not HAS_XGBOOST:
                raise RuntimeError("xgboost is not installed — required to load "
                                    f"'{path}' for region '{region_id}'.")
            booster = xgb.Booster()
            booster.load_model(path)
            _MODEL_CACHE[region_id] = booster
            _MODEL_KIND[region_id] = "xgboost"
    return _MODEL_CACHE[region_id]


def predict_fire_risk_generic(region_id: str, lat: float, lon: float, doy: int,
                               t2m_max: float, t2m_min: float, rh2m: float,
                               ws2m: float, prectotcorr: float) -> dict:
    """Region-agnostic prediction — same **kwargs-in, dict-out shape as
    congo_predict.predict_fire_risk(), so callers don't need to care which
    region — or which underlying model format — they're calling."""
    model = load_model(region_id)
    row = pd.DataFrame([{
        "LAT": lat, "LON": lon, "DOY": doy,
        "T2M_MAX": t2m_max, "T2M_MIN": t2m_min,
        "RH2M": rh2m, "WS2M": ws2m, "PRECTOTCORR": prectotcorr,
    }])[_FEATURE_ORDER]

    if _MODEL_KIND[region_id] == "sklearn":
        prob = float(model.predict_proba(row)[:, 1][0])
    else:
        dmat = xgb.DMatrix(row, feature_names=_FEATURE_ORDER)
        prob = float(model.predict(dmat)[0])

    return {"fire_probability": round(prob, 4), "risk_level": classify_risk(prob)}
