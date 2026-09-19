"""
regions.py
==========
Single source of truth for per-region configuration (Congo + Algeria, and
any region added later). Both congo_api.py (FastAPI) and
congo_streamlit_app.py (Streamlit) import from this module — adding a
third region later means editing REGIONS here, not the app code.

GENERIC PREDICTOR
-----------------
NOTE: Congo and Algeria do NOT share an identical feature schema. Congo's
existing model expects 8 raw features (LAT, LON, DOY, T2M_MAX, T2M_MIN,
RH2M, WS2M, PRECTOTCORR) and is served via congo_predict.py, unchanged.

Algeria's model was trained on 9 features, with day-of-year encoded
cyclically (doy_sin/doy_cos instead of raw DOY) to better capture
seasonality — LAT, LON, PRECTOTCORR, RH2M, T2M_MAX, T2M_MIN, WS2M,
doy_sin, doy_cos. Its model file is also saved as a bundle (metadata +
model_comparison + the raw xgboost model nested under "xgboost_model"),
not a bare XGBoost booster file.

Because of this, predict_fire_risk_generic() does NOT assume a shared
schema across regions. Each region's model bundle carries its own
"feature_order" in its metadata; the predictor loads that, builds
whichever engineered features the order calls for (currently just the
cyclic DOY encoding), and orders the row accordingly. This keeps the
function correctly "generic" as more regions with their own schemas get
added, rather than silently mispredicting when a new region's model
doesn't match Congo's original 8-feature layout.

congo_api.py keeps using the existing congo_predict.py for the "congo"
region specifically (unchanged behavior for the model that's already been
tested), and only routes to predict_fire_risk_generic() for every other
region. If congo_predict.py does anything beyond raw XGBoost inference
(extra calibration, different thresholds), that stays exactly as it was —
this module only affects newly-added regions.
"""

import json
import math
import os
import tempfile
from typing import Dict, List, Tuple

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
# Cache holds, per region: {"booster": xgb.Booster, "feature_order": [...]}
_MODEL_CACHE: Dict[str, dict] = {}


def _load_bundle(model_json_path: str) -> dict:
    with open(model_json_path) as f:
        bundle = json.load(f)

    # Support both a plain XGBoost booster file (old/Congo-style) and the
    # newer bundle format {"metadata": {...}, "xgboost_model": {...}, ...}.
    if "xgboost_model" in bundle:
        raw_model = bundle["xgboost_model"]
        feature_order = bundle["metadata"]["feature_order"]
    else:
        raise ValueError(
            f"'{model_json_path}' doesn't look like a recognized model bundle "
            "(missing 'xgboost_model' key). If this is a bare XGBoost booster "
            "file, load it directly with load_model() from congo_predict.py "
            "instead of the generic predictor."
        )
    return raw_model, feature_order


def load_model(region_id: str) -> dict:
    """Loads (and caches) the XGBoost booster + its feature order for a
    region. Cached so the model file is only read from disk once per
    region per running process."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is not installed — required for any region other than "
                            "'congo' (which uses congo_predict.py instead).")
    if region_id not in _MODEL_CACHE:
        cfg = get_region(region_id)
        raw_model, feature_order = _load_bundle(cfg["model_json"])

        booster = xgb.Booster()
        # Booster.load_model wants a real file (or a buffer produced by
        # save_raw); round-tripping through a temp file guarantees we hand
        # it back exactly the bytes that were originally saved with
        # xgb_model.save_model(...), regardless of xgboost version quirks
        # around in-memory JSON buffers.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(raw_model, tmp)
            tmp_path = tmp.name
        try:
            booster.load_model(tmp_path)
        finally:
            os.remove(tmp_path)

        _MODEL_CACHE[region_id] = {"booster": booster, "feature_order": feature_order}
    return _MODEL_CACHE[region_id]


def _build_feature_row(feature_order: List[str], lat: float, lon: float, doy: int,
                        t2m_max: float, t2m_min: float, rh2m: float,
                        ws2m: float, prectotcorr: float) -> List[float]:
    """Builds a feature row in whatever order this region's model expects,
    deriving engineered features (currently: cyclic day-of-year) on demand
    rather than assuming every region encodes DOY the same way."""
    values = {
        "LAT": lat, "LON": lon, "DOY": doy,
        "T2M_MAX": t2m_max, "T2M_MIN": t2m_min,
        "RH2M": rh2m, "WS2M": ws2m, "PRECTOTCORR": prectotcorr,
    }
    if "doy_sin" in feature_order:
        values["doy_sin"] = math.sin(2 * math.pi * doy / 365.25)
    if "doy_cos" in feature_order:
        values["doy_cos"] = math.cos(2 * math.pi * doy / 365.25)

    missing = [c for c in feature_order if c not in values]
    if missing:
        raise ValueError(
            f"Model expects features {missing} that predict_fire_risk_generic() "
            "doesn't know how to compute yet — add them to _build_feature_row()."
        )
    return [values[c] for c in feature_order]


def predict_fire_risk_generic(region_id: str, lat: float, lon: float, doy: int,
                               t2m_max: float, t2m_min: float, rh2m: float,
                               ws2m: float, prectotcorr: float) -> dict:
    """Region-agnostic prediction — same **kwargs-in, dict-out shape as
    congo_predict.predict_fire_risk(), so callers don't need to care which
    path a given region takes. Internally, each region's own feature_order
    (stored in its model bundle) decides what gets fed to the model and in
    what order."""
    model = load_model(region_id)
    booster, feature_order = model["booster"], model["feature_order"]

    row = _build_feature_row(feature_order, lat, lon, doy, t2m_max, t2m_min,
                              rh2m, ws2m, prectotcorr)
    dmat = xgb.DMatrix([row], feature_names=feature_order)
    prob = float(booster.predict(dmat)[0])
    return {"fire_probability": round(prob, 4), "risk_level": classify_risk(prob)}
