"""
Congo (Katanga) Fire Risk Prediction — Model Package
=========================================================
Same pattern as phoenix_predict.py (Algeria), pointed at the Congo-trained
XGBoost model, using the same threshold logic (0.35 / 0.65) that both
regions independently converged on.

Usage:
    from congo_predict import predict_fire_risk
    result = predict_fire_risk(lat=-9.9, lon=27.5, doy=213,
                                t2m_max=32.0, t2m_min=19.0,
                                rh2m=35.0, ws2m=3.5, prectotcorr=0.0)
    print(result)
    # {'fire_probability': 0.87, 'risk_level': 'High'}
"""

from xgboost import XGBClassifier
import pandas as pd

import os

# Reuses the same BASE path defined earlier in your notebook's pipeline cell.
# If BASE isn't defined in this session (e.g. running this cell standalone),
# falls back to the full known path on your Drive.
try:
    MODEL_PATH = os.path.join(BASE, "congo_fire_risk_model.json")
except NameError:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "congo_fire_risk_model.json")

THRESHOLD_LOW = 0.35
THRESHOLD_HIGH = 0.65

_model = XGBClassifier()
_model.load_model(MODEL_PATH)


def risk_level(p: float) -> str:
    """Convert a fire probability (0-1) into a Low/Medium/High risk label."""
    if p < THRESHOLD_LOW:
        return "Low"
    elif p < THRESHOLD_HIGH:
        return "Medium"
    else:
        return "High"


def predict_fire_risk(lat, lon, doy, t2m_max, t2m_min, rh2m, ws2m, prectotcorr):
    """
    Run the trained Congo model on a single point's weather data and
    return both the raw probability and the risk category.
    """
    X = pd.DataFrame([{
        "LAT": lat, "LON": lon, "DOY": doy,
        "T2M_MAX": t2m_max, "T2M_MIN": t2m_min,
        "RH2M": rh2m, "WS2M": ws2m, "PRECTOTCORR": prectotcorr
    }])
    proba = float(_model.predict_proba(X)[0, 1])
    return {
        "fire_probability": round(proba, 4),
        "risk_level": risk_level(proba)
    }


if __name__ == "__main__":
    # quick self-test — a hot, dry, windy day in the Katanga dry season (June-Aug)
    example = predict_fire_risk(
        lat=-9.9, lon=27.5, doy=213,
        t2m_max=32.0, t2m_min=19.0,
        rh2m=35.0, ws2m=3.5, prectotcorr=0.0
    )
    print("Example prediction:", example)
