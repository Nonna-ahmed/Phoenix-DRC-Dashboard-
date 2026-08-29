"""
risk_engine.py
==============
Single source of truth for risk classification — imported by the dashboard,
the API, and the SMS alert trigger, so all three always agree.

This file shows BOTH approaches side by side so you can compare them,
plus my recommendation at the bottom (see `get_alert()`).
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Existing fire-risk classification (XGBoost probability -> 3-level output)
# Replace FIRE_THRESHOLDS with your actual tuned thresholds from the model.
# ---------------------------------------------------------------------------

FIRE_THRESHOLDS = {
    "low": 0.35,     # probability < 0.35  -> low     (matches PHOENIX tuned XGBoost threshold)
    "medium": 0.65,  # 0.35 <= probability < 0.65 -> medium
    # probability >= 0.65 -> high
}


def fire_risk_level(fire_probability: float) -> str:
    """Your existing logic — kept as-is, just referenced here for clarity."""
    if fire_probability < FIRE_THRESHOLDS["low"]:
        return "low"
    elif fire_probability < FIRE_THRESHOLDS["medium"]:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# OPTION A — Separate health_risk_level() driven purely by PM2.5
# Breakpoints follow the US EPA PM2.5 AQI scale (a defensible, published
# standard — not something we invented), collapsed to match your 3-level
# low/medium/high scheme.
# ---------------------------------------------------------------------------

def health_risk_level(pm2_5: Optional[float]) -> str:
    """
    Independent of the fire model. Based only on measured/estimated PM2.5.
    US EPA breakpoints (µg/m³, 24h basis):
      0.0  - 12.0   Good / Moderate           -> low
      12.1 - 35.4   Moderate / Unhealthy(SG)  -> low
      35.5 - 150.4  Unhealthy(SG) / Unhealthy -> medium
      150.5+        Very Unhealthy / Hazardous-> high
    """
    if pm2_5 is None:
        return "unknown"
    if pm2_5 <= 35.4:
        return "low"
    elif pm2_5 <= 150.4:
        return "medium"
    return "high"


def health_advice_text(level: str) -> str:
    return {
        "low": "Air quality is currently acceptable.",
        "medium": "Air quality is moderate — sensitive groups (elderly, children, "
                  "people with respiratory conditions) should take caution.",
        "high": "Air quality is dangerous — stay indoors and avoid outdoor exertion.",
        "unknown": "Air-quality data is not currently available.",
    }[level]


# ---------------------------------------------------------------------------
# OPTION B — Merge PM2.5 as a feature that shifts the fire risk score itself
# (illustrative only — the actual weight below is arbitrary and NOT
# statistically justified; doing this properly means retraining XGBoost
# with pm2_5 as an input feature and re-validating thresholds.)
# ---------------------------------------------------------------------------

def combined_risk_level_naive(fire_probability: float, pm2_5: Optional[float]) -> str:
    """
    Nudges the fire probability upward if PM2.5 is elevated, then re-applies
    the same thresholds. This is a quick heuristic, not a validated model.
    """
    adjusted = fire_probability
    if pm2_5 is not None and pm2_5 > 35.4:
        boost = min(0.15, (pm2_5 - 35.4) / 1000)  # arbitrary, small, capped
        adjusted = min(1.0, fire_probability + boost)
    return fire_risk_level(adjusted)


# ---------------------------------------------------------------------------
# Recommended pattern: keep the two signals SEPARATE and let each consumer
# (dashboard / API / SMS) decide how to present them together. This is
# what get_alert() below returns — a single dict, one source of truth,
# both signals visible and independently actionable.
# ---------------------------------------------------------------------------

@dataclass
class SiteAlert:
    fire_level: str
    health_level: str
    health_advice: str
    fire_probability: float
    pm2_5: Optional[float]

    def sms_text(self) -> str:
        fire_label = {"low": "Low", "medium": "Medium", "high": "High"}[self.fire_level]
        lines = [f"⚠️ Fire risk: {fire_label}."]
        if self.health_level in ("medium", "high"):
            lines.append(self.health_advice)
        return " ".join(lines)


def get_alert(fire_probability: float, pm2_5: Optional[float]) -> SiteAlert:
    """This is the ONE function the dashboard, API, and SMS trigger should
    all import and call. It never duplicates threshold logic elsewhere."""
    f_level = fire_risk_level(fire_probability)
    h_level = health_risk_level(pm2_5)
    return SiteAlert(
        fire_level=f_level,
        health_level=h_level,
        health_advice=health_advice_text(h_level),
        fire_probability=fire_probability,
        pm2_5=pm2_5,
    )


if __name__ == "__main__":
    # Quick sanity check
    example = get_alert(fire_probability=0.72, pm2_5=180)
    print(example)
    print(example.sms_text())
