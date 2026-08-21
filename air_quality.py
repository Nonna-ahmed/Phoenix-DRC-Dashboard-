"""
PHOENIX — Live Air Quality Fetcher (Open-Meteo)
===================================================
Open-Meteo's Air Quality API is free, requires no API key, and gives
real current + short-term forecast PM2.5/AQI for any lat/lon worldwide.

Limitation (important, and shown wherever this is used): it only
covers TODAY/near-term — it cannot give real air quality for past
dates like 2021-06-15. For historical dates, the dashboard/API should
say "not available" rather than guess.
"""

import requests

OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_live_pm25(lat: float, lon: float, timeout: int = 5):
    """
    Returns the current PM2.5 (µg/m³) for a point, or None if the
    request fails (no internet, API down, etc.) — callers must handle None.
    """
    try:
        r = requests.get(
            OPEN_METEO_AQ_URL,
            params={"latitude": lat, "longitude": lon, "current": "pm2_5"},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["current"]["pm2_5"]
    except Exception:
        return None


if __name__ == "__main__":
    # quick self-test (needs internet access to run)
    print("Kabylie area PM2.5:", fetch_live_pm25(36.7, 4.5))
