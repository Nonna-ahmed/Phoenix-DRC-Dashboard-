"""
Update Shelter Air Quality — Open-Meteo auto-refresh
=========================================================
Refreshes the pm2_5 / us_aqi / observation_time columns in
drc_katanga_shelters_final.csv for every shelter/facility, using
Open-Meteo's free Air Quality API (batched — one request per chunk of
locations, not one request per shelter, since there can be thousands
of shelters and Open-Meteo supports many coordinates per request via
comma-separated lat/lon lists).

Meant to be run on a schedule (see .github/workflows/update-climate-data.yml,
which runs this alongside the climate update). Air quality changes much
faster than climate/fire risk, so if you want it fresher than once a day,
add a second `schedule:` entry with a shorter interval in that workflow.

Run manually:
    python update_shelter_air_quality.py
"""

import sys
import time

import pandas as pd
import requests

SHELTERS_CSV = "drc_katanga_shelters_final.csv"
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo supports many locations per request via comma-separated
# lat/lon lists. Keep chunks modest to stay well within URL length and
# response-time limits.
CHUNK_SIZE = 100
REQUEST_DELAY_SECONDS = 1.0


def fetch_chunk(lats, lons):
    """Fetches current pm2_5 + us_aqi for a batch of points in ONE request.
    Returns a list of dicts, same order as the input lats/lons. On failure,
    returns None values for the whole chunk rather than crashing the run —
    callers should just move on to the next chunk."""
    try:
        resp = requests.get(
            OPEN_METEO_AQ_URL,
            params={
                "latitude": ",".join(str(x) for x in lats),
                "longitude": ",".join(str(x) for x in lons),
                "current": "pm2_5,us_aqi",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [!] Chunk request failed: {e}", file=sys.stderr)
        return [{"pm2_5": None, "us_aqi": None, "observation_time": None} for _ in lats]

    # Open-Meteo returns a LIST of per-location objects for multi-location
    # requests, and a single object for a single-location request.
    entries = data if isinstance(data, list) else [data]
    results = []
    for entry in entries:
        current = entry.get("current", {})
        results.append({
            "pm2_5": current.get("pm2_5"),
            "us_aqi": current.get("us_aqi"),
            "observation_time": current.get("time"),
        })
    while len(results) < len(lats):  # pad defensively if the API returned fewer entries
        results.append({"pm2_5": None, "us_aqi": None, "observation_time": None})
    return results


def main():
    print(f"Loading {SHELTERS_CSV} ...")
    df = pd.read_csv(SHELTERS_CSV)
    print(f"Found {len(df)} shelters/facilities.")

    pm25_col, aqi_col, obs_col = [], [], []
    for start in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[start:start + CHUNK_SIZE]
        print(f"  Fetching {start + 1}-{start + len(chunk)} of {len(df)} ...")
        results = fetch_chunk(chunk["lat"].tolist(), chunk["lon"].tolist())
        for r in results:
            pm25_col.append(r["pm2_5"])
            aqi_col.append(r["us_aqi"])
            obs_col.append(r["observation_time"])
        time.sleep(REQUEST_DELAY_SECONDS)

    df["pm2_5"] = pm25_col
    df["us_aqi"] = aqi_col
    df["observation_time"] = obs_col

    df.to_csv(SHELTERS_CSV, index=False)
    updated = df["pm2_5"].notna().sum()
    print(f"Saved. {updated}/{len(df)} shelters got fresh air-quality data.")


if __name__ == "__main__":
    main()
