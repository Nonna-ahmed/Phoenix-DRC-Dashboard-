"""
Update Climate Data — NASA POWER auto-refresh (multi-region)
==================================================================
Fetches the most recent days of weather from NASA POWER for every grid
point already present in the SELECTED REGION's climate CSV, and merges
them in — replacing any existing rows for the same (LAT, LON, date) so
that rows which were previously "-999" (not yet processed) get filled in
once NASA POWER catches up, and today's/yesterday's data gets added as
soon as it becomes available.

Meant to be run on a schedule (see .github/workflows/update-data.yml),
once per region. Can also be run manually:
    python update_climate_data.py --region congo
    python update_climate_data.py --region algeria

IMPORTANT — inherent limitation, not a bug:
NASA POWER's near-real-time data has a ~3-5 day processing lag. Running
this script daily will NOT make today's data appear today — it just keeps
the file moving forward automatically instead of staying frozen at a
single old date. The most recent 3-5 days will always show as unavailable
until NASA POWER finishes processing them, no matter how often this runs.

Adding a region: this script needs no changes — it reads whichever CSV
path regions.py's REGIONS[<id>]["climate_csv"] points to, and re-derives
the grid points from whatever (LAT, LON) pairs are already in that file.
"""

import argparse
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

import regions as region_config

# How many days back to re-fetch each run. Wider than the ~3-5 day lag so
# that rows which were "-999" last time get a chance to be filled in once
# NASA POWER finishes processing them.
LOOKBACK_DAYS = 10

# NASA POWER community — must match whatever produced the original CSV, or
# values may not line up with the historical data. "RE" (Renewable Energy)
# is the common choice for this parameter set; check power.larc.nasa.gov
# docs and adjust per-region below if a region's numbers look inconsistent
# with its own historical data (e.g. if Algeria's file was built with a
# different community setting than Congo's).
COMMUNITY = "RE"

PARAMETERS = "T2M_MAX,T2M_MIN,RH2M,WS2M,WD2M,PRECTOTCORR"

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Be polite to NASA POWER's servers between requests (also required —
# hammering the same points back-to-back risks getting rate-limited/blocked).
REQUEST_DELAY_SECONDS = 1.0


def fetch_point(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Fetches T2M_MAX/T2M_MIN/RH2M/WS2M/PRECTOTCORR for one grid point over
    [start, end]. Returns a DataFrame with columns LAT, LON, YEAR, DOY, and
    the 5 weather columns. Returns an empty DataFrame on any failure —
    callers should just skip that point and move on, not crash the run."""
    params = {
        "parameters": PARAMETERS,
        "community": COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    try:
        resp = requests.get(POWER_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        param_data = data["properties"]["parameter"]
    except Exception as e:
        print(f"  [!] Failed to fetch ({lat}, {lon}): {e}", file=sys.stderr)
        return pd.DataFrame()

    dates = sorted(param_data.get("T2M_MAX", {}).keys())
    rows = []
    for d in dates:
        d_obj = pd.Timestamp(d)
        rows.append({
            "LAT": lat,
            "LON": lon,
            "YEAR": d_obj.year,
            "DOY": d_obj.dayofyear,
            "T2M_MAX": param_data.get("T2M_MAX", {}).get(d),
            "T2M_MIN": param_data.get("T2M_MIN", {}).get(d),
            "RH2M": param_data.get("RH2M", {}).get(d),
            "WS2M": param_data.get("WS2M", {}).get(d),
            "WD2M": param_data.get("WD2M", {}).get(d),
            "PRECTOTCORR": param_data.get("PRECTOTCORR", {}).get(d),
        })
    return pd.DataFrame(rows)


def update_region(region_id: str):
    cfg = region_config.get_region(region_id)
    climate_csv = cfg["climate_csv"]

    print(f"=== Region: {cfg['flag']} {cfg['label']} ({climate_csv}) ===")
    print(f"Loading existing data from {climate_csv} ...")
    existing = pd.read_csv(climate_csv)
    existing["YEAR"] = existing["YEAR"].astype(int)
    existing["DOY"] = existing["DOY"].astype(int)

    grid_points = existing[["LAT", "LON"]].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(grid_points)} unique grid points.")

    end_date = date.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    print(f"Fetching {start_date} to {end_date} for each point (this can take a few minutes)...")

    new_rows = []
    for i, point in grid_points.iterrows():
        lat, lon = float(point["LAT"]), float(point["LON"])
        print(f"  [{i + 1}/{len(grid_points)}] ({lat}, {lon})")
        df = fetch_point(lat, lon, start_date, end_date)
        if not df.empty:
            new_rows.append(df)
        time.sleep(REQUEST_DELAY_SECONDS)

    if not new_rows:
        print("No new data fetched — leaving the file unchanged.")
        return

    fresh = pd.concat(new_rows, ignore_index=True)
    print(f"Fetched {len(fresh)} fresh rows.")

    # Replace: drop any existing rows matching (LAT, LON, YEAR, DOY) that we
    # just re-fetched, then append the fresh ones — this is how -999 rows
    # from earlier runs get corrected once NASA POWER finishes processing.
    key_cols = ["LAT", "LON", "YEAR", "DOY"]
    fresh_keys = set(map(tuple, fresh[key_cols].values))
    existing_keys = existing[key_cols].apply(tuple, axis=1)
    kept = existing[~existing_keys.isin(fresh_keys)]

    merged = pd.concat([kept, fresh], ignore_index=True)
    # Preserve any extra engineered columns already in the file (e.g.
    # Algeria's temp_avg_7d / rain_sum_30d / season) for rows we kept —
    # freshly-fetched rows simply won't have those columns populated,
    # which is fine: the app only actually needs the core weather columns.
    merged = merged.sort_values(["YEAR", "DOY", "LAT", "LON"]).reset_index(drop=True)

    merged.to_csv(climate_csv, index=False)
    print(f"Saved {len(merged)} total rows to {climate_csv} "
          f"({len(existing)} before -> {len(merged)} after).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=list(region_config.REGIONS.keys()),
                         help="Which region's climate file to refresh.")
    args = parser.parse_args()
    update_region(args.region)


if __name__ == "__main__":
    main()
