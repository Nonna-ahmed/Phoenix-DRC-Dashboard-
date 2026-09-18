"""
merge_algeria_fire_climate.py
================================
Cleans north_algeria_climate_final.csv (removes NASA POWER's -999 fill
values, fixes rows the auto-update job appended without the engineered
features) and merges in fire_nrt_SV-C2_808728.csv (NASA FIRMS VIIRS
active-fire detections) to produce a labeled training dataset — one row
per (grid cell, date), with a binary FIRE column, matching the same
label-building approach used for the original Congo model.

WHY THE CLEANING STEPS EXIST
-----------------------------
1. -999 fill values: NASA POWER's near-real-time feed uses -999 as a
   placeholder for a value it hasn't finished processing yet (~3-5 day
   lag). These are NOT real measurements — left in place, they'd corrupt
   any statistics or model trained on them. Converted to NaN instead of
   dropping the row, so the date/grid-cell still exists in the file
   (consistent with how every other script in this project handles the
   same -999 convention).

2. Missing `date` / `temp_avg_7d` / `rain_sum_30d` / `season` on 576 rows:
   these are rows appended by the scheduled update_climate_data.py job.
   That script fetches YEAR/DOY/weather columns directly from NASA POWER
   but was never told to also recompute the derived `date` string or the
   engineered rolling features — so every auto-appended row had those
   columns blank. This script recomputes all four for the WHOLE file
   (not just the gaps), so the rolling windows are correct at every
   boundary between the original bulk export and the auto-appended rows.

FIRE LABEL — HOW IT'S BUILT
-----------------------------
Each FIRMS detection is a precise lat/lon point; the climate file is a
0.5-degree grid. Every detection is assigned to its NEAREST grid cell
(haversine distance — same approach used everywhere else in this
project for zone-to-facility matching), then grouped by (grid cell,
date). A grid-cell/date gets FIRE=1 if at least one detection matched
it that day.

`fire_detections` (count) and `frp_max` (max fire radiative power) are
also kept, for reference and QA — but per this project's own documented
finding for the Congo model, THESE TWO COLUMNS ARE LEAKAGE if used as
model inputs: they are only known once a fire is already burning and
detected, so they can't be used to predict risk *before* ignition.
Drop them before training exactly as the Congo model did; they're
included here only so you can sanity-check the label.

Run:
    python3 merge_algeria_fire_climate.py
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

CLIMATE_CSV = "north_algeria_climate_final.csv"
FIRE_CSV = "fire_nrt_SV-C2_808728.csv"
OUTPUT_CSV = "north_algeria_climate_fire_labeled.csv"

WEATHER_COLS = ["PRECTOTCORR", "RH2M", "T2M_MAX", "T2M_MIN", "WS2M", "WD2M"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def clean_climate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_total = len(df)

    # --- Step 1: -999 (and any other NASA POWER fill value <= -900) -> NaN ---
    present_weather_cols = [c for c in WEATHER_COLS if c in df.columns]
    fill_counts = {}
    for c in present_weather_cols:
        mask = df[c] <= -900
        fill_counts[c] = int(mask.sum())
        df.loc[mask, c] = np.nan

    # --- Step 2: rebuild `date`, `season` for every row from YEAR/DOY ---
    # (fixes the 576 rows the scheduled job appended without a `date`
    # column — see module docstring)
    df["YEAR"] = df["YEAR"].astype(int)
    df["DOY"] = df["DOY"].astype(int)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + \
                 pd.to_timedelta(df["DOY"] - 1, unit="D")

    month_to_season = {12: "Winter", 1: "Winter", 2: "Winter",
                        3: "Spring", 4: "Spring", 5: "Spring",
                        6: "Summer", 7: "Summer", 8: "Summer",
                        9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["date"].dt.month.map(month_to_season)

    # --- Step 3: rebuild the rolling engineered features for every row,
    # per grid cell, sorted by date (fixes the same 576-row gap and
    # keeps every window consistent across the whole file) ---
    df = df.sort_values(["LAT", "LON", "date"]).reset_index(drop=True)
    df["temp_avg_7d"] = (
        df.groupby(["LAT", "LON"])["T2M_MAX"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )
    df["rain_sum_30d"] = (
        df.groupby(["LAT", "LON"])["PRECTOTCORR"]
        .transform(lambda s: s.rolling(window=30, min_periods=1).sum())
    )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    print(f"Loaded {n_total} rows from {path}")
    print("NASA POWER -999 fill values found and converted to NaN:")
    for c, n in fill_counts.items():
        print(f"  {c}: {n}")
    print(f"Rebuilt date/season/temp_avg_7d/rain_sum_30d for all {len(df)} rows "
          f"(fixes the rows the scheduled job appended without them).")
    return df


def load_fire_detections(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_total = len(df)
    # Sanity check only — this file came back completely clean (no
    # nulls, no -999-style fill values), but check anyway rather than
    # assuming silently.
    numeric_cols = ["latitude", "longitude", "brightness", "scan", "track", "bright_t31", "frp"]
    bad = (df[numeric_cols] <= -900).any(axis=1) | df[numeric_cols].isna().any(axis=1)
    if bad.any():
        print(f"[!] Dropping {int(bad.sum())} FIRMS row(s) with invalid/fill values.")
        df = df[~bad].copy()
    print(f"Loaded {len(df)}/{n_total} usable fire detections from {path} "
          f"({df['acq_date'].min()} to {df['acq_date'].max()}).")
    return df


def assign_nearest_grid_cell(fire_df: pd.DataFrame, grid_points: pd.DataFrame) -> pd.DataFrame:
    """Vectorized nearest-grid-cell assignment (haversine) for every fire
    detection — same nearest-match approach used throughout this project."""
    R = 6371.0
    flat = np.radians(fire_df["latitude"].to_numpy())[:, None]
    flon = np.radians(fire_df["longitude"].to_numpy())[:, None]
    glat = np.radians(grid_points["LAT"].to_numpy())[None, :]
    glon = np.radians(grid_points["LON"].to_numpy())[None, :]

    dlat = glat - flat
    dlon = glon - flon
    a = np.sin(dlat / 2) ** 2 + np.cos(flat) * np.cos(glat) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    nearest_idx = np.argmin(dist, axis=1)
    out = fire_df.copy()
    out["LAT"] = grid_points["LAT"].to_numpy()[nearest_idx]
    out["LON"] = grid_points["LON"].to_numpy()[nearest_idx]
    out["match_distance_km"] = dist[np.arange(len(fire_df)), nearest_idx]
    return out


def main():
    climate = clean_climate(CLIMATE_CSV)
    fire = load_fire_detections(FIRE_CSV)

    grid_points = climate[["LAT", "LON"]].drop_duplicates().reset_index(drop=True)
    fire_matched = assign_nearest_grid_cell(fire, grid_points)

    print(f"\nFire-detection match distance to nearest grid cell: "
          f"mean={fire_matched['match_distance_km'].mean():.1f} km, "
          f"max={fire_matched['match_distance_km'].max():.1f} km")
    far = (fire_matched["match_distance_km"] > 40).sum()
    if far:
        print(f"[i] {far} detection(s) are >40 km from their nearest grid cell "
              f"(likely near the edge of the covered area) — still matched, not dropped.")

    fire_daily = (
        fire_matched.groupby(["LAT", "LON", "acq_date"])
        .agg(fire_detections=("latitude", "count"), frp_max=("frp", "max"))
        .reset_index()
        .rename(columns={"acq_date": "date"})
    )

    merged = climate.merge(fire_daily, on=["LAT", "LON", "date"], how="left")
    merged["fire_detections"] = merged["fire_detections"].fillna(0).astype(int)
    merged["FIRE"] = (merged["fire_detections"] > 0).astype(int)

    # QA numbers computed BEFORE dropping the leakage columns, since
    # fire_detections/frp_max are still useful for sanity-checking the
    # label right here — they just shouldn't reach the model as features.
    print(f"\nSaved rows: {len(merged)}")
    print(f"FIRE label distribution: {merged['FIRE'].sum()} fire-days "
          f"({merged['FIRE'].mean()*100:.2f}%), {(merged['FIRE']==0).sum()} non-fire-days "
          f"({(merged['FIRE']==0).mean()*100:.2f}%).")

    # Drop the leakage columns before saving — matches the Congo model's
    # own documented decision to drop `detections`/`frp_max` for the same
    # reason: both are only knowable once a fire is already detected, so
    # keeping them as model inputs would let the model "predict" a fire
    # using data that only exists because the fire already happened.
    merged = merged.drop(columns=["fire_detections", "frp_max"])
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Dropped fire_detections/frp_max (leakage) before saving — "
          f"{OUTPUT_CSV} is training-ready as-is.")


if __name__ == "__main__":
    main()
