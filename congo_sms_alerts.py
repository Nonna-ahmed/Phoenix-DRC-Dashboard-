"""
PHOENIX — Congo (Katanga) SMS/USSD Alert Sender (Africa's Talking)
=======================================================================
Same pattern as the Algeria sms_alerts.py. Sends a real SMS to registered
contacts whenever a zone is flagged "High" risk, including the nearest
available shelter and a health advisory when live air-quality data is
reachable.

Africa's Talking confirmed to support DRC — same integration pattern
applies, just register a Sender ID with DRC's regulator (ARPTC) when
moving beyond the free Sandbox.

SETUP (same as Algeria — skip if you already have an account):
  1. https://account.africastalking.com/auth/register
  2. Sandbox app -> Settings -> copy API Key (username is "sandbox")
  3. Sandbox -> Simulator -> add & verify test phone numbers
  4. export AT_USERNAME="sandbox"
     export AT_API_KEY="your_api_key_here"

DRY RUN (default, no account needed):
    python congo_sms_alerts.py

LIVE SEND:
    python congo_sms_alerts.py --live

Specific date:
    python congo_sms_alerts.py --date 2026-08-12 --live
"""

import argparse
import os
import sys
import pandas as pd

from congo_predict import predict_fire_risk
from risk_engine import get_alert
from air_quality import fetch_live_pm25
from math import radians, sin, cos, sqrt, atan2


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_climate_clean(path="phoenix_climate_2020_2026.csv"):
    """Load climate data. NASA POWER has a ~3-5 day processing lag; unprocessed
    recent days come back as the fill value -999 instead of real numbers.
    Those are marked NaN per-column (not dropped as whole rows), so the most
    recent date still counts as 'available' — get_high_risk_zones() skips
    just the specific points that are missing, instead of the script falling
    back to an older date."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["YEAR", "DOY"])  # a few rows have genuinely missing YEAR/DOY
    df["YEAR"] = df["YEAR"].astype(int)
    df["DOY"] = df["DOY"].astype(int)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + \
                 pd.to_timedelta(df["DOY"] - 1, unit="D")
    weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
    df[weather_cols] = df[weather_cols].where(df[weather_cols] >= -900)  # -999 -> NaN
    return df


def get_high_risk_zones(date_str: str) -> pd.DataFrame:
    climate = load_climate_clean()
    day = climate[climate["date"] == pd.Timestamp(date_str)]
    if day.empty:
        raise ValueError(f"No climate data available for {date_str}")
    doy = pd.Timestamp(date_str).dayofyear

    weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
    skipped = 0
    rows = []
    for _, r in day.iterrows():
        if r[weather_cols].isna().any():
            skipped += 1
            continue  # No data yet for this point — don't alert on it, don't crash
        pred = predict_fire_risk(
            lat=r["LAT"], lon=r["LON"], doy=doy,
            t2m_max=r["T2M_MAX"], t2m_min=r["T2M_MIN"],
            rh2m=r["RH2M"], ws2m=r["WS2M"], prectotcorr=r["PRECTOTCORR"],
        )
        rows.append({"lat": r["LAT"], "lon": r["LON"], **pred})
    if skipped:
        print(f"[i] Skipped {skipped} point(s) with no data yet for {date_str}.")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["risk_level"] == "High"].reset_index(drop=True)


def nearest_shelter(lat, lon):
    shelters = pd.read_csv("drc_katanga_shelters_final.csv")
    shelters = shelters.rename(columns={"capacity_estimate": "capacity"})
    shelters = shelters[shelters["is_shelter"]]
    if shelters.empty:
        return None, None, None
    shelters = shelters.copy()
    shelters["dist"] = shelters.apply(lambda s: haversine_km(lat, lon, s["lat"], s["lon"]), axis=1)
    nearest = shelters.loc[shelters["dist"].idxmin()]
    return nearest["name"], round(nearest["dist"], 1), nearest.get("province")


def load_contacts(path="congo_contacts.csv"):
    """
    congo_contacts.csv format:
        phone_number,region_name
        +243800000001,Haut-Katanga Community Rep
        +243800000002,Lualaba Civil Protection
    Replace with real verified numbers once you have an Africa's Talking account.
    DRC country code is +243.
    """
    if not os.path.exists(path):
        print(f"[!] No {path} found — creating a sample file. Edit it with real numbers before --live sending.")
        pd.DataFrame([
            {"phone_number": "+243800000001", "region_name": "Sample Contact 1"},
            {"phone_number": "+243800000002", "region_name": "Sample Contact 2"},
        ]).to_csv(path, index=False)

    # Force phone_number to be read as TEXT, never a number — otherwise pandas
    # can silently turn "+243800000001" into a float, which the SDK rejects.
    df = pd.read_csv(path, dtype={"phone_number": str})
    df["phone_number"] = df["phone_number"].str.strip()
    df = df.dropna(subset=["phone_number"])
    df = df[df["phone_number"] != ""]

    invalid = df[~df["phone_number"].str.startswith("+")]
    if not invalid.empty:
        print(f"[!] Warning: {len(invalid)} phone number(s) in {path} don't start with '+' "
              f"(international format required, e.g. +243800000001): {invalid['phone_number'].tolist()}")

    return df


def send_sms(recipients, message, live=False):
    if not live:
        print("\n--- DRY RUN: no SMS actually sent ---")
        for r in recipients:
            print(f"  [WOULD SEND] to {r}: {message}")
        return

    import africastalking
    username = os.environ.get("AT_USERNAME")
    api_key = os.environ.get("AT_API_KEY")
    if not username or not api_key:
        print("[ERROR] AT_USERNAME / AT_API_KEY environment variables not set. "
              "See the setup instructions at the top of this file.")
        sys.exit(1)

    africastalking.initialize(username, api_key)
    sms = africastalking.SMS
    try:
        response = sms.send(message, recipients)
        print("Africa's Talking response:", response)
    except Exception as e:
        print(f"[ERROR] Failed to send SMS: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date to check, e.g. 2026-08-12 (default: latest available)")
    parser.add_argument("--live", action="store_true", help="Actually send SMS via Africa's Talking (default: dry run)")
    args = parser.parse_args()

    climate = load_climate_clean()
    date_str = args.date or str(climate["date"].max().date())

    print(f"Checking fire risk for {date_str} (Katanga region) ...")
    high_risk = get_high_risk_zones(date_str)
    print(f"Found {len(high_risk)} high-risk zone(s).")

    if high_risk.empty:
        print("No alerts to send.")
        return

    contacts = load_contacts()
    recipients = contacts["phone_number"].tolist()

    for _, zone in high_risk.iterrows():
        shelter_name, dist, province = nearest_shelter(zone["lat"], zone["lon"])
        shelter_txt = f"Nearest shelter: {shelter_name} ({dist} km, {province})." if shelter_name else "No nearby shelter found."
        health_txt = ""
        pm25 = fetch_live_pm25(zone["lat"], zone["lon"])
        if pm25 is not None:
            alert = get_alert(zone["fire_probability"], pm25)
            health_txt = f" {alert.health_advice}"

        message = (
            f"[PHOENIX ALERT] High wildfire risk near ({zone['lat']}, {zone['lon']}). "
            f"Probability: {zone['fire_probability']*100:.0f}%. {shelter_txt} "
            f"Move livestock/valuables now.{health_txt}"
        )
        print(f"\nZone ({zone['lat']}, {zone['lon']}) -> {len(recipients)} recipient(s)")
        send_sms(recipients, message, live=args.live)


if __name__ == "__main__":
    main()
