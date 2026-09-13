"""
PHOENIX — Multi-Region SMS/USSD Alert Sender (Africa's Talking)
=======================================================================
Sends a real SMS to registered contacts whenever a zone is flagged
"High" risk, including the nearest available shelter and a health
advisory when live air-quality data is reachable — for ANY region
registered in regions.py (currently Congo and Algeria).

This used to be Congo-only (and there was a separate, near-duplicate
Algeria script following "the same pattern"). It's now one script that
takes --region, so both regions share one source of truth instead of two
scripts that can quietly drift apart.

Africa's Talking confirmed to support DRC — same integration pattern
applies, just register a Sender ID with DRC's regulator (ARPTC) when
moving beyond the free Sandbox. Algeria's own regulatory requirements for
a production Sender ID may differ — check before going live there too.

SETUP (same for every region — skip if you already have an account):
  1. https://account.africastalking.com/auth/register
  2. Sandbox app -> Settings -> copy API Key (username is "sandbox")
  3. Sandbox -> Simulator -> add & verify test phone numbers
  4. export AT_USERNAME="sandbox"
     export AT_API_KEY="your_api_key_here"

DRY RUN (default, no account needed):
    python congo_sms_alerts.py --region congo
    python congo_sms_alerts.py --region algeria

LIVE SEND:
    python congo_sms_alerts.py --region congo --live

Specific date:
    python congo_sms_alerts.py --region congo --date 2026-08-12 --live
"""

import argparse
import os
import sys
from math import radians, sin, cos, sqrt, atan2

import pandas as pd

from congo_predict import predict_fire_risk as congo_predict_fire_risk
from risk_engine import get_alert
from air_quality import fetch_live_pm25
import regions as region_config


def predict_fire_risk_for_region(region_id: str, **kwargs) -> dict:
    """Same routing as the dashboard/API: Congo keeps using
    congo_predict.py unchanged; any other region uses the generic
    XGBoost predictor in regions.py."""
    cfg = region_config.get_region(region_id)
    if cfg.get("use_congo_predict", region_id == "congo"):
        return congo_predict_fire_risk(**kwargs)
    return region_config.predict_fire_risk_generic(region_id, **kwargs)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_climate_clean(region_id: str) -> pd.DataFrame:
    """Load the given region's climate data. NASA POWER has a ~3-5 day
    processing lag; unprocessed recent days come back as the fill value
    -999 instead of real numbers. Those are marked NaN per-column (not
    dropped as whole rows), so the most recent date still counts as
    'available' — get_high_risk_zones() skips just the specific points
    that are missing, instead of the script falling back to an older
    date."""
    path = region_config.get_region(region_id)["climate_csv"]
    df = pd.read_csv(path)
    df = df.dropna(subset=["YEAR", "DOY"])  # a few rows have genuinely missing YEAR/DOY
    df["YEAR"] = df["YEAR"].astype(int)
    df["DOY"] = df["DOY"].astype(int)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + \
                 pd.to_timedelta(df["DOY"] - 1, unit="D")
    weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
    df[weather_cols] = df[weather_cols].where(df[weather_cols] >= -900)  # -999 -> NaN
    return df


def get_high_risk_zones(region_id: str, date_str: str) -> pd.DataFrame:
    climate = load_climate_clean(region_id)
    day = climate[climate["date"] == pd.Timestamp(date_str)]
    if day.empty:
        raise ValueError(f"No climate data available for {date_str} in region '{region_id}'")
    doy = pd.Timestamp(date_str).dayofyear

    weather_cols = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
    skipped = 0
    rows = []
    for _, r in day.iterrows():
        if r[weather_cols].isna().any():
            skipped += 1
            continue  # No data yet for this point — don't alert on it, don't crash
        pred = predict_fire_risk_for_region(
            region_id, lat=r["LAT"], lon=r["LON"], doy=doy,
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


def nearest_shelter(region_id: str, lat, lon):
    shelters_csv = region_config.get_region(region_id)["shelters_csv"]
    shelters = pd.read_csv(shelters_csv)
    shelters = shelters.rename(columns={"capacity_estimate": "capacity"})  # no-op if already renamed
    shelters = shelters[shelters["is_shelter"]]
    if shelters.empty:
        return None, None, None
    shelters = shelters.copy()
    shelters["dist"] = shelters.apply(lambda s: haversine_km(lat, lon, s["lat"], s["lon"]), axis=1)
    nearest = shelters.loc[shelters["dist"].idxmin()]
    return nearest["name"], round(nearest["dist"], 1), nearest.get("province")


def _default_contacts_path(region_id: str) -> str:
    """congo -> congo_contacts.csv (unchanged, so nothing breaks for
    anyone already using that file); every other region gets its own
    "<region>_contacts.csv" the same way."""
    return f"{region_id}_contacts.csv"


def load_contacts(region_id: str, path: str = None) -> pd.DataFrame:
    """
    <region>_contacts.csv format (same for every region):
        phone_number,region_name
        +243800000001,Haut-Katanga Community Rep
        +213500000001,Sétif Civil Protection

    Replace with real verified numbers once you have an Africa's Talking
    account. Congo's country code is +243, Algeria's is +213.

    NOTE ON WHERE THIS FILE LIVES: this script (and the scheduled GitHub
    Action that runs it) reads this CSV from the git checkout — NOT from
    whatever the live Streamlit dashboard has in memory. If contacts are
    meant to be added/edited from inside the dashboard, that UI needs to
    write to this same file AND commit/push it (or read from a shared
    database/API instead) for the scheduled job to ever see those
    contacts — a Streamlit Cloud session's local disk writes don't sync
    back into this repo by themselves.
    """
    path = path or _default_contacts_path(region_id)
    if not os.path.exists(path):
        print(f"[!] No {path} found — creating a sample file. Edit it with real numbers before --live sending.")
        pd.DataFrame([
            {"phone_number": "+000000000001", "region_name": "Sample Contact 1"},
            {"phone_number": "+000000000002", "region_name": "Sample Contact 2"},
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


# -------------------------------------------------------------
# Multilingual message building — one message per language the region
# actually supports (region_cfg["languages"]), joined together, matching
# the dashboard's own alert format. The USSD line is only included when
# the region has a registered code (regions.py's "ussd_code" isn't None).
# -------------------------------------------------------------
_TEMPLATES = {
    "en": "[PHOENIX ALERT] High wildfire risk near ({lat}, {lon}). Probability: {prob:.0f}%. "
          "{shelter_txt}{health_txt} Move livestock/valuables now.{ussd_line}",
    "fr": "[ALERTE PHOENIX] Risque eleve d'incendie pres de ({lat}, {lon}). Probabilite : {prob:.0f}%. "
          "{shelter_txt}{health_txt} Deplacez le betail/les biens de valeur maintenant.{ussd_line}",
    "sw": "[TAHADHARI YA PHOENIX] Hatari kubwa ya moto karibu na ({lat}, {lon}). Uwezekano: {prob:.0f}%. "
          "{shelter_txt}{health_txt} Hamisha mifugo/vitu vya thamani sasa.{ussd_line}",
    "ar": "[تنبيه فينيكس] خطر حريق مرتفع بالقرب من ({lat}, {lon}). الاحتمال: {prob:.0f}%. "
          "{shelter_txt}{health_txt} انقل الماشية/الممتلكات القيمة الآن.{ussd_line}",
}

_SHELTER_TEXT = {
    "en": lambda name, dist: (f"Nearest shelter: {name} ({dist:.1f} km)." if name
                               else "No nearby shelter found."),
    "fr": lambda name, dist: (f"Abri le plus proche : {name} ({dist:.1f} km)." if name
                               else "Aucun abri a proximite trouve."),
    "sw": lambda name, dist: (f"Makazi ya karibu: {name} ({dist:.1f} km)." if name
                               else "Hakuna makazi ya karibu yaliyopatikana."),
    "ar": lambda name, dist: (f"أقرب ملجأ: {name} ({dist:.1f} كم)." if name
                               else "لم يتم العثور على ملجأ قريب."),
}

_USSD_LINE_TEMPLATES = {
    "en": " Dial {ussd} for shelter info — no internet needed.",
    "fr": " Composez {ussd} pour les infos abris — sans internet.",
    "sw": " Piga {ussd} kwa taarifa za makazi — hauitaji intaneti.",
    "ar": " اتصل بـ {ussd} لمعلومات الملجأ — بدون إنترنت.",
}


def build_message(region_id: str, zone, shelter_name, shelter_dist_km, health_advice: str = None) -> str:
    """One message per language the region supports, joined together —
    matches the dashboard's own multi-language alert format exactly."""
    cfg = region_config.get_region(region_id)
    ussd = cfg["ussd_code"]

    parts = []
    for lang in cfg["languages"]:
        shelter_txt = _SHELTER_TEXT[lang](shelter_name, shelter_dist_km)
        health_txt = f" {health_advice}" if health_advice else ""
        ussd_line = _USSD_LINE_TEMPLATES[lang].format(ussd=ussd) if ussd else ""
        parts.append(_TEMPLATES[lang].format(
            lat=round(zone["lat"], 3), lon=round(zone["lon"], 3),
            prob=zone["fire_probability"] * 100, shelter_txt=shelter_txt,
            health_txt=health_txt, ussd_line=ussd_line,
        ))
    return "\n---\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, choices=list(region_config.REGIONS.keys()),
                         help="Which region to check and send alerts for.")
    parser.add_argument("--date", default=None, help="Date to check, e.g. 2026-08-12 (default: latest available)")
    parser.add_argument("--live", action="store_true", help="Actually send SMS via Africa's Talking (default: dry run)")
    args = parser.parse_args()

    cfg = region_config.get_region(args.region)
    climate = load_climate_clean(args.region)
    date_str = args.date or str(climate["date"].max().date())

    print(f"Checking fire risk for {date_str} ({cfg['label']}) ...")
    high_risk = get_high_risk_zones(args.region, date_str)
    print(f"Found {len(high_risk)} high-risk zone(s).")

    if high_risk.empty:
        print("No alerts to send.")
        return

    contacts = load_contacts(args.region)
    recipients = contacts["phone_number"].tolist()
    if not recipients:
        print("[!] No contacts registered — nothing to send.", file=sys.stderr)
        return

    for _, zone in high_risk.iterrows():
        shelter_name, dist, province = nearest_shelter(args.region, zone["lat"], zone["lon"])
        health_advice = None
        pm25 = fetch_live_pm25(zone["lat"], zone["lon"])
        if pm25 is not None:
            alert = get_alert(zone["fire_probability"], pm25)
            health_advice = alert.health_advice

        message = build_message(args.region, zone, shelter_name, dist, health_advice)
        print(f"\nZone ({zone['lat']}, {zone['lon']}) -> {len(recipients)} recipient(s)")
        send_sms(recipients, message, live=args.live)


if __name__ == "__main__":
    main()
