"""
Send Scheduled Alerts — automatic daily wildfire SMS trigger (multi-region)
=================================================================================
Meant to run on a schedule (see .github/workflows/update-data.yml, which
runs this AFTER the daily data refresh so alerts always use fresh data),
once per region. Checks the latest available date for High-risk zones and
sends a real, multi-language SMS to every contact registered for that
region — no manual button click needed.

Reuses congo_sms_alerts.py's existing logic (climate loading, high-risk
detection, nearest-shelter lookup, contact loading, message building, SMS
sending) instead of duplicating it, so both scripts and every region share
one source of truth. congo_sms_alerts.py's message language(s) and USSD
code line come from regions.py automatically per region.

Requires AT_USERNAME / AT_API_KEY as environment variables — in GitHub
Actions, set these as repository secrets (Settings -> Secrets and
variables -> Actions) and pass them through `env:` in the workflow step.

Run manually:
    python send_scheduled_alerts.py --region congo
    python send_scheduled_alerts.py --region algeria
"""

import argparse
import sys

from congo_sms_alerts import (
    load_climate_clean, get_high_risk_zones, nearest_shelter, load_contacts,
    build_message, send_sms,
)
from air_quality import fetch_live_pm25
from risk_engine import get_alert
import regions as region_config


def run_for_region(region_id: str):
    cfg = region_config.get_region(region_id)
    climate = load_climate_clean(region_id)
    date_str = str(climate["date"].max().date())
    print(f"[scheduled-alerts:{region_id}] Checking fire risk for {date_str} ({cfg['label']}) ...")

    high_risk = get_high_risk_zones(region_id, date_str)
    print(f"[scheduled-alerts:{region_id}] Found {len(high_risk)} high-risk zone(s).")
    if high_risk.empty:
        print(f"[scheduled-alerts:{region_id}] No alerts to send today.")
        return

    contacts = load_contacts(region_id)
    recipients = contacts["phone_number"].tolist()
    if not recipients:
        print(f"[scheduled-alerts:{region_id}] No contacts registered — nothing to send.", file=sys.stderr)
        return

    for _, zone in high_risk.iterrows():
        shelter_name, dist, _province = nearest_shelter(region_id, zone["lat"], zone["lon"])

        health_advice = None
        pm25 = fetch_live_pm25(zone["lat"], zone["lon"])
        if pm25 is not None:
            alert = get_alert(zone["fire_probability"], pm25)
            health_advice = alert.health_advice

        message = build_message(region_id, zone, shelter_name, dist, health_advice)
        print(f"[scheduled-alerts:{region_id}] Sending for zone ({zone['lat']}, {zone['lon']}) "
              f"to {len(recipients)} recipient(s)")
        send_sms(recipients, message, live=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=list(region_config.REGIONS.keys()),
                         help="Which region to check and send scheduled alerts for.")
    args = parser.parse_args()
    run_for_region(args.region)


if __name__ == "__main__":
    main()
