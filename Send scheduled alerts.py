"""
Send Scheduled Alerts — automatic daily wildfire SMS trigger
=================================================================
Meant to run on a schedule (see .github/workflows/update-climate-data.yml,
which runs this AFTER the daily data refresh so alerts always use fresh
data). Checks the latest available date for High-risk zones and sends a
REAL bilingual (English + French) SMS to every contact in
congo_contacts.csv — no manual button click needed.

Reuses congo_sms_alerts.py's existing logic (climate loading, high-risk
detection, nearest-shelter lookup, contact loading, SMS sending) instead of
duplicating it, so both scripts share one source of truth.

Requires AT_USERNAME / AT_API_KEY as environment variables — in GitHub
Actions, set these as repository secrets (Settings -> Secrets and
variables -> Actions) and pass them through `env:` in the workflow step.

Run manually:
    python send_scheduled_alerts.py
"""

import sys

from congo_sms_alerts import (
    load_climate_clean, get_high_risk_zones, nearest_shelter, load_contacts, send_sms,
)

# Keep this in sync with USSD_SERVICE_CODE in congo_streamlit_app.py.
USSD_SERVICE_CODE = "*384*99838#"

_TEMPLATES = {
    "en": "[PHOENIX ALERT] High wildfire risk near ({lat}, {lon}). Probability: {prob:.0f}%. "
          "{shelter_txt} Move livestock/valuables now. Dial {ussd} for shelter info — no internet needed.",
    "fr": "[ALERTE PHOENIX] Risque eleve d'incendie pres de ({lat}, {lon}). Probabilite : {prob:.0f}%. "
          "{shelter_txt} Deplacez le betail/les biens de valeur maintenant. Composez {ussd} pour les infos abris.",
    "sw": "[TAHADHARI YA PHOENIX] Hatari kubwa ya moto karibu na ({lat}, {lon}). Uwezekano: {prob:.0f}%. "
          "{shelter_txt} Hamisha mifugo/vitu vya thamani sasa. Piga {ussd} kwa taarifa za makazi.",
}

_SHELTER_TEXT = {
    "en": lambda name, dist: (f"Nearest shelter: {name} ({dist:.1f} km)." if name
                               else "No nearby shelter found."),
    "fr": lambda name, dist: (f"Abri le plus proche : {name} ({dist:.1f} km)." if name
                               else "Aucun abri a proximite trouve."),
    "sw": lambda name, dist: (f"Makazi ya karibu: {name} ({dist:.1f} km)." if name
                               else "Hakuna makazi ya karibu yaliyopatikana."),
}


def build_message(zone, shelter_name, shelter_dist_km) -> str:
    """Trilingual EN+FR+SW alert text for one high-risk zone, matching the
    dashboard's alert format."""
    parts = []
    for lang in ("en", "fr", "sw"):
        shelter_txt = _SHELTER_TEXT[lang](shelter_name, shelter_dist_km)
        parts.append(_TEMPLATES[lang].format(
            lat=round(zone["lat"], 3), lon=round(zone["lon"], 3),
            prob=zone["fire_probability"] * 100, shelter_txt=shelter_txt, ussd=USSD_SERVICE_CODE,
        ))
    return "\n---\n".join(parts)


def main():
    climate = load_climate_clean()
    date_str = str(climate["date"].max().date())
    print(f"[scheduled-alerts] Checking fire risk for {date_str} ...")

    high_risk = get_high_risk_zones(date_str)
    print(f"[scheduled-alerts] Found {len(high_risk)} high-risk zone(s).")
    if high_risk.empty:
        print("[scheduled-alerts] No alerts to send today.")
        return

    contacts = load_contacts()
    recipients = contacts["phone_number"].tolist()
    if not recipients:
        print("[scheduled-alerts] No contacts registered — nothing to send.", file=sys.stderr)
        return

    for _, zone in high_risk.iterrows():
        shelter_name, dist, _province = nearest_shelter(zone["lat"], zone["lon"])
        message = build_message(zone, shelter_name, dist)
        print(f"[scheduled-alerts] Sending for zone ({zone['lat']}, {zone['lon']}) "
              f"to {len(recipients)} recipient(s)")
        send_sms(recipients, message, live=True)


if __name__ == "__main__":
    main()
