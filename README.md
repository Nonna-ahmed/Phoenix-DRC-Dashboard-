# 🔥 PHOENIX — Multi-Region Wildfire Early Warning & Shelter Matching

PHOENIX is an early-warning system that predicts next-day wildfire risk from
weather data, cross-checks it against real satellite fire detections,
matches at-risk zones to the nearest evacuee shelter, and delivers alerts
over SMS, Voice, and USSD — including to people with **no smartphone and no
internet connection**. Built for the **AI for All Hackathon**.

> **The core idea:** an early-warning system is only useful if the warning
> actually reaches people. So alongside the dashboard, PHOENIX reaches
> people through whatever they actually have — smartphone, basic phone, or
> nothing but a signal — via SMS, voice calls, and USSD.

Two regions are live today, sharing one codebase:

| Region | Coverage | Grid resolution |
|---|---|---|
| 🇨🇩 Congo — Katanga (DRC) | Haut-Katanga, Lualaba, Tanganyika | 168 grid cells @ 0.5° |
| 🇩🇿 Algeria — North-East | Sétif, Constantine, Annaba, Béjaïa, Jijel, Guelma, Skikda and 6 more wilayas | 36 grid cells @ 0.5° |

Adding a third region means editing `regions.py` — the rest of the app
(dashboard, API, alert pipeline, scheduled jobs) doesn't need to change.

---

## 🌐 Live Demo

| | |
|---|---|
| **Dashboard** | `[(https://phoenix-drc-dashboard.streamlit.app/#all-citizen-fire-reports)]` |
| **API docs (Swagger)** | `[(https://phoenix-drc-dashboard-production.up.railway.app/)]/docs` |
| **USSD (sandbox)** | Dial `*384*99838#` (Congo) or `*384*96678#` (Algeria) in the [Africa's Talking simulator](https://developers.africastalking.com/simulator) |

---

## ❓ The Problem

Wildfires often spread faster than warnings can reach the people in their
path — and most existing fire-risk tools stop at "here's a map," assuming
everyone has a smartphone and a data connection to see it. In practice,
many at-risk residents don't.

## ✅ What PHOENIX Does About It

- **Predicts** next-day fire risk using a trained XGBoost model on live
  NASA weather data
- **Confirms** that prediction against real satellite fire detections
  (NASA FIRMS) and an independent, internationally-used fire-danger index
  (the Canadian FWI System) — so it's not a black box
- **Routes** people to their nearest shelter with real road directions and
  distance, walking or driving
- **Reaches everyone**, not just smartphone owners: SMS, voice calls, and a
  USSD menu that works on any basic phone with zero data
- **Listens back**: anyone can report a fire they see, or request
  evacuation help for an elderly or disabled person, from the same USSD
  menu — no app required

---

## ✨ Key Features

### 🗺️ Live Risk Map (Current Monitoring)
- Fire-risk grid (Low / Medium / High), updated daily from NASA POWER
  weather data
- 🔥 **Confirmed active fires** overlay (NASA FIRMS satellite detections)
  — separate from the *predicted* risk, so you can see where a fire is
  actually happening right now
- 📢 Citizen-reported fires and ♿ evacuation-assistance requests, both
  crowd-sourced via USSD
- 🧭 Wind-direction arrows (where data is available)
- Live air-quality (PM2.5) readings alongside fire risk
- Click any risk zone to see its nearest shelter, a **real routed path**
  (walking or driving, with distance & time), an on-demand **population
  estimate** (WorldPop), and a **Canadian FWI cross-check** against the
  ML prediction
- 📄 One-click **offline PDF report** for a selected zone — risk, air
  quality, nearest shelter, nearby fires — for field teams heading
  somewhere with no signal
- 📈 Seasonal fire-risk analysis and a simple prediction-vs-reality
  validation view
- 🌍 Region picker in the sidebar (Congo / Algeria today)

### 🔮 Future Prediction
- Forecasts fire risk for any future date using climatology (the
  historical weather average for that day of the year), for dates/places
  NASA doesn't have live weather for yet

### 📱 Multi-Channel Alerts
- **SMS** — multi-language alerts (Congo: English/French/Swahili;
  Algeria: English/French/Arabic), sent manually from the dashboard or
  **automatically every day** via a scheduled job for any newly
  High-risk zone
- **Voice calls** — for people who can't read text at all; real outbound
  calls require a live, paid Africa's Talking production account. A
  free, audible **Demo Audio** preview (generated live via gTTS, clearly
  labeled as a demo) is always available regardless of account status,
  so the accessibility value is always demonstrable
- **USSD** — works on any phone, no internet, no app. Menu flow: country
  → language → action (check fire risk & nearest shelter / report a fire
  you saw / request evacuation help) → province
  - Congo: `*384*99838#`
  - Algeria: `*384*96678#`

### 🏠 Shelter Management
- Shelter staff can update live availability and report accessibility
  info (wheelchair access, ground floor, medical staff on-site) — only
  ever set by a real person confirming it, never guessed

### 🔐 Admin Dashboard
- Password-protected tab: usage stats, all citizen reports, shelter
  status, Excel export

### ♿ Accessibility
- Large-text / high-contrast display mode
- Multilingual reach across every channel (English/French/Swahili for
  Congo, English/French/Arabic for Algeria)
- Voice + USSD specifically so no one is excluded for lacking a
  smartphone, literacy, or data access

---

## 🏗️ How It Works

```
┌─────────────────────┐         ┌──────────────────────┐
│  Streamlit Dashboard │◄───────►│   FastAPI Backend     │
│  (congo_streamlit_   │  HTTP   │   (congo_api.py)      │
│   app.py)            │         │   every endpoint      │
│  region picker       │         │   takes ?region=      │
└─────────────────────┘         └──────────┬─────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
             ┌──────▼──────┐        ┌───────▼───────┐       ┌───────▼───────┐
             │ NASA POWER  │        │ Africa's       │       │ NASA FIRMS /   │
             │ (weather)   │        │ Talking        │       │ WorldPop /     │
             │             │        │ (SMS/Voice/    │       │ Open-Meteo /   │
             │             │        │  USSD)         │       │ OSRM           │
             └─────────────┘        └───────────────┘       └───────────────┘
```

A **daily GitHub Actions workflow** refreshes climate + air-quality data
per region and triggers automatic alerts — the system runs itself without
anyone needing to click a button.

---

## 🧠 How Risk Is Predicted

### 🇨🇩 Congo (Katanga)

**Data:** 280,257 rows × 18 columns — daily weather + fire detections, DR
Congo, 2020–2026 (2026 partial). Balanced classes (50.4% fire-days / 49.6%
non-fire-days).

Two columns (`detections`, `frp_max`) were dropped during feature
selection — they're only known *after* a fire is already detected by
satellite, so keeping them would mean the model is re-encoding the label
itself rather than predicting ahead of it, which defeats the purpose of an
early-warning system.

**Models compared:** rule-based threshold, Logistic Regression, Random
Forest, HistGradientBoosting, XGBoost, LightGBM. The three
gradient-boosting variants were statistically indistinguishable;
**XGBoost** was selected on PR-AUC (0.9687) and Brier score (0.0774), using
PR-AUC as the primary KPI since missing a real fire-risk day (a false
negative) is far more costly than one unnecessary alert (a false
positive) — while also avoiding so many false alarms that the community
stops trusting the system (the "cry wolf" effect).

**Validation:** model selection used the 2024 holdout; final evaluation
ran on 2025–2026 (years the model never trained on).

| Metric | Value |
|---|---|
| PR-AUC | 0.9687 |
| Brier score | 0.0774 |
| ROC-AUC | 0.96 |
| Fire-risk days correctly flagged | 90% |
| False-alarm rate | 10% |

Confusion matrix on the 99,455-row test set: 5,268 false alarms, 4,764
missed fire-risk days (the costliest error type).

**Top drivers (feature importance + SHAP agree):** 30-day accumulated
rainfall and relative humidity dominate; wind speed ranks lowest in both
— it affects fire *spread*, not fire *occurrence*, which is what this
model predicts.

**Model file:** `fire_risk_model_XGBoost.joblib`
(scikit-learn/XGBoost binary classifier). Load with `joblib.load(...)`,
call `.predict_proba(X)[:, 1]` for the risk score.

**Input — 15 columns, in order:**

| Column | Type | Notes |
|---|---|---|
| `LAT`, `LON` | float | grid cell |
| `doy_sin`, `doy_cos` | float | `sin`/`cos(2π × day_of_year / 366)` |
| `T2M_MAX`, `T2M_MIN` | float | daily max/min temp (°C) |
| `RH2M` | float | relative humidity (%) |
| `WS2M` | float | wind speed |
| `PRECTOTCORR` | float | daily precipitation |
| `temp_avg_7d` | float | 7-day mean of `T2M_MAX` |
| `rain_sum_30d` | float | 30-day sum of `PRECTOTCORR` |
| `season_Autumn/Spring/Summer/Winter` | 0/1 | one-hot, exactly one = 1 |

**Scope & limitations:**
- Geography: **DR Congo only**, 168 grid cells at 0.5° resolution — don't
  imply national/continental validation in copy or marketing language.
- Predicts risk from *today's* conditions — there's no "days until
  ignition" countdown.
- Horizontal rollout to other African regions, other hazards, or finer
  spatial resolution is an untested hypothesis, not a validated claim.

### 🇩🇿 Algeria (North-East)

Same early-warning framing and the same leakage discipline (no
post-detection features). Four models were compared head-to-head:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RandomForest v1 | 0.871 | 0.689 | 0.932 | 0.792 | 0.959 |
| RandomForest v2 | 0.861 | 0.670 | 0.936 | 0.781 | 0.954 |
| **XGBoost v1** | **0.899** | **0.762** | 0.902 | **0.826** | **0.964** |
| XGBoost v2 | 0.899 | 0.767 | 0.887 | 0.823 | 0.962 |

**Selected: XGBoost v1** — trained on raw features only (no extra
engineered features). It wins on precision (76% vs. 69% for Random Forest
— far fewer false alarms), F1 (0.826, the best balance of precision and
recall), and ROC-AUC (0.964, the strongest overall discrimination).

**Interesting finding:** feature engineering (v2) didn't improve
XGBoost's performance either — confirmed independently across two model
families (Random Forest and XGBoost). Geographic location is the dominant
signal; accumulated-weather engineering on top of it adds little. That's
a finding worth stating confidently, not a modeling shortfall.

**The trade-off:** XGBoost v1's recall (90.2%) is a bit lower than Random
Forest's (93.2–93.6%) — it misses slightly more real fire-risk days in
exchange for far fewer false alarms. For an early-warning system the
safer default is usually to bias toward recall (missing a real fire is
worse than one extra alert), so this trade-off is worth revisiting as a
team decision if false negatives start to matter more in practice than
they do today.

**Model file:** `north_algeria_fire_risk_model.json` (XGBoost native
format, loaded via `regions.py`'s generic predictor — confirmed to use
the same 8 raw weather features as Congo's model: `LAT, LON, DOY,
T2M_MAX, T2M_MIN, RH2M, WS2M, PRECTOTCORR`).

**Scope & limitations:**
- Geography: **north-eastern Algeria only** (~13 wilayas), 36 grid cells
  at 0.5° resolution.
- Wilaya assignment for shelters uses nearest-administrative-capital
  matching (straight-line distance), not real administrative boundaries
  — a reasonable approximation, not survey-grade.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | [Streamlit](https://streamlit.io) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com) |
| ML models | XGBoost (`congo_predict.py` / `regions.py` generic predictor) |
| Hosting | [Railway](https://railway.app) |
| Automation | GitHub Actions (daily data refresh + alerts) |
| Messaging | [Africa's Talking](https://africastalking.com) (SMS, Voice, USSD) |
| Demo voice audio | [gTTS](https://pypi.org/project/gTTS/) (free, unofficial Google TTS) |
| Weather data | [NASA POWER](https://power.larc.nasa.gov) |
| Fire detection | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) |
| Air quality | [Open-Meteo](https://open-meteo.com) |
| Population | [WorldPop](https://www.worldpop.org) |
| Routing | [OSRM](http://project-osrm.org) (public demo server) |
| Fire-danger cross-check | Canadian FWI System (Van Wagner, 1987) |

See [`phoenix_drc_financial_cost_breakdown.md`](./phoenix_drc_financial_cost_breakdown.md)
for a full, sourced cost breakdown — the prototype runs on **under
$10/month**, almost entirely free tiers.

---

## 📁 Repository Layout

```
regions.py                          # per-region config: file paths, model, map center,
                                     # FIRMS bbox, languages, USSD code — add a region here only
congo_api.py                        # FastAPI backend, every endpoint takes ?region=
congo_streamlit_app.py              # Streamlit dashboard (region picker + 3 tabs:
                                     # Monitoring, Forecast, Admin)
congo_predict.py                    # Congo's model wrapper (joblib/XGBoost)
risk_engine.py                      # fire-probability + air-quality -> health advisory
                                     # (shared logic)
air_quality.py                      # live PM2.5 lookup (Open-Meteo)

phoenix_climate_2020_2026.csv       # Congo weather history (auto-refreshed)
drc_katanga_shelters_final.csv      # Congo shelters (schools, places of worship,
                                     # health facilities)
congo_fire_risk_model.json / fire_risk_model_XGBoost.joblib   # Congo model

north_algeria_climate_final.csv     # Algeria weather history
north_algeria_shelters_final.csv    # Algeria shelters (+ fire stations, emergency
                                     # shelters)
north_algeria_fire_risk_model.json  # Algeria model
prepare_algeria_shelters.py         # one-off script that built the shelters file above

congo_sms_alerts.py                 # multi-region SMS alert sender (Africa's Talking)
send_scheduled_alerts.py            # wraps congo_sms_alerts.py for the scheduled job
update_climate_data.py              # daily NASA POWER refresh, per region
update_shelter_air_quality.py       # daily Open-Meteo AQI refresh, per region
congo_contacts.csv / algeria_contacts.csv   # SMS recipient lists, per region

requirements.txt
railpack.json                       # Railway build config
.github/workflows/update-data.yml   # daily cron: refresh data -> commit -> send alerts
```

---

## ⚙️ Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Run the API locally
```bash
uvicorn congo_api:app --reload
```
Open `http://127.0.0.1:8000/docs` for interactive API docs.

### Run the dashboard locally
```bash
streamlit run congo_streamlit_app.py
```

### Secrets you'll need
Create `.streamlit/secrets.toml` (dashboard) and set the equivalents as
GitHub Actions repo secrets (for scheduled alerts):

```toml
AT_USERNAME = "sandbox"              # Africa's Talking — SMS/USSD (free sandbox works)
AT_API_KEY = "your_key_here"
AT_VOICE_USERNAME = "your_production_username"   # Voice needs a LIVE production app — no sandbox exists
AT_VOICE_API_KEY = "your_production_key"
AT_VOICE_CALLER_ID = "+243XXXXXXXXX"
FIRMS_MAP_KEY = "your_free_firms_key"            # firms.modaps.eosdis.nasa.gov/api/map_key
ADMIN_PASSWORD = "choose_a_password"             # gates the Admin tab
```

---

## ⚠️ Known Limitations (Being Upfront)

- **Citizen reports, assistance requests, and shelter updates** are
  stored on the API server's local disk — this persists while the server
  runs, but is wiped on every redeploy. Fine for a demo; a real
  deployment needs a proper database.
- **OSRM's free public routing server** only has road/driving data
  loaded — walking time is calculated separately (distance ÷ average
  walking speed), not from a real pedestrian-routing profile.
- **gTTS** (demo voice audio) is a free but *unofficial* wrapper around
  Google Translate's TTS endpoint — great for demos, not guaranteed
  production infrastructure.
- **Real Voice calls** require a live, paid Africa's Talking production
  account — there is no free sandbox for Voice at all. Until that's set
  up, the "Call Now" button honestly falls back to real, audible demo
  speech instead of pretending a call went through.
- **A nationwide dedicated USSD code** (vs. the shared sandbox code used
  today) is a separate, larger investment — see the cost breakdown doc.
- Both models predict **today's** risk from **today's** weather — they
  are not multi-day forecasts (the "Future Prediction" tab uses
  historical climatology, not a live forecast).
- NASA POWER has a ~3–5 day processing lag; the most recent few days will
  show "No data" until NASA finishes processing them — this is expected,
  not a bug.
- Neither model has been validated outside its own region — don't
  extrapolate performance claims from one region to the other, or to any
  hazard other than wildfire.
- Wilaya/province assignment for shelters (both regions) uses
  nearest-reference-point matching, not authoritative administrative
  boundaries.

---

## 🙏 Data & Acknowledgements

Built for the **AI for All Hackathon**. NASA POWER · NASA FIRMS ·
Open-Meteo · WorldPop (University of Southampton) · OpenStreetMap
contributors · Project OSRM · Africa's Talking · Van Wagner (1987),
Canadian Forest Fire Weather Index System

---

## 📄 License

MIT License

Copyright (c) 2026 Nehad Hesham Fathy Ramadan Ahmed

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
