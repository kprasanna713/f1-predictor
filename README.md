---
title: F1 Race Predictor
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# F1 Race Winner Prediction

End-to-end ML pipeline that predicts F1 race winners using FastF1 telemetry + Ergast historical data.
Designed to run **weekly before each Grand Prix**, store predictions, then automatically compare them
against actual results once the race is over.

## Highlights

- **No AWS** — runs locally or in any container.
- **FastF1** for recent telemetry (lap times, sectors, weather, tyre data).
- **Ergast/Jolpica** for full historical race + qualifying + standings data.
- **30+ engineered features** — driver form, constructor form, circuit affinity, qualifying gap,
  weather, tyre pace, DNF rate, momentum, head-to-head vs teammate, and more.
- **XGBoost** classifier with cross-validated AUC reporting.
- **Weekly job** that detects the next race, predicts, and persists results.
- **Backfill comparison** that reconciles past predictions with actual finishing positions.
- **Docker + GitHub Actions** for CI/CD deployment.

## Quick start

```bash
pip install -r requirements.txt

# 1. Train (uses 2018-2025 by default; predicts 2026 season)
python -m f1_predictor.train

# 2. Predict the next race
python -m f1_predictor.predict

# 3. Compare past predictions to actual results
python -m f1_predictor.compare
```

## Weekly workflow

A single entry point runs the full weekly cycle:

```bash
python -m f1_predictor.weekly
```

This will:
1. Compare any unresolved past predictions against actual race results.
2. Refresh the training set with the latest completed race.
3. Re-train the model.
4. Predict the next upcoming race and write a timestamped JSON to `data/predictions/`.

Schedule it via cron, Windows Task Scheduler, or GitHub Actions (see `.github/workflows/weekly.yml`).

## Web UI

```bash
python -m f1_predictor serve
# open http://localhost:8000
```

## Docker

```bash
docker build -t f1-predictor .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" f1-predictor
```

## CI/CD

`.github/workflows/ci.yml` runs lint + tests on every push.
`.github/workflows/deploy.yml` builds and pushes the Docker image to GHCR on tag.
`.github/workflows/weekly.yml` runs the prediction job every Friday 12:00 UTC.
