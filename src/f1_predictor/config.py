"""Central configuration for the F1 predictor."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("F1_DATA_DIR", str(ROOT_DIR / "data")))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RESULTS_DIR = DATA_DIR / "results"
CACHE_DIR = DATA_DIR / "fastf1_cache"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, PREDICTIONS_DIR, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

ERGAST_BASE_URL = os.getenv("ERGAST_BASE_URL", "https://api.jolpi.ca/ergast/f1")

TRAIN_SEASONS = list(range(2018, 2026))
PREDICT_SEASON = int(os.getenv("F1_PREDICT_SEASON", "2026"))

MODEL_PATH = MODELS_DIR / "f1_winner_model.pkl"

FEATURE_COLS = [
    "grid_position",
    "quali_gap_to_pole",
    "quali_gap_to_teammate",
    "driver_form_3",
    "driver_form_5",
    "driver_form_10",
    "driver_avg_grid_3",
    "driver_avg_finish_3",
    "constructor_form_3",
    "constructor_form_5",
    "circuit_avg_finish",
    "circuit_best_finish",
    "circuit_wins_here",
    "season_wins_so_far",
    "season_podiums_so_far",
    "season_points_so_far",
    "championship_position",
    "races_since_win",
    "races_since_podium",
    "dnf_rate_10",
    "teammate_h2h_quali",
    "teammate_h2h_race",
    "experience_races",
    "is_home_race",
    "weather_is_wet",
    "track_temp_c",
    "air_temp_c",
    "fp_avg_lap_rank",
    "fp_long_run_pace_rank",
]
