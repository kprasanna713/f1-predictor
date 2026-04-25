"""Predict winner + podium probabilities for the next race."""
from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path

import joblib
import pandas as pd

from f1_predictor.config import MODEL_PATH, PREDICT_SEASON, PREDICTIONS_DIR
from f1_predictor.ergast_client import ErgastClient
from f1_predictor.fastf1_client import driver_code_to_id_map, practice_pace, race_weather
from f1_predictor.features import build_feature_table
from f1_predictor.ingest import load_raw

logger = logging.getLogger(__name__)


def predict_next_race(season: int = PREDICT_SEASON, round_num: int | None = None) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    bundle = joblib.load(MODEL_PATH)
    feature_cols = bundle["features"]

    client = ErgastClient()
    schedule = client.schedule(season)
    season_results = client.race_results(season)
    season_quali = client.qualifying(season)
    season_standings = client.driver_standings(season)

    round_num = round_num or _next_round(schedule, season_results)
    race_info = schedule[schedule["round"] == round_num].iloc[0].to_dict()
    logger.info("Predicting %s round %s — %s", season, round_num, race_info["race_name"])

    weather = race_weather(season, round_num)
    weather_df = pd.DataFrame([weather]) if weather else pd.DataFrame()
    fp_df = practice_pace(season, round_num)
    code_map = driver_code_to_id_map(season, round_num)

    raw = load_raw()
    full_results = pd.concat([raw["results"], season_results], ignore_index=True).drop_duplicates(
        subset=["season", "round", "driver_id"], keep="last"
    )
    full_quali = pd.concat([raw["qualifying"], season_quali], ignore_index=True).drop_duplicates(
        subset=["season", "round", "driver_id"], keep="last"
    )
    full_standings = pd.concat([raw["standings"], season_standings], ignore_index=True).drop_duplicates(
        subset=["season", "round", "driver_id"], keep="last"
    )
    full_weather = pd.concat([raw.get("weather", pd.DataFrame()), weather_df], ignore_index=True)
    full_fp = pd.concat([raw.get("fp_pace", pd.DataFrame()), fp_df], ignore_index=True)

    feat = build_feature_table(
        results=full_results,
        qualifying=full_quali,
        standings=full_standings,
        weather=full_weather,
        fp_pace=full_fp,
        driver_code_map=code_map,
    )

    pred_rows = feat[(feat["season"] == season) & (feat["round"] == round_num)].copy()
    if pred_rows.empty:
        # Qualifying hasn't happened yet — fall back to the last round with data
        available = feat[feat["season"] == season]["round"].max()
        if pd.isna(available):
            raise RuntimeError(f"No feature data available for season {season}.")
        logger.warning(
            "No data for round %s — falling back to last available round %s", round_num, int(available)
        )
        round_num = int(available)
        race_info = schedule[schedule["round"] == round_num].iloc[0].to_dict()
        pred_rows = feat[(feat["season"] == season) & (feat["round"] == round_num)].copy()

    pred_rows = pred_rows.fillna(pred_rows[feature_cols].median(numeric_only=True))
    pred_rows = pred_rows.fillna(0)

    X = pred_rows[feature_cols].values
    pred_rows["win_prob"] = bundle["winner"].predict_proba(X)[:, 1]
    pred_rows["podium_prob"] = bundle["podium"].predict_proba(X)[:, 1]
    pred_rows = pred_rows.sort_values("win_prob", ascending=False).reset_index(drop=True)

    payload = {
        "predicted_at": datetime.utcnow().isoformat(),
        "season": season,
        "round": int(round_num),
        "race_name": race_info["race_name"],
        "circuit_id": race_info["circuit_id"],
        "race_date": race_info.get("date", ""),
        "model_trained_at": bundle.get("trained_at"),
        "model_metrics": bundle.get("metrics"),
        "predictions": pred_rows[
            ["driver_id", "constructor_id", "grid_position", "win_prob", "podium_prob"]
        ].to_dict(orient="records"),
    }

    out_path = PREDICTIONS_DIR / f"{season}_round{int(round_num):02d}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Saved predictions → %s", out_path)
    return payload


def _next_round(schedule: pd.DataFrame, results: pd.DataFrame) -> int:
    completed = set(results["round"].unique()) if not results.empty else set()
    today = date.today().isoformat()
    upcoming = schedule[(~schedule["round"].isin(completed)) & (schedule["date"] >= today)]
    if upcoming.empty:
        # All races run or schedule lacks dates: fall back to first round not yet completed.
        remaining = sorted(set(schedule["round"].unique()) - completed)
        if not remaining:
            raise RuntimeError("Season complete — no upcoming race.")
        return int(remaining[0])
    return int(upcoming.sort_values("date").iloc[0]["round"])


def print_top10(payload: dict) -> None:
    print(f"\n-- {payload['race_name']} (round {payload['round']}) --")
    rows = payload["predictions"][:10]
    print(f"{'#':>2}  {'driver':<20} {'team':<22} {'grid':>4}  {'win%':>6}  {'pod%':>6}")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>2}  {r['driver_id']:<20} {r['constructor_id']:<22} "
            f"{int(r['grid_position'] or 0):>4}  "
            f"{r['win_prob']*100:>5.1f}%  {r['podium_prob']*100:>5.1f}%"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print_top10(predict_next_race())
