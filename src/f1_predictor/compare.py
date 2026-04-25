"""Reconcile saved predictions against actual race results."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from f1_predictor.config import PREDICTIONS_DIR, RESULTS_DIR
from f1_predictor.ergast_client import ErgastClient

logger = logging.getLogger(__name__)


def compare_all() -> pd.DataFrame:
    """Walk every saved prediction file, attach actual results, write a leaderboard."""
    client = ErgastClient()
    rows = []
    actuals_cache: dict[int, pd.DataFrame] = {}

    for path in sorted(PREDICTIONS_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        season = payload["season"]
        rnd = payload["round"]

        if season not in actuals_cache:
            try:
                actuals_cache[season] = client.race_results(season)
            except Exception as exc:
                logger.warning("Could not fetch %s results: %s", season, exc)
                actuals_cache[season] = pd.DataFrame()

        actuals = actuals_cache[season]
        race_actuals = actuals[actuals["round"] == rnd] if not actuals.empty else pd.DataFrame()

        if race_actuals.empty:
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "race_name": payload["race_name"],
                    "status": "pending",
                }
            )
            continue

        ranked = sorted(payload["predictions"], key=lambda x: -x["win_prob"])
        predicted_winner = ranked[0]["driver_id"]
        predicted_podium = {r["driver_id"] for r in ranked[:3]}

        actual_winner_row = race_actuals[race_actuals["is_winner"] == 1]
        actual_winner = (
            actual_winner_row.iloc[0]["driver_id"] if not actual_winner_row.empty else None
        )
        actual_podium = set(race_actuals[race_actuals["is_podium"] == 1]["driver_id"].tolist())

        result_path = RESULTS_DIR / path.name
        result = {
            **payload,
            "actual_winner": actual_winner,
            "actual_podium": sorted(actual_podium),
            "predicted_winner": predicted_winner,
            "predicted_podium": sorted(predicted_podium),
            "winner_correct": int(predicted_winner == actual_winner),
            "podium_overlap": len(predicted_podium & actual_podium),
        }
        result_path.write_text(json.dumps(result, indent=2, default=str))

        rows.append(
            {
                "season": season,
                "round": rnd,
                "race_name": payload["race_name"],
                "status": "resolved",
                "predicted_winner": predicted_winner,
                "actual_winner": actual_winner,
                "winner_correct": result["winner_correct"],
                "podium_overlap": result["podium_overlap"],
            }
        )

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        out = RESULTS_DIR / "leaderboard.csv"
        leaderboard.to_csv(out, index=False)
        logger.info("Wrote leaderboard → %s", out)
    return leaderboard


def summary(leaderboard: pd.DataFrame) -> dict:
    resolved = leaderboard[leaderboard["status"] == "resolved"]
    if resolved.empty:
        return {"resolved_races": 0}
    return {
        "resolved_races": int(len(resolved)),
        "winner_accuracy": float(resolved["winner_correct"].mean()),
        "avg_podium_overlap": float(resolved["podium_overlap"].mean()),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lb = compare_all()
    if lb.empty:
        print("No predictions on disk yet.")
    else:
        print(lb.to_string(index=False))
        print("\nSummary:", json.dumps(summary(lb), indent=2))
