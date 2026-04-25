"""Smoke tests for the feature engineering pipeline."""
from __future__ import annotations

import pandas as pd

from f1_predictor.features import build_feature_table


def _toy_results() -> pd.DataFrame:
    rows = []
    drivers = [
        ("verstappen", "red_bull", "Dutch"),
        ("perez", "red_bull", "Mexican"),
        ("hamilton", "mercedes", "British"),
        ("russell", "mercedes", "British"),
    ]
    for season in (2023, 2024):
        for rnd in range(1, 4):
            for i, (drv, team, nat) in enumerate(drivers):
                pos = (i + (rnd % len(drivers))) % len(drivers) + 1
                rows.append(
                    {
                        "season": season, "round": rnd, "race_name": f"R{rnd}",
                        "circuit_id": f"c{rnd}", "country": "UK", "date": f"{season}-04-{rnd:02d}",
                        "driver_id": drv, "driver_nationality": nat,
                        "constructor_id": team,
                        "grid": i + 1, "position": pos,
                        "points": max(26 - pos * 5, 0), "laps": 50, "status": "Finished",
                        "is_winner": int(pos == 1),
                        "is_podium": int(pos <= 3),
                        "is_dnf": 0,
                    }
                )
    return pd.DataFrame(rows)


def _toy_qualifying() -> pd.DataFrame:
    base = _toy_results()[["season", "round", "circuit_id", "driver_id", "constructor_id"]].copy()
    base["quali_pos"] = base.groupby(["season", "round"]).cumcount() + 1
    base["q1"] = 90.0 + base["quali_pos"] * 0.1
    base["q2"] = base["q1"] - 0.5
    base["q3"] = base["q2"] - 0.3
    return base


def test_feature_table_has_required_columns():
    from f1_predictor.config import FEATURE_COLS

    df = build_feature_table(
        results=_toy_results(),
        qualifying=_toy_qualifying(),
        standings=pd.DataFrame(),
    )
    assert not df.empty
    for col in FEATURE_COLS:
        assert col in df.columns, f"missing feature: {col}"


def test_no_lookahead_in_form_features():
    df = build_feature_table(
        results=_toy_results(),
        qualifying=_toy_qualifying(),
        standings=pd.DataFrame(),
    )
    first_round = df[(df["season"] == 2023) & (df["round"] == 1)]
    # Before any race has been seen for a driver, rolling-mean form must be NaN.
    assert first_round["driver_form_3"].isna().all()
    assert (first_round["season_wins_so_far"] == 0).all()
