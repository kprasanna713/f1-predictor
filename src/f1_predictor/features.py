"""Feature engineering: turn raw race / qualifying / standings into model-ready rows."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from f1_predictor.config import FEATURE_COLS

logger = logging.getLogger(__name__)


HOME_RACE_BY_NATIONALITY = {
    "British": "UK",
    "Dutch": "Netherlands",
    "German": "Germany",
    "Spanish": "Spain",
    "Italian": "Italy",
    "French": "France",
    "Mexican": "Mexico",
    "Australian": "Australia",
    "Finnish": "Finland",
    "Monegasque": "Monaco",
    "Japanese": "Japan",
    "Canadian": "Canada",
    "American": "USA",
    "Brazilian": "Brazil",
    "Thai": "Thailand",
    "Chinese": "China",
    "Danish": "Denmark",
}


def build_feature_table(
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    standings: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    fp_pace: pd.DataFrame | None = None,
    driver_code_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Join all sources and compute per-(season, round, driver) features.

    Every rolling/historical feature uses `.shift(1)` to avoid look-ahead leakage.
    """
    if results.empty:
        return pd.DataFrame()

    df = results.copy().sort_values(["season", "round"]).reset_index(drop=True)

    df = _merge_qualifying(df, qualifying)
    df = _merge_standings(df, standings)
    df = _merge_weather(df, weather)
    df = _merge_fp_pace(df, fp_pace, driver_code_map)

    df = _add_form_features(df)
    df = _add_circuit_features(df)
    df = _add_season_features(df)
    df = _add_momentum_features(df)
    df = _add_teammate_features(df)
    df = _add_experience_feature(df)
    df = _add_home_race_feature(df)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df


def _merge_qualifying(df: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    if q.empty:
        df["quali_pos"] = df["grid"]
        df["grid_position"] = df["grid"]
        df["quali_gap_to_pole"] = np.nan
        df["quali_gap_to_teammate"] = np.nan
        return df

    q = q.copy()
    q["best_quali_time"] = q[["q1", "q2", "q3"]].min(axis=1)
    pole = q.groupby(["season", "round"])["best_quali_time"].transform("min")
    q["quali_gap_to_pole"] = q["best_quali_time"] - pole

    teammate = q.groupby(["season", "round", "constructor_id"])["best_quali_time"].transform("min")
    q["quali_gap_to_teammate"] = q["best_quali_time"] - teammate

    df = df.merge(
        q[[
            "season", "round", "driver_id",
            "quali_pos", "quali_gap_to_pole", "quali_gap_to_teammate",
        ]],
        on=["season", "round", "driver_id"], how="left",
    )
    df["grid_position"] = df["quali_pos"].fillna(df["grid"])
    return df


def _merge_standings(df: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    if s.empty:
        df["championship_position"] = np.nan
        df["season_points_so_far"] = 0.0
        return df
    s = s.copy()
    s["round_next"] = s["round"] + 1  # standings AFTER round N → applies BEFORE round N+1
    df = df.merge(
        s[["season", "round_next", "driver_id", "championship_position", "championship_points"]]
        .rename(columns={"round_next": "round", "championship_points": "season_points_so_far"}),
        on=["season", "round", "driver_id"], how="left",
    )
    df["championship_position"] = df["championship_position"].fillna(20)
    df["season_points_so_far"] = df["season_points_so_far"].fillna(0.0)
    return df


def _merge_weather(df: pd.DataFrame, w: pd.DataFrame | None) -> pd.DataFrame:
    if w is None or w.empty:
        df["weather_is_wet"] = 0
        df["track_temp_c"] = np.nan
        df["air_temp_c"] = np.nan
        return df
    return df.merge(w, on=["season", "round"], how="left").assign(
        weather_is_wet=lambda d: d["weather_is_wet"].fillna(0).astype(int)
    )


def _merge_fp_pace(
    df: pd.DataFrame,
    fp: pd.DataFrame | None,
    driver_code_map: dict[str, str] | None,
) -> pd.DataFrame:
    if fp is None or fp.empty:
        df["fp_avg_lap_rank"] = np.nan
        df["fp_long_run_pace_rank"] = np.nan
        return df
    fp = fp.copy()
    if driver_code_map:
        fp["driver_id"] = fp["driver_code"].str.upper().map(driver_code_map)
    else:
        fp["driver_id"] = fp["driver_code"].str.lower()
    return df.merge(
        fp[["season", "round", "driver_id", "fp_avg_lap_rank", "fp_long_run_pace_rank"]],
        on=["season", "round", "driver_id"], how="left",
    )


def _add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("driver_id")
    for n in (3, 5, 10):
        df[f"driver_form_{n}"] = g["points"].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean()
        )
    df["driver_avg_grid_3"] = g["grid_position"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["driver_avg_finish_3"] = g["position"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["dnf_rate_10"] = g["is_dnf"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    constructor_pts = (
        df.groupby(["season", "round", "constructor_id"])["points"]
        .sum().reset_index(name="constructor_pts")
        .sort_values(["season", "round"])
    )
    for n in (3, 5):
        constructor_pts[f"constructor_form_{n}"] = (
            constructor_pts.groupby("constructor_id")["constructor_pts"]
            .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        )
    df = df.merge(
        constructor_pts.drop(columns="constructor_pts"),
        on=["season", "round", "constructor_id"], how="left",
    )
    return df


def _add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["driver_id", "circuit_id"])
    df["circuit_avg_finish"] = g["position"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["circuit_best_finish"] = g["position"].transform(
        lambda x: x.shift(1).expanding().min()
    )
    df["circuit_wins_here"] = g["is_winner"].transform(
        lambda x: x.shift(1).expanding().sum().fillna(0)
    )
    return df


def _add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["driver_id", "season"])
    df["season_wins_so_far"] = g["is_winner"].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    df["season_podiums_so_far"] = g["is_podium"].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    def races_since(series: pd.Series) -> pd.Series:
        out, count = [], 0
        for v in series:
            out.append(count)
            count = 0 if v == 1 else count + 1
        return pd.Series(out, index=series.index)

    df["races_since_win"] = df.groupby("driver_id")["is_winner"].transform(races_since)
    df["races_since_podium"] = df.groupby("driver_id")["is_podium"].transform(races_since)
    return df


def _add_teammate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative head-to-head record vs current teammate (qualifying + race).

    History is stored from pair_key[0]'s perspective (pair_key is sorted).
    Each driver reads the history from their own perspective to avoid look-ahead
    and double-append bugs: outcomes are written once per pair after both drivers
    in a round have had their history values read.
    """
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    teammate_quali = [np.nan] * len(df)
    teammate_race = [np.nan] * len(df)

    # History stored as: fraction of rounds won by pair_key[0]
    pair_quali: dict[tuple, list[int]] = {}
    pair_race: dict[tuple, list[int]] = {}

    for (_, _round), group in df.groupby(["season", "round"]):
        # --- Phase 1: read history for every driver in this round ---
        for idx, row in group.iterrows():
            teammates = group[
                (group["constructor_id"] == row["constructor_id"])
                & (group["driver_id"] != row["driver_id"])
            ]
            if teammates.empty:
                continue
            teammate_id = teammates.iloc[0]["driver_id"]
            pair_key = tuple(sorted([row["driver_id"], teammate_id]))

            history_q = pair_quali.get(pair_key, [])
            if history_q:
                mean_q = float(np.mean(history_q))
                # Flip perspective for pair_key[1] so each driver sees their own win rate
                teammate_quali[idx] = mean_q if row["driver_id"] == pair_key[0] else 1.0 - mean_q

            history_r = pair_race.get(pair_key, [])
            if history_r:
                mean_r = float(np.mean(history_r))
                teammate_race[idx] = mean_r if row["driver_id"] == pair_key[0] else 1.0 - mean_r

        # --- Phase 2: write this round's outcome once per pair ---
        processed: set[tuple] = set()
        for _, row in group.iterrows():
            teammates = group[
                (group["constructor_id"] == row["constructor_id"])
                & (group["driver_id"] != row["driver_id"])
            ]
            if teammates.empty:
                continue
            teammate_id = teammates.iloc[0]["driver_id"]
            pair_key = tuple(sorted([row["driver_id"], teammate_id]))
            if pair_key in processed:
                continue
            processed.add(pair_key)

            drv0, drv1 = pair_key
            row0 = group[group["driver_id"] == drv0]
            row1 = group[group["driver_id"] == drv1]
            if row0.empty or row1.empty:
                continue

            q0 = row0.iloc[0].get("quali_pos")
            q1 = row1.iloc[0].get("quali_pos")
            if pd.notna(q0) and pd.notna(q1):
                pair_quali.setdefault(pair_key, []).append(int(q0 < q1))

            p0 = row0.iloc[0].get("position")
            p1 = row1.iloc[0].get("position")
            if pd.notna(p0) and pd.notna(p1):
                pair_race.setdefault(pair_key, []).append(int(p0 < p1))

    df["teammate_h2h_quali"] = teammate_quali
    df["teammate_h2h_race"] = teammate_race
    return df


def _add_experience_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["experience_races"] = df.groupby("driver_id").cumcount()
    return df


def _add_home_race_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "country" not in df.columns:
        df["is_home_race"] = 0
        return df

    def home(row) -> int:
        nat = row.get("driver_nationality", "")
        target = HOME_RACE_BY_NATIONALITY.get(nat)
        if not target:
            return 0
        return int(row.get("country", "") == target)

    df["is_home_race"] = df.apply(home, axis=1)
    return df
