"""FastF1 wrapper that pulls weather, practice pace, and tyre data per race."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from f1_predictor.config import CACHE_DIR

logger = logging.getLogger(__name__)

try:
    import fastf1
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    _FASTF1_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning("FastF1 unavailable: %s", exc)
    _FASTF1_AVAILABLE = False


def _safe_load(season: int, rnd: int, session: str):
    if not _FASTF1_AVAILABLE:
        return None
    try:
        s = fastf1.get_session(season, rnd, session)
        s.load(laps=True, telemetry=False, weather=True, messages=False)
        return s
    except Exception as exc:
        logger.info("FastF1 %s %s %s unavailable: %s", season, rnd, session, exc)
        return None


def practice_pace(season: int, rnd: int) -> pd.DataFrame:
    """Average lap-time rank per driver across FP1/FP2/FP3, plus long-run pace.

    Long-run pace = mean of laps slower than the driver's median lap (proxies race stints).
    """
    rows: list[dict] = []
    for sess_name in ("FP1", "FP2", "FP3"):
        sess = _safe_load(season, rnd, sess_name)
        if sess is None:
            continue
        try:
            if sess.laps is None or sess.laps.empty:
                continue
            laps = sess.laps.pick_quicklaps()
        except Exception as exc:
            logger.info("Laps unavailable for %s %s %s: %s", season, rnd, sess_name, exc)
            continue
        if laps.empty:
            continue
        for drv_code, drv_laps in laps.groupby("Driver"):
            lap_secs = drv_laps["LapTime"].dt.total_seconds().dropna()
            if lap_secs.empty:
                continue
            median = lap_secs.median()
            long_run = lap_secs[lap_secs >= median].mean()
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "session": sess_name,
                    "driver_code": drv_code,
                    "fp_avg_lap": lap_secs.mean(),
                    "fp_long_run_pace": long_run,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["season", "round", "driver_code", "fp_avg_lap_rank", "fp_long_run_pace_rank"]
        )

    df = pd.DataFrame(rows)
    agg = df.groupby(["season", "round", "driver_code"], as_index=False).agg(
        fp_avg_lap=("fp_avg_lap", "mean"),
        fp_long_run_pace=("fp_long_run_pace", "mean"),
    )
    agg["fp_avg_lap_rank"] = agg.groupby(["season", "round"])["fp_avg_lap"].rank(method="min")
    agg["fp_long_run_pace_rank"] = agg.groupby(["season", "round"])["fp_long_run_pace"].rank(method="min")
    return agg[["season", "round", "driver_code", "fp_avg_lap_rank", "fp_long_run_pace_rank"]]


def race_weather(season: int, rnd: int) -> Optional[dict]:
    """Pull race-day weather summary from FastF1. Returns None if race hasn't run yet."""
    sess = _safe_load(season, rnd, "R")
    if sess is None:
        return None
    try:
        w = sess.weather_data
        if w is None or w.empty:
            return None
    except Exception as exc:
        logger.info("Weather data unavailable for %s round %s: %s", season, rnd, exc)
        return None
    return {
        "season": season,
        "round": rnd,
        "weather_is_wet": int(w["Rainfall"].any()) if "Rainfall" in w else 0,
        "track_temp_c": float(w["TrackTemp"].mean()) if "TrackTemp" in w else None,
        "air_temp_c": float(w["AirTemp"].mean()) if "AirTemp" in w else None,
    }


def driver_code_to_id_map(season: int, rnd: int) -> dict[str, str]:
    """FastF1 uses 3-letter codes (VER, HAM); Ergast uses driverId (verstappen, hamilton)."""
    sess = _safe_load(season, rnd, "R") or _safe_load(season, rnd, "Q")
    if sess is None:
        return {}
    try:
        results = sess.results
        return {row["Abbreviation"]: row["DriverId"].lower() for _, row in results.iterrows()}
    except Exception:
        return {}
