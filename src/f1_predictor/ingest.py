"""Pull historical race results / qualifying / standings into local parquet files."""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from f1_predictor.config import RAW_DIR, TRAIN_SEASONS
from f1_predictor.ergast_client import ErgastClient
from f1_predictor.fastf1_client import practice_pace, race_weather

logger = logging.getLogger(__name__)


def ingest_seasons(seasons: Iterable[int], use_fastf1: bool = True) -> dict[str, pd.DataFrame]:
    client = ErgastClient()
    results, quali, standings, schedules, weather, fp = [], [], [], [], [], []

    for season in seasons:
        logger.info("Ingesting season %s", season)
        try:
            results.append(client.race_results(season))
        except Exception as exc:
            logger.warning("results %s failed: %s", season, exc)
        try:
            quali.append(client.qualifying(season))
        except Exception as exc:
            logger.warning("qualifying %s failed: %s", season, exc)
        try:
            standings.append(client.driver_standings(season))
        except Exception as exc:
            logger.warning("standings %s failed: %s", season, exc)
        try:
            sched = client.schedule(season)
            schedules.append(sched)
        except Exception as exc:
            logger.warning("schedule %s failed: %s", season, exc)
            continue

        if not use_fastf1:
            continue

        # FastF1 only goes back to 2018 reliably
        if season < 2018:
            continue
        for rnd in sched["round"].tolist():
            w = race_weather(season, rnd)
            if w:
                weather.append(w)
            fp_df = practice_pace(season, rnd)
            if not fp_df.empty:
                fp.append(fp_df)

    out = {
        "results": _concat(results),
        "qualifying": _concat(quali),
        "standings": _concat(standings),
        "schedule": _concat(schedules),
        "weather": pd.DataFrame(weather) if weather else pd.DataFrame(),
        "fp_pace": _concat(fp),
    }

    for name, df in out.items():
        if df.empty:
            continue
        path = RAW_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Wrote %s rows → %s", len(df), path)
    return out


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_raw() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in ("results", "qualifying", "standings", "schedule", "weather", "fp_pace"):
        path = RAW_DIR / f"{name}.parquet"
        out[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest_seasons(TRAIN_SEASONS)
