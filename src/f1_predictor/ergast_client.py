"""Thin client for the Ergast/Jolpica F1 REST API."""
from __future__ import annotations

import time

import pandas as pd
import requests

from f1_predictor.config import ERGAST_BASE_URL


class ErgastClient:
    def __init__(self, base_url: str = ERGAST_BASE_URL, timeout: int = 30, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{endpoint}"
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, params=params or {}, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"Ergast request failed: {url}") from last_err

    def race_results(self, season: int) -> pd.DataFrame:
        data = self._get(f"{season}/results.json", {"limit": 1000})
        rows = []
        for race in data["MRData"]["RaceTable"]["Races"]:
            for r in race.get("Results", []):
                pos = r.get("position", "")
                rows.append(
                    {
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "race_name": race["raceName"],
                        "circuit_id": race["Circuit"]["circuitId"],
                        "country": race["Circuit"]["Location"]["country"],
                        "date": race.get("date", ""),
                        "driver_id": r["Driver"]["driverId"],
                        "driver_nationality": r["Driver"].get("nationality", ""),
                        "constructor_id": r["Constructor"]["constructorId"],
                        "grid": int(r.get("grid", 0) or 0),
                        "position": int(pos) if pos.isdigit() else None,
                        "points": float(r.get("points", 0) or 0),
                        "laps": int(r.get("laps", 0) or 0),
                        "status": r.get("status", ""),
                        "is_winner": 1 if pos == "1" else 0,
                        "is_podium": 1 if pos in ("1", "2", "3") else 0,
                        "is_dnf": 0 if pos.isdigit() else 1,
                    }
                )
        return pd.DataFrame(rows)

    def qualifying(self, season: int) -> pd.DataFrame:
        data = self._get(f"{season}/qualifying.json", {"limit": 1000})
        rows = []
        for race in data["MRData"]["RaceTable"]["Races"]:
            for q in race.get("QualifyingResults", []):
                rows.append(
                    {
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "circuit_id": race["Circuit"]["circuitId"],
                        "driver_id": q["Driver"]["driverId"],
                        "constructor_id": q["Constructor"]["constructorId"],
                        "quali_pos": int(q["position"]),
                        "q1": _parse_time(q.get("Q1", "")),
                        "q2": _parse_time(q.get("Q2", "")),
                        "q3": _parse_time(q.get("Q3", "")),
                    }
                )
        return pd.DataFrame(rows)

    def driver_standings(self, season: int) -> pd.DataFrame:
        data = self._get(f"{season}/driverStandings.json", {"limit": 100})
        rows = []
        for sl in data["MRData"]["StandingsTable"]["StandingsLists"]:
            rnd = int(sl.get("round", 0))
            for s in sl["DriverStandings"]:
                rows.append(
                    {
                        "season": int(sl["season"]),
                        "round": rnd,
                        "driver_id": s["Driver"]["driverId"],
                        "championship_position": int(s["position"]),
                        "championship_points": float(s["points"]),
                        "season_wins": int(s["wins"]),
                    }
                )
        return pd.DataFrame(rows)

    def schedule(self, season: int) -> pd.DataFrame:
        data = self._get(f"{season}/races.json", {"limit": 30})
        races = data["MRData"]["RaceTable"]["Races"]
        return pd.DataFrame(
            [
                {
                    "season": int(r["season"]),
                    "round": int(r["round"]),
                    "race_name": r["raceName"],
                    "circuit_id": r["Circuit"]["circuitId"],
                    "country": r["Circuit"]["Location"]["country"],
                    "date": r.get("date", ""),
                }
                for r in races
            ]
        )


def _parse_time(value: str) -> float | None:
    """Parse mm:ss.SSS → seconds (float)."""
    if not value:
        return None
    try:
        if ":" in value:
            mins, secs = value.split(":", 1)
            return int(mins) * 60 + float(secs)
        return float(value)
    except ValueError:
        return None
