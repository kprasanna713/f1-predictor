"""FastAPI app that serves the prediction UI + JSON API."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from f1_predictor.compare import compare_all, summary
from f1_predictor.config import MODEL_PATH, PREDICTIONS_DIR
from f1_predictor.predict import predict_next_race

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="F1 Race Predictor", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_trained": MODEL_PATH.exists(),
    }


@app.get("/api/predict")
def api_predict(season: int | None = None, round: int | None = None) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m f1_predictor train` first.",
        )
    try:
        from f1_predictor.config import PREDICT_SEASON
        return predict_next_race(season or PREDICT_SEASON, round)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/leaderboard")
def api_leaderboard() -> dict:
    lb = compare_all()
    if lb.empty:
        return {"summary": {"resolved_races": 0}, "races": []}
    resolved = lb[lb["status"] == "resolved"]
    return {
        "summary": summary(lb),
        "races": resolved.to_dict(orient="records") if not resolved.empty else [],
    }


@app.get("/api/predictions")
def api_predictions() -> dict:
    """List every saved prediction file."""
    files = sorted(PREDICTIONS_DIR.glob("*.json"))
    return {
        "count": len(files),
        "items": [
            {
                "file": f.name,
                **{k: v for k, v in json.loads(f.read_text()).items() if k != "predictions"},
            }
            for f in files
        ],
    }


def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    import uvicorn
    uvicorn.run("f1_predictor.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
