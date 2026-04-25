"""Weekly entry point: reconcile previous predictions, retrain, predict next race."""
from __future__ import annotations

import json
import logging

from f1_predictor.compare import compare_all, summary
from f1_predictor.predict import predict_next_race, print_top10
from f1_predictor.train import train

logger = logging.getLogger(__name__)


def run() -> None:
    logger.info("=== Step 1/3: comparing past predictions to actual results ===")
    lb = compare_all()
    if not lb.empty:
        print(lb.to_string(index=False))
        print("\nLeaderboard summary:", json.dumps(summary(lb), indent=2))

    logger.info("=== Step 2/3: retraining on fresh data ===")
    metrics = train(refresh=True)
    print("\nTraining metrics:", json.dumps(metrics, indent=2))

    logger.info("=== Step 3/3: predicting next race ===")
    payload = predict_next_race()
    print_top10(payload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
