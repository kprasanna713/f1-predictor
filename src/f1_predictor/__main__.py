"""CLI entrypoint: `python -m f1_predictor <command>`."""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F1 race winner predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Pull historical data from Ergast + FastF1")
    sub.add_parser("train", help="Train the XGBoost models")

    p_predict = sub.add_parser("predict", help="Predict next race")
    p_predict.add_argument("--season", type=int, default=None)
    p_predict.add_argument("--round", type=int, default=None)

    sub.add_parser("compare", help="Compare past predictions to actual results")
    sub.add_parser("weekly", help="Run the weekly cycle (compare → retrain → predict)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "ingest":
        from f1_predictor.config import TRAIN_SEASONS
        from f1_predictor.ingest import ingest_seasons
        ingest_seasons(TRAIN_SEASONS)
    elif args.cmd == "train":
        from f1_predictor.train import train
        print(json.dumps(train(refresh=False), indent=2))
    elif args.cmd == "predict":
        from f1_predictor.config import PREDICT_SEASON
        from f1_predictor.predict import predict_next_race, print_top10
        payload = predict_next_race(args.season or PREDICT_SEASON, args.round)
        print_top10(payload)
    elif args.cmd == "compare":
        from f1_predictor.compare import compare_all, summary
        lb = compare_all()
        if lb.empty:
            print("No predictions on disk yet.")
        else:
            print(lb.to_string(index=False))
            print("\nSummary:", json.dumps(summary(lb), indent=2))
    elif args.cmd == "weekly":
        from f1_predictor.weekly import run
        run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
