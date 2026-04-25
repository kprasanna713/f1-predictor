#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m f1_predictor ingest
python -m f1_predictor train
python -m f1_predictor predict
python -m f1_predictor compare
