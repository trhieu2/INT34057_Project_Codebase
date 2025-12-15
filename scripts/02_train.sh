#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m cafa6ml.cli --config configs/base.yaml train
