#!/usr/bin/env bash
# Every pretrained model in configs/infer/, one after the other (one heavy job at a time). Each run is resumable, so a rerun continues.
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in configs/infer/*.yml; do
  uv run python -m src.infer --config "$cfg"
done
