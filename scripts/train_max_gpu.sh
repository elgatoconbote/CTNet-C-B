#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python ctnet_run_max.py train \
  --profile max \
  --synthetic \
  --steps "${CTNET_STEPS:-100000}" \
  --log-every "${CTNET_LOG_EVERY:-50}" \
  --cuda \
  --coherence-grad-scale \
  --save-final checkpoints/ctnet_max_final.pt
