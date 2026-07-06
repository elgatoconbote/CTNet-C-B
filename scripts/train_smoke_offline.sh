#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python ctnet_run_max.py train --profile smoke --synthetic --steps 10 --log-every 1 --save-final checkpoints/ctnet_smoke_final.pt
