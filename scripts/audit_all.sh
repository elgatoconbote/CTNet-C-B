#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/test_ctnet_invariants.py
python ctnet_functional_benchmark.py --profile smoke --steps 3 --batch 2
