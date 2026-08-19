#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec /home/qluai/miniconda3/envs/ACL2027-sglang/bin/python -B \
  -m unittest discover -s "${PROJECT_DIR}/tests" -v
