#!/usr/bin/env bash
set -euo pipefail

SGLANG_ENV="/home/qluai/miniconda3/envs/ACL2027-sglang"
SGLANG_BIN="${SGLANG_ENV}/bin/sglang"
MODEL_PATH="/home/qluai/.cache/modelscope/hub/models/Qwen/Qwen3___8-27B-FP8"

if [[ ! -x "${SGLANG_BIN}" ]]; then
  echo "SGLang executable not found: ${SGLANG_BIN}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Judge model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

export PATH="${SGLANG_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

EXTRA_ARGS=()
# Default on: temperature=0.7 plus dynamic batching is not reproducible
# from seed alone. Set DETERMINISTIC_INFERENCE=0 to opt out.
if [[ "${DETERMINISTIC_INFERENCE:-1}" != "0" ]]; then
  EXTRA_ARGS+=(--enable-deterministic-inference)
fi

exec "${SGLANG_BIN}" serve \
  --model-path "${MODEL_PATH}" \
  --served-model-name qwen3.8-27b-judge \
  --host "${BIND_HOST:-127.0.0.1}" \
  --port "${BIND_PORT:-18000}" \
  --tp-size 2 \
  --language-only \
  --mm-feature-transport cpu \
  --context-length "${CONTEXT_LENGTH:-8192}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.88}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-5}" \
  --grammar-backend xgrammar \
  --reasoning-parser qwen3 \
  --sampling-defaults openai \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --random-seed "${SEED:-20260819}" \
  --enable-p2p-check \
  --disable-custom-all-reduce \
  "${EXTRA_ARGS[@]}"
