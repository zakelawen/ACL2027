#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:-}"
if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: $0 {qwen2.5-7b|llama3.1-8b|mistral-7b}" >&2
  exit 2
fi

case "${MODEL_KEY}" in
  qwen2.5-7b)
    MODEL_PATH="/mnt/model/Qwen2.5-7B-Instruct"
    SERVED_MODEL="qwen2.5-7b"
    ;;
  llama3.1-8b)
    MODEL_PATH="/mnt/model/Meta-Llama-3.1-8B-Instruct"
    SERVED_MODEL="llama3.1-8b"
    ;;
  mistral-7b)
    MODEL_PATH="/mnt/model/Mistral-7B-Instruct-v0.3"
    SERVED_MODEL="mistral-7b"
    ;;
  *)
    echo "Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

VLLM_ENV="/home/qluai/miniconda3/envs/ACL2027-vllm"
VLLM_BIN="${VLLM_ENV}/bin/vllm"
if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vLLM executable not found: ${VLLM_BIN}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

export PATH="${VLLM_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host "${BIND_HOST:-127.0.0.1}" \
  --port "${BIND_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --generation-config vllm \
  --seed "${SEED:-20260819}"
