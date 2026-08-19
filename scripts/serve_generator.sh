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

# Isolate JIT/link from the caller conda env. run_experiment.sh is launched
# from ACL2027-sglang, whose CC/CXX would compile FlashInfer against the
# wrong libcuda (-lcuda not found).
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
VLLM_CC="${VLLM_ENV}/bin/x86_64-conda-linux-gnu-cc"
VLLM_CXX="${VLLM_ENV}/bin/x86_64-conda-linux-gnu-c++"
if [[ ! -x "${VLLM_CC}" || ! -x "${VLLM_CXX}" ]]; then
  echo "vLLM compilers not found under ${VLLM_ENV}/bin" >&2
  exit 1
fi
export CC="${VLLM_CC}"
export CXX="${VLLM_CXX}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${HOME}/.cache/flashinfer-workspaces/ACL2027-vllm}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host "${BIND_HOST:-127.0.0.1}" \
  --port "${BIND_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --generation-config vllm \
  --seed "${SEED:-20260819}"
