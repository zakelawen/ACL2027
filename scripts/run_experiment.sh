#!/usr/bin/env bash
# One-command formal run: prepare → generate (3 models) → judge → score.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-/home/qluai/miniconda3/envs/ACL2027-sglang/bin/python}"
GEN_HOST="${HOST:-127.0.0.1}"
GEN_PORT="${PORT:-8000}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"
JUDGE_PORT="${JUDGE_PORT:-18000}"
GEN_GPU="${GEN_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-0,1}"
WAIT_SECS="${WAIT_SECS:-600}"
RUN_NAME="${RUN_NAME:-}"

MODELS=(qwen2.5-7b llama3.1-8b mistral-7b)
DO_PREPARE=1
DO_GENERATE=1
DO_JUDGE=1
DO_SCORE=1

usage() {
  cat <<'EOF'
Usage: bash scripts/run_experiment.sh [options]

  --skip-prepare     Do not download/normalize data
  --generate-only    Only start generators and produce answers
  --judge-only       Only start the Judge and score existing generations
  --score-only       Only compute metrics (no servers)
  --models LIST      Comma-separated generators (default: all three)
  --run-name NAME    Passed through as --run-name
  -h, --help

Environment: GEN_GPU (default 0), JUDGE_GPU (default 0,1),
WAIT_SECS (default 600), PYTHON, HOST, PORT, JUDGE_HOST, JUDGE_PORT.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-prepare) DO_PREPARE=0; shift ;;
    --generate-only) DO_JUDGE=0; DO_SCORE=0; shift ;;
    --judge-only) DO_PREPARE=0; DO_GENERATE=0; shift ;;
    --score-only) DO_PREPARE=0; DO_GENERATE=0; DO_JUDGE=0; shift ;;
    --models)
      IFS=',' read -r -a MODELS <<< "${2:?}"
      shift 2
      ;;
    --run-name) RUN_NAME="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found: ${PYTHON}" >&2
  exit 1
fi

CLI=( "${PYTHON}" -B -m clapnq_eval )
if [[ -n "${RUN_NAME}" ]]; then
  CLI+=( --run-name "${RUN_NAME}" )
fi
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${PROJECT_DIR}/runs/${RUN_NAME:-main}/logs"
mkdir -p "${LOG_DIR}"

GEN_PID=""
JUDGE_PID=""

stop_pid() {
  local pid="${1:-}"
  local name="${2:-process}"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  echo "Stopping ${name} (pid ${pid})"
  kill "${pid}" 2>/dev/null || true
  local i
  for i in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Force-killing ${name} (pid ${pid})"
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_pid "${GEN_PID}" "generator"
  stop_pid "${JUDGE_PID}" "judge"
}
trap cleanup EXIT INT TERM

wait_http() {
  local url="$1"
  local name="$2"
  local deadline=$((SECONDS + WAIT_SECS))
  echo "Waiting for ${name} at ${url}"
  while (( SECONDS < deadline )); do
    if curl -sf --max-time 2 "${url}" >/dev/null; then
      echo "${name} is up"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${name} (${WAIT_SECS}s): ${url}" >&2
  return 1
}

if [[ "${DO_PREPARE}" == "1" ]]; then
  echo "=== prepare ==="
  "${CLI[@]}" prepare
fi

if [[ "${DO_GENERATE}" == "1" ]]; then
  for model in "${MODELS[@]}"; do
    echo "=== generate ${model} ==="
    CUDA_VISIBLE_DEVICES="${GEN_GPU}" \
      bash "${PROJECT_DIR}/scripts/serve_generator.sh" "${model}" \
      > "${LOG_DIR}/generator.${model}.log" 2>&1 &
    GEN_PID=$!
    wait_http "http://${GEN_HOST}:${GEN_PORT}/v1/models" "generator ${model}"
    "${CLI[@]}" generate --model "${model}"
    stop_pid "${GEN_PID}" "generator ${model}"
    GEN_PID=""
    sleep 3
  done
fi

if [[ "${DO_JUDGE}" == "1" ]]; then
  echo "=== judge ==="
  CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" \
    bash "${PROJECT_DIR}/scripts/serve_judge.sh" \
    > "${LOG_DIR}/judge.log" 2>&1 &
  JUDGE_PID=$!
  wait_http "http://${JUDGE_HOST}:${JUDGE_PORT}/v1/models" "judge"
  "${CLI[@]}" judge --model all
  stop_pid "${JUDGE_PID}" "judge"
  JUDGE_PID=""
  sleep 3
fi

if [[ "${DO_SCORE}" == "1" ]]; then
  echo "=== score ==="
  "${CLI[@]}" score
fi

echo "=== status ==="
"${CLI[@]}" status
echo "Done. Metrics under runs/${RUN_NAME:-main}/metrics/"
