#!/usr/bin/env bash
set -euo pipefail

#-----------------------------------------
# Server port for parallel-safe runs
#-----------------------------------------
SERVER_PORT="${SERVER_PORT:-8005}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
SERVER_START_INTERVAL="${SERVER_START_INTERVAL:-5}"
SERVER_LOG="${SERVER_LOG:-logs/vllm_server_${SERVER_PORT}.log}"

# Default to a safe attention backend for older GPUs unless overridden.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

#-----------------------------------------
# Graceful shutdown on Ctrl-C / SIGTERM
#-----------------------------------------
cleanup() {
  echo "[start_rl_training.sh] Caught signal, stopping VLLM server..."
  ./stop_vllm_server.sh || true
  exit 130
}
trap cleanup INT TERM

#-----------------------------------------
# Buckets for the three kinds of flags
#-----------------------------------------
ACCELERATE_ARGS=()
SERVER_ARGS=()
TRAIN_ARGS=()

#-----------------------------------------
# Parse CLI
#-----------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    # -------------------------------
    # Flags that belong to Accelerate
    # -------------------------------
    --config_file|--num_processes|--num_machines|--machine_rank|\
    --main_process_ip|--main_process_port)
        ACCELERATE_ARGS+=("$1" "$2")
        shift 2
        ;;

    # ---------------------------------------------------------
    # The model-specific YAML should go to both buckets
    # ---------------------------------------------------------
    --config-name)
        SERVER_ARGS+=("$1" "$2")      # for start_vllm_server.sh
        TRAIN_ARGS+=("$1" "$2")       # for train_rl.py via accelerate
        shift 2
        ;;

    # ---------------------------------
    # Separator: everything after " -- "
    # goes only to TRAIN_ARGS
    # ---------------------------------
    --)
        shift
        TRAIN_ARGS+=("$@")
        break
        ;;

    # ---------------------------------
    # Anything else is a server flag
    # ---------------------------------
    *)
        SERVER_ARGS+=("$1")
        shift
        ;;
  esac
done

#-----------------------------------------
# Make sure we at least got a config file
#-----------------------------------------
if ! printf '%s\n' "${ACCELERATE_ARGS[@]}" | grep -q -- '--config_file'; then
  echo "[start_rl_training.sh] ERROR: --config_file is required." >&2
  exit 1
fi

#-----------------------------------------
# Start the VLLM server
#-----------------------------------------
echo "[start_rl_training.sh] Launching VLLM server..."
./stop_vllm_server.sh || true
sleep 2
mkdir -p "$(dirname "${SERVER_LOG}")"
./start_vllm_server.sh "${SERVER_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > ".vllm_${SERVER_PORT}.pid"

#-----------------------------------------
# Wait until the server responds
#-----------------------------------------
start_ts="$(date +%s)"
until curl -fsS "http://localhost:${SERVER_PORT}/docs" >/dev/null ; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[start_rl_training.sh] ERROR: VLLM server process exited before it became ready." >&2
    echo "[start_rl_training.sh] See ${SERVER_LOG} for details." >&2
    exit 1
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts > SERVER_START_TIMEOUT )); then
    echo "[start_rl_training.sh] ERROR: Timed out waiting for VLLM server on port ${SERVER_PORT}." >&2
    exit 1
  fi
  echo "[start_rl_training.sh] Waiting for VLLM server..."
  sleep "${SERVER_START_INTERVAL}"
done
echo "[start_rl_training.sh] VLLM server is up."

#-----------------------------------------
# Run training
#-----------------------------------------
ACCEL_CMD=(accelerate launch "${ACCELERATE_ARGS[@]}" train_rl.py "${TRAIN_ARGS[@]}")
echo "[start_rl_training.sh] About to run:"
printf '  %q ' "${ACCEL_CMD[@]}"
echo    

"${ACCEL_CMD[@]}"

#-----------------------------------------
# Cleanup
#-----------------------------------------
./stop_vllm_server.sh
echo "[start_rl_training.sh] Done."
