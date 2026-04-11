#!/usr/bin/env bash
set -euo pipefail

SERVER_PORT="${SERVER_PORT:-8005}"
./stop_vllm_server.sh
exec python vllm_server.py "$@"
