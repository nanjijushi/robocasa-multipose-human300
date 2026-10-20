#!/usr/bin/env bash
set -euo pipefail

HOST=${1:-127.0.0.1}
PORT=${2:-8000}
LOG=${3:-/dev/null}
TIMEOUT=${4:-300}
DEADLINE=$((SECONDS + TIMEOUT))

while ! curl --silent --show-error --fail --max-time 2 "http://$HOST:$PORT/healthz" >/dev/null 2>&1; do
  if [[ -f "$LOG" ]] && grep -qE "Traceback|ValueError:|RuntimeError:|CUDA_ERROR|RESOURCE_EXHAUSTED" "$LOG"; then
    echo "Policy server exited or reported an error:" >&2
    tail -n 80 "$LOG" >&2
    exit 1
  fi
  if (( SECONDS >= DEADLINE )); then
    echo "Timed out after ${TIMEOUT}s waiting for http://$HOST:$PORT/healthz" >&2
    [[ -f "$LOG" ]] && tail -n 80 "$LOG" >&2
    exit 1
  fi
  sleep 2
done

echo "Policy server is healthy at $HOST:$PORT"
