#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/wangxinhao/envs/robocasa/bin/python" ]]; then
    PYTHON_BIN="/wangxinhao/envs/robocasa/bin/python"
  elif [[ -x "/opt/conda/envs/robocasa/bin/python" ]]; then
    PYTHON_BIN="/opt/conda/envs/robocasa/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the collected OpenDrawer result directory}"
WORK_ROOT="${WORK_ROOT:-$REPO/outputs/opendrawer_multipose_adapter_v1}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-$REPO/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO/.cache/openpi}"

mkdir -p "$WORK_ROOT"

"$PYTHON_BIN" "$REPO/adapter/prepare_opendrawer_multipose_adapter_split_v1.py" \
  --data-root "$DATA_ROOT" --output "$WORK_ROOT/split"

"$PYTHON_BIN" "$REPO/adapter/cache_opendrawer_multipose_features_v1.py" \
  --manifest-dir "$WORK_ROOT/split" --output "$WORK_ROOT/cache" \
  --checkpoint "$PI05_CHECKPOINT" --resume

"$PYTHON_BIN" "$REPO/adapter/train_opendrawer_multipose_adapter_v1.py" \
  --dataset "$WORK_ROOT/cache/opendrawer_multipose_features.npz" \
  --output "$WORK_ROOT/training" "$@"
