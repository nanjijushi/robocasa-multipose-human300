#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the collected OpenDrawer result directory}"
WORK_ROOT="${WORK_ROOT:-$REPO/outputs/opendrawer_multipose_adapter_v1}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-$REPO/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000}"

mkdir -p "$WORK_ROOT"

"$PYTHON_BIN" "$REPO/adapter/prepare_opendrawer_multipose_adapter_split_v1.py" \
  --data-root "$DATA_ROOT" --output "$WORK_ROOT/split"

if [[ ! -f "$WORK_ROOT/cache/opendrawer_multipose_features.npz" ]]; then
  "$PYTHON_BIN" "$REPO/adapter/cache_opendrawer_multipose_features_v1.py" \
    --manifest-dir "$WORK_ROOT/split" --output "$WORK_ROOT/cache" \
    --checkpoint "$PI05_CHECKPOINT"
fi

"$PYTHON_BIN" "$REPO/adapter/train_opendrawer_multipose_adapter_v1.py" \
  --dataset "$WORK_ROOT/cache/opendrawer_multipose_features.npz" \
  --output "$WORK_ROOT/training" "$@"
