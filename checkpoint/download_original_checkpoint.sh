#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DEST_ROOT="${1:-$REPO_ROOT/robocasa365_checkpoints}"
REPO="robocasa/robocasa365_checkpoints"
REVISION="c484448aba1a9b60a04c9b0ca117241518ea69f3"
PREFIX="pi05_pretrain_human300/multitask_learning/75000/**"

mkdir -p "$DEST_ROOT"

if command -v hf >/dev/null 2>&1; then
  HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=huggingface-cli
else
  echo "Missing Hugging Face CLI. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

"$HF" download "$REPO" \
  --revision "$REVISION" \
  --include "$PREFIX" \
  --local-dir "$DEST_ROOT"

CHECKPOINT="$DEST_ROOT/pi05_pretrain_human300/multitask_learning/75000"
test -f "$CHECKPOINT/_CHECKPOINT_METADATA"
test -f "$CHECKPOINT/assets/norm_stats.json"
test -f "$CHECKPOINT/params/_METADATA"
test -f "$CHECKPOINT/train_state/_METADATA"
echo "Checkpoint ready: $CHECKPOINT"

