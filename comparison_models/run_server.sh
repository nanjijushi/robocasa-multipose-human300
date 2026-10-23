#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
MODE="${1:-}"
if [[ "$MODE" != lora && "$MODE" != external ]]; then
  echo 'Usage: bash comparison_models/run_server.sh lora|external [server options]' >&2
  exit 2
fi
shift
"$PYTHON_BIN" "$COMPARISON_DIR/download_weights.py" --output "$WEIGHTS_DIR" --verify-only
COMMON=(--stage full --host 0.0.0.0 --port 8001 --base-checkpoint "$PI05_CHECKPOINT" --adapter "$WEIGHTS_DIR/pure_mobility_lora_v2.npz" --num-steps 5)
if [[ "$MODE" == lora ]]; then
  exec "$PYTHON_BIN" -u "$COMPARISON_DIR/src/serve_seed17_drawer_stage12.py" "${COMMON[@]}" \
    --mobility-reset-idle-seconds 30 --stop-command-threshold 0.001 \
    --motion-start-threshold 0.18 --maximum-mobility-inferences 0 \
    --base-translation-deadband 0.08 --base-yaw-deadband 0.075 \
    --decision-action-window 4 \
    --movement-prompt 'Move the mobile base to a suitable viewpoint for grasping the drawer handle.' "$@"
else
  exec "$PYTHON_BIN" -u "$COMPARISON_DIR/src/serve_joint_offline_value_reposition_v1.py" "${COMMON[@]}" \
    --view-quality-checkpoint "$WEIGHTS_DIR/external_joint_reposition_v1.pt" "$@"
fi
