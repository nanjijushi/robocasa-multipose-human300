#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
MODE="${1:-}"
case "$MODE" in
  lora) SCRIPT=demo_pure_mobility_lora_seed25_v1.py ;;
  external) SCRIPT=demo_joint_offline_value_reposition_v1.py ;;
  *) echo 'Usage: bash comparison_models/run_client.sh lora|external [client options]' >&2; exit 2 ;;
esac
shift
# Identical placement defaults for both methods. Override identically when comparing.
exec "$PYTHON_BIN" -u "$COMPARISON_DIR/src/$SCRIPT" \
  --seed 25 --placement-seed 2551 --host 127.0.0.1 --port 8001 \
  --allow-policy-base-motion --attempt-max-steps 600 \
  --start-distance-min 0.55 --start-distance-max 1.05 \
  --start-bearing-deg 40 --side-min-bearing-deg 20 --center-bearing-deg 8 \
  --bearing-bins 5 --distance-bins 4 --start-yaw-noise-deg 10 \
  --visibility-margin-px 8 --min-drawer-visible-px 80 --placement-max-tries 240 \
  --out-root "$REPO_ROOT/outputs/mobility_comparison/$MODE" "$@"
