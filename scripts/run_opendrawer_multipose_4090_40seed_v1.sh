#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROBOCASA_WORK_ROOT:-$REPO_ROOT}
OPENPI_DIR=${OPENPI_DIR:-$REPO_ROOT/third_party/openpi}
ROBOCASA_DIR=${ROBOCASA_DIR:-$REPO_ROOT/third_party/robocasa}
ROBOSUITE_DIR=${ROBOSUITE_DIR:-$REPO_ROOT/third_party/robosuite}
PY=${PYTHON_BIN:-$(command -v python)}
CHECKPOINT=${PI05_CHECKPOINT:-$ROOT/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000}
OUT=${OUT:-"$ROOT/results/OpenDrawer_40seed_5pose_5trial_4090_v1"}
SESSION=${SESSION:-opendrawer_40seed_4090_v1}
PORT=${PORT:-8000}
mkdir -p "$OUT/logs"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session exists: $SESSION" >&2; exit 2; }
if ss -ltn | awk '{print $4}' | grep -qE ":${PORT}$"; then echo "port $PORT is occupied" >&2; exit 3; fi
test -x "$PY" || { echo "Python executable not found: $PY" >&2; exit 4; }
test -d "$OPENPI_DIR" || { echo "OpenPI submodule missing; run git submodule update --init --recursive" >&2; exit 5; }
test -f "$CHECKPOINT/_CHECKPOINT_METADATA" || { echo "Checkpoint missing: $CHECKPOINT" >&2; exit 6; }
ENV_CMD="export CUDA_VISIBLE_DEVICES=0; export XLA_PYTHON_CLIENT_PREALLOCATE=false; export MUJOCO_GL=osmesa; export PYOPENGL_PLATFORM=osmesa; unset MUJOCO_EGL_DEVICE_ID; export ROBOCASA_WORK_ROOT=$ROOT; export OPENPI_DIR=$OPENPI_DIR; export ROBOCASA_DIR=$ROBOCASA_DIR; export ROBOSUITE_DIR=$ROBOSUITE_DIR; export PI05_CHECKPOINT=$CHECKPOINT; export PYTHONPATH=$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src:$ROBOCASA_DIR:$ROBOSUITE_DIR;"
SERVER="$ENV_CMD cd $OPENPI_DIR; $PY scripts/serve_policy.py --port=$PORT policy:checkpoint --policy.config=pi05_pretrain_human300_inference --policy.dir=$CHECKPOINT 2>&1 | tee $OUT/logs/server.log"
CLIENT="$ENV_CMD $REPO_ROOT/scripts/wait_for_policy_server.sh 127.0.0.1 $PORT $OUT/logs/server.log 300; cd $REPO_ROOT; $PY -u $REPO_ROOT/scripts/collect_opendrawer_multipose_v1.py --seeds 0-39 --worker 0 --trials-per-pose 5 --host 127.0.0.1 --port $PORT --horizon 330 --resume --out-root '$OUT' 2>&1 | tee $OUT/logs/client.log"
tmux new-session -d -s "$SESSION" -n server "$SERVER"
tmux new-window -t "$SESSION" -n collect "$CLIENT"
echo "Started: $SESSION"
echo "Monitor: tmux attach -t $SESSION"
echo "Output: $OUT"
