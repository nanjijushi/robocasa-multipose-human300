#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/gpufree-data
PY=/opt/conda/envs/robocasa/bin/python
OUT=${OUT:-"$ROOT/实验结果/20261020_OpenDrawer_40seed_每seed5位置_每位置5次_4090_v1"}
SESSION=${SESSION:-opendrawer_40seed_4090_v1}
PORT=${PORT:-8000}
mkdir -p "$OUT/logs"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session exists: $SESSION" >&2; exit 2; }
if ss -ltn | awk '{print $4}' | grep -qE ":${PORT}$"; then echo "port $PORT is occupied" >&2; exit 3; fi
ENV_CMD="export CUDA_VISIBLE_DEVICES=0; export XLA_PYTHON_CLIENT_PREALLOCATE=false; export MUJOCO_GL=osmesa; export PYOPENGL_PLATFORM=osmesa; unset MUJOCO_EGL_DEVICE_ID; export PYTHONPATH=$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$ROOT/robocasa:$ROOT/robosuite;"
SERVER="$ENV_CMD cd $ROOT/openpi; $PY scripts/serve_policy.py --port=$PORT policy:checkpoint --policy.config=pi05_pretrain_human300_inference --policy.dir=$ROOT/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000 2>&1 | tee $OUT/logs/server.log"
CLIENT="$ENV_CMD until grep -q 'Websocket server started' $OUT/logs/server.log 2>/dev/null; do sleep 2; done; cd $ROOT; $PY -u $ROOT/view_discriminator_pilot/collect_opendrawer_multipose_v1.py --seeds 0-39 --worker 0 --trials-per-pose 5 --host 127.0.0.1 --port $PORT --horizon 330 --resume --out-root '$OUT' 2>&1 | tee $OUT/logs/client.log"
tmux new-session -d -s "$SESSION" -n server "$SERVER"
tmux new-window -t "$SESSION" -n collect "$CLIENT"
echo "Started: $SESSION"
echo "Monitor: tmux attach -t $SESSION"
echo "Output: $OUT"
