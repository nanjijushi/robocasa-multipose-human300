#!/usr/bin/env bash
# Source from launchers; installed repository dependencies are still required.
set -euo pipefail
COMPARISON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$COMPARISON_DIR/.." && pwd)"
export ROBOCASA_DATA_ROOT="${ROBOCASA_DATA_ROOT:-$REPO_ROOT}"
export OPENPI_ROOT="${OPENPI_ROOT:-$REPO_ROOT/third_party/openpi}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$COMPARISON_DIR/src:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$REPO_ROOT/third_party/robocasa:$REPO_ROOT/third_party/robosuite${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/.cache/openpi}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
NVIDIA_LIBS="$("$PYTHON_BIN" -c 'import site,pathlib; print(":".join(str(p) for s in site.getsitepackages() for p in sorted(pathlib.Path(s).glob("nvidia/*/lib"))))')"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COMPARISON_DIR/weights}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-$ROBOCASA_DATA_ROOT/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
