#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON_BIN:-$(command -v python)}

cd "$REPO_ROOT"
git submodule update --init --recursive

"$PY" -m pip install -e third_party/robosuite
"$PY" -m pip install -e third_party/robocasa
"$PY" -m pip install -e third_party/openpi
"$PY" -m pip install -e third_party/openpi/packages/openpi-client
"$PY" -m pip install pillow numpy imageio imageio-ffmpeg gymnasium==0.29.1

PATCH=patches/openpi_robocasa_multipose.patch
if git -C third_party/openpi apply --reverse --check "$REPO_ROOT/$PATCH" >/dev/null 2>&1; then
  echo "OpenPI compatibility patch is already applied."
else
  git -C third_party/openpi apply --check "$REPO_ROOT/$PATCH"
  git -C third_party/openpi apply "$REPO_ROOT/$PATCH"
  echo "Applied OpenPI compatibility patch."
fi

"$PY" -m robocasa.scripts.setup_macros
echo "Environment installed. Download assets with:"
echo "  python -m robocasa.scripts.download_kitchen_assets"
