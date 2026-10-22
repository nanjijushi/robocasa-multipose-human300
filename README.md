# RoboCasa OpenDrawer Multi-Pose Dataset Toolkit

This public repository packages the reproducible OpenDrawer base-pose generator,
the fixed-scene multi-pose evaluator, the RoboCasa/OpenPI source revisions, and the
official original `pi05_pretrain_human300` checkpoint locator.

## Included components

- `scripts/preview_opendrawer_same_scene_multipose_v2.py`: generates five
  collision-checked, target-visible base poses in one fixed RoboCasa scene and
  saves the first-person preview, base joint position, and geometry metadata.
- `scripts/collect_opendrawer_multipose_v1.py`: evaluates each generated pose
  repeatedly while masking the right external camera and records success and
  geometry data.
- `scripts/run_opendrawer_multipose_4090_40seed_v1.sh`: 40 scenes x 5 poses x
  5 trials launcher for one RTX 4090.
- `third_party/robocasa`: RoboCasa pinned at commit
  `b4684e6ee37d377cc392e98302a6b916d588b415`.
- `third_party/openpi`: RoboCasa OpenPI pinned at commit
  `5a6beda9ff99da30b4e1b59320f6a32971d7c397`.
- `third_party/robosuite`: robosuite pinned at the tested server commit
  `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.
- `checkpoint/download_original_checkpoint.sh`: downloads the complete official
  original checkpoint from Hugging Face into the layout used by the scripts.
- `adapter/`: task-conditioned embodied-feasibility Adapter, deterministic
  dataset split, frozen Pi0.5 feature cache, grouped multi-pose trainer, and
  method documentation.

## Checkout

```bash
git clone --recurse-submodules git@github.com:nanjijushi/robocasa-multipose-human300.git
cd robocasa-multipose-human300
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Environment setup

The tested server environment is Ubuntu 22.04, Python 3.11, CUDA with one RTX
4090, RoboCasa 1.0.1, robosuite 1.5.2, Gymnasium 0.29.1, MuJoCo 3.3.1, JAX
0.5.3, and jaxlib 0.5.3. GPU drivers and the CUDA-enabled JAX build must match
the CUDA installation on the target machine.

Create the Python environment, then let the repository install the pinned
source trees and apply the included OpenPI compatibility patch:

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa

python -m pip install --upgrade pip
bash scripts/setup_repository.sh
```

The data-collection utilities additionally use Pillow, NumPy, ImageIO, and
Gymnasium. Install them if they were not pulled in by the upstream packages:

```bash
python -m pip install pillow numpy imageio imageio-ffmpeg gymnasium==0.29.1
```

Initialize RoboCasa's local macros and download the official kitchen assets.
The upstream downloader reports roughly 10 GB; allow additional working and
extraction space. The assets come from NVIDIA's official Hugging Face dataset,
not from this Git repository.

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

Official asset source:
https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF

On Ubuntu, the headless OSMesa renderer may additionally require:

```bash
sudo apt-get update
sudo apt-get install -y libosmesa6 libosmesa6-dev libgl1-mesa-glx libglfw3
```

Headless collection uses software-rendered MuJoCo in the tested launcher:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/third_party/openpi/src:$PWD/third_party/openpi/packages/openpi-client/src:$PWD/third_party/robocasa:$PWD/third_party/robosuite"
```

The launcher computes all project paths from its own repository location. For a
nonstandard layout, override `ROBOCASA_WORK_ROOT`, `OPENPI_DIR`,
`ROBOCASA_DIR`, `ROBOSUITE_DIR`, `PYTHON_BIN`, or `PI05_CHECKPOINT`.

## Download the original checkpoint

The full checkpoint is about 44.7 GB and contains individual files larger than
GitHub's regular Git/LFS limits. It therefore remains in RoboCasa's official
Hugging Face model repository rather than being copied into Git history.

```bash
bash checkpoint/download_original_checkpoint.sh
```

Expected policy path after download:

```text
./robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000
```

Official source: https://huggingface.co/robocasa/robocasa365_checkpoints

Pinned model-repository revision:
`c484448aba1a9b60a04c9b0ca117241518ea69f3`.

Verify the required checkpoint files:

```bash
CKPT="$PWD/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"
test -f "$CKPT/_CHECKPOINT_METADATA"
test -f "$CKPT/assets/norm_stats.json"
test -f "$CKPT/params/_METADATA"
test -f "$CKPT/train_state/_METADATA"
```

## Run the 40-scene collection

The launcher starts the OpenPI policy server and collector in a detached tmux
session. It evaluates 40 scene seeds, five generated base poses per scene, and
five repeated trials per pose.

```bash
cd robocasa-multipose-human300
conda activate robocasa
bash scripts/run_opendrawer_multipose_4090_40seed_v1.sh
```

Monitor it with:

```bash
tmux attach -t opendrawer_40seed_4090_v1
```

## Train the embodied-feasibility Adapter

The Adapter learns current-pose OpenDrawer success from repeated rollouts while
keeping the original Pi0.5 checkpoint frozen. Five poses from the same scene are
ranked together, distance is used only as a success-tied secondary preference,
and an auxiliary head requires the learned tokens to decode robot-centered
geometry. Raw experiment results and generated feature caches remain outside
Git because they are machine-specific and large.

```bash
export DATA_ROOT="/absolute/path/to/OpenDrawer_100seed_5pose_10trials"
export PI05_CHECKPOINT="$PWD/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"
bash adapter/run_opendrawer_multipose_adapter_v1.sh \
  --steps 2000 --scene-batch-size 8
```

See [`adapter/README.md`](adapter/README.md) for the architecture, exact loss,
split policy, expected dataset layout, outputs, and the distinction between
static feasibility learning and later repositioning-advantage learning.

## Repository-specific OpenPI files

`scripts/main_camera_sweep.py` is included directly because it is the evaluator
used by this data collector and is not part of the pinned upstream OpenPI
commit. `patches/openpi_robocasa_multipose.patch` adds the inference-only
`pi05_pretrain_human300_inference` config and deterministic sample-noise support
required by that evaluator. `scripts/setup_repository.sh` applies the patch and
is safe to rerun.

## Licenses

RoboCasa is distributed under MIT and OpenPI under Apache-2.0. The checkpoint is
downloaded from the official RoboCasa Hugging Face repository and is not stored
in this Git repository. Review the upstream repositories and model card before
redistribution or public release.

