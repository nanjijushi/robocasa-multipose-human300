# RoboCasa OpenDrawer Multi-Pose Dataset Toolkit

This private repository packages the reproducible OpenDrawer base-pose generator,
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
- `checkpoint/download_original_checkpoint.sh`: downloads the complete official
  original checkpoint from Hugging Face into the layout used by the scripts.

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

Create the Python environment and install the pinned source trees:

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa

python -m pip install --upgrade pip
python -m pip install -e third_party/robocasa
python -m pip install -e third_party/openpi
python -m pip install -e third_party/openpi/packages/openpi-client
```

The data-collection utilities additionally use Pillow, NumPy, ImageIO, and
Gymnasium. Install them if they were not pulled in by the upstream packages:

```bash
python -m pip install pillow numpy imageio imageio-ffmpeg gymnasium==0.29.1
```

For the established `/root/gpufree-data` server layout, create links so the
absolute paths used by the scripts resolve without editing them:

```bash
ROOT=/root/gpufree-data
REPO="$ROOT/robocasa-multipose-human300"

ln -sfn "$REPO/third_party/robocasa" "$ROOT/robocasa"
ln -sfn "$REPO/third_party/openpi" "$ROOT/openpi"
mkdir -p "$ROOT/view_discriminator_pilot"
cp "$REPO"/scripts/* "$ROOT/view_discriminator_pilot/"
```

Headless collection uses software-rendered MuJoCo in the tested launcher:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/root/gpufree-data/openpi/src:/root/gpufree-data/openpi/packages/openpi-client/src:/root/gpufree-data/robocasa:/root/gpufree-data/robosuite
```

If RoboCasa uses its bundled/upstream robosuite checkout rather than a pip
installation, place or link that checkout at `/root/gpufree-data/robosuite` and
keep it on `PYTHONPATH`, matching the tested server layout.

## Download the original checkpoint

The full checkpoint is about 44.7 GB and contains individual files larger than
GitHub's regular Git/LFS limits. It therefore remains in RoboCasa's official
Hugging Face model repository rather than being copied into Git history.

```bash
bash checkpoint/download_original_checkpoint.sh /root/gpufree-data/robocasa365_checkpoints
```

Expected policy path after download:

```text
/root/gpufree-data/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000
```

Official source: https://huggingface.co/robocasa/robocasa365_checkpoints

Pinned model-repository revision:
`c484448aba1a9b60a04c9b0ca117241518ea69f3`.

Verify the required checkpoint files:

```bash
CKPT=/root/gpufree-data/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000
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
cd /root/gpufree-data
conda activate robocasa
bash view_discriminator_pilot/run_opendrawer_multipose_4090_40seed_v1.sh
```

Monitor it with:

```bash
tmux attach -t opendrawer_40seed_4090_v1
```

## Server layout

The scripts preserve the established server layout under `/root/gpufree-data`.
After cloning, link or copy the repository scripts into
`/root/gpufree-data/view_discriminator_pilot`, and place the two submodules at
`/root/gpufree-data/robocasa` and `/root/gpufree-data/openpi`, or adjust the
`ROOT` constants in the scripts.

## Licenses

RoboCasa is distributed under MIT and OpenPI under Apache-2.0. The checkpoint is
downloaded from the official RoboCasa Hugging Face repository and is not stored
in this Git repository. Review the upstream repositories and model card before
redistribution or public release.

