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

