# Task-conditioned embodied feasibility Adapter

This directory contains the first-stage Adapter for learning whether the current
robot-centered observation is suitable for executing the downstream OpenDrawer
policy. The original Pi0.5 `pi05_pretrain_human300` checkpoint is frozen.

## Architecture

The feature cache reads three policy-input views (main, masked right external,
and wrist), extracts frozen Pi0.5 SigLIP spatial tokens, and extracts the final
task-conditioned PaliGemma prompt tokens. `EmbodiedFeasibilityAdapter` projects
the four sources to a shared space. Four learned embodied queries cross-attend
to them and produce Adapter tokens. Two heads use the same tokens:

1. the ACT value head predicts downstream task success probability;
2. the auxiliary geometry head decodes robot-centered `dx`, `dy`, horizontal
   distance, `sin(bearing)`, and `cos(bearing)`.

The geometry is supervision only. It is not fed to the Adapter as an input.
This forces its tokens to retain robot-target spatial information visible in
the frozen VLA representations.

The query-resampler design follows the general experience of Visual Query
Tuning, BLIP-2 Q-Former, and Perceiver-style learned queries. This repository
contains an independent JAX/Flax NNX implementation and does not copy their
model code.

## Objective

For each pose with `k` successes in `n` frozen-policy rollouts, training uses:

```text
L = L_success + 0.5 L_rank + 0.1 L_distance + 0.2 L_geometry
    + 1e-5 L_token
```

- `L_success` is binomial negative log likelihood and preserves uncertainty in
  observations such as 7/10 instead of converting labels to zero or one.
- `L_rank` compares the five poses from the same scene seed. It is enabled only
  when observed success rates differ by at least 0.20.
- `L_distance` is a secondary rule. It prefers the nearer safe pose only when
  success rates differ by no more than 0.10 and distance differs by at least
  5 cm. It never overrides a clear success-rate difference.
- `L_geometry` is Huber loss on normalized robot-centered geometry.
- `L_token` is a small Adapter-token magnitude regularizer.

The split is made by `scene_seed`, so no pose from a validation or test kitchen
appears in training. The test split is evaluated once after training and is not
used for checkpoint selection. Partial seeds are used only by pointwise success
and geometry supervision; they are excluded from five-pose ranking.

## Files

- `prepare_opendrawer_multipose_adapter_split_v1.py`: audits collected records
  and creates deterministic train/validation/test manifests.
- `cache_opendrawer_multipose_features_v1.py`: runs the frozen Pi0.5 backbone
  once and stores compact features.
- `embodied_feasibility_adapter_v1.py`: learned-query Adapter and value head.
- `embodied_feasibility_adapter_multipose_v1.py`: geometry-head extension.
- `feasibility_losses_v1.py`: rollout, ranking, and distance losses.
- `train_opendrawer_multipose_adapter_v1.py`: grouped multi-pose trainer.
- `run_opendrawer_multipose_adapter_v1.sh`: end-to-end launcher.

## Dataset layout

The raw result directory is intentionally outside Git. It must contain worker
manifests and the saved first-frame images produced by the collection scripts:

```text
DATA_ROOT/
  worker_00/
    feasibility_observations_worker00.jsonl
    scene_seed_000/P0_default/*.jpg
    ...
```

The split script discovers all complete five-pose seeds and deterministically
uses approximately 70/15/15 percent for train/validation/test. Explicit counts
can still be supplied with all three split arguments. Partial seeds remain
training-only and never enter grouped ranking.

## Run

Install this repository and download the original checkpoint as described in
the root README. Then run on a CUDA-capable machine:

```bash
cd /path/to/robocasa-multipose-human300
conda activate robocasa

export DATA_ROOT="/wangxinhao/tmp/实验结果/OpenDrawer_8卡_100seed_5位置_每位置10次"
export PI05_CHECKPOINT="$PWD/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"
export WORK_ROOT="$PWD/outputs/opendrawer_multipose_adapter_v1"

bash adapter/run_opendrawer_multipose_adapter_v1.sh \
  --steps 2000 --scene-batch-size 8
```

The backbone feature cache requires the GPU and original checkpoint. Adapter
training itself reads the cached NPZ and does not update the checkpoint.
Outputs are written to `$WORK_ROOT/split`, `$WORK_ROOT/cache`, and
`$WORK_ROOT/training`. The selected weights are
`$WORK_ROOT/training/best_adapter.npz`.

The launcher selects `/wangxinhao/envs/robocasa/bin/python` when present, then
the tested `/opt/conda` environment, and finally `python` from `PATH`. Override
`PYTHON_BIN` explicitly when needed. It also places tokenizer/download caches in
the repository's persistent `.cache/openpi` directory. Feature extraction uses
atomic per-batch shards and can resume after interruption with the same manifest.

This stage learns current-pose feasibility `V(o,g)`. A later movement dataset
with state transitions `(o_t, a_t, o_{t+1})` is required to learn repositioning
advantage `V(o_{t+1},g)-V(o_t,g)`; static rollouts do not fabricate that label.
