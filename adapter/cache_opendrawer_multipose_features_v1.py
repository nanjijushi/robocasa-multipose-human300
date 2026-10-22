#!/usr/bin/env python3
"""Cache frozen Pi0.5 features for the OpenDrawer multi-pose manifest."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OPENPI = REPO / "third_party/openpi/src"
DEFAULT_CKPT = REPO / "robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"
POSE_INDEX = {name: index for index, name in enumerate((
    "P0_default", "P1_near_center", "P2_far_center", "P3_lateral_mid", "P4_lateral_wide"
))}
SPLIT_INDEX = {"train": 0, "train_partial": 0, "validation": 1, "test": 2}


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", default="pi05_pretrain_human300_inference")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-examples", type=int)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_rgb(path: str, resize_with_pad):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return resize_with_pad(image, 224, 224).astype(np.float32) / 127.5 - 1.0


def main():
    a = cli()
    manifest = a.manifest_dir / "adapter_records.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.max_examples is not None:
        rows = rows[:a.max_examples]
    if a.output.exists():
        if not a.overwrite:
            raise FileExistsError(f"{a.output} exists; pass --overwrite")
        shutil.rmtree(a.output)
    a.output.mkdir(parents=True)
    if not (a.checkpoint / "params").exists():
        raise FileNotFoundError(a.checkpoint / "params")
    sys.path.insert(0, str(OPENPI))

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training import config as config_lib
    from openpi_client import image_tools

    params = model_lib.restore_params(a.checkpoint / "params", dtype=jnp.bfloat16)
    model = config_lib.get_config(a.config).model.load(params)
    graph, state = nnx.split(model)
    tokenizer = PaligemmaTokenizer(max_len=200)

    def encode_batch(st, images, prompt_tokens, prompt_mask):
        mdl = nnx.merge(graph, st)
        visual, prefix_parts, prefix_masks = [], [], []
        for image in images:
            tok, _ = mdl.PaliGemma.img(image, train=False)
            pooled = tok.reshape(tok.shape[0], 4, 4, 4, 4, tok.shape[-1]).mean(axis=(2, 4))
            visual.append(pooled.reshape(tok.shape[0], 16, tok.shape[-1]))
            prefix_parts.append(tok)
            prefix_masks.append(jnp.ones(tok.shape[:2], dtype=jnp.bool_))
        text = mdl.PaliGemma.llm(prompt_tokens, method="embed")
        prefix_parts.append(text); prefix_masks.append(prompt_mask)
        prefix = jnp.concatenate(prefix_parts, axis=1)
        mask = jnp.concatenate(prefix_masks, axis=1)
        attn = make_attn_mask(mask, jnp.zeros((prefix.shape[1],), dtype=jnp.bool_))
        positions = jnp.cumsum(mask, axis=1) - 1
        (prefix_out, suffix_out), _ = mdl.PaliGemma.llm([prefix, None], mask=attn, positions=positions)
        if suffix_out is not None:
            raise AssertionError("unexpected suffix output")
        return tuple(x.astype(jnp.float32) for x in visual), prefix_out[:, -prompt_tokens.shape[1]:].astype(jnp.float32)

    encode = jax.jit(encode_batch)
    sources, masks = [[], [], [], []], [[], [], [], []]
    started = time.time()
    for begin in range(0, len(rows), a.batch_size):
        batch_rows = rows[begin:begin + a.batch_size]
        camera_batches, token_batch, token_mask_batch = [[], [], []], [], []
        for row in batch_rows:
            for dst, key in zip(camera_batches, ("main", "right", "wrist"), strict=True):
                dst.append(read_rgb(row["images"][key], image_tools.resize_with_pad))
            prompt = f"Open the {row['drawer_side'].lower()} drawer."
            tok, mask = tokenizer.tokenize(prompt)
            token_batch.append(tok); token_mask_batch.append(mask)
        visual, task = encode(state, tuple(jnp.asarray(np.stack(x)) for x in camera_batches),
                              jnp.asarray(np.stack(token_batch)), jnp.asarray(np.stack(token_mask_batch)))
        for index, array in enumerate((*visual, task)):
            sources[index].append(np.asarray(array, dtype=np.float16))
        for index, array in enumerate(visual):
            masks[index].append(np.ones(array.shape[:2], dtype=bool))
        masks[3].append(np.asarray(token_mask_batch, dtype=bool))
        print(f"encoded {begin + len(batch_rows)}/{len(rows)}", flush=True)

    source_arrays = [np.concatenate(x) for x in sources]
    mask_arrays = [np.concatenate(x) for x in masks]
    local = np.asarray([r["handle_minus_base_local_xyz_m"] for r in rows], dtype=np.float32)
    bearing = np.deg2rad(np.asarray([r["handle_bearing_in_base_deg"] for r in rows], dtype=np.float32))
    geometry = np.column_stack((local[:, 0], local[:, 1],
                                np.asarray([r["base_to_handle_xy_m"] for r in rows], dtype=np.float32),
                                np.sin(bearing), np.cos(bearing))).astype(np.float32)
    train = np.asarray([SPLIT_INDEX[r["split"]] == 0 for r in rows])
    geometry_mean = geometry[train].mean(axis=0)
    geometry_std = np.maximum(geometry[train].std(axis=0), 1e-4)
    payload = {
        **{f"source_{i}": x for i, x in enumerate(source_arrays)},
        **{f"source_mask_{i}": x for i, x in enumerate(mask_arrays)},
        "candidate_actions": np.tile(np.asarray([[[0, 0, 0, 1]]], np.float32), (len(rows), 1, 1)),
        "successes": np.asarray([r["successes"] for r in rows], np.float32),
        "trials": np.asarray([r["trials"] for r in rows], np.float32),
        "distances_m": np.asarray([r["base_to_handle_xy_m"] for r in rows], np.float32),
        "geometry_targets": geometry,
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "split": np.asarray([SPLIT_INDEX[r["split"]] for r in rows], np.uint8),
        "complete_group": np.asarray([r["split"] != "train_partial" for r in rows], bool),
        "scene_seed": np.asarray([r["scene_seed"] for r in rows], np.int32),
        "pose_index": np.asarray([POSE_INDEX[r["pose"]] for r in rows], np.int8),
    }
    dataset = a.output / "opendrawer_multipose_features.npz"
    np.savez_compressed(dataset, **payload)
    metadata = {"status": "complete", "examples": len(rows), "dataset": str(dataset),
                "checkpoint": str(a.checkpoint), "checkpoint_modified": False,
                "source_shapes": [list(x.shape) for x in source_arrays],
                "geometry_order": ["dx_local_m", "dy_local_m", "xy_distance_m", "sin_bearing", "cos_bearing"],
                "elapsed_seconds": time.time() - started, "jax_backend": jax.default_backend()}
    (a.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
