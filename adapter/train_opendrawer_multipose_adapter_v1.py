#!/usr/bin/env python3
"""Train ACT feasibility on same-scene, five-pose OpenDrawer groups."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from flax import traverse_util
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from embodied_feasibility_adapter_v1 import FeasibilityAdapterConfig
from embodied_feasibility_adapter_multipose_v1 import (
    MultiPoseFeasibilityAdapter, huber_geometry_loss, token_regularization,
)
from feasibility_losses_v1 import (
    rollout_value_loss, reliable_pairwise_rank_loss, distance_tiebreak_loss,
)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--scene-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--ff-dim", type=int, default=768)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--rank-weight", type=float, default=0.5)
    p.add_argument("--distance-weight", type=float, default=0.1)
    p.add_argument("--geometry-weight", type=float, default=0.2)
    p.add_argument("--token-reg-weight", type=float, default=1e-5)
    return p.parse_args()


def grouped_indices(scene_seed, pose_index, split, complete, split_value):
    groups = []
    for seed in sorted(set(scene_seed[(split == split_value) & complete].tolist())):
        ids = np.flatnonzero((scene_seed == seed) & (split == split_value) & complete)
        order = ids[np.argsort(pose_index[ids])]
        if len(order) != 5 or pose_index[order].tolist() != [0, 1, 2, 3, 4]:
            raise ValueError(f"seed {seed} is not an exact five-pose group")
        groups.append(order)
    return np.stack(groups)


def save_state(path, state, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = traverse_util.flatten_dict(state.filter(nnx.Param).to_pure_dict())
    arrays = {"/".join(map(str, key)): np.asarray(value.value if hasattr(value, "value") else value)
              for key, value in flat.items() if value is not None}
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    a = cli(); a.output.mkdir(parents=True, exist_ok=True)
    data = np.load(a.dataset, allow_pickle=False)
    source_ids = sorted(int(k.rsplit("_", 1)[1]) for k in data.files
                        if k.startswith("source_") and not k.startswith("source_mask_"))
    sources = [np.asarray(data[f"source_{i}"], np.float32) for i in source_ids]
    masks = [np.asarray(data[f"source_mask_{i}"], bool) for i in source_ids]
    scene_seed = np.asarray(data["scene_seed"], np.int32)
    pose_index = np.asarray(data["pose_index"], np.int8)
    split = np.asarray(data["split"], np.uint8)
    complete = np.asarray(data["complete_group"], bool)
    successes = np.asarray(data["successes"], np.float32)
    trials = np.asarray(data["trials"], np.float32)
    distances = np.asarray(data["distances_m"], np.float32)
    geometry = (np.asarray(data["geometry_targets"], np.float32) - data["geometry_mean"]) / data["geometry_std"]
    actions = np.asarray(data["candidate_actions"], np.float32)
    train_groups = grouped_indices(scene_seed, pose_index, split, complete, 0)
    val_groups = grouped_indices(scene_seed, pose_index, split, complete, 1)
    test_groups = grouped_indices(scene_seed, pose_index, split, complete, 2)
    partial_ids = np.flatnonzero((split == 0) & ~complete)
    cfg = FeasibilityAdapterConfig(source_dims=tuple(x.shape[-1] for x in sources),
        action_dim=actions.shape[-1], model_dim=a.model_dim, num_queries=a.num_queries,
        num_heads=a.num_heads, num_blocks=a.num_blocks, ff_dim=a.ff_dim)
    model = MultiPoseFeasibilityAdapter(cfg, rngs=nnx.Rngs(a.seed))
    graph, state = nnx.split(model)
    tx = optax.adamw(a.lr, weight_decay=a.weight_decay)
    optimizer = tx.init(state.filter(nnx.Param))

    def batch(ids):
        return {"sources": tuple(jnp.asarray(x[ids]) for x in sources),
                "masks": tuple(jnp.asarray(x[ids]) for x in masks),
                "actions": jnp.asarray(actions[ids]), "successes": jnp.asarray(successes[ids]),
                "trials": jnp.asarray(trials[ids]), "distances": jnp.asarray(distances[ids]),
                "geometry": jnp.asarray(geometry[ids])}

    def loss_fn(m, b, grouped_count):
        _, logits, tokens, geom = m(b["sources"], b["masks"], b["actions"])
        logits = logits[:, 0]
        point_mask = jnp.ones_like(logits, dtype=jnp.bool_)
        value = rollout_value_loss(logits[:, None], b["successes"][:, None], b["trials"][:, None], point_mask[:, None])
        group_logits = logits[:grouped_count].reshape((-1, 5))
        group_successes = b["successes"][:grouped_count].reshape((-1, 5))
        group_trials = b["trials"][:grouped_count].reshape((-1, 5))
        group_distances = b["distances"][:grouped_count].reshape((-1, 5))
        valid = jnp.ones_like(group_logits, dtype=jnp.bool_)
        rank, rank_pairs = reliable_pairwise_rank_loss(group_logits, group_successes, group_trials, valid,
            margin=0.10, rate_delta=0.20, min_trials=5)
        distance, distance_pairs = distance_tiebreak_loss(group_logits, group_successes, group_trials,
            group_distances, valid, valid, margin=0.05, tie_delta=0.10,
            min_distance_gap_m=0.05, min_trials=5)
        geo = huber_geometry_loss(geom, b["geometry"])
        token_reg = token_regularization(tokens)
        total = value + a.rank_weight * rank + a.distance_weight * distance + a.geometry_weight * geo + a.token_reg_weight * token_reg
        return total, {"loss": total, "success_nll": value, "rank_loss": rank,
            "distance_tiebreak_loss": distance, "geometry_loss": geo, "token_regularization": token_reg,
            "rank_pairs": rank_pairs, "distance_pairs": distance_pairs}

    def train_step(st, opt, b, grouped_count):
        mdl = nnx.merge(graph, st)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(mdl, b, grouped_count)
        params = st.filter(nnx.Param)
        updates, opt = tx.update(grads, opt, params)
        nnx.update(mdl, optax.apply_updates(params, updates))
        return nnx.state(mdl), opt, metrics, optax.global_norm(grads)
    train_step = jax.jit(train_step, static_argnums=(3,))

    def evaluate(st, groups):
        ids = groups.reshape(-1); b = batch(ids); mdl = nnx.merge(graph, st)
        _, logits, tokens, geom = mdl(b["sources"], b["masks"], b["actions"])
        total, metrics = loss_fn(mdl, b, len(ids))
        rates = successes[ids] / trials[ids]
        probs = np.asarray(jax.nn.sigmoid(logits[:, 0]))
        out = {k: float(v) for k, v in metrics.items()}
        out["mae_success_rate"] = float(np.mean(np.abs(probs - rates)))
        out["brier"] = float(np.mean((probs - rates) ** 2))
        return out

    rng = np.random.default_rng(a.seed); best = math.inf; history = []; started = time.time()
    for step in range(1, a.steps + 1):
        chosen = rng.choice(len(train_groups), a.scene_batch_size, replace=len(train_groups) < a.scene_batch_size)
        group_ids = train_groups[chosen].reshape(-1)
        ids = np.concatenate((group_ids, partial_ids))
        state, optimizer, metrics, grad_norm = train_step(state, optimizer, batch(ids), len(group_ids))
        if step == 1 or step % a.eval_interval == 0 or step == a.steps:
            val = evaluate(state, val_groups)
            row = {"step": step, "train_loss": float(metrics["loss"]), "grad_norm": float(grad_norm),
                   **{f"val_{k}": v for k, v in val.items()}}
            history.append(row); print(json.dumps(row), flush=True)
            if val["loss"] < best:
                best = val["loss"]
                save_state(a.output / "best_adapter.npz", state, {"step": step, "val_loss": best,
                    "config": cfg.__dict__, "loss_weights": {"success": 1.0, "rank": a.rank_weight,
                    "distance": a.distance_weight, "geometry": a.geometry_weight,
                    "token_regularization": a.token_reg_weight}})
    test = evaluate(state, test_groups)
    summary = {"status": "complete", "dataset": str(a.dataset), "complete_train_scenes": len(train_groups),
        "partial_train_positions": len(partial_ids), "validation_scenes": len(val_groups), "test_scenes": len(test_groups),
        "best_val_loss": best, "final_test": test, "history": history, "elapsed_seconds": time.time() - started}
    (a.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
