"""Read-only Pi0.5 feature taps for the feasibility adapter.

The original Pi0.5 checkpoint is never modified. V1 exposes each camera's
projected SigLIP spatial tokens and the final task-conditioned PaliGemma prefix
representation. Intermediate scanned Gemma layers are intentionally deferred to
an ablation because exposing them requires modifying the checkpoint execution
path.
"""
from __future__ import annotations
from typing import NamedTuple
import os
from pathlib import Path

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask

_REPO = Path(__file__).resolve().parents[1]
ORIGINAL_ROBOCASA_PI05_CHECKPOINT = os.environ.get(
    "PI05_CHECKPOINT",
    str(_REPO / "robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"),
)


class FeasibilityFeatures(NamedTuple):
    sources: tuple[jax.Array, ...]
    source_masks: tuple[jax.Array, ...]
    source_names: tuple[str, ...]


def extract_frozen_pi05_features(pi_model, observation, *, preprocess_rng=None) -> FeasibilityFeatures:
    """Extract frozen visual and task-conditioned features from a Pi0.5 model."""
    observation = model_lib.preprocess_observation(preprocess_rng, observation, train=False)
    image_tokens = []
    image_masks = []
    source_names = []
    prefix_parts = []
    prefix_masks = []

    for name in observation.images:
        tokens, _ = pi_model.PaliGemma.img(observation.images[name], train=False)
        mask = einops.repeat(observation.image_masks[name], "b -> b s", s=tokens.shape[1])
        image_tokens.append(jax.lax.stop_gradient(tokens))
        image_masks.append(mask)
        source_names.append(f"siglip:{name}")
        prefix_parts.append(tokens)
        prefix_masks.append(mask)

    if observation.tokenized_prompt is None:
        raise ValueError("Pi0.5 feasibility extraction requires a tokenized task prompt")
    prompt_tokens = pi_model.PaliGemma.llm(observation.tokenized_prompt, method="embed")
    prefix_parts.append(prompt_tokens)
    prefix_masks.append(observation.tokenized_prompt_mask)
    prefix_tokens = jnp.concatenate(prefix_parts, axis=1)
    prefix_mask = jnp.concatenate(prefix_masks, axis=1)
    prefix_ar_mask = jnp.zeros((prefix_tokens.shape[1],), dtype=jnp.bool_)
    attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = pi_model.PaliGemma.llm(
        [prefix_tokens, None], mask=attn_mask, positions=positions
    )
    if suffix_out is not None:
        raise AssertionError("unexpected action-expert output during frozen prefix extraction")
    image_tokens.append(jax.lax.stop_gradient(prefix_out))
    image_masks.append(prefix_mask)
    source_names.append("paligemma:final_task_conditioned_prefix")
    return FeasibilityFeatures(tuple(image_tokens), tuple(image_masks), tuple(source_names))
