"""Multi-pose extension of the task-conditioned feasibility adapter."""
from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from embodied_feasibility_adapter_v1 import EmbodiedFeasibilityAdapter, FeasibilityAdapterConfig


class MultiPoseFeasibilityAdapter(nnx.Module):
    """Predict ACT success and decode robot-target geometry from shared tokens."""

    def __init__(self, config: FeasibilityAdapterConfig, geometry_dim: int = 5, *, rngs: nnx.Rngs):
        self.core = EmbodiedFeasibilityAdapter(config, rngs=rngs)
        self.geometry_in = nnx.Linear(config.model_dim, config.model_dim, rngs=rngs)
        self.geometry_out = nnx.Linear(config.model_dim, geometry_dim, rngs=rngs)

    def __call__(self, sources, source_masks, candidate_actions):
        probabilities, logits, tokens = self.core(sources, source_masks, candidate_actions)
        pooled = jnp.mean(tokens, axis=1)
        geometry = self.geometry_out(nnx.gelu(self.geometry_in(pooled)))
        return probabilities, logits, tokens, geometry


def huber_geometry_loss(predicted, target, delta: float = 1.0):
    error = predicted - target
    absolute = jnp.abs(error)
    return jnp.mean(jnp.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta)))


def token_regularization(tokens):
    return jnp.mean(jnp.square(tokens.astype(jnp.float32)))
