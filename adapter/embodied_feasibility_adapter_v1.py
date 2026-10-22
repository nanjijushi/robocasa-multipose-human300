"""Success-conditioned multi-source query adapter for Pi0.5.

Architecture inspiration:
- Visual Query Tuning: learnable queries aggregate frozen intermediate features.
- BLIP-2 Q-Former / Perceiver Resampler: fixed query tokens cross-attend to a
  variable number of source tokens.

This is a clean JAX/Flax NNX implementation for OpenPI. It does not copy those
projects' model code. It predicts downstream task success for arbitrary
candidate actions, including ACT, instead of a fixed seven/eight-way class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class FeasibilityAdapterConfig:
    source_dims: tuple[int, ...]
    action_dim: int = 4  # e.g. [vx, vy, wz, is_act]
    model_dim: int = 256
    num_queries: int = 4
    num_heads: int = 8
    num_blocks: int = 2
    ff_dim: int = 768
    dropout_rate: float = 0.0

    def __post_init__(self):
        if not self.source_dims:
            raise ValueError("source_dims must contain at least one source")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")


class QueryBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.attn_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.ff_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.ff_in = nnx.Linear(dim, ff_dim, rngs=rngs)
        self.ff_out = nnx.Linear(ff_dim, dim, rngs=rngs)

    def __call__(self, queries: Array, context: Array, context_mask: Array) -> Array:
        b, qn, _ = queries.shape
        sn = context.shape[1]
        q = self.q_proj(self.attn_norm(queries)).reshape(b, qn, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(b, sn, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(b, sn, self.num_heads, self.head_dim)
        logits = jnp.einsum("bqhd,bshd->bhqs", q, k) * (self.head_dim ** -0.5)
        logits = jnp.where(context_mask[:, None, None, :], logits, jnp.finfo(logits.dtype).min)
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
        attended = jnp.einsum("bhqs,bshd->bqhd", weights, v).reshape(b, qn, self.dim)
        queries = queries + self.o_proj(attended)
        ff = self.ff_out(nnx.gelu(self.ff_in(self.ff_norm(queries))))
        return queries + ff


class EmbodiedFeasibilityAdapter(nnx.Module):
    """Aggregates frozen features and scores arbitrary candidate actions.

    Args to __call__:
      sources: feature arrays [B, S_i, D_i], e.g. projected SigLIP spatial
        tokens, PaliGemma middle-layer tokens, and final-layer tokens.
      source_masks: corresponding boolean masks [B, S_i].
      candidate_actions: [B, A, action_dim]. The last component may mark ACT;
        the remaining components may describe continuous base displacement.
    Returns:
      probabilities [B, A], logits [B, A], embodiment tokens [B, Q, model_dim].
    """

    def __init__(self, config: FeasibilityAdapterConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.source_projections = [nnx.Linear(d, config.model_dim, rngs=rngs) for d in config.source_dims]
        self.source_embeddings = nnx.Param(
            jax.random.normal(rngs.params(), (len(config.source_dims), config.model_dim)) * 0.02
        )
        self.queries = nnx.Param(
            jax.random.normal(rngs.params(), (config.num_queries, config.model_dim)) * 0.02
        )
        self.blocks = [
            QueryBlock(config.model_dim, config.num_heads, config.ff_dim, rngs=rngs)
            for _ in range(config.num_blocks)
        ]
        self.output_norm = nnx.LayerNorm(config.model_dim, rngs=rngs)
        self.action_in = nnx.Linear(config.action_dim, config.model_dim, rngs=rngs)
        self.action_out = nnx.Linear(config.model_dim, config.model_dim, rngs=rngs)
        self.value_in = nnx.Linear(2 * config.model_dim, config.model_dim, rngs=rngs)
        self.value_out = nnx.Linear(config.model_dim, 1, rngs=rngs)

    def __call__(
        self,
        sources: Sequence[Array],
        source_masks: Sequence[Array],
        candidate_actions: Array,
    ) -> tuple[Array, Array, Array]:
        if len(sources) != len(self.source_projections) or len(source_masks) != len(sources):
            raise ValueError("sources, source_masks and configured source_dims must have equal length")
        contexts = []
        masks = []
        for idx, (source, mask, projection) in enumerate(zip(sources, source_masks, self.source_projections, strict=True)):
            if source.ndim != 3 or mask.shape != source.shape[:2]:
                raise ValueError(f"invalid source or mask shape for source {idx}: {source.shape}, {mask.shape}")
            token = projection(source.astype(jnp.float32))
            token = token + self.source_embeddings.value[idx][None, None, :]
            contexts.append(token)
            masks.append(mask.astype(jnp.bool_))
        context = jnp.concatenate(contexts, axis=1)
        context_mask = jnp.concatenate(masks, axis=1)
        batch_size = context.shape[0]
        queries = jnp.broadcast_to(self.queries.value[None, :, :], (batch_size,) + self.queries.value.shape)
        for block in self.blocks:
            queries = block(queries, context, context_mask)
        embodiment_tokens = self.output_norm(queries)
        pooled = jnp.mean(embodiment_tokens, axis=1)

        action_features = self.action_out(nnx.gelu(self.action_in(candidate_actions.astype(jnp.float32))))
        pooled = jnp.broadcast_to(pooled[:, None, :], action_features.shape)
        joint = jnp.concatenate([pooled, action_features], axis=-1)
        logits = self.value_out(nnx.gelu(self.value_in(joint)))[..., 0]
        probabilities = jax.nn.sigmoid(logits)
        return probabilities, logits, embodiment_tokens
