"""Losses for the success-conditioned feasibility adapter.

Value supervision uses rollout counts rather than hand-authored scalar scores.
Ranking is enabled only for candidate pairs with a sufficiently large observed
success-rate difference. Distance is a secondary tie-breaker only.
"""
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class FeasibilityLossConfig:
    rank_weight: float = 0.5
    rank_margin: float = 0.10
    rank_rate_delta: float = 0.20
    min_trials_for_rank: int = 5
    distance_weight: float = 0.10
    distance_margin: float = 0.05
    distance_tie_delta: float = 0.10
    distance_min_gap_m: float = 0.05
    epsilon: float = 1e-6


def _safe_divide(numerator: Array, denominator: Array) -> Array:
    return numerator / jnp.maximum(denominator, 1)


def rollout_value_loss(logits: Array, successes: Array, trials: Array, candidate_mask: Array) -> Array:
    """Mean Bernoulli NLL over all valid rollout attempts.

    A 50/100 observation receives more evidence weight than 1/2, while both
    retain the same target success rate. The binomial combinatorial constant is
    omitted because it does not affect gradients.
    """
    successes = successes.astype(jnp.float32)
    trials = trials.astype(jnp.float32)
    failures = trials - successes
    nll = successes * jax.nn.softplus(-logits) + failures * jax.nn.softplus(logits)
    valid = candidate_mask.astype(jnp.float32) * (trials > 0)
    return jnp.sum(nll * valid) / jnp.maximum(jnp.sum(trials * valid), 1.0)


def reliable_pairwise_rank_loss(
    logits: Array,
    successes: Array,
    trials: Array,
    candidate_mask: Array,
    *,
    margin: float,
    rate_delta: float,
    min_trials: int,
) -> tuple[Array, Array]:
    """Hinge ranking for pairs whose measured rates differ enough to trust."""
    rates = _safe_divide(successes.astype(jnp.float32), trials.astype(jnp.float32))
    rate_diff = rates[:, :, None] - rates[:, None, :]
    logit_diff = logits[:, :, None] - logits[:, None, :]
    valid_candidate = candidate_mask & (trials >= min_trials)
    pair_mask = valid_candidate[:, :, None] & valid_candidate[:, None, :]
    upper = jnp.triu(jnp.ones(rate_diff.shape[-2:], dtype=jnp.bool_), k=1)[None, :, :]
    reliable = pair_mask & upper & (jnp.abs(rate_diff) >= rate_delta)
    direction = jnp.sign(rate_diff)
    loss = jax.nn.relu(margin - direction * logit_diff)
    weight = jnp.abs(rate_diff)
    denom = jnp.maximum(jnp.sum(reliable * weight), 1.0)
    return jnp.sum(loss * reliable * weight) / denom, jnp.sum(reliable)


def distance_tiebreak_loss(
    logits: Array,
    successes: Array,
    trials: Array,
    distances_m: Array,
    candidate_mask: Array,
    safe_mask: Array,
    *,
    margin: float,
    tie_delta: float,
    min_distance_gap_m: float,
    min_trials: int,
) -> tuple[Array, Array]:
    """Prefer the nearer safe candidate only when measured success is tied.

    This term never overrides a clear success-rate difference. Unsafe candidates
    are excluded rather than rewarded for being geometrically close.
    """
    rates = _safe_divide(successes.astype(jnp.float32), trials.astype(jnp.float32))
    rate_diff = rates[:, :, None] - rates[:, None, :]
    distance_diff = distances_m[:, :, None] - distances_m[:, None, :]
    logit_diff = logits[:, :, None] - logits[:, None, :]
    valid_candidate = candidate_mask & safe_mask & (trials >= min_trials)
    pair_mask = valid_candidate[:, :, None] & valid_candidate[:, None, :]
    upper = jnp.triu(jnp.ones(rate_diff.shape[-2:], dtype=jnp.bool_), k=1)[None, :, :]
    tied = pair_mask & upper & (jnp.abs(rate_diff) <= tie_delta) & (jnp.abs(distance_diff) >= min_distance_gap_m)
    # Negative distance_diff means candidate i is nearer and should have a higher logit.
    nearer_direction = -jnp.sign(distance_diff)
    loss = jax.nn.relu(margin - nearer_direction * logit_diff)
    weight = jnp.minimum(jnp.abs(distance_diff) / jnp.maximum(min_distance_gap_m, 1e-6), 4.0)
    denom = jnp.maximum(jnp.sum(tied * weight), 1.0)
    return jnp.sum(loss * tied * weight) / denom, jnp.sum(tied)


def feasibility_adapter_loss(
    logits: Array,
    successes: Array,
    trials: Array,
    candidate_mask: Array,
    *,
    config: FeasibilityLossConfig = FeasibilityLossConfig(),
    distances_m: Array | None = None,
    safe_mask: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    value = rollout_value_loss(logits, successes, trials, candidate_mask)
    rank, rank_pairs = reliable_pairwise_rank_loss(
        logits, successes, trials, candidate_mask,
        margin=config.rank_margin,
        rate_delta=config.rank_rate_delta,
        min_trials=config.min_trials_for_rank,
    )
    distance = jnp.asarray(0.0, dtype=jnp.float32)
    distance_pairs = jnp.asarray(0, dtype=jnp.int32)
    if distances_m is not None:
        if safe_mask is None:
            raise ValueError("safe_mask is required when distances_m is provided")
        distance, distance_pairs = distance_tiebreak_loss(
            logits, successes, trials, distances_m, candidate_mask, safe_mask,
            margin=config.distance_margin,
            tie_delta=config.distance_tie_delta,
            min_distance_gap_m=config.distance_min_gap_m,
            min_trials=config.min_trials_for_rank,
        )
    total = value + config.rank_weight * rank + config.distance_weight * distance
    metrics = {
        "loss": total,
        "value_loss": value,
        "rank_loss": rank,
        "distance_tiebreak_loss": distance,
        "rank_pairs": rank_pairs,
        "distance_pairs": distance_pairs,
    }
    return total, metrics
