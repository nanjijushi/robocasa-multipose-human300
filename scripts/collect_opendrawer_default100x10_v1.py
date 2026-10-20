#!/usr/bin/env python3
"""Collect OpenDrawer feasibility observations at RoboCasa default positions.

One scene seed defines one fixed scene and one fixed default robot position.
For every seed, the exact episode metadata is frozen and restored for ten
rollouts. The right external camera is blacked out before policy inference.

This script reuses the verified main_camera_sweep evaluator, but deliberately:
  * keeps RoboCasa's default camera pose (no look-at / camera sweep),
  * records initial robot-handle geometry,
  * stores compact two-frame diagnostic videos to control disk usage,
  * writes an aggregate JSONL/CSV suitable for later adapter preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio_v2
import numpy as np


ROOT = Path("/root/gpufree-data")
EXAMPLES = ROOT / "openpi/examples/robocasa"
# These must be set before importing RoboCasa / MuJoCo.  This container exposes
# CUDA compute but currently enumerates zero NVIDIA EGL devices, so use the
# verified OSMesa headless renderer. Policy inference still runs on GPU.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import main_camera_sweep as sweep  # noqa: E402


DEFAULT_OUT = ROOT / "实验结果/20261018_OpenDrawer默认位姿_100seed每seed10次_右主视角黑_v1"
BASE_CHECKPOINT = ROOT / "robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"
_original_mimwrite = imageio_v2.mimwrite


def parse_seed_spec(spec: str) -> list[int]:
    result: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(x) for x in part.split("-", 1))
            step = 1 if last >= first else -1
            result.extend(range(first, last + step, step))
        else:
            result.append(int(part))
    result = list(dict.fromkeys(result))
    if not result:
        raise ValueError("--seeds produced no seeds")
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def compact_mimwrite(uri, images, *args, **kwargs):
    """Write a valid tiny diagnostic video containing only first/last frames."""
    frames = list(images)
    if not frames:
        return None
    selected = [np.asarray(frames[0])]
    if len(frames) > 1:
        selected.append(np.asarray(frames[-1]))
    kwargs = dict(kwargs)
    kwargs["fps"] = 2
    return _original_mimwrite(uri, selected, *args, **kwargs)


def default_camera_metadata(env, args) -> dict[str, Any]:
    config = sweep.CamUtils.CAM_CONFIGS["DEFAULT"].get(args.camera_name, {})
    serializable = {}
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return {
        "mode": "robocasa_default_camera_unchanged",
        "camera_name": args.camera_name,
        "camera_config": serializable,
    }


_original_geometry_snapshot = sweep._geometry_snapshot


def geometry_snapshot_with_relation(env, obs, initial_handle_pos, initial_open_fraction):
    result = _original_geometry_snapshot(env, obs, initial_handle_pos, initial_open_fraction)
    base = np.asarray(result["base_xyz"], dtype=np.float64)
    handle = np.asarray(result["handle_xyz"], dtype=np.float64)
    delta = handle - base
    rotation = np.asarray(obs.get("state.base_rotation", []), dtype=np.float64).reshape(-1)
    result.update(
        {
            "handle_minus_base_world_xyz_m": delta.tolist(),
            "handle_bearing_world_deg": float(math.degrees(math.atan2(delta[1], delta[0]))),
            "base_rotation_raw": rotation.tolist(),
        }
    )

    # RoboCasa's mobile-base observation is normally planar. Preserve the raw
    # rotation above and mark the convention used for this convenience field.
    yaw = None
    convention = "unavailable"
    if rotation.size == 1:
        yaw = float(rotation[0])
        convention = "single_planar_yaw_rad"
    elif rotation.size == 3:
        yaw = float(rotation[-1])
        convention = "three_vector_last_component_as_planar_yaw_rad"
    elif rotation.size == 4:
        x, y, z, w = (float(v) for v in rotation)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        convention = "xyzw_quaternion_to_yaw_rad"
    if yaw is not None:
        c, s = math.cos(yaw), math.sin(yaw)
        local_x = c * delta[0] + s * delta[1]
        local_y = -s * delta[0] + c * delta[1]
        bearing = math.atan2(local_y, local_x)
        result.update(
            {
                "base_yaw_rad": yaw,
                "base_yaw_convention": convention,
                "handle_minus_base_local_xyz_m": [float(local_x), float(local_y), float(delta[2])],
                "handle_bearing_in_base_deg": float(math.degrees(bearing)),
            }
        )
    return result


def completed_seed(scene_dir: Path, trials: int) -> bool:
    summary = read_json(scene_dir / "summary.json")
    if not summary:
        return False
    if summary.get("same_scene") is not True or summary.get("disable_right_image") is not True:
        return False
    variant = (summary.get("variants") or {}).get("baseline") or {}
    stats = variant.get("stats") or {}
    return (
        int(stats.get("num_episodes", -1)) == trials
        and len(stats.get("episode_results") or []) == trials
        and bool(stats.get("record_geometry"))
    )


def split_for_seed(index: int, total: int) -> str:
    # Deterministic seed-level split; trials from one seed never cross splits.
    train_end = round(total * 0.70)
    val_end = train_end + round(total * 0.15)
    return "train" if index < train_end else ("val" if index < val_end else "test")


def collect_rows(
    out_root: Path,
    seeds: list[int],
    trials: int,
    evaluation_horizon: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        scene_dir = out_root / f"scene_seed_{seed:03d}"
        summary = read_json(scene_dir / "summary.json") or {}
        variant = (summary.get("variants") or {}).get("baseline") or {}
        stats = variant.get("stats") or {}
        episodes = stats.get("episode_results") or []
        if not episodes:
            continue
        raw_successes = sum(bool(ep.get("success")) for ep in episodes)
        def success_at_horizon(ep: dict[str, Any]) -> bool:
            if not bool(ep.get("success")):
                return False
            if evaluation_horizon is None:
                return True
            return int(ep.get("steps", evaluation_horizon + 1)) <= evaluation_horizon
        successes = sum(success_at_horizon(ep) for ep in episodes)
        initial = ((episodes[0].get("geometry") or {}).get("initial") or {})
        row = {
            "observed": True,
            "task": "OpenDrawer",
            "task_description": stats.get("original_task_description"),
            "scene_seed": seed,
            "split": split_for_seed(index, len(seeds)),
            "position_source": "robocasa_default",
            "same_scene_across_trials": True,
            "right_external_camera_masked": True,
            "successes": successes,
            "successes_raw": raw_successes,
            "trials": len(episodes),
            "success_rate": successes / len(episodes),
            "success_rate_raw": raw_successes / len(episodes),
            "success_rate_beta11": (successes + 1) / (len(episodes) + 2),
            "evaluation_horizon": evaluation_horizon,
            "initial_geometry": initial,
            "initial_policy_input_images": episodes[0].get("initial_policy_input_images"),
            "trial_results": [
                {
                    "trial_index": int(ep.get("episode_idx", i)),
                    "success_raw": bool(ep.get("success")),
                    "success_at_evaluation_horizon": success_at_horizon(ep),
                    "steps": ep.get("steps"),
                    "horizon": ep.get("horizon"),
                    "video_path": ep.get("video_path"),
                }
                for i, ep in enumerate(episodes)
            ],
            "scene_dir": str(scene_dir),
            "base_checkpoint": str(BASE_CHECKPOINT),
        }
        rows.append(row)
    return rows


def write_aggregate(
    out_root: Path,
    seeds: list[int],
    trials: int,
    evaluation_horizon: int | None,
) -> None:
    rows = collect_rows(out_root, seeds, trials, evaluation_horizon)
    jsonl = out_root / "feasibility_observations.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    csv_path = out_root / "feasibility_summary.csv"
    fields = [
        "scene_seed", "split", "task_description", "successes", "trials",
        "success_rate", "successes_raw", "success_rate_raw",
        "success_rate_beta11", "evaluation_horizon", "base_to_handle_xy_m",
        "base_to_handle_3d_m", "handle_dx_world_m", "handle_dy_world_m",
        "handle_dz_world_m", "handle_bearing_world_deg", "handle_bearing_in_base_deg",
        "scene_dir",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            g = row["initial_geometry"]
            d = g.get("handle_minus_base_world_xyz_m") or [None, None, None]
            writer.writerow(
                {
                    "scene_seed": row["scene_seed"],
                    "split": row["split"],
                    "task_description": row["task_description"],
                    "successes": row["successes"],
                    "trials": row["trials"],
                    "success_rate": row["success_rate"],
                    "successes_raw": row["successes_raw"],
                    "success_rate_raw": row["success_rate_raw"],
                    "success_rate_beta11": row["success_rate_beta11"],
                    "evaluation_horizon": row["evaluation_horizon"],
                    "base_to_handle_xy_m": g.get("base_to_handle_xy_m"),
                    "base_to_handle_3d_m": g.get("base_to_handle_3d_m"),
                    "handle_dx_world_m": d[0], "handle_dy_world_m": d[1], "handle_dz_world_m": d[2],
                    "handle_bearing_world_deg": g.get("handle_bearing_world_deg"),
                    "handle_bearing_in_base_deg": g.get("handle_bearing_in_base_deg"),
                    "scene_dir": row["scene_dir"],
                }
            )
    completed = sum(completed_seed(out_root / f"scene_seed_{s:03d}", trials) for s in seeds)
    atomic_json(
        out_root / "collection_status.json",
        {
            "status": "complete" if completed == len(seeds) else "partial",
            "completed_seeds": completed,
            "requested_seeds": len(seeds),
            "requested_trials_per_seed": trials,
            "completed_observations": len(rows),
            "requested_rollouts": len(seeds) * trials,
            "evaluation_horizon": evaluation_horizon,
            "base_checkpoint": str(BASE_CHECKPOINT),
            "right_external_camera_masked": True,
            "default_robot_position": True,
            "default_camera_pose": True,
            "updated_at": time.time(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-99", help="e.g. 0-99 or 0,3,8")
    parser.add_argument("--trials-per-seed", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    if args.trials_per_seed <= 0:
        raise ValueError("--trials-per-seed must be positive")
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Expected original checkpoint is missing: {BASE_CHECKPOINT}")

    seeds = parse_seed_spec(args.seeds)
    args.out_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out_root / "experiment_config.json",
        {
            "seeds": seeds,
            "trials_per_seed": args.trials_per_seed,
            "task": "OpenDrawer",
            "split": "pretrain",
            "base_checkpoint": str(BASE_CHECKPOINT),
            "position": "RoboCasa default; one fixed position per scene seed",
            "camera": "RoboCasa default pose; right external/main camera zeroed before policy",
            "render_backend": os.environ.get("MUJOCO_GL"),
            "repetition": "fixed episode metadata and reset seed; policy server sampling continues across trials",
            "unobserved_positions_are_not_negative": True,
            "horizon_override": args.horizon,
            "time_limit_note": (
                "330 steps is approximately 90 seconds with the current OSMesa renderer; "
                "the evaluator enforces a step limit, so wall-clock duration can vary"
                if args.horizon == 330 else None
            ),
        },
    )

    # Preserve default camera extrinsics and enrich geometry without forking the
    # proven evaluator. Two-frame videos are enough to audit initial/final state.
    sweep._aim_camera_at_target = default_camera_metadata
    sweep._geometry_snapshot = geometry_snapshot_with_relation
    sweep.imageio.mimwrite = compact_mimwrite

    for ordinal, seed in enumerate(seeds, start=1):
        scene_dir = args.out_root / f"scene_seed_{seed:03d}"
        if args.resume and completed_seed(scene_dir, args.trials_per_seed):
            print(f"[skip {ordinal}/{len(seeds)}] seed={seed} complete", flush=True)
            continue
        print(f"[run {ordinal}/{len(seeds)}] seed={seed}, trials={args.trials_per_seed}", flush=True)
        eval_args = sweep.Args(
            host=args.host,
            port=args.port,
            task="OpenDrawer",
            split="pretrain",
            num_trials=args.trials_per_seed,
            seed=seed,
            same_scene=True,
            freeze_episode_meta=True,
            horizon_override=args.horizon,
            disable_right_image=True,
            record_geometry=True,
            variants=["baseline"],
            # Default camera is intentionally left unchanged, so no look-at
            # target should be resolved while freezing episode metadata.
            look_at="none",
            log_dir=str(scene_dir),
            render_every=1_000_000,
            resume=args.resume,
        )
        sweep.eval_main(eval_args)
        write_aggregate(args.out_root, seeds, args.trials_per_seed, args.horizon)

    write_aggregate(args.out_root, seeds, args.trials_per_seed, args.horizon)
    print(f"COMPLETE: {args.out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
