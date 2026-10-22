#!/usr/bin/env python3
"""Prepare a deterministic scene-level split for OpenDrawer adapter training."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

POSES = (
    "P0_default", "P1_near_center", "P2_far_center",
    "P3_lateral_mid", "P4_lateral_wide",
)
IMAGE_NAMES = {
    "main": "OpenDrawer_baseline_initial_policy_input_main.jpg",
    "right": "OpenDrawer_baseline_initial_policy_input_right.jpg",
    "wrist": "OpenDrawer_baseline_initial_policy_input_wrist.jpg",
    "all": "OpenDrawer_baseline_initial_policy_input_all.jpg",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--train-scenes", type=int, default=60)
    p.add_argument("--val-scenes", type=int, default=12)
    p.add_argument("--test-scenes", type=int, default=13)
    return p.parse_args()


def difficulty(mean_rate: float) -> str:
    if mean_rate < 0.10:
        return "hard"
    if mean_rate < 0.40:
        return "medium"
    return "easy"


def largest_remainder(sizes, total, capacity=None):
    n = sum(sizes.values())
    quota = {k: sizes[k] * total / n for k in sizes}
    out = {
        k: min(math.floor(quota[k]), sizes[k] if capacity is None else capacity[k])
        for k in sizes
    }
    remaining = total - sum(out.values())
    order = sorted(sizes, key=lambda k: (-(quota[k] - math.floor(quota[k])), -sizes[k], k))
    while remaining:
        progressed = False
        for k in order:
            cap = sizes[k] if capacity is None else capacity[k]
            if out[k] < cap:
                out[k] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Cannot satisfy split capacities")
    return out


def ensure_minimum_representation(allocation, sizes, total, minimum=1):
    """Give every stratum with at least three scenes representation in a split."""
    quota = {k: sizes[k] * total / sum(sizes.values()) for k in sizes}
    eligible = [k for k in sizes if sizes[k] >= 3]
    for receiver in eligible:
        if allocation[receiver] >= minimum:
            continue
        donors = [k for k in sizes if allocation[k] > minimum]
        if not donors:
            raise RuntimeError("Cannot give every eligible stratum split representation")
        donor = max(donors, key=lambda k: (allocation[k] - quota[k], allocation[k], sizes[k]))
        allocation[donor] -= 1
        allocation[receiver] += 1
    return allocation


def resolved_record(data_root: Path, raw: dict) -> dict:
    worker = int(raw["worker"])
    scene_seed = int(raw["scene_seed"])
    pose = str(raw["pose"])
    pose_dir = data_root / f"worker_{worker:02d}" / f"scene_seed_{scene_seed:03d}" / pose
    images = {k: str((pose_dir / name).resolve()) for k, name in IMAGE_NAMES.items()}
    meta = raw["pose_metadata"]
    geom = raw["initial_geometry"]
    rel = geom["handle_minus_base_local_xyz_m"]
    return {
        "scene_seed": scene_seed,
        "pose": pose,
        "worker": worker,
        "successes": int(raw["successes"]),
        "trials": int(raw["trials"]),
        "success_rate": float(raw["success_rate"]),
        "images": images,
        "pose_dir": str(pose_dir.resolve()),
        "base_world_xy_m": [float(meta["base_x_m"]), float(meta["base_y_m"])],
        "base_yaw_rad": float(meta["base_yaw_rad"]),
        "base_to_handle_xy_m": float(meta["actual_horizontal_distance_m"]),
        "sample_bearing_deg": float(meta["actual_sample_bearing_deg"]),
        "handle_minus_base_local_xyz_m": [float(x) for x in rel],
        "handle_bearing_in_base_deg": float(geom["handle_bearing_in_base_deg"]),
        "drawer_side": str(geom["drawer_side"]),
        "drawer_fixture_name": str(geom["drawer_fixture_name"]),
        "right_external_camera_masked": bool(raw["right_external_camera_masked"]),
    }


def main():
    a = parse_args()
    root = a.data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    raw_rows = []
    jsonl_files = sorted(root.glob("worker_*/feasibility_observations_worker*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No worker JSONL files under {root}")
    for file in jsonl_files:
        with file.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    raw_rows.append(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"Invalid JSON at {file}:{line_no}") from exc

    rows, rejected, seen = [], [], set()
    for raw in raw_rows:
        key = (int(raw["scene_seed"]), str(raw["pose"]))
        if key in seen:
            raise ValueError(f"Duplicate seed/pose record: {key}")
        seen.add(key)
        row = resolved_record(root, raw)
        missing = [k for k, p in row["images"].items() if not Path(p).is_file()]
        if missing:
            rejected.append({"scene_seed": key[0], "pose": key[1], "reason": "missing_images", "missing": missing})
            continue
        if row["trials"] != 10:
            rejected.append({"scene_seed": key[0], "pose": key[1], "reason": "trials_not_10", "trials": row["trials"]})
            continue
        if not row["right_external_camera_masked"]:
            rejected.append({"scene_seed": key[0], "pose": key[1], "reason": "right_camera_not_masked"})
            continue
        rows.append(row)

    by_seed = collections.defaultdict(list)
    for row in rows:
        by_seed[row["scene_seed"]].append(row)
    complete, partial = {}, {}
    expected = set(POSES)
    for seed, group in sorted(by_seed.items()):
        pose_set = {r["pose"] for r in group}
        target = complete if pose_set == expected and len(group) == len(POSES) else partial
        target[seed] = sorted(group, key=lambda r: POSES.index(r["pose"]))

    expected_complete = a.train_scenes + a.val_scenes + a.test_scenes
    if len(complete) != expected_complete:
        raise ValueError(
            f"Expected {expected_complete} complete five-pose seeds, found {len(complete)}; "
            f"partial={ {k: [r['pose'] for r in v] for k, v in partial.items()} }"
        )

    summaries = {}
    strata = collections.defaultdict(list)
    for seed, group in complete.items():
        mean_rate = float(np.mean([r["success_rate"] for r in group]))
        side = collections.Counter(r["drawer_side"] for r in group).most_common(1)[0][0]
        diff = difficulty(mean_rate)
        summaries[seed] = {
            "mean_success_rate": mean_rate,
            "difficulty": diff,
            "drawer_side": side,
            "drawer_fixture_name": group[0]["drawer_fixture_name"],
        }
        strata[(side, diff)].append(seed)

    rng = np.random.default_rng(a.seed)
    for key in strata:
        ordered = np.asarray(sorted(strata[key]), dtype=np.int64)
        strata[key] = list(ordered[rng.permutation(len(ordered))])
    sizes = {k: len(v) for k, v in strata.items()}
    val_n = largest_remainder(sizes, a.val_scenes)
    val_n = ensure_minimum_representation(val_n, sizes, a.val_scenes)
    test_n = largest_remainder(sizes, a.test_scenes, {k: sizes[k] - val_n[k] for k in sizes})
    test_n = ensure_minimum_representation(test_n, sizes, a.test_scenes)

    split_seeds = {"train": [], "validation": [], "test": []}
    for key in sorted(strata):
        seeds = strata[key]
        nv, nt = val_n[key], test_n[key]
        split_seeds["validation"].extend(seeds[:nv])
        split_seeds["test"].extend(seeds[nv:nv + nt])
        split_seeds["train"].extend(seeds[nv + nt:])
    for name in split_seeds:
        split_seeds[name] = sorted(int(x) for x in split_seeds[name])
    actual_counts = [len(split_seeds[x]) for x in ("train", "validation", "test")]
    if actual_counts != [a.train_scenes, a.val_scenes, a.test_scenes]:
        raise AssertionError(actual_counts)

    seed_to_split = {seed: name for name, seeds in split_seeds.items() for seed in seeds}
    for seed in partial:
        seed_to_split[seed] = "train_partial"
    manifest_rows = []
    for row in rows:
        row = dict(row)
        row["split"] = seed_to_split[row["scene_seed"]]
        manifest_rows.append(row)
    manifest_rows.sort(key=lambda r: (r["scene_seed"], POSES.index(r["pose"])))

    a.output.mkdir(parents=True, exist_ok=True)
    split_doc = {
        "schema_version": 1,
        "created_by": Path(__file__).name,
        "data_root": str(root),
        "random_seed": a.seed,
        "split_unit": "scene_seed",
        "expected_poses": list(POSES),
        "complete_seed_counts": {k: len(v) for k, v in split_seeds.items()},
        "complete_position_counts": {k: len(v) * len(POSES) for k, v in split_seeds.items()},
        "seeds": split_seeds,
        "partial_train_seeds": {str(seed): [r["pose"] for r in group] for seed, group in sorted(partial.items())},
        "difficulty_thresholds": {"hard_max_exclusive": 0.10, "medium_max_exclusive": 0.40},
        "seed_summaries": {str(k): v for k, v in sorted(summaries.items())},
        "rejected_records": rejected,
    }
    (a.output / "split.json").write_text(json.dumps(split_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    with (a.output / "adapter_records.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "status": "complete",
        "raw_position_records": len(raw_rows),
        "usable_position_records": len(rows),
        "complete_five_pose_seeds": len(complete),
        "partial_train_seeds": {str(k): len(v) for k, v in partial.items()},
        "records_by_split": dict(collections.Counter(r["split"] for r in manifest_rows)),
        "successes_by_split": {name: sum(r["successes"] for r in manifest_rows if r["split"] == name) for name in ("train", "train_partial", "validation", "test")},
        "trials_by_split": {name: sum(r["trials"] for r in manifest_rows if r["split"] == name) for name in ("train", "train_partial", "validation", "test")},
        "strata": {
            f"{side}/{diff}": {
                "total": sizes[(side, diff)],
                "validation": val_n[(side, diff)],
                "test": test_n[(side, diff)],
                "train": sizes[(side, diff)] - val_n[(side, diff)] - test_n[(side, diff)],
            }
            for side, diff in sorted(sizes)
        },
    }
    (a.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
