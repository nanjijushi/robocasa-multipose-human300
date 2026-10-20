#!/usr/bin/env python3
"""Preview multiple reproducible robot birth poses in one fixed OpenDrawer scene."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path("/root/gpufree-data")
for part in ("view_discriminator_pilot", "robocasa", "robosuite"):
    sys.path.insert(0, str(ROOT / part))

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=25,
                        help="RoboCasa scene seed; kept fixed for all poses")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--visibility-margin-px", type=int, default=20)
    parser.add_argument("--min-drawer-visible-px", type=int, default=250)
    parser.add_argument("--collision-tolerance-m", type=float, default=0.002)
    parser.add_argument("--out-root", type=Path, default=
                        ROOT / "实验结果" / "20261020_OpenDrawer同场景多出生点检查_v1")
    args = parser.parse_args()

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robosuite.utils.camera_utils import (
        get_camera_segmentation,
        get_camera_transform_matrix,
    )
    from base_handle_distance_hud import find_drawer_handle_site

    out = args.out_root / f"seed{args.seed:03d}"
    out.mkdir(parents=True, exist_ok=True)

    wrapped = gym.make(
        "robocasa/OpenDrawer", split="pretrain", seed=args.seed,
        camera_widths=args.image_size, camera_heights=args.image_size,
    )
    wrapped.unwrapped.env.hard_reset = False
    mapped_obs, _ = wrapped.reset(seed=args.seed)
    env = wrapped.unwrapped.env
    robot = env.robots[0]
    base_qpos_ids = np.asarray(robot._ref_base_joint_pos_indexes, dtype=np.int64)
    base_qvel_ids = np.asarray([
        env.sim.model.get_joint_qvel_addr(name)
        for name in robot.robot_model.base_joints
    ], dtype=np.int64)
    if base_qpos_ids.size != 3 or base_qvel_ids.size != 3:
        raise RuntimeError("Expected a planar mobile base with x/y/yaw joints")

    initial_state = env.sim.get_state().flatten().copy()
    anchor_qpos = np.asarray(env.sim.data.qpos[base_qpos_ids], dtype=np.float64).copy()
    handle_site = find_drawer_handle_site(env)
    base_site = robot.robot_model.base.correct_naming("center")
    base_site_id = env.sim.model.site_name2id(base_site)
    handle_xyz = np.asarray(env.sim.data.get_site_xpos(handle_site), dtype=np.float64).copy()
    anchor_base_xy = np.asarray(env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64).copy()
    anchor_distance = float(np.linalg.norm(anchor_base_xy - handle_xyz[:2]))
    outward = anchor_base_xy - handle_xyz[:2]
    outward /= np.linalg.norm(outward)

    camera_name = "robot0_agentview_left"
    size = args.image_size
    active_drawer_geom_ids = np.asarray([
        i for i, name in enumerate(env.sim.model.geom_names)
        if name and name.startswith(env.drawer.name)
    ], dtype=np.int64)
    if active_drawer_geom_ids.size == 0:
        raise RuntimeError("Could not identify active drawer geoms")

    robot_collision_geom_ids = {
        i for i, name in enumerate(env.sim.model.geom_names)
        if name
        and (name.startswith("robot0_") or name.startswith("mobilebase0_"))
        and ("collision" in name or name.endswith("_col") or "feet_col" in name)
    }

    def restore_scene():
        env.sim.set_state_from_flattened(initial_state.copy())
        env.sim.forward()

    def projection():
        transform = get_camera_transform_matrix(env.sim, camera_name, size, size)
        xyz = np.asarray(env.sim.data.get_site_xpos(handle_site), dtype=np.float64)
        projected = transform @ np.concatenate([xyz, [1.0]])
        if projected[2] <= 0:
            return None
        return np.asarray(projected[:2] / projected[2]), float(projected[2])

    def bad_contacts():
        bad = []
        for idx in range(env.sim.data.ncon):
            contact = env.sim.data.contact[idx]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            r1, r2 = g1 in robot_collision_geom_ids, g2 in robot_collision_geom_ids
            if r1 == r2 or float(contact.dist) >= -args.collision_tolerance_m:
                continue
            other = g2 if r1 else g1
            other_name = env.sim.model.geom_id2name(other) or ""
            if "floor" in other_name.lower():
                continue
            bad.append({
                "robot_geom": env.sim.model.geom_id2name(g1 if r1 else g2),
                "environment_geom": other_name,
                "penetration_m": float(-contact.dist),
            })
        return bad

    restore_scene()
    baseline_contacts = {
        (item["robot_geom"], item["environment_geom"]): item["penetration_m"]
        for item in bad_contacts()
    }

    def novel_bad_contacts():
        """Ignore contacts already present in RoboCasa's untouched default state."""
        novel = []
        for item in bad_contacts():
            key = (item["robot_geom"], item["environment_geom"])
            baseline = baseline_contacts.get(key)
            if baseline is None or item["penetration_m"] > baseline + args.collision_tolerance_m:
                novel.append(item)
        return novel

    def set_base_at(radius, bearing_deg, preserve_default_yaw=False):
        restore_scene()
        theta = np.deg2rad(bearing_deg)
        c, s = np.cos(theta), np.sin(theta)
        direction = np.array([
            c * outward[0] - s * outward[1],
            s * outward[0] + c * outward[1],
        ])
        target_xy = handle_xyz[:2] + radius * direction
        qpos = anchor_qpos.copy()
        qpos[:2] += target_xy - anchor_base_xy

        if preserve_default_yaw:
            yaw_candidates = [anchor_qpos[2]]
        else:
            yaw_candidates = anchor_qpos[2] + np.linspace(-np.pi, np.pi, 145, endpoint=False)
        best = None
        for yaw in yaw_candidates:
            qpos[2] = yaw
            env.sim.data.qpos[base_qpos_ids] = qpos
            env.sim.data.qvel[base_qvel_ids] = 0.0
            env.sim.forward()
            proj = projection()
            if proj is None:
                continue
            uv, depth = proj
            in_safe_frame = (
                args.visibility_margin_px <= uv[0] < size - args.visibility_margin_px and
                args.visibility_margin_px <= uv[1] < size - args.visibility_margin_px
            )
            if not preserve_default_yaw and not in_safe_frame:
                continue
            score = abs(float(uv[0]) - size / 2.0) + 0.15 * abs(float(uv[1]) - size / 2.0)
            if best is None or score < best[0]:
                best = (score, float(yaw), uv.copy(), depth)
        if best is None:
            return None

        qpos[2] = best[1]
        env.sim.data.qpos[base_qpos_ids] = qpos
        env.sim.data.qvel[base_qvel_ids] = 0.0
        env.sim.forward()
        # The translational base joints are not world x/y in every kitchen.
        # Estimate their local Jacobian and solve the world-space correction.
        for _ in range(4):
            env.sim.data.qpos[base_qpos_ids] = qpos
            env.sim.data.qvel[base_qvel_ids] = 0.0
            env.sim.forward()
            actual_xy = np.asarray(env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64).copy()
            error_xy = target_xy - actual_xy
            if np.linalg.norm(error_xy) < 1e-4:
                break
            eps = 1e-4
            jac = np.zeros((2, 2), dtype=np.float64)
            for joint_axis in range(2):
                probe = qpos.copy()
                probe[joint_axis] += eps
                env.sim.data.qpos[base_qpos_ids] = probe
                env.sim.forward()
                probe_xy = np.asarray(env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64)
                jac[:, joint_axis] = (probe_xy - actual_xy) / eps
            try:
                delta_q = np.linalg.solve(jac, error_xy)
            except np.linalg.LinAlgError:
                delta_q = np.linalg.lstsq(jac, error_xy, rcond=None)[0]
            if not np.isfinite(delta_q).all() or np.max(np.abs(delta_q)) > 2.0:
                return None
            qpos[:2] += delta_q
        env.sim.data.qpos[base_qpos_ids] = qpos
        env.sim.data.qvel[base_qvel_ids] = 0.0
        env.sim.forward()
        actual_xy = np.asarray(env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64)
        if abs(float(np.linalg.norm(actual_xy - handle_xyz[:2])) - radius) > 0.012:
            return None

        proj = projection()
        if proj is None:
            return None
        uv, depth = proj
        seg = get_camera_segmentation(env.sim, camera_name, size, size)[..., 1]
        visible = int(np.isin(seg, active_drawer_geom_ids).sum())
        contacts = novel_bad_contacts()
        required_visible = 0 if preserve_default_yaw else args.min_drawer_visible_px
        if visible < required_visible or contacts:
            return None
        return qpos.copy(), uv, depth, visible, contacts

    near = float(np.clip(anchor_distance - 0.12, 0.48, 0.62))
    middle = float(np.clip(anchor_distance + 0.06, 0.60, 0.74))
    far = float(np.clip(anchor_distance + 0.24, 0.78, 0.95))
    specifications = [
        ("P0_default", anchor_distance, 0.0, True),
        ("P1_near_center", near, 0.0, False),
        ("P2_far_center", far, 0.0, False),
        ("P3_lateral_mid", middle, 25.0, False),
        ("P4_lateral_wide", min(0.90, middle + 0.10), 45.0, False),
    ]

    rows, panels = [], []
    selected_qpos = []
    for index, (name, wanted_radius, wanted_bearing, keep_yaw) in enumerate(specifications):
        accepted = None
        # Small deterministic fallback grid handles counters or islands near a requested pose.
        bearing_centres = [wanted_bearing]
        if wanted_bearing != 0.0:
            bearing_centres.append(-wanted_bearing)
        for dr in (0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12, 0.16, -0.16):
            for bearing_centre in bearing_centres:
                for db in (0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0, 25.0, -25.0):
                    if keep_yaw and (dr != 0.0 or db != 0.0):
                        continue
                    candidate_bearing = bearing_centre + db
                    if bearing_centre > 0 and candidate_bearing <= 0:
                        continue
                    if bearing_centre < 0 and candidate_bearing >= 0:
                        continue
                    accepted = set_base_at(max(0.45, wanted_radius + dr), candidate_bearing,
                                           preserve_default_yaw=keep_yaw)
                    if accepted is not None:
                        actual_radius = max(0.45, wanted_radius + dr)
                        actual_bearing = candidate_bearing
                        break
                if accepted is not None:
                    break
            if accepted is not None:
                break
        if accepted is None:
            # Some kitchens cannot support a requested distance or side. Search
            # the full feasible grid and retain a pose that is geometrically
            # distinct from all poses already selected for this scene.
            fallback_radii = sorted(set([
                round(float(x), 3) for x in np.linspace(0.48, 0.95, 13)
            ] + [round(near, 3), round(middle, 3), round(far, 3), round(anchor_distance, 3)]))
            fallback_bearings = [0.0, 15.0, -15.0, 25.0, -25.0, 35.0, -35.0,
                                 45.0, -45.0, 55.0, -55.0, 65.0, -65.0]
            for fallback_radius in fallback_radii:
                for fallback_bearing in fallback_bearings:
                    candidate = set_base_at(fallback_radius, fallback_bearing, preserve_default_yaw=False)
                    if candidate is None:
                        continue
                    cq = candidate[0]
                    if any(np.linalg.norm(cq[:2] - old[:2]) < 0.055 and
                           abs(np.arctan2(np.sin(cq[2]-old[2]), np.cos(cq[2]-old[2]))) < 0.12
                           for old in selected_qpos):
                        continue
                    accepted = candidate
                    actual_radius = fallback_radius
                    actual_bearing = fallback_bearing
                    break
                if accepted is not None:
                    break
        if accepted is None:
            raise RuntimeError(f"No fifth diverse safe visible pose found for {name}")

        qpos, uv, depth, visible, contacts = accepted
        selected_qpos.append(qpos.copy())
        raw = env._get_observations(force_update=True)
        mapped_obs = wrapped.unwrapped.get_observation(raw)
        image = np.asarray(mapped_obs["video.robot0_agentview_left"], dtype=np.uint8)
        if image.shape[:2] != (size, size):
            raise RuntimeError(f"Unexpected image shape: {image.shape}")
        Image.fromarray(image).save(out / f"{index}_{name}_main.png")
        np.save(out / f"{index}_{name}_base_qpos.npy", qpos)

        base_xy = np.asarray(env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64)
        handle_now = np.asarray(env.sim.data.get_site_xpos(handle_site), dtype=np.float64)
        horizontal = float(np.linalg.norm(base_xy - handle_now[:2]))
        distance_3d = float(np.linalg.norm(
            np.r_[base_xy, env.sim.data.site_xpos[base_site_id][2]] - handle_now))
        row = {
            "pose": name,
            "requested_distance_m": float(wanted_radius),
            "actual_horizontal_distance_m": horizontal,
            "base_handle_3d_distance_m": distance_3d,
            "requested_bearing_deg": float(wanted_bearing),
            "actual_sample_bearing_deg": float(actual_bearing),
            "base_x_m": float(base_xy[0]),
            "base_y_m": float(base_xy[1]),
            "base_yaw_rad": float(qpos[2]),
            "handle_u_px": float(uv[0]),
            "handle_v_px": float(uv[1]),
            "handle_camera_depth_m": float(depth),
            "drawer_visible_pixels": visible,
            "collision_free_over_2mm": len(contacts) == 0,
            "scene_seed": args.seed,
            "task": str(mapped_obs["annotation.human.task_description"]),
        }
        rows.append(row)
        panel = Image.fromarray(image).resize((384, 384))
        canvas = Image.new("RGB", (384, 430), "white")
        canvas.paste(panel, (0, 46))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"{name}  d={horizontal:.3f}m  bearing={actual_bearing:+.0f}deg", fill="black")
        draw.text((8, 24), f"handle=({uv[0]:.0f},{uv[1]:.0f}) visible_px={visible} collision=PASS", fill="black")
        panels.append(canvas)

    with (out / "poses.json").open("w", encoding="utf-8") as f:
        json.dump({
            "fixed_scene_seed": args.seed,
            "same_scene_guarantee": "one env reset; full initial simulator state restored before each base pose",
            "task": rows[0]["task"],
            "poses": rows,
        }, f, ensure_ascii=False, indent=2)
    with (out / "poses.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    montage = Image.new("RGB", (384 * len(panels), 430), (230, 230, 230))
    for i, panel in enumerate(panels):
        montage.paste(panel, (384 * i, 0))
    montage.save(out / "all_poses_montage.png")
    print(json.dumps({"status": "complete", "output": str(out), "poses": rows},
                     ensure_ascii=False, indent=2), flush=True)
    wrapped.close()


if __name__ == "__main__":
    main()



