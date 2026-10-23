#!/usr/bin/env python3
"""Continuous manual / policy control; viewer HUD is excluded from observations."""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import os
import queue
import sys
import threading
import time

ROOT = Path(os.environ.get('ROBOCASA_DATA_ROOT', Path(__file__).resolve().parents[2]))
for p in ('comparison_models/src','third_party/robocasa','third_party/robosuite','third_party/openpi/packages/openpi-client/src'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--min-distance', type=float, default=.50)
    ap.add_argument('--max-distance', type=float, default=.65)
    ap.add_argument('--position-sensitivity', type=float, default=1.)
    ap.add_argument('--rotation-sensitivity', type=float, default=1.)
    ap.add_argument('--start-distance-min', type=float, default=.60,
                    help='random initial horizontal base-to-handle distance in metres')
    ap.add_argument('--start-distance-max', type=float, default=1.20)
    ap.add_argument('--start-bearing-deg', type=float, default=70.,
                    help='half width of the random front-side sector around the drawer')
    ap.add_argument('--side-min-bearing-deg', type=float, default=35.,
                    help='minimum absolute bearing for a clearly lateral start')
    ap.add_argument('--center-bearing-deg', type=float, default=12.,
                    help='maximum absolute bearing for a centre start')
    ap.add_argument('--bearing-bins', type=int, default=7,
                    help='number of stratified left/right bearing bands')
    ap.add_argument('--distance-bins', type=int, default=4,
                    help='number of stratified distance bands')
    ap.add_argument('--start-yaw-noise-deg', type=float, default=15.,
                    help='random heading error around facing the handle')
    ap.add_argument('--placement-seed', type=int, default=None,
                    help='optional reproducible seed for the sequence of robot placements')
    ap.add_argument('--visibility-margin-px', type=int, default=24,
                    help='minimum image-border margin for the projected handle')
    ap.add_argument('--min-drawer-visible-px', type=int, default=300,
                    help='minimum active-drawer segmentation pixels in the main camera')
    ap.add_argument('--placement-max-tries', type=int, default=60)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--replan-steps', type=int, default=5)
    ap.add_argument('--attempt-max-steps', type=int, default=600,
                    help='warn after this many policy steps; the operator still decides success/failure')
    ap.add_argument('--enable-right-image', action='store_true')
    ap.add_argument('--allow-policy-base-motion', action='store_true',
                    help='allow PI policy to move the mobile base; disabled by default')
    ap.add_argument('--smoke-steps', type=int, default=0)
    ap.add_argument('--smoke-policy', action='store_true')
    ap.add_argument('--out-root', type=Path, default=ROOT/'实验结果'/'数据采集')
    a = ap.parse_args()
    if not 1 <= a.replan_steps <= 50: ap.error('replan-steps must be 1..50')
    if a.attempt_max_steps < 1: ap.error('attempt-max-steps must be positive')
    if not 0 < a.start_distance_min < a.start_distance_max:
        ap.error('start-distance-min must be positive and smaller than start-distance-max')
    if not 0 <= a.start_bearing_deg <= 75: ap.error('start-bearing-deg must be 0..75')
    if not 0 <= a.center_bearing_deg < a.side_min_bearing_deg < a.start_bearing_deg:
        ap.error('require 0 <= center-bearing-deg < side-min-bearing-deg < start-bearing-deg')
    if a.bearing_bins < 3: ap.error('bearing-bins must be at least 3')
    if a.distance_bins < 2: ap.error('distance-bins must be at least 2')
    if not 0 <= a.start_yaw_noise_deg <= 90: ap.error('start-yaw-noise-deg must be 0..90')
    if not 0 <= a.visibility_margin_px < 128: ap.error('visibility-margin-px must be 0..127')
    if a.min_drawer_visible_px < 1: ap.error('min-drawer-visible-px must be positive')
    if a.placement_max_tries < 1: ap.error('placement-max-tries must be positive')
    import gymnasium as gym
    import robocasa
    import mujoco.viewer
    from robosuite.devices import Keyboard
    from robosuite.utils.camera_utils import get_camera_segmentation, get_camera_transform_matrix
    from robocasa.utils.env_utils import convert_action
    from openpi_client import image_tools, msgpack_numpy
    from websockets.sync.client import connect
    from base_handle_distance_hud import find_drawer_handle_site, install_distance_hud

    # Use the same Gym creation/reset path and observation mapping as evaluation.
    g = gym.make('robocasa/OpenDrawer', split='pretrain', seed=a.seed,
                 camera_widths=256, camera_heights=256)
    # This RoboCasa Gym wrapper does not expose hard_reset as a constructor
    # argument. Set it on the underlying robosuite environment before the
    # explicit reset so the passive viewer keeps valid model/data pointers.
    g.unwrapped.env.hard_reset = False
    obs, _ = g.reset(seed=a.seed)
    env = g.unwrapped.env
    task_prompt = str(obs['annotation.human.task_description'])
    drawer_side = str(env.drawer_side)
    if drawer_side not in ('left', 'right') or 'drawer' not in task_prompt.lower():
        g.close()
        raise RuntimeError(
            f'OpenDrawer seed {a.seed} produced an unsupported target: '
            f'prompt={task_prompt!r}, drawer_side={drawer_side!r}.')
    drawer_label_cn = '左侧抽屉' if drawer_side == 'left' else '右侧抽屉'
    robot = env.robots[0]
    base_qpos_ids = np.asarray(robot._ref_base_joint_pos_indexes, dtype=np.int64)
    base_qvel_ids = np.asarray([
        env.sim.model.get_joint_qvel_addr(name)
        for name in robot.robot_model.base_joints
    ], dtype=np.int64)
    if base_qpos_ids.size != 3 or base_qvel_ids.size != 3:
        g.close()
        raise RuntimeError(f'Expected planar base x/y/yaw joints, got {base_qpos_ids.size}.')
    anchor_base_qpos = np.asarray(env.sim.data.qpos[base_qpos_ids], dtype=np.float64).copy()
    handle_site_name = find_drawer_handle_site(env)
    base_site_name = robot.robot_model.base.correct_naming('center')
    base_site_id = env.sim.model.site_name2id(base_site_name)
    anchor_base_xy = np.asarray(env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64).copy()
    anchor_handle_xy = np.asarray(env.sim.data.get_site_xpos(handle_site_name)[:2], dtype=np.float64).copy()
    outward = anchor_base_xy - anchor_handle_xy
    outward /= np.linalg.norm(outward)
    placement_rng = np.random.default_rng(a.placement_seed)
    a.out_root.mkdir(parents=True, exist_ok=True)
    placement_cycle_path = a.out_root/f'.placement_cycle_v2_OpenDrawer_seed{a.seed}.txt'
    try:
        placement_cycle_index = int(placement_cycle_path.read_text().strip())
    except (FileNotFoundError, ValueError):
        placement_cycle_index = 0
    previous_start_xy = None
    current_placement = None
    camera_name = 'robot0_agentview_left'
    image_size = 256
    active_drawer_geom_ids = np.asarray([
        i for i, name in enumerate(env.sim.model.geom_names)
        if name and name.startswith(env.drawer.name)
    ], dtype=np.int64)
    if active_drawer_geom_ids.size == 0:
        g.close()
        raise RuntimeError('Could not identify active drawer geometry for visibility checking.')
    robot_collision_geom_ids = {
        i for i, name in enumerate(env.sim.model.geom_names)
        if name and (
            name.startswith('robot0_') or name.startswith('mobilebase0_')
        ) and (
            'collision' in name or name.endswith('_col') or 'feet_col' in name
        )
    }

    def penetrating_environment_contacts(tolerance=-2e-3):
        """Return robot-vs-environment penetrations at the current pose."""
        bad = []
        for contact_index in range(env.sim.data.ncon):
            contact = env.sim.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            robot1 = geom1 in robot_collision_geom_ids
            robot2 = geom2 in robot_collision_geom_ids
            if robot1 == robot2 or float(contact.dist) >= tolerance:
                continue
            other = geom2 if robot1 else geom1
            other_name = env.sim.model.geom_id2name(other) or ''
            if 'floor' in other_name.lower():
                continue
            bad.append(dict(
                robot_geom=env.sim.model.geom_id2name(geom1 if robot1 else geom2),
                environment_geom=other_name,
                penetration_m=float(-contact.dist),
            ))
        return bad

    def next_stratified_placement_cell():
        """Cycle a 2-D bearing/distance grid, persisting across restarts."""
        nonlocal placement_cycle_index
        cycle_index = placement_cycle_index
        cell_count = a.bearing_bins*a.distance_bins
        within_cycle = cycle_index % cell_count
        # Sweep all bearing bands early instead of producing several visually
        # similar positions in a row. This mapping still covers every bearing x
        # distance cell exactly once per full cycle.
        bearing_bin_index = within_cycle % a.bearing_bins
        distance_round = within_cycle // a.bearing_bins
        distance_bin_index = (distance_round+bearing_bin_index) % a.distance_bins
        placement_cycle_index += 1
        placement_cycle_path.write_text(str(placement_cycle_index), encoding='utf-8')
        return bearing_bin_index, distance_bin_index, cycle_index

    def handle_projection():
        transform = get_camera_transform_matrix(env.sim, camera_name, image_size, image_size)
        handle_xyz = np.asarray(env.sim.data.get_site_xpos(handle_site_name), dtype=np.float64)
        projected = transform @ np.concatenate([handle_xyz, [1.]])
        if projected[2] <= 0:
            return None
        return np.asarray(projected[:2]/projected[2], dtype=np.float64), float(projected[2])

    def randomize_robot_base():
        """Sample a balanced pose whose main camera really sees the active drawer."""
        nonlocal obs, previous_start_xy, current_placement
        bearing_bin_index, distance_bin_index, cycle_index = next_stratified_placement_cell()
        bearing_edges = np.linspace(-a.start_bearing_deg, a.start_bearing_deg,
                                    a.bearing_bins+1)
        bearing_low = float(bearing_edges[bearing_bin_index])
        bearing_high = float(bearing_edges[bearing_bin_index+1])
        bearing_mid = .5*(bearing_low+bearing_high)
        if bearing_mid <= -a.side_min_bearing_deg:
            drawer_relative_region = 'drawer_far_right_side'
        elif bearing_mid < -a.center_bearing_deg:
            drawer_relative_region = 'drawer_right_side'
        elif bearing_mid <= a.center_bearing_deg:
            drawer_relative_region = 'drawer_center'
        elif bearing_mid < a.side_min_bearing_deg:
            drawer_relative_region = 'drawer_left_side'
        else:
            drawer_relative_region = 'drawer_far_left_side'
        sector = ('negative_side' if bearing_mid < -a.center_bearing_deg else
                  'positive_side' if bearing_mid > a.center_bearing_deg else 'center')
        distance_edges = np.linspace(a.start_distance_min, a.start_distance_max,
                                     a.distance_bins+1)
        radius_low = float(distance_edges[distance_bin_index])
        radius_high = float(distance_edges[distance_bin_index+1])
        accepted = None
        rejected = dict(repeat=0, projection=0, distance=0,
                        collision=0, visibility=0)
        collision_examples = []
        relax_after = max(3, a.placement_max_tries//4)
        feasible_side_abs = max(
            a.center_bearing_deg+2., min(48., a.start_bearing_deg))
        feasible_side_inner = min(
            a.side_min_bearing_deg, feasible_side_abs-1.)
        for placement_try in range(1, a.placement_max_tries+1):
            # Prefer the scheduled bearing x distance cell.  Some cells do not
            # physically exist in seed19 because a counter or wall occupies the
            # corresponding floor area.  In the second half, search the complete
            # requested 1--2 m corridor rather than aborting on an impossible cell.
            global_fallback = placement_try > a.placement_max_tries//2
            boundary_fallback = placement_try > 3*a.placement_max_tries//4
            sample_radius_low = a.start_distance_min if global_fallback else radius_low
            sample_radius_high = a.start_distance_max if global_fallback else radius_high
            if boundary_fallback:
                # seed19's target drawer is close to the right room wall.  The
                # physically valid part of the requested range is near 1 m and
                # may require approaching on a wider arc.
                sample_radius_low = max(.55, a.start_distance_min - .05)
                sample_radius_high = min(a.start_distance_max, .85)
            radius = float(placement_rng.uniform(sample_radius_low, sample_radius_high))
            bearing_range_relaxed = placement_try > relax_after
            if boundary_fallback:
                actual_bearing_low = -180.
                actual_bearing_high = 180.
            elif global_fallback:
                actual_bearing_low = -a.start_bearing_deg
                actual_bearing_high = a.start_bearing_deg
            elif not bearing_range_relaxed:
                actual_bearing_low, actual_bearing_high = bearing_low, bearing_high
            elif sector == 'negative_side':
                actual_bearing_low = -feasible_side_abs
                actual_bearing_high = -feasible_side_inner
            elif sector == 'positive_side':
                actual_bearing_low = feasible_side_inner
                actual_bearing_high = feasible_side_abs
            else:
                actual_bearing_low = -a.center_bearing_deg
                actual_bearing_high = a.center_bearing_deg
            bearing_deg = float(placement_rng.uniform(
                actual_bearing_low, actual_bearing_high))
            bearing = np.deg2rad(bearing_deg)
            c, s = np.cos(bearing), np.sin(bearing)
            direction = np.array([c*outward[0]-s*outward[1],
                                  s*outward[0]+c*outward[1]])
            target_xy = anchor_handle_xy + radius*direction
            if previous_start_xy is not None and np.linalg.norm(target_xy-previous_start_xy) < .08:
                rejected['repeat'] += 1
                continue

            # The left agent camera is offset and not aligned with the base yaw.
            # Search yaw in simulation instead of assuming an analytic sign.
            desired_view_offset_deg = float(placement_rng.uniform(
                -a.start_yaw_noise_deg, a.start_yaw_noise_deg))
            fovy = float(env.sim.model.cam_fovy[env.sim.model.camera_name2id(camera_name)])
            focal = .5*image_size/np.tan(np.deg2rad(fovy)/2.)
            desired_u = image_size/2. + focal*np.tan(np.deg2rad(desired_view_offset_deg))
            qpos = anchor_base_qpos.copy()
            qpos[:2] += target_xy-anchor_base_xy
            best = None
            for yaw_delta in np.linspace(-np.pi, np.pi, 73, endpoint=False):
                qpos[2] = anchor_base_qpos[2]+yaw_delta
                env.sim.data.qpos[base_qpos_ids] = qpos
                env.sim.data.qvel[base_qvel_ids] = 0.
                env.sim.forward()
                projection = handle_projection()
                if projection is None:
                    continue
                uv, depth = projection
                margin = a.visibility_margin_px
                if not (margin <= uv[0] < image_size-margin and
                        margin <= uv[1] < image_size-margin):
                    continue
                score = abs(float(uv[0])-desired_u)
                if best is None or score < best[0]:
                    best = (score, float(yaw_delta), uv.copy(), depth)
            if best is None:
                rejected['projection'] += 1
                continue
            _, yaw_delta, uv, depth = best
            qpos[2] = anchor_base_qpos[2]+yaw_delta
            env.sim.data.qpos[base_qpos_ids] = qpos
            env.sim.data.qvel[base_qvel_ids] = 0.
            env.sim.forward()
            # Base joint coordinates are not guaranteed to be world x/y in
            # every RoboCasa layout. Estimate the local 2x2 Jacobian from the
            # two translational joints to the named base-centre site and solve
            # for the world-space correction. This avoids the large invalid
            # qpos values produced by assuming an identity mapping.
            for _ in range(3):
                env.sim.data.qpos[base_qpos_ids] = qpos
                env.sim.data.qvel[base_qvel_ids] = 0.
                env.sim.forward()
                actual_xy = np.asarray(
                    env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64).copy()
                error_xy = target_xy-actual_xy
                if np.linalg.norm(error_xy) < 1e-4:
                    break
                eps = 1e-4
                jac = np.zeros((2, 2), dtype=np.float64)
                for joint_axis in range(2):
                    probe = qpos.copy()
                    probe[joint_axis] += eps
                    env.sim.data.qpos[base_qpos_ids] = probe
                    env.sim.forward()
                    probe_xy = np.asarray(
                        env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64)
                    jac[:, joint_axis] = (probe_xy-actual_xy)/eps
                try:
                    delta_q = np.linalg.solve(jac, error_xy)
                except np.linalg.LinAlgError:
                    delta_q = np.linalg.lstsq(jac, error_xy, rcond=None)[0]
                if not np.isfinite(delta_q).all() or np.max(np.abs(delta_q)) > 2.0:
                    break
                qpos[:2] += delta_q
            env.sim.data.qpos[base_qpos_ids] = qpos
            env.sim.data.qvel[base_qvel_ids] = 0.
            env.sim.forward()
            actual_xy = np.asarray(
                env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64).copy()
            actual_radius = float(np.linalg.norm(actual_xy-anchor_handle_xy))
            valid_radius_low = sample_radius_low
            valid_radius_high = sample_radius_high
            if not valid_radius_low-0.01 <= actual_radius <= valid_radius_high+0.01:
                rejected['distance'] += 1
                continue
            penetrations = penetrating_environment_contacts()
            if penetrations:
                rejected['collision'] += 1
                if len(collision_examples) < 8:
                    collision_examples.extend(penetrations[:2])
                continue
            projection = handle_projection()
            if projection is None:
                rejected['projection'] += 1
                continue
            uv, depth = projection
            margin = a.visibility_margin_px
            if not (margin <= uv[0] < image_size-margin and
                    margin <= uv[1] < image_size-margin):
                rejected['projection'] += 1
                continue
            segmentation = get_camera_segmentation(
                env.sim, camera_name, image_size, image_size)[..., 1]
            visible_pixels = int(np.isin(segmentation, active_drawer_geom_ids).sum())
            if visible_pixels < a.min_drawer_visible_px:
                rejected['visibility'] += 1
                continue
            accepted = (qpos.copy(), radius, bearing_deg, desired_view_offset_deg,
                        uv, depth, visible_pixels, placement_try,
                        bearing_range_relaxed, global_fallback, boundary_fallback,
                        [actual_bearing_low, actual_bearing_high])
            break
        if accepted is None:
            raise RuntimeError(
                f'Could not find a visible {sector} base pose after '
                f'{a.placement_max_tries} attempts; rejection counts={rejected}; '
                f'collision examples={collision_examples}.')
        (qpos, radius, bearing_deg, desired_view_offset_deg, uv, depth,
         visible_pixels, placement_try, bearing_range_relaxed,
         global_fallback, boundary_fallback,
         actual_bearing_range_deg) = accepted
        raw_obs = env._get_observations(force_update=True)
        obs = g.unwrapped.get_observation(raw_obs)
        actual_base_xy = np.asarray(env.sim.data.site_xpos[base_site_id][:2],
                                    dtype=np.float64).copy()
        previous_start_xy = actual_base_xy
        current_placement = dict(
            base_xy_m=actual_base_xy.tolist(),
            handle_xy_m=np.asarray(env.sim.data.get_site_xpos(handle_site_name)[:2],
                                   dtype=np.float64).tolist(),
            horizontal_distance_m=float(np.linalg.norm(
                actual_base_xy-env.sim.data.get_site_xpos(handle_site_name)[:2])),
            sampling_sector=sector, drawer_relative_region=drawer_relative_region,
            placement_cycle_index=cycle_index,
            bearing_bin=f'{bearing_bin_index+1}_of_{a.bearing_bins}',
            requested_bearing_bin_range_deg=[bearing_low, bearing_high],
            actual_bearing_sampling_range_deg=actual_bearing_range_deg,
            bearing_range_relaxed_for_visibility=bearing_range_relaxed,
            global_safe_corridor_fallback=global_fallback,
            boundary_safe_arc_fallback=boundary_fallback,
            distance_bin=f'{distance_bin_index+1}_of_{a.distance_bins}',
            distance_bin_range_m=[radius_low, radius_high], bearing_deg=bearing_deg,
            desired_view_offset_deg=desired_view_offset_deg,
            handle_pixel_uv=uv.tolist(), handle_camera_depth_m=depth,
            drawer_visible_pixels=visible_pixels,
            visibility_margin_px=a.visibility_margin_px,
            sampling_attempts=placement_try,
            collision_check='no non-floor robot-environment penetration deeper than 2 mm',
            base_qpos=qpos.tolist())
        return current_placement

    randomize_robot_base()
    rows, event_rows, attempts = [], [], []
    def save_episode():
        """Persist the current episode only after an explicit operator command."""
        result_kind = ('A失败回退_B纠正成功配对'
                       if pair_complete else '人工判定成功_非配对')
        out = a.out_root / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') +
                            f'_{drawer_label_cn}_seed{a.seed}_{result_kind}_v6')
        out.mkdir(parents=True, exist_ok=False)
        attempt_meta = []
        for item in attempts:
            image_bundle = item.get('handoff_images')
            position_label = 'A_bad' if item['status'] == 'failed' else 'B_good'
            image_dir_name = f'attempt_{item["index"]:02d}_{position_label}'
            meta = {k: v for k, v in item.items()
                    if k not in ('checkpoint_state', 'handoff_images')}
            meta['position_label'] = position_label
            meta['image_directory'] = image_dir_name
            attempt_meta.append(meta)
            np.save(out/f'attempt_{item["index"]:02d}_handoff_state.npy',
                    item['checkpoint_state'])
            if image_bundle is None:
                raise RuntimeError(f'Missing handoff images for attempt {item["index"]}.')
            image_dir = out/image_dir_name
            image_dir.mkdir()
            exact_arrays = {}
            for image_kind in ('raw', 'policy_input'):
                for camera_short_name, image in image_bundle[image_kind].items():
                    array = np.asarray(image, dtype=np.uint8)
                    exact_arrays[f'{image_kind}_{camera_short_name}'] = array
                    Image.fromarray(array).save(
                        image_dir/f'{image_kind}_{camera_short_name}.png')
            np.savez_compressed(image_dir/'handoff_images_exact.npz', **exact_arrays)
        (out/'model.xml').write_text(env.sim.model.get_xml(), encoding='utf-8')
        (out/'manifest.json').write_text(json.dumps(dict(
            seed=a.seed, task_prompt=task_prompt, drawer_side=drawer_side,
            ep_meta=env.get_ep_meta(),
            right_image_disabled=not a.enable_right_image, replan_steps=a.replan_steps,
            policy_base_frozen=not a.allow_policy_base_motion,
            steps=len(rows), save_trigger='operator_pressed_s_marks_success_and_saves',
            outcome_authority='operator_only; RoboCasa success is recorded for diagnostics but never decides the label',
            pair_complete=pair_complete,
            pair_definition='one or more operator-labeled failed policy attempts restored to their handoff states, followed by manual correction and an operator-labeled successful policy attempt',
            attempts=attempt_meta,
            random_initial_base=dict(
                placement_seed=a.placement_seed,
                distance_range_m=[a.start_distance_min, a.start_distance_max],
                bearing_range_deg=[-a.start_bearing_deg, a.start_bearing_deg],
                center_bearing_range_deg=[-a.center_bearing_deg, a.center_bearing_deg],
                lateral_bearing_abs_range_deg=[a.side_min_bearing_deg,
                                               a.start_bearing_deg],
                desired_view_offset_range_deg=[-a.start_yaw_noise_deg, a.start_yaw_noise_deg],
                stratified_sampling=dict(
                    cells=a.bearing_bins*a.distance_bins,
                    bearing_bins=a.bearing_bins,
                    distance_bins=a.distance_bins,
                    persistent_cycle_file=str(placement_cycle_path),
                    guarantee='every full cycle covers each bearing x distance cell exactly once, including across program restarts; exact pose inside each cell is random'),
                main_camera=camera_name,
                visibility_margin_px=a.visibility_margin_px,
                min_drawer_visible_px=a.min_drawer_visible_px,
                sampled=current_placement),
            handoff_images=dict(
                cameras=['main', 'wrist', 'right'],
                raw_resolution=[256, 256], policy_input_resolution=[224, 224],
                format='lossless PNG plus exact uint8 arrays in handoff_images_exact.npz',
                capture_time='the instant V is pressed, before the first policy action',
                right_policy_input_masked=not a.enable_right_image),
            recording='sim state before action, raw robosuite action, controller source, pair phase, and exact A/B handoff images; failed branches contain an intentional state restore described by events and attempt metadata',
            policy_server=f'{a.host}:{a.port}'), ensure_ascii=False, indent=2), encoding='utf-8')
        with (out/'events.jsonl').open('w', encoding='utf-8') as f:
            for item in event_rows:
                f.write(json.dumps(item, ensure_ascii=False)+'\n')
        for chunk, begin in enumerate(range(0, len(rows), 100)):
            part = rows[begin:begin+100]
            np.savez_compressed(out/f'trajectory_{chunk:05d}.npz',
                states=np.stack([r[0] for r in part]), actions=np.stack([r[1] for r in part]),
                sources=np.array([r[2] for r in part]), success=np.array([r[3] for r in part]),
                phases=np.array([r[4] for r in part]))
        np.save(out/'final_state.npy', env.sim.get_state().flatten())
        return out
    commands = queue.Queue()
    class ControlKeyboard(Keyboard):
        def on_press(self, key):
            if getattr(self,'manual',True): super().on_press(key)
        def on_release(self, key):
            char = getattr(key,'char',None)
            char = char.lower() if char else None
            if char in ('v','s','x','d','q'): commands.put(char); return
            if getattr(self,'manual',True): super().on_release(key)
    device = ControlKeyboard(env=env, pos_sensitivity=a.position_sensitivity,
                             rot_sensitivity=a.rotation_sensitivity)
    device.start_control()
    hud = install_distance_hud(env, recommended_min_m=a.min_distance, recommended_max_m=a.max_distance)
    viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data,
                                          show_left_ui=False, show_right_ui=False)
    # Use the robot's real third-person camera. It is guaranteed to look into
    # the workspace and follows the base while the operator moves it.
    viewer.opt.geomgroup[0] = 0
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = env.sim.model.camera_name2id("robot0_agentview_left")
    mode, epoch, pending, ticks = 'manual', 0, False, 0
    phase, active_attempt, pair_complete = 'initial_to_A', None, False
    plan, replies = deque(), queue.Queue()
    def event(kind, **kw):
        event_rows.append(dict(event=kind, step=ticks, time=time.time(), **kw))
    def switch(next_mode):
        nonlocal mode,epoch
        mode=next_mode; epoch+=1; plan.clear()
        device.manual = mode == 'manual'
        if device.manual:
            device.start_control()
            # Synchronize gripper command with actual current opening.
            grip = np.asarray(obs['state.gripper_qpos'])
            device.grasp_states[0][device.active_arm_index] = bool(np.sum(np.abs(grip)) < .04)
        event('mode',mode=mode)
        print('控制模式：'+('人工' if mode=='manual' else '模型'),flush=True)
    def current_base_pose():
        base_xy = np.asarray(env.sim.data.site_xpos[base_site_id][:2], dtype=np.float64)
        handle_xy = np.asarray(env.sim.data.get_site_xpos(handle_site_name)[:2], dtype=np.float64)
        return dict(base_xy_m=base_xy.tolist(), handle_xy_m=handle_xy.tolist(),
                    horizontal_distance_m=float(np.linalg.norm(base_xy-handle_xy)),
                    base_qpos=np.asarray(env.sim.data.qpos[base_qpos_ids],
                                         dtype=np.float64).tolist())
    def capture_handoff_images():
        """Freeze raw and actual policy-input images at a V-key handoff."""
        camera_names = dict(main='agentview_left', wrist='eye_in_hand',
                            right='agentview_right')
        raw_images = {
            short_name: np.ascontiguousarray(
                obs['video.robot0_'+camera_name]).copy()
            for short_name, camera_name in camera_names.items()
        }
        policy_images = {
            short_name: np.ascontiguousarray(image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, 224, 224))).copy()
            for short_name, image in raw_images.items()
        }
        if not a.enable_right_image:
            policy_images['right'] = np.zeros_like(policy_images['right'])
        return dict(raw=raw_images, policy_input=policy_images)
    def begin_policy_attempt():
        nonlocal active_attempt, phase
        if active_attempt is not None or mode == 'policy':
            print('已有模型尝试正在进行。', flush=True); return
        if pair_complete:
            print('配对已经完成但尚未保存；请按 s 保存，或按 d 丢弃。', flush=True); return
        # Remove residual manual base velocity before taking the handoff
        # checkpoint. During policy control this pose is held exactly fixed.
        if not a.allow_policy_base_motion:
            env.sim.data.qvel[base_qvel_ids] = 0.
            env.sim.forward()
        index = len(attempts)
        active_attempt = dict(index=index, start_row=len(rows),
                              checkpoint_state=env.sim.get_state().flatten().copy(),
                              handoff_pose=current_base_pose(),
                              handoff_images=capture_handoff_images(),
                              policy_base_frozen=not a.allow_policy_base_motion,
                              frozen_base_qpos=np.asarray(
                                  env.sim.data.qpos[base_qpos_ids],
                                  dtype=np.float64).tolist(),
                              policy_requested_base_motion_max_abs=[0., 0., 0.],
                              policy_requested_base_mode_steps=0,
                              status='running')
        phase = f'attempt_{index:02d}_policy'
        event('attempt_started', attempt=index,
              handoff_pose=active_attempt['handoff_pose'])
        print(f'开始模型尝试 {index}：接管距离 '
              f'{active_attempt["handoff_pose"]["horizontal_distance_m"]:.3f} m；'
              f'底盘{" 已锁定" if not a.allow_policy_base_motion else "允许模型控制"}。',
              flush=True)
        switch('policy')
    def restore_failed_attempt(reason):
        nonlocal obs, active_attempt, phase, pending
        if mode != 'policy' or active_attempt is None:
            print('当前没有模型尝试；只有模型运行时才能标记失败。', flush=True); return
        item = active_attempt
        item['end_row'] = len(rows)
        item['policy_steps'] = item['end_row']-item['start_row']
        item['status'] = 'failed'
        item['failure_reason'] = reason
        attempts.append(item)
        event('attempt_failed', attempt=item['index'], reason=reason,
              start_row=item['start_row'], end_row=item['end_row'])
        # Invalidate in-flight replies before restoring the exact handoff state.
        switch('manual')
        pending = False
        with viewer.lock():
            env.sim.set_state_from_flattened(item['checkpoint_state'])
            env.sim.forward()
            env.robots[0].composite_controller.update_state()
            env.robots[0].composite_controller.reset()
            raw_obs = env._get_observations(force_update=True)
            obs = g.unwrapped.get_observation(raw_obs)
        device.start_control()
        grip = np.asarray(obs['state.gripper_qpos'])
        device.grasp_states[0][device.active_arm_index] = bool(np.sum(np.abs(grip)) < .04)
        active_attempt = None
        phase = f'correction_after_attempt_{item["index"]:02d}'
        event('handoff_state_restored', attempt=item['index'],
              restored_pose=current_base_pose())
        hud.update(force=True)
        print(f'尝试 {item["index"]} 已标记失败并恢复到接管点 A；'
              '现在人工移动到更好的位置 B，再按 v。', flush=True)
    def complete_policy_attempt(success_source):
        nonlocal active_attempt, phase, pair_complete
        if mode != 'policy' or active_attempt is None:
            print('当前没有模型尝试，不能标记成功。', flush=True); return
        item = active_attempt
        item['end_row'] = len(rows)
        item['policy_steps'] = item['end_row']-item['start_row']
        item['status'] = 'success'
        item['success_source'] = success_source
        item['success_pose'] = current_base_pose()
        # The success array is an operator label. Mark the final executed action
        # of this attempt as the terminal success; do not use RoboCasa's drawer
        # completion threshold as ground truth.
        if item['end_row'] <= item['start_row']:
            print('模型还没有执行动作，不能标记成功。', flush=True)
            return False
        state, action, source, _, row_phase = rows[item['end_row']-1]
        rows[item['end_row']-1] = (state, action, source, True, row_phase)
        attempts.append(item)
        active_attempt = None
        pair_complete = any(x['status']=='failed' for x in attempts[:-1])
        phase = 'pair_complete' if pair_complete else 'unpaired_success'
        event('attempt_success', attempt=item['index'],
              start_row=item['start_row'], end_row=item['end_row'],
              success_source=success_source, pair_complete=pair_complete)
        switch('manual')
        if pair_complete:
            print('B 点已由操作者判定成功。', flush=True)
        else:
            print('第一个接管点已由操作者判定成功；将作为非配对成功样本保存。', flush=True)
        return True
    def reset_episode(reason):
        """Return to the same deterministic scene and invalidate old policy replies."""
        nonlocal obs, mode, epoch, pending, ticks, current_placement
        nonlocal phase, active_attempt, pair_complete
        epoch += 1
        pending = False
        plan.clear()
        mode = 'manual'
        device.manual = True
        rows.clear()
        event_rows.clear()
        attempts.clear()
        ticks = 0
        phase = 'initial_to_A'
        active_attempt = None
        pair_complete = False
        with viewer.lock():
            obs, _ = g.reset(seed=a.seed)
            if (str(obs['annotation.human.task_description']) != task_prompt or
                    str(env.drawer_side) != drawer_side):
                raise RuntimeError('Reset changed the target drawer or task prompt; stopped.')
            current_placement = randomize_robot_base()
        device.start_control()
        grip = np.asarray(obs['state.gripper_qpos'])
        device.grasp_states[0][device.active_arm_index] = bool(np.sum(np.abs(grip)) < .04)
        event('start', reset_reason=reason, random_initial_base=current_placement)
        hud.update(force=True)
        print(f'场景已重置并随机放置机器人：水平距离 '
              f'{current_placement["horizontal_distance_m"]:.3f} m，'
              f'相对区域 {current_placement["drawer_relative_region"]}，'
              f'距离组 {current_placement["distance_bin"]}，'
              f'左右角 {current_placement["bearing_deg"]:+.1f}°；'
              f'把手像素 {np.round(current_placement["handle_pixel_uv"],1).tolist()}，'
              f'抽屉可见像素 {current_placement["drawer_visible_pixels"]}。', flush=True)
    policy_ws=None
    def request_policy(score_only=False,force_grasp_handoff=False,reset_mobility=False):
        nonlocal policy_ws
        images=capture_handoff_images()['policy_input']
        state=np.concatenate([obs['state.'+k] for k in ('end_effector_position_relative',
            'end_effector_rotation_relative','base_position','base_rotation','gripper_qpos')])
        req={'observation/image':images['main'],'observation/wrist_image':images['wrist'],
             'observation/right_image':images['right'],'observation/state':state,'prompt':task_prompt}
        req['observation/base_qpos']=np.asarray(env.sim.data.qpos[base_qpos_ids],dtype=np.float32).copy()
        if score_only:req['score_only']=True
        if force_grasp_handoff:req['force_grasp_handoff']=True
        if reset_mobility:req['reset_mobility']=True
        for attempt in range(2):
            try:
                if policy_ws is None:
                    policy_ws=connect(f'ws://{a.host}:{a.port}',compression=None,max_size=None,
                                      proxy=None,open_timeout=10,ping_interval=None)
                    policy_ws.recv(timeout=15)
                policy_ws.send(msgpack_numpy.Packer().pack(req));answer=policy_ws.recv(timeout=180)
                if isinstance(answer,str):raise RuntimeError(answer)
                return msgpack_numpy.unpackb(answer)
            except Exception:
                if policy_ws is not None:
                    try:policy_ws.close()
                    except Exception:pass
                    policy_ws=None
                if attempt:raise

    diagnostic_path=Path(a.out_root)/'joint_offline_value_reposition_v1_live.jsonl'
    diagnostic_path.parent.mkdir(parents=True,exist_ok=True)
    diagnostic_file=diagnostic_path.open('a',encoding='utf-8',buffering=1)
    cc=env.robots[0].composite_controller
    if 'base' not in cc._action_split_indexes:raise RuntimeError(f'Base controller missing: {list(cc._action_split_indexes)}')
    base_lo,base_hi=cc._action_split_indexes['base']
    stop_threshold=.50;local_confirmations=3;high_count=0;grasp_mode=False;iteration=0;last_action=None
    trans_ema=None;last_translation=None;translation_hold=0
    quality_history=deque(maxlen=16);recovery_remaining=0;recovery_cooldown=0;recovery_reference=None

    def advance(command,steps):
        nonlocal obs
        raw=np.zeros(cc.action_limits[0].shape,dtype=np.float64);raw[base_lo:base_hi]=command;raw[-1]=1.
        for _ in range(steps):
            raw_obs,_,_,_=env.step(raw);obs=g.unwrapped.get_observation(raw_obs)

    print(f'联合TD价值/动作/停止模型测试：按v开始；停止概率连续3次>={stop_threshold:.2f}后切换原始VLA抓取；x回退，q退出。',flush=True)
    print('随机初始水平/三维距离：',hud.distances(),flush=True);print('随机位姿参数：',current_placement,flush=True)
    event('start',reset_reason='program_start',random_initial_base=current_placement)
    try:
        while viewer.is_running():
            start=time.monotonic();quit_requested=False
            while not commands.empty():
                command=commands.get()
                if command=='q':quit_requested=True
                elif command=='v':
                    print('正在同步服务端并准备移动阶段……',flush=True)
                    rr=request_policy(reset_mobility=True)
                    if rr.get('stage')!='mobility' or not rr.get('mobility_reset'):
                        raise RuntimeError(f'Server rejected mobility reset: {rr.get("stage")}')
                    high_count=0;grasp_mode=False;iteration=0;last_action=None
                    trans_ema=None;last_translation=None;translation_hold=0
                    quality_history.clear();recovery_remaining=0;recovery_cooldown=0;recovery_reference=None
                    begin_policy_attempt()
                elif command=='x':
                    if mode=='policy' and active_attempt is not None:restore_failed_attempt('operator_pressed_x')
                    else:reset_episode('operator_pressed_x_during_manual')
                    high_count=0;grasp_mode=False;iteration=0;last_action=None
                elif command=='d':
                    reset_episode('discarded');high_count=0;grasp_mode=False;iteration=0;last_action=None
            if quit_requested:break
            raw_action=None
            if mode=='manual':
                inputs=device.input2action(mirror_actions=True,goal_update_mode='achieved')
                if inputs is None:break
                for arm in env.robots[0].arms:
                    suffix='delta' if env.robots[0].part_controllers[arm].input_type=='delta' else 'abs'
                    inputs[arm]=inputs[f'{arm}_{suffix}']
                raw_action=env.robots[0].create_action_vector(inputs)
            elif grasp_mode:
                result=request_policy(False);actions=np.asarray(result['actions'])
                if result.get('stage')!='grasp':raise RuntimeError(f'Expected grasp stage, got {result.get("stage")}')
                for act in actions[:min(24,len(actions))]:
                    mapped=g.unwrapped.key_converter.unmap_action(convert_action(act));prefix_name=env.robots[0].robot_model.naming_prefix
                    raw=np.zeros(cc.action_limits[0].shape)
                    for part in cc.part_controllers:
                        lo,hi=cc._action_split_indexes[part];raw[lo:hi]=mapped.pop(prefix_name+part)
                    raw[-1]=mapped.pop(prefix_name+'base_mode');raw_obs,_,_,_=env.step(raw);obs=g.unwrapped.get_observation(raw_obs)
            else:
                result=request_policy(True);quality=float(result['view_quality'])
                stop_p=float(result['joint_stop_probability'])
                probs=np.asarray(result['joint_action_probabilities'],np.float32).reshape(-1)
                names=list(result.get('joint_action_names') or [])
                if probs.shape!=(6,) or len(names)!=6:raise RuntimeError('Server did not return valid joint outputs')
                high_count=high_count+1 if stop_p>=stop_threshold else 0
                ai=int(np.argmax(probs));choice=names[ai];command=np.zeros(3,np.float32)
                if stop_p>=stop_threshold:
                    choice='confirm_stop'
                else:
                    command[:]=np.asarray([[.30,0,0],[-.30,0,0],[0,.30,0],[0,-.30,0],[0,0,.12],[0,0,-.12]],np.float32)[ai]
                diagnostic_file.write(json.dumps({'time':time.time(),'iteration':iteration,'view_quality':quality,'stop_probability':stop_p,'stop_count':high_count,'action_names':names,'action_probabilities':probs.tolist(),'choice':choice,'applied':command.tolist()},ensure_ascii=False)+'\n')
                print(f'[joint-td {iteration:03d}] value={quality:.3f} stop={stop_p:.3f} '
                      f'confirm={high_count}/{local_confirmations} actions={dict(zip(names,np.round(probs,3)))} '
                      f'-> {choice} applied={command.tolist()} x4',flush=True)
                if high_count>=local_confirmations:
                    rr=request_policy(False,True)
                    if rr.get('stage')!='grasp' or not rr.get('switching_to_grasp'):raise RuntimeError(f'Server rejected grasp handoff: {rr.get("stage")}')
                    grasp_mode=True;print('联合模型连续3次判定停止，切换原始VLA抓取阶段。',flush=True)
                else:
                    with viewer.lock():advance(command,4)
                    iteration+=1
            if raw_action is not None:
                with viewer.lock():raw_obs,_,_,_=env.step(raw_action);obs=g.unwrapped.get_observation(raw_obs)
            hud.update(force=True);viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'VIEW ACTION GAIN\nv start | x rollback | q quit',('GRASP' if grasp_mode else mode.upper())+f' | iter {iteration}'))
            viewer.sync();ticks+=1
            if a.smoke_policy and ticks==1:commands.put('v')
            if a.smoke_steps and ticks>=a.smoke_steps:break
            time.sleep(max(0,.05-(time.monotonic()-start)))
    finally:
        try:diagnostic_file.close()
        except Exception:pass
        if policy_ws is not None:
            try:policy_ws.close()
            except Exception:pass
        device.listener.stop();viewer.close();hud.close();g.close();print('动作增益测试已退出。',flush=True)
if __name__=='__main__':main()
