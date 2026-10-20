import collections
import dataclasses
import csv
import json
import logging
import math
import pathlib
from copy import deepcopy

import gymnasium as gym
import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import robocasa  # noqa: F401 - registers RoboCasa gym environments
from robocasa.models.fixtures import FixtureType
from robocasa.utils import camera_utils as CamUtils
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from robosuite.utils import transform_utils as T
import tqdm
import tyro


CAMERA_POS_DELTAS = {
    "baseline": (0.0, 0.0, 0.0),
    "x_plus_10cm": (0.10, 0.0, 0.0),
    "x_minus_10cm": (-0.10, 0.0, 0.0),
    "y_plus_10cm": (0.0, 0.10, 0.0),
    "y_minus_10cm": (0.0, -0.10, 0.0),
    "z_plus_10cm": (0.0, 0.0, 0.10),
    "z_minus_10cm": (0.0, 0.0, -0.10),
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # Evaluation parameters
    #################################################################################################################
    task: str = "AddIceCubes"
    split: str = "pretrain"
    num_trials: int = 3
    seed: int = 7
    same_scene: bool = False
    freeze_episode_meta: bool = False
    horizon_override: int | None = None
    disable_main_image: bool = False
    disable_right_image: bool = False
    disable_wrist_image: bool = False
    disable_all_images: bool = False
    only_wrist_image: bool = False
    record_geometry: bool = False
    prompt_override: str | None = None

    #################################################################################################################
    # Camera sweep parameters
    #################################################################################################################
    camera_name: str = "robot0_agentview_left"
    variants: list[str] = dataclasses.field(
        default_factory=lambda: [
            "baseline",
            "x_plus_10cm",
            "x_minus_10cm",
            "y_plus_10cm",
            "y_minus_10cm",
            "z_plus_10cm",
            "z_minus_10cm",
        ]
    )
    custom_deltas: list[str] | None = None
    look_at: str = "objects:ice_cube*"
    look_at_z_offset: float = 0.05
    sphere_sweep: bool = False
    num_camera_points: int = 20
    include_baseline: bool = True
    azimuth_limit_deg: float = 90.0
    elevation_min_deg: float = -30.0
    elevation_max_deg: float = 60.0
    radius_scale: float = 1.0
    sphere_views_csv: str | None = None
    camera_metadata_json: str | None = None
    camera_metadata_scene_idx: int = 0

    #################################################################################################################
    # Output
    #################################################################################################################
    log_dir: str = "./results/camera_sweep"
    render_every: int = 2
    resume: bool = False


def _parse_custom_delta(spec: str) -> tuple[str, tuple[float, float, float]]:
    try:
        name, raw_delta = spec.split(":", 1)
        values = tuple(float(v) for v in raw_delta.split(","))
    except ValueError as exc:
        raise ValueError(
            f"Bad custom delta '{spec}'. Use name:x,y,z, for example close:0.10,0,0."
        ) from exc

    if len(values) != 3:
        raise ValueError(f"Bad custom delta '{spec}'. Expected exactly three numbers.")
    if not name:
        raise ValueError(f"Bad custom delta '{spec}'. Name cannot be empty.")

    return name, values


def _selected_camera_deltas(args: Args) -> list[tuple[str, tuple[float, float, float]]]:
    deltas = []
    for variant in args.variants:
        if variant not in CAMERA_POS_DELTAS:
            raise ValueError(
                f"Unknown camera variant '{variant}'. Available: {sorted(CAMERA_POS_DELTAS)}"
            )
        deltas.append((variant, CAMERA_POS_DELTAS[variant]))

    for spec in args.custom_deltas or []:
        deltas.append(_parse_custom_delta(spec))

    return deltas


def _apply_camera_delta(
    original_configs: dict,
    camera_name: str,
    delta: tuple[float, float, float],
) -> dict:
    if camera_name not in original_configs:
        raise ValueError(
            f"Unknown camera '{camera_name}'. Available: {sorted(original_configs)}"
        )

    base_cfg = original_configs[camera_name]
    new_cfg = deepcopy(base_cfg)
    new_cfg["pos"] = (
        np.asarray(base_cfg["pos"], dtype=np.float64)
        + np.asarray(delta, dtype=np.float64)
    ).tolist()
    CamUtils.CAM_CONFIGS["DEFAULT"][camera_name] = new_cfg

    return {
        "camera_name": camera_name,
        "base_pos": base_cfg["pos"],
        "delta": list(delta),
        "pos": new_cfg["pos"],
        "quat": new_cfg["quat"],
    }


def _restore_camera_configs(original_configs: dict) -> None:
    CamUtils.CAM_CONFIGS["DEFAULT"] = deepcopy(original_configs)


def _unwrap_robocasa_gym_env(env):
    gym_env = getattr(env, "unwrapped", env)
    if hasattr(gym_env, "key_converter"):
        return gym_env

    current = env
    while hasattr(current, "env"):
        current = current.env
        if hasattr(current, "key_converter"):
            return current

    raise RuntimeError("Could not find the RoboCasaGymEnv wrapper.")


def _refresh_observation(env) -> dict:
    gym_env = _unwrap_robocasa_gym_env(env)
    base_env = gym_env.env
    raw_obs = (
        base_env.viewer._get_observations(force_update=True)
        if base_env.viewer_get_obs
        else base_env._get_observations(force_update=True)
    )
    return gym_env.get_observation(raw_obs)


def _base_robocasa_env(env):
    return _unwrap_robocasa_gym_env(env).env


def _capture_fixed_episode_meta(
    args: Args,
    log_path: pathlib.Path,
) -> tuple[dict | None, dict | None]:
    if not args.freeze_episode_meta:
        return None, None

    meta_path = log_path / "fixed_episode_meta.json"
    existing = _read_json(meta_path)
    if isinstance(existing, dict) and isinstance(existing.get("episode_meta"), dict):
        return deepcopy(existing["episode_meta"]), existing.get("metadata")

    env = gym.make(f"robocasa/{args.task}", split=args.split, seed=args.seed)
    try:
        obs, _ = env.reset(seed=args.seed)
        base_env = _base_robocasa_env(env)
        base_env.sim.forward()
        meta = deepcopy(base_env.get_ep_meta())
        look_at_target = _resolve_look_at_position(base_env, args)
        metadata = {
            "source": "captured_once_and_reused_for_resume",
            "task": args.task,
            "split": args.split,
            "seed": args.seed,
            "task_description": obs.get("annotation.human.task_description"),
            "look_at": args.look_at,
            "look_at_z_offset": args.look_at_z_offset,
            "look_at_target_position": (
                None if look_at_target is None else np.asarray(look_at_target).tolist()
            ),
            "initial_state": {
                "base_position": np.asarray(
                    obs.get("state.base_position", []), dtype=np.float64
                ).tolist(),
                "base_rotation": np.asarray(
                    obs.get("state.base_rotation", []), dtype=np.float64
                ).tolist(),
            },
        }
        _atomic_write_json(
            meta_path,
            {
                "episode_meta": meta,
                "metadata": metadata,
            },
        )
        return meta, metadata
    finally:
        env.close()


def _reset_for_episode(
    env,
    args: Args,
    fixed_episode_meta: dict | None,
) -> tuple[dict, dict]:
    if fixed_episode_meta is not None:
        base_env = _base_robocasa_env(env)
        base_env.set_ep_meta(deepcopy(fixed_episode_meta))
        return env.reset(seed=args.seed)
    return env.reset(seed=args.seed) if args.same_scene else env.reset()


def _drawer_fixture(base_env):
    fixture = getattr(base_env, "drawer", None)
    if fixture is None and "drawer" in getattr(base_env, "fixture_refs", {}):
        ref = base_env.fixture_refs["drawer"]
        fixture = ref[0] if isinstance(ref, tuple) else ref
    if fixture is None:
        raise RuntimeError("Could not resolve drawer fixture.")
    return fixture


def _handle_site(base_env, fixture) -> tuple[str, np.ndarray]:
    candidates = [
        f"{fixture.name}_door_handle_default_site",
        f"{fixture.name}_left_door_handle_default_site",
        f"{fixture.name}_right_door_handle_default_site",
    ]
    site_names = set(base_env.sim.model.site_names)
    for site_name in candidates:
        if site_name in site_names:
            site_id = base_env.sim.model.site_name2id(site_name)
            return site_name, np.array(base_env.sim.data.site_xpos[site_id], dtype=np.float64)

    matches = [
        site_name
        for site_name in base_env.sim.model.site_names
        if site_name and site_name.startswith(fixture.name) and "handle" in site_name
    ]
    if not matches:
        raise RuntimeError(f"No handle site found for fixture {fixture.name}.")
    site_id = base_env.sim.model.site_name2id(matches[0])
    return matches[0], np.array(base_env.sim.data.site_xpos[site_id], dtype=np.float64)


def _eef_site(base_env) -> tuple[str, np.ndarray]:
    for site_name in (
        "gripper0_right_grip_site",
        "gripper0_grip_site",
        "gripper0_left_grip_site",
    ):
        if site_name in base_env.sim.model.site_names:
            site_id = base_env.sim.model.site_name2id(site_name)
            return site_name, np.array(base_env.sim.data.site_xpos[site_id], dtype=np.float64)
    raise RuntimeError("No known gripper site found.")


def _drawer_open_fraction(base_env, fixture) -> float:
    try:
        door_state = fixture.get_door_state(env=base_env)
    except Exception:
        return math.nan
    values = [float(value) for value in door_state.values()]
    return float(np.mean(values)) if values else math.nan


def _geometry_snapshot(
    env,
    obs: dict,
    initial_handle_pos: np.ndarray | None,
    initial_open_fraction: float | None,
) -> dict:
    base_env = _base_robocasa_env(env)
    base_env.sim.forward()
    fixture = _drawer_fixture(base_env)
    handle_name, handle_pos = _handle_site(base_env, fixture)
    eef_name, eef_pos = _eef_site(base_env)
    base_pos = np.array(obs["state.base_position"], dtype=np.float64)
    drawer_pos = np.array(fixture.pos, dtype=np.float64)
    gripper_qpos = np.asarray(obs.get("state.gripper_qpos", []), dtype=np.float64)
    open_fraction = _drawer_open_fraction(base_env, fixture)
    handle_delta = (
        float(np.linalg.norm(handle_pos - initial_handle_pos))
        if initial_handle_pos is not None
        else 0.0
    )
    open_delta = (
        float(open_fraction - initial_open_fraction)
        if initial_open_fraction is not None and math.isfinite(open_fraction)
        else 0.0
    )
    drawer_side = getattr(env.unwrapped, "drawer_side", getattr(base_env, "drawer_side", ""))
    return {
        "drawer_side": str(drawer_side),
        "drawer_fixture_name": str(getattr(fixture, "name", "")),
        "handle_site_name": handle_name,
        "eef_site_name": eef_name,
        "base_xyz": [float(x) for x in base_pos],
        "drawer_xyz": [float(x) for x in drawer_pos],
        "handle_xyz": [float(x) for x in handle_pos],
        "eef_xyz": [float(x) for x in eef_pos],
        "base_to_handle_xy_m": float(np.linalg.norm(base_pos[:2] - handle_pos[:2])),
        "base_to_handle_3d_m": float(np.linalg.norm(base_pos - handle_pos)),
        "eef_to_handle_xy_m": float(np.linalg.norm(eef_pos[:2] - handle_pos[:2])),
        "eef_to_handle_3d_m": float(np.linalg.norm(eef_pos - handle_pos)),
        "eef_handle_dx_m": float(handle_pos[0] - eef_pos[0]),
        "eef_handle_dy_m": float(handle_pos[1] - eef_pos[1]),
        "eef_handle_dz_m": float(handle_pos[2] - eef_pos[2]),
        "gripper_qpos": [float(x) for x in gripper_qpos.reshape(-1)],
        "gripper_abs_sum": float(np.abs(gripper_qpos).sum()) if gripper_qpos.size else math.nan,
        "drawer_open_fraction": open_fraction,
        "drawer_open_delta": open_delta,
        "handle_displacement_from_initial_m": handle_delta,
    }


def _episode_geometry_complete(payload: dict) -> bool:
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        return False
    for section in ("initial", "nearest_eef_to_handle", "final"):
        section_payload = geometry.get(section)
        if not isinstance(section_payload, dict):
            return False
        if "eef_to_handle_3d_m" not in section_payload:
            return False
        if "base_to_handle_3d_m" not in section_payload:
            return False
    return True


def _finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean_or_none(values: list[float]) -> float | None:
    clean = _finite_values(values)
    return float(np.mean(clean)) if clean else None


def _std_or_none(values: list[float]) -> float | None:
    clean = _finite_values(values)
    return float(np.std(clean)) if clean else None


def _expand_object_names(base_env, raw_names: str) -> list[str]:
    object_names = []
    available_names = sorted(base_env.obj_body_id.keys())

    for raw_name in raw_names.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name.endswith("*"):
            prefix = name[:-1]
            matches = [obj_name for obj_name in available_names if obj_name.startswith(prefix)]
            if not matches:
                raise ValueError(
                    f"No objects match '{name}'. Available objects: {available_names}"
                )
            object_names.extend(matches)
        else:
            if name not in base_env.obj_body_id:
                raise ValueError(
                    f"Unknown object '{name}'. Available objects: {available_names}"
                )
            object_names.append(name)

    if not object_names:
        raise ValueError(f"No object names parsed from '{raw_names}'.")
    return object_names


def _resolve_look_at_position(base_env, args: Args) -> np.ndarray | None:
    spec = args.look_at
    if spec is None or spec.lower() in {"", "none", "false"}:
        return None

    if spec.startswith("object:"):
        object_names = _expand_object_names(base_env, spec.split(":", 1)[1])
        positions = [
            np.array(base_env.sim.data.body_xpos[base_env.obj_body_id[name]], dtype=np.float64)
            for name in object_names
        ]
        target = np.mean(positions, axis=0)
    elif spec.startswith("objects:"):
        object_names = _expand_object_names(base_env, spec.split(":", 1)[1])
        positions = [
            np.array(base_env.sim.data.body_xpos[base_env.obj_body_id[name]], dtype=np.float64)
            for name in object_names
        ]
        target = np.mean(positions, axis=0)
    elif spec.startswith("fixture:"):
        fixture_name = spec.split(":", 1)[1].strip()
        fixture = getattr(base_env, fixture_name, None)
        if fixture is None and fixture_name in getattr(base_env, "fixture_refs", {}):
            ref = base_env.fixture_refs[fixture_name]
            fixture = ref[0] if isinstance(ref, tuple) else ref
        if fixture is None or not hasattr(fixture, "pos"):
            raise ValueError(f"Unknown fixture '{fixture_name}'.")
        target = np.array(fixture.pos, dtype=np.float64)
    elif spec.startswith("handle:"):
        fixture_name = spec.split(":", 1)[1].strip()
        fixture = getattr(base_env, fixture_name, None)
        if fixture is None and fixture_name in getattr(base_env, "fixture_refs", {}):
            ref = base_env.fixture_refs[fixture_name]
            fixture = ref[0] if isinstance(ref, tuple) else ref
        if fixture is None or not hasattr(fixture, "name"):
            raise ValueError(f"Unknown fixture '{fixture_name}'.")

        candidates = [
            f"{fixture.name}_door_handle_default_site",
            f"{fixture.name}_left_door_handle_default_site",
            f"{fixture.name}_right_door_handle_default_site",
        ]
        site_names = set(base_env.sim.model.site_names)
        for site_name in candidates:
            if site_name in site_names:
                site_id = base_env.sim.model.site_name2id(site_name)
                target = np.array(base_env.sim.data.site_xpos[site_id], dtype=np.float64)
                break
        else:
            matches = [
                site_name
                for site_name in base_env.sim.model.site_names
                if site_name
                and site_name.startswith(fixture.name)
                and "handle" in site_name
            ]
            if not matches:
                raise ValueError(
                    f"Could not find a handle site for fixture '{fixture_name}' ({fixture.name})."
                )
            site_id = base_env.sim.model.site_name2id(matches[0])
            target = np.array(base_env.sim.data.site_xpos[site_id], dtype=np.float64)
    elif spec.startswith("geom:"):
        geom_name = spec.split(":", 1)[1].strip()
        if geom_name not in base_env.sim.model.geom_names:
            raise ValueError(f"Unknown geom '{geom_name}'.")
        geom_id = base_env.sim.model.geom_name2id(geom_name)
        target = np.array(base_env.sim.data.geom_xpos[geom_id], dtype=np.float64)
    elif spec.startswith("faucet_handle:"):
        fixture_name = spec.split(":", 1)[1].strip() or "sink"
        fixture = getattr(base_env, fixture_name, None)
        if fixture is None and fixture_name in getattr(base_env, "fixture_refs", {}):
            ref = base_env.fixture_refs[fixture_name]
            fixture = ref[0] if isinstance(ref, tuple) else ref
        if fixture is None and fixture_name == "sink":
            fixture = base_env.get_fixture(FixtureType.SINK)
        if fixture is None or not hasattr(fixture, "name"):
            raise ValueError(f"Unknown faucet fixture '{fixture_name}'.")

        geom_names = set(base_env.sim.model.geom_names)
        site_names = set(base_env.sim.model.site_names)
        candidates = [
            f"{fixture.name}_handle_main",
            f"{fixture.name}_g16",
            f"{fixture.name}_g17",
        ]
        for geom_name in candidates:
            if geom_name in geom_names:
                geom_id = base_env.sim.model.geom_name2id(geom_name)
                target = np.array(base_env.sim.data.geom_xpos[geom_id], dtype=np.float64)
                break
        else:
            matches = [
                geom_name
                for geom_name in base_env.sim.model.geom_names
                if geom_name
                and geom_name.startswith(fixture.name)
                and "handle" in geom_name
            ]
            if matches:
                geom_id = base_env.sim.model.geom_name2id(matches[0])
                target = np.array(base_env.sim.data.geom_xpos[geom_id], dtype=np.float64)
            else:
                site_name = f"{fixture.name}_water"
                if site_name not in site_names:
                    raise ValueError(
                        f"Could not find faucet handle geom or water site for fixture '{fixture_name}' ({fixture.name})."
                    )
                site_id = base_env.sim.model.site_name2id(site_name)
                target = np.array(base_env.sim.data.site_xpos[site_id], dtype=np.float64)
                target[2] += 0.06
    elif spec.startswith("site:"):
        site_name = spec.split(":", 1)[1].strip()
        if site_name not in base_env.sim.model.site_names:
            raise ValueError(f"Unknown site '{site_name}'.")
        site_id = base_env.sim.model.site_name2id(site_name)
        target = np.array(base_env.sim.data.site_xpos[site_id], dtype=np.float64)
    elif spec.startswith("xyz:"):
        try:
            target = np.array(
                [float(v) for v in spec.split(":", 1)[1].split(",")],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise ValueError(
                f"Bad xyz look-at target '{spec}'. Use xyz:x,y,z."
            ) from exc
        if target.shape != (3,):
            raise ValueError(f"Bad xyz look-at target '{spec}'. Use xyz:x,y,z.")
    else:
        raise ValueError(
            f"Bad look_at '{spec}'. Use object:name, objects:name1,name2, fixture:name, handle:name, faucet_handle:sink, geom:name, site:name, xyz:x,y,z, or none."
        )

    target = target.copy()
    target[2] += args.look_at_z_offset
    return target


def _camera_look_at_rotation(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    forward = target_pos - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        raise ValueError("Camera position and look-at target are too close.")
    forward = forward / forward_norm

    z_axis = -forward
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, z_axis))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    return np.column_stack([x_axis, y_axis, z_axis])


def _aim_camera_at_target(env, args: Args) -> dict:
    gym_env = _unwrap_robocasa_gym_env(env)
    base_env = gym_env.env
    sim = base_env.sim
    sim.forward()

    camera_cfg = CamUtils.CAM_CONFIGS["DEFAULT"][args.camera_name]
    camera_local_pos = np.array(camera_cfg["pos"], dtype=np.float64)
    parent_body = camera_cfg.get("parent_body")

    if parent_body is None:
        parent_pos = np.zeros(3, dtype=np.float64)
        parent_rot = np.eye(3, dtype=np.float64)
    else:
        parent_pos = np.array(sim.data.get_body_xpos(parent_body), dtype=np.float64)
        parent_rot = np.array(sim.data.get_body_xmat(parent_body), dtype=np.float64).reshape(3, 3)

    camera_world_pos = parent_pos + parent_rot @ camera_local_pos
    target_pos = _resolve_look_at_position(base_env, args)
    camera_local_quat = np.array(camera_cfg["quat"], dtype=np.float64)

    if target_pos is not None:
        camera_world_rot = _camera_look_at_rotation(camera_world_pos, target_pos)
        camera_local_rot = parent_rot.T @ camera_world_rot
        camera_local_quat = T.convert_quat(T.mat2quat(camera_local_rot), to="wxyz")

    cam_id = sim.model.camera_name2id(args.camera_name)
    sim.model.cam_pos[cam_id] = camera_local_pos
    sim.model.cam_quat[cam_id] = camera_local_quat
    sim.forward()

    camera_world_pos_after = np.array(sim.data.cam_xpos[cam_id], dtype=np.float64)
    camera_world_rot_after = np.array(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    target_camera_frame = None
    if target_pos is not None:
        target_camera_frame = camera_world_rot_after.T @ (target_pos - camera_world_pos_after)

    return {
        "camera_name": args.camera_name,
        "parent_body": parent_body,
        "look_at": args.look_at,
        "look_at_z_offset": args.look_at_z_offset,
        "target_world_pos": None if target_pos is None else target_pos.tolist(),
        "camera_local_pos": camera_local_pos.tolist(),
        "camera_local_quat": camera_local_quat.tolist(),
        "camera_world_pos": camera_world_pos_after.tolist(),
        "target_camera_frame": None
        if target_camera_frame is None
        else target_camera_frame.tolist(),
    }


def _parent_pose(sim, camera_name: str) -> tuple[str | None, np.ndarray, np.ndarray]:
    cam_id = sim.model.camera_name2id(camera_name)
    body_id = int(sim.model.cam_bodyid[cam_id])
    parent_body = sim.model.body_id2name(body_id)
    if parent_body == "world":
        return None, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    parent_pos = np.array(sim.data.body_xpos[body_id], dtype=np.float64)
    parent_rot = np.array(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
    return parent_body, parent_pos, parent_rot


def _set_camera_world_pose(
    sim,
    camera_name: str,
    camera_world_pos: np.ndarray,
    target_world_pos: np.ndarray,
) -> dict:
    cam_id = sim.model.camera_name2id(camera_name)
    parent_body, parent_pos, parent_rot = _parent_pose(sim, camera_name)

    camera_world_rot = _camera_look_at_rotation(camera_world_pos, target_world_pos)
    camera_local_pos = parent_rot.T @ (camera_world_pos - parent_pos)
    camera_local_rot = parent_rot.T @ camera_world_rot
    camera_local_quat = T.convert_quat(T.mat2quat(camera_local_rot), to="wxyz")

    sim.model.cam_pos[cam_id] = camera_local_pos
    sim.model.cam_quat[cam_id] = camera_local_quat
    sim.forward()

    camera_world_pos_after = np.array(sim.data.cam_xpos[cam_id], dtype=np.float64)
    camera_world_rot_after = np.array(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    target_camera_frame = camera_world_rot_after.T @ (
        target_world_pos - camera_world_pos_after
    )

    return {
        "parent_body": parent_body,
        "camera_local_pos": camera_local_pos.tolist(),
        "camera_local_quat": camera_local_quat.tolist(),
        "camera_world_pos": camera_world_pos_after.tolist(),
        "target_camera_frame": target_camera_frame.tolist(),
    }


def _camera_direction_basis(default_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = world_up - np.dot(world_up, default_direction) * default_direction
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = up - np.dot(up, default_direction) * default_direction
    up = up / np.linalg.norm(up)

    right = np.cross(up, default_direction)
    right = right / np.linalg.norm(right)
    return right, up


def _direction_from_offsets(
    default_direction: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    direction = (
        math.cos(elevation)
        * (math.cos(azimuth) * default_direction + math.sin(azimuth) * right)
        + math.sin(elevation) * up
    )
    return direction / np.linalg.norm(direction)


def _selected_sphere_views(args: Args) -> list[dict]:
    if args.sphere_views_csv:
        views = []
        with pathlib.Path(args.sphere_views_csv).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                name = row.get("view_id") or row.get("name") or f"view_{idx:02d}_az_custom"
                views.append(
                    {
                        "name": name,
                        "azimuth_deg": float(row["azimuth_deg"]),
                        "elevation_deg": float(row["elevation_deg"]),
                    }
                )
        if not views:
            raise ValueError(f"No sphere views found in {args.sphere_views_csv}")
        return views

    if args.num_camera_points < 1:
        raise ValueError("--args.num-camera-points must be at least 1.")

    views = []
    if args.include_baseline:
        views.append({"name": "view_00_baseline", "azimuth_deg": 0.0, "elevation_deg": 0.0})

    remaining = args.num_camera_points - len(views)
    if remaining <= 0:
        return views[: args.num_camera_points]

    # Same 20-point extreme layout as preview_camera_sphere.py. No filtering.
    az = float(args.azimuth_limit_deg)
    elev_min = float(args.elevation_min_deg)
    elev_max = float(args.elevation_max_deg)
    elevs = np.linspace(elev_min, elev_max, 4).tolist()
    rows = [
        [-az, -az * 0.5, 0.0, az * 0.5, az],
        [-az * 0.8, -az * 0.4, az * 0.4, az * 0.8],
        [-az, -az * 0.5, 0.0, az * 0.5, az],
        [-az * 0.8, -az * 0.4, 0.0, az * 0.4, az * 0.8],
    ]

    candidates = []
    for elevation, row in zip(elevs, rows):
        for azimuth in row:
            candidates.append((azimuth, elevation))

    if remaining > len(candidates):
        extra = remaining - len(candidates)
        sin_min = math.sin(math.radians(elev_min))
        sin_max = math.sin(math.radians(elev_max))
        golden = 0.6180339887498949
        for i in range(extra):
            frac = (i + 0.5) / extra
            elevation = math.degrees(math.asin(sin_min + frac * (sin_max - sin_min)))
            azimuth = ((i * golden) % 1.0 * 2.0 - 1.0) * az
            candidates.append((azimuth, elevation))

    start_idx = len(views)
    for idx, (azimuth, elevation) in enumerate(candidates[:remaining], start=start_idx):
        name = f"view_{idx:02d}_az_{azimuth:+.0f}_el_{elevation:+.0f}"
        name = name.replace("+", "p").replace("-", "m")
        views.append(
            {
                "name": name,
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
            }
        )

    return views


def _aim_camera_sphere_view(env, args: Args, sphere_view: dict) -> dict:
    gym_env = _unwrap_robocasa_gym_env(env)
    base_env = gym_env.env
    sim = base_env.sim
    sim.forward()

    camera_cfg = CamUtils.CAM_CONFIGS["DEFAULT"][args.camera_name]
    default_camera_local_pos = np.array(camera_cfg["pos"], dtype=np.float64)
    parent_body = camera_cfg.get("parent_body")

    if parent_body is None:
        parent_pos = np.zeros(3, dtype=np.float64)
        parent_rot = np.eye(3, dtype=np.float64)
    else:
        parent_pos = np.array(sim.data.get_body_xpos(parent_body), dtype=np.float64)
        parent_rot = np.array(sim.data.get_body_xmat(parent_body), dtype=np.float64).reshape(3, 3)

    default_camera_world_pos = parent_pos + parent_rot @ default_camera_local_pos
    target_pos = _resolve_look_at_position(base_env, args)
    if target_pos is None:
        raise ValueError("Sphere sweep requires a look-at target.")

    radius = np.linalg.norm(default_camera_world_pos - target_pos) * args.radius_scale
    default_direction = (default_camera_world_pos - target_pos) / np.linalg.norm(
        default_camera_world_pos - target_pos
    )
    right, up = _camera_direction_basis(default_direction)
    direction = _direction_from_offsets(
        default_direction,
        right,
        up,
        sphere_view["azimuth_deg"],
        sphere_view["elevation_deg"],
    )
    camera_world_pos = target_pos + radius * direction
    pose_metadata = _set_camera_world_pose(
        sim, args.camera_name, camera_world_pos, target_pos
    )

    return {
        "camera_name": args.camera_name,
        "look_at": args.look_at,
        "look_at_z_offset": args.look_at_z_offset,
        "sphere_view": sphere_view,
        "target_world_pos": target_pos.tolist(),
        "default_camera_world_pos": default_camera_world_pos.tolist(),
        "radius": float(radius),
        "default_direction": default_direction.tolist(),
        **pose_metadata,
    }


def _select_camera_metadata_scene(metadata: dict, scene_idx: int) -> dict:
    scenes = metadata.get("scenes") or metadata.get("views")
    if not scenes:
        raise ValueError("Camera metadata must contain a non-empty 'scenes' or 'views' list.")

    for scene in scenes:
        if int(scene.get("scene_idx", -1)) == scene_idx:
            return scene

    if scene_idx < 0 or scene_idx >= len(scenes):
        raise ValueError(
            f"camera_metadata_scene_idx={scene_idx} is out of range for {len(scenes)} scenes."
        )
    return scenes[scene_idx]


def _aim_camera_from_metadata(env, args: Args) -> dict:
    if args.camera_metadata_json is None:
        raise ValueError("camera_metadata_json is required.")

    metadata_path = pathlib.Path(args.camera_metadata_json)
    metadata = json.loads(metadata_path.read_text())
    scene = _select_camera_metadata_scene(metadata, args.camera_metadata_scene_idx)

    if "camera_world_pos" not in scene:
        raise ValueError(f"Scene metadata is missing camera_world_pos: {scene}")
    target_world_pos = scene.get("target_world_pos") or scene.get("handle_world_pos")
    if target_world_pos is None:
        target_world_pos = metadata.get("target_world_pos")
    if target_world_pos is None:
        raise ValueError(
            "Camera metadata must contain handle_world_pos or target_world_pos."
        )

    gym_env = _unwrap_robocasa_gym_env(env)
    base_env = gym_env.env
    sim = base_env.sim
    sim.forward()

    camera_world_pos = np.array(scene["camera_world_pos"], dtype=np.float64)
    target_world_pos = np.array(target_world_pos, dtype=np.float64)
    pose_metadata = _set_camera_world_pose(
        sim, args.camera_name, camera_world_pos, target_world_pos
    )

    return {
        "camera_name": args.camera_name,
        "camera_metadata_json": str(metadata_path),
        "camera_metadata_scene_idx": args.camera_metadata_scene_idx,
        "metadata_scene": scene.get("scene_idx", args.camera_metadata_scene_idx),
        "metadata_prompt": scene.get("prompt"),
        "metadata_camera_world_pos": camera_world_pos.tolist(),
        "metadata_target_world_pos": target_world_pos.tolist(),
        **pose_metadata,
    }


def _atomic_write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    tmp_path.replace(path)


def _read_json(path: pathlib.Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _episode_result_path(
    log_path: pathlib.Path,
    variant_name: str,
    episode_idx: int,
) -> pathlib.Path:
    return log_path / "episodes" / variant_name / f"trial_{episode_idx:03d}.json"


def _initial_policy_input_paths(
    log_path: pathlib.Path,
    task: str,
    variant_name: str,
    episode_idx: int,
    same_scene: bool,
) -> dict[str, pathlib.Path]:
    if same_scene:
        prefix = log_path / f"{task}_{variant_name}_initial_policy_input"
    else:
        prefix = log_path / f"{task}_{variant_name}_rollout_{episode_idx}_initial_policy_input"
    return {
        "main": prefix.with_name(f"{prefix.name}_main.jpg"),
        "right": prefix.with_name(f"{prefix.name}_right.jpg"),
        "wrist": prefix.with_name(f"{prefix.name}_wrist.jpg"),
        "all": prefix.with_name(f"{prefix.name}_all.jpg"),
    }


def _save_initial_policy_inputs(
    log_path: pathlib.Path,
    task: str,
    variant_name: str,
    episode_idx: int,
    same_scene: bool,
    main_img: np.ndarray,
    right_img: np.ndarray,
    wrist_img: np.ndarray,
) -> dict[str, str]:
    paths = _initial_policy_input_paths(
        log_path, task, variant_name, episode_idx, same_scene
    )
    if all(path.exists() for path in paths.values()):
        return {key: str(path) for key, path in paths.items()}

    log_path.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(paths["main"], np.asarray(main_img))
    imageio.imwrite(paths["right"], np.asarray(right_img))
    imageio.imwrite(paths["wrist"], np.asarray(wrist_img))

    h = max(main_img.shape[0], right_img.shape[0], wrist_img.shape[0])
    gutter = np.full((h, 8, 3), 255, dtype=np.uint8)
    parts = []
    for image in (main_img, right_img, wrist_img):
        image = np.asarray(image)
        if image.shape[0] != h:
            pad = np.full((h - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
            image = np.concatenate([image, pad], axis=0)
        parts.extend([image, gutter])
    imageio.imwrite(paths["all"], np.concatenate(parts[:-1], axis=1))
    return {key: str(path) for key, path in paths.items()}


def _existing_rollout_video(
    log_path: pathlib.Path,
    task: str,
    variant_name: str,
    episode_idx: int,
) -> pathlib.Path | None:
    matches = sorted(
        log_path.glob(f"{task}_{variant_name}_rollout_{episode_idx}_*.mp4")
    )
    if len(matches) != 1:
        return None
    return matches[0]


def _image_flags_match(payload: dict, args: Args) -> bool:
    expected = {
        "disable_main_image": args.disable_main_image,
        "disable_right_image": args.disable_right_image,
        "disable_wrist_image": args.disable_wrist_image,
        "disable_all_images": args.disable_all_images,
        "only_wrist_image": args.only_wrist_image,
    }
    return all(payload.get(key) == value for key, value in expected.items())


def _initial_policy_inputs_exist(
    payload: dict,
    expected_paths: dict[str, pathlib.Path] | None = None,
) -> bool:
    paths = payload.get("initial_policy_input_images")
    if not isinstance(paths, dict):
        return False
    required = ("main", "right", "wrist", "all")
    if expected_paths is not None:
        return all(
            pathlib.Path(str(paths.get(key, ""))) == expected_paths[key]
            and expected_paths[key].exists()
            for key in required
        )
    return all(pathlib.Path(str(paths.get(key, ""))).exists() for key in required)


def _load_completed_episode(
    log_path: pathlib.Path,
    task: str,
    variant_name: str,
    episode_idx: int,
    num_trials: int,
    args: Args,
) -> dict | None:
    result_path = _episode_result_path(log_path, variant_name, episode_idx)
    payload = _read_json(result_path)
    if payload is not None:
        if int(payload.get("episode_idx", -1)) != episode_idx:
            return None
        if payload.get("variant_name") != variant_name:
            return None
        if "success" not in payload:
            return None
        video_path = payload.get("video_path")
        if video_path is not None and not pathlib.Path(video_path).exists():
            return None
        if not _image_flags_match(payload, args):
            return None
        if args.record_geometry and not _episode_geometry_complete(payload):
            return None
        expected_paths = _initial_policy_input_paths(
            log_path, task, variant_name, episode_idx, args.same_scene
        )
        if not _initial_policy_inputs_exist(payload, expected_paths):
            return None
        return payload

    stats = _read_json(log_path / f"{task}_{variant_name}_stats.json")
    if stats is None:
        return None
    if int(stats.get("num_episodes", -1)) != num_trials:
        return None
    if "success_rate" not in stats:
        return None
    if not _image_flags_match(stats, args):
        return None
    if args.record_geometry:
        return None

    video_path = _existing_rollout_video(log_path, task, variant_name, episode_idx)
    if video_path is None:
        return None
    image_paths = _initial_policy_input_paths(
        log_path, task, variant_name, episode_idx, args.same_scene
    )
    if not all(path.exists() for path in image_paths.values()):
        return None
    suffix = video_path.stem.rsplit("_", 1)[-1]
    if suffix not in {"success", "failure"}:
        return None
    return {
        "task": task,
        "variant_name": variant_name,
        "episode_idx": episode_idx,
        "success": suffix == "success",
        "video_path": str(video_path),
        "disable_main_image": args.disable_main_image,
        "disable_right_image": args.disable_right_image,
        "disable_wrist_image": args.disable_wrist_image,
        "disable_all_images": args.disable_all_images,
        "only_wrist_image": args.only_wrist_image,
        "initial_policy_input_images": {
            key: str(path) for key, path in image_paths.items()
        },
        "legacy_video_only": True,
    }


def _build_variant_stats(
    args: Args,
    horizon: int,
    episode_results: list[dict],
    original_task_description: str | None,
) -> dict:
    ordered_results = sorted(
        episode_results, key=lambda result: int(result["episode_idx"])
    )
    total_episodes = len(ordered_results)
    total_successes = sum(1 for result in ordered_results if result.get("success"))
    success_rate = (
        float(total_successes) / float(total_episodes) if total_episodes else 0.0
    )
    stats = {
        "num_episodes": total_episodes,
        "success_rate": success_rate,
        "look_at": args.look_at,
        "original_task_description": original_task_description,
        "prompt": args.prompt_override or original_task_description,
        "prompt_override": args.prompt_override,
        "freeze_episode_meta": args.freeze_episode_meta,
        "horizon_override": args.horizon_override,
        "horizon": horizon,
        "camera_metadata_json": args.camera_metadata_json,
        "camera_metadata_scene_idx": args.camera_metadata_scene_idx,
        "disable_main_image": args.disable_main_image,
        "disable_right_image": args.disable_right_image,
        "disable_wrist_image": args.disable_wrist_image,
        "disable_all_images": args.disable_all_images,
        "only_wrist_image": args.only_wrist_image,
        "record_geometry": args.record_geometry,
        "resume": args.resume,
        "completed_episode_indices": [
            int(result["episode_idx"]) for result in ordered_results
        ],
        "episode_cameras": [
            result.get("camera_metadata") for result in ordered_results
        ],
        "episode_results": ordered_results,
    }

    if any(isinstance(result.get("geometry"), dict) for result in ordered_results):
        initial_eef = []
        nearest_eef = []
        final_eef = []
        initial_base = []
        final_open_delta = []
        for result in ordered_results:
            geometry = result.get("geometry") or {}
            initial = geometry.get("initial") or {}
            nearest = geometry.get("nearest_eef_to_handle") or {}
            final = geometry.get("final") or {}
            initial_eef.append(float(initial.get("eef_to_handle_3d_m", math.nan)))
            nearest_eef.append(float(nearest.get("eef_to_handle_3d_m", math.nan)))
            final_eef.append(float(final.get("eef_to_handle_3d_m", math.nan)))
            initial_base.append(float(initial.get("base_to_handle_3d_m", math.nan)))
            final_open_delta.append(float(final.get("drawer_open_delta", math.nan)))
        stats["geometry_summary"] = {
            "initial_eef_to_handle_3d_m_mean": _mean_or_none(initial_eef),
            "initial_eef_to_handle_3d_m_std": _std_or_none(initial_eef),
            "nearest_eef_to_handle_3d_m_mean": _mean_or_none(nearest_eef),
            "nearest_eef_to_handle_3d_m_std": _std_or_none(nearest_eef),
            "final_eef_to_handle_3d_m_mean": _mean_or_none(final_eef),
            "final_eef_to_handle_3d_m_std": _std_or_none(final_eef),
            "initial_base_to_handle_3d_m_mean": _mean_or_none(initial_base),
            "initial_base_to_handle_3d_m_std": _std_or_none(initial_base),
            "final_drawer_open_delta_mean": _mean_or_none(final_open_delta),
            "final_drawer_open_delta_std": _std_or_none(final_open_delta),
        }
    return stats


def eval_main(args: Args) -> None:
    if args.camera_metadata_json is not None and args.sphere_sweep:
        raise ValueError(
            "--args.camera-metadata-json cannot be combined with --args.sphere-sweep."
        )

    logging.info("Waiting for server at ws://%s:%s...", args.host, args.port)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    log_path = pathlib.Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    fixed_episode_meta, fixed_episode_metadata = _capture_fixed_episode_meta(args, log_path)
    original_configs = deepcopy(CamUtils.CAM_CONFIGS["DEFAULT"])
    summary = {
        "task": args.task,
        "split": args.split,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "same_scene": args.same_scene,
        "freeze_episode_meta": args.freeze_episode_meta,
        "fixed_episode_metadata": fixed_episode_metadata,
        "horizon_override": args.horizon_override,
        "prompt_override": args.prompt_override,
        "camera_name": args.camera_name,
        "look_at": args.look_at,
        "look_at_z_offset": args.look_at_z_offset,
        "sphere_sweep": args.sphere_sweep,
        "num_camera_points": args.num_camera_points,
        "include_baseline": args.include_baseline,
        "azimuth_limit_deg": args.azimuth_limit_deg,
        "elevation_min_deg": args.elevation_min_deg,
        "elevation_max_deg": args.elevation_max_deg,
        "radius_scale": args.radius_scale,
        "sphere_views_csv": args.sphere_views_csv,
        "camera_metadata_json": args.camera_metadata_json,
        "camera_metadata_scene_idx": args.camera_metadata_scene_idx,
        "disable_main_image": args.disable_main_image,
        "disable_right_image": args.disable_right_image,
        "disable_wrist_image": args.disable_wrist_image,
        "disable_all_images": args.disable_all_images,
        "only_wrist_image": args.only_wrist_image,
        "record_geometry": args.record_geometry,
        "variants": {},
    }

    try:
        if args.sphere_sweep:
            for sphere_view in _selected_sphere_views(args):
                variant_name = sphere_view["name"]
                logging.info("Running %s with sphere view: %s", variant_name, sphere_view)
                stats = eval_variant(
                    args,
                    client,
                    variant_name,
                    log_path,
                    sphere_view=sphere_view,
                    fixed_episode_meta=fixed_episode_meta,
                )
                summary["variants"][variant_name] = {
                    "sphere_view": sphere_view,
                    "stats": stats,
                }
                with (log_path / "summary.json").open("w") as f:
                    json.dump(summary, f, indent=4)
        else:
            for variant_name, delta in _selected_camera_deltas(args):
                camera_pose = _apply_camera_delta(original_configs, args.camera_name, delta)
                logging.info("Running %s with camera pose: %s", variant_name, camera_pose)
                stats = eval_variant(
                    args,
                    client,
                    variant_name,
                    log_path,
                    fixed_episode_meta=fixed_episode_meta,
                )
                summary["variants"][variant_name] = {
                    "camera": camera_pose,
                    "stats": stats,
                }
                with (log_path / "summary.json").open("w") as f:
                    json.dump(summary, f, indent=4)
    finally:
        _restore_camera_configs(original_configs)

    logging.info("Saved camera sweep results to %s", log_path)


def eval_variant(
    args: Args,
    client: _websocket_client_policy.WebsocketClientPolicy,
    variant_name: str,
    log_path: pathlib.Path,
    sphere_view: dict | None = None,
    fixed_episode_meta: dict | None = None,
) -> dict:
    assert args.split in ["pretrain", "target"]
    if args.horizon_override is not None and args.horizon_override <= 0:
        raise ValueError("--args.horizon-override must be positive.")
    horizon = args.horizon_override or get_task_horizon(args.task)

    np.random.seed(args.seed)
    env = gym.make(f"robocasa/{args.task}", split=args.split, seed=args.seed)

    episode_results = []
    original_task_description = None

    try:
        for episode_idx in tqdm.tqdm(
            range(args.num_trials), desc=f"{args.task}/{variant_name}"
        ):
            completed_episode = None
            if args.resume:
                completed_episode = _load_completed_episode(
                    log_path,
                    args.task,
                    variant_name,
                    episode_idx,
                    args.num_trials,
                    args,
                )

            obs, info = _reset_for_episode(env, args, fixed_episode_meta)
            if args.camera_metadata_json is not None:
                camera_metadata = _aim_camera_from_metadata(env, args)
            elif sphere_view is None:
                camera_metadata = _aim_camera_at_target(env, args)
            else:
                camera_metadata = _aim_camera_sphere_view(env, args, sphere_view)
            obs = _refresh_observation(env)
            task_lang = obs["annotation.human.task_description"]
            if original_task_description is None:
                original_task_description = task_lang
            if args.prompt_override is not None:
                task_lang = args.prompt_override
            if completed_episode is not None:
                completed_episode.setdefault("split", args.split)
                completed_episode.setdefault("seed", args.seed)
                completed_episode.setdefault("same_scene", args.same_scene)
                completed_episode.setdefault("freeze_episode_meta", args.freeze_episode_meta)
                completed_episode.setdefault("horizon", horizon)
                if completed_episode.get("camera_metadata") is None:
                    completed_episode["camera_metadata"] = camera_metadata
                if completed_episode.get("prompt") is None:
                    completed_episode["prompt"] = task_lang
                if completed_episode.get("legacy_video_only"):
                    _atomic_write_json(
                        _episode_result_path(log_path, variant_name, episode_idx),
                        completed_episode,
                    )
                logging.info(
                    "Skipping completed %s episode %d (%s).",
                    variant_name,
                    episode_idx + 1,
                    "success" if completed_episode.get("success") else "failure",
                )
                episode_results.append(completed_episode)
                continue

            action_plan = collections.deque()
            replay_images = []
            initial_policy_input_images = None
            initial_geometry = None
            nearest_geometry = None
            final_geometry = None
            nearest_geometry_step = 0
            initial_handle_pos = None
            initial_open_fraction = None

            if args.record_geometry:
                initial_geometry = _geometry_snapshot(env, obs, None, None)
                nearest_geometry = initial_geometry
                final_geometry = initial_geometry
                initial_handle_pos = np.asarray(initial_geometry["handle_xyz"], dtype=np.float64)
                initial_open_fraction = float(initial_geometry.get("drawer_open_fraction", math.nan))

            if episode_idx == 0:
                preview = np.ascontiguousarray(obs["video.robot0_agentview_left"])
                imageio.imwrite(
                    log_path / f"{args.task}_{variant_name}_main_view_reset.jpg",
                    preview,
                )

            done = False
            t = 0
            logging.info("Starting %s episode %d...", variant_name, episode_idx + 1)
            while t < horizon:
                img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
                wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
                img_right = np.ascontiguousarray(obs["video.robot0_agentview_right"])

                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        wrist_img, args.resize_size, args.resize_size
                    )
                )
                img_right = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        img_right, args.resize_size, args.resize_size
                    )
                )
                if args.disable_all_images or args.disable_main_image or args.only_wrist_image:
                    img = np.zeros_like(img)
                if args.disable_all_images or args.disable_right_image or args.only_wrist_image:
                    img_right = np.zeros_like(img_right)
                if args.disable_all_images or args.disable_wrist_image:
                    wrist_img = np.zeros_like(wrist_img)

                if initial_policy_input_images is None:
                    initial_policy_input_images = _save_initial_policy_inputs(
                        log_path,
                        args.task,
                        variant_name,
                        episode_idx,
                        args.same_scene,
                        img,
                        img_right,
                        wrist_img,
                    )

                if not action_plan:
                    state = np.concatenate(
                        (
                            obs["state.end_effector_position_relative"],
                            obs["state.end_effector_rotation_relative"],
                            obs["state.base_position"],
                            obs["state.base_rotation"],
                            obs["state.gripper_qpos"],
                        ),
                        axis=0,
                    )

                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/right_image": img_right,
                        "observation/state": state,
                        "prompt": task_lang,
                    }

                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        "We want to replan every "
                        f"{args.replan_steps} steps, but policy only predicts "
                        f"{len(action_chunk)} steps."
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = convert_action(action_plan.popleft())
                obs, reward, done, truncated, info = env.step(action)
                done = info["success"]
                if args.record_geometry:
                    step_geometry = _geometry_snapshot(
                        env,
                        obs,
                        initial_handle_pos,
                        initial_open_fraction,
                    )
                    final_geometry = step_geometry
                    if (
                        float(step_geometry["eef_to_handle_3d_m"])
                        < float(nearest_geometry["eef_to_handle_3d_m"])
                    ):
                        nearest_geometry = step_geometry
                        nearest_geometry_step = t + 1

                replay_img = env.render()
                replay_img = np.ascontiguousarray(replay_img)
                replay_img = image_tools.convert_to_uint8(replay_img)
                if t % args.render_every == 0 or t == horizon - 1 or done:
                    replay_images.append(replay_img)

                if done:
                    break
                t += 1

            suffix = "success" if done else "failure"
            video_path = (
                log_path
                / f"{args.task}_{variant_name}_rollout_{episode_idx}_{suffix}.mp4"
            )
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=20,
            )
            episode_result = {
                "task": args.task,
                "split": args.split,
                "seed": args.seed,
                "same_scene": args.same_scene,
                "freeze_episode_meta": args.freeze_episode_meta,
                "variant_name": variant_name,
                "episode_idx": episode_idx,
                "success": bool(done),
                "steps": int(t + 1 if done else t),
                "horizon": horizon,
                "prompt": task_lang,
                "prompt_override": args.prompt_override,
                "camera_metadata": camera_metadata,
                "disable_main_image": args.disable_main_image,
                "disable_right_image": args.disable_right_image,
                "disable_wrist_image": args.disable_wrist_image,
                "disable_all_images": args.disable_all_images,
                "only_wrist_image": args.only_wrist_image,
                "record_geometry": args.record_geometry,
                "initial_policy_input_images": initial_policy_input_images,
                "video_path": str(video_path),
            }
            if args.record_geometry:
                episode_result["geometry"] = {
                    "initial": initial_geometry,
                    "nearest_eef_to_handle": nearest_geometry,
                    "nearest_eef_to_handle_step": int(nearest_geometry_step),
                    "final": final_geometry,
                }
            _atomic_write_json(
                _episode_result_path(log_path, variant_name, episode_idx),
                episode_result,
            )
            episode_results.append(episode_result)

            total_episodes = len(episode_results)
            total_successes = sum(
                1 for result in episode_results if result.get("success")
            )

            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %d", total_episodes)
            logging.info(
                "# successes: %d (%.1f%%)",
                total_successes,
                total_successes / total_episodes * 100,
            )

        stats = _build_variant_stats(
            args,
            horizon,
            episode_results,
            original_task_description,
        )
        _atomic_write_json(log_path / f"{args.task}_{variant_name}_stats.json", stats)
        logging.info("[%s/%s] Total success rate: %.3f", args.task, variant_name, stats["success_rate"])
        return stats
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_main)
