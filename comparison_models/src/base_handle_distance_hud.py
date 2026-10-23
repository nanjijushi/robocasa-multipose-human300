"""Real-time desktop HUD for RoboCasa base-to-drawer-handle distance.

Call ``install_distance_hud(env)`` after creating the environment.  The helper
wraps ``env.render`` so an always-on-top box on DISPLAY is refreshed whenever
the simulation is rendered.  It never changes the MuJoCo scene and therefore
does not appear in policy camera observations.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


def unwrap_env(env: Any) -> Any:
    """Return the innermost RoboCasa / robosuite environment."""
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        next_env = getattr(current, "env")
        if next_env is None or next_env is current:
            break
        current = next_env
    return current


def _drawer_fixture(base_env: Any) -> Any | None:
    fixture = getattr(base_env, "drawer", None)
    if fixture is not None:
        return fixture
    refs = getattr(base_env, "fixture_refs", {})
    fixture = refs.get("drawer") if isinstance(refs, dict) else None
    if isinstance(fixture, (tuple, list)) and fixture:
        fixture = fixture[0]
    return fixture


def find_drawer_handle_site(base_env: Any, explicit_name: str | None = None) -> str:
    """Find the handle site belonging to the active drawer fixture."""
    site_names = [name for name in base_env.sim.model.site_names if name]
    if explicit_name is not None:
        if explicit_name not in site_names:
            raise ValueError(f"Unknown target site: {explicit_name}")
        return explicit_name

    fixture = _drawer_fixture(base_env)
    fixture_name = getattr(fixture, "name", None)
    candidates = [name for name in site_names if "handle" in name.lower()]
    if fixture_name:
        fixture_candidates = [name for name in candidates if name.startswith(fixture_name)]
        if fixture_candidates:
            candidates = fixture_candidates

    preferred_suffixes = (
        "_door_handle_default_site",
        "_right_door_handle_default_site",
        "_left_door_handle_default_site",
    )
    for suffix in preferred_suffixes:
        for name in candidates:
            if name.endswith(suffix):
                return name
    if candidates:
        return sorted(candidates)[0]
    raise RuntimeError("No drawer handle site was found in the current environment.")


class DesktopBaseHandleDistanceHUD:
    """Always-on-top distance box placed at the desktop's upper-left corner."""

    def __init__(
        self,
        env: Any,
        target_site_name: str | None = None,
        recommended_min_m: float = 0.50,
        recommended_max_m: float = 0.65,
        refresh_hz: float = 20.0,
        screen_x: int = 12,
        screen_y: int = 12,
    ) -> None:
        if recommended_min_m >= recommended_max_m:
            raise ValueError("recommended_min_m must be smaller than recommended_max_m")
        if refresh_hz <= 0:
            raise ValueError("refresh_hz must be positive")

        import tkinter as tk

        self.env = env
        self.base_env = unwrap_env(env)
        self.target_site_name = find_drawer_handle_site(
            self.base_env, target_site_name
        )
        self.recommended_min_m = float(recommended_min_m)
        self.recommended_max_m = float(recommended_max_m)
        self.min_period_s = 1.0 / float(refresh_hz)
        self.last_update_s = -math.inf
        self.closed = False

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.geometry(f"+{int(screen_x)}+{int(screen_y)}")
        self.root.configure(bg="#171717")

        self.label = tk.Label(
            self.root,
            text="正在读取底座—把手距离…",
            justify="left",
            anchor="w",
            font=("DejaVu Sans", 18, "bold"),
            fg="white",
            bg="#171717",
            padx=14,
            pady=10,
        )
        self.label.pack()
        self.root.update_idletasks()
        self.root.update()

    def _positions(self) -> tuple[np.ndarray, np.ndarray]:
        robot = self.base_env.robots[0]
        # Match RoboCasa's robot0_base_pos observable exactly.  The robot model
        # root body is a fixed reference for a mobile robot and does not follow
        # planar base joints; the mobile-base center site does.
        base_site_name = robot.robot_model.base.correct_naming("center")
        base_site_id = self.base_env.sim.model.site_name2id(base_site_name)
        base_xyz = np.asarray(
            self.base_env.sim.data.site_xpos[base_site_id], dtype=np.float64
        ).copy()
        handle_xyz = np.asarray(
            self.base_env.sim.data.get_site_xpos(self.target_site_name),
            dtype=np.float64,
        ).copy()
        return base_xyz, handle_xyz

    def distances(self) -> tuple[float, float]:
        base_xyz, handle_xyz = self._positions()
        delta = handle_xyz - base_xyz
        return float(np.linalg.norm(delta[:2])), float(np.linalg.norm(delta))

    def update(self, force: bool = False) -> tuple[float, float] | None:
        if self.closed:
            return None
        now = time.monotonic()
        if not force and now - self.last_update_s < self.min_period_s:
            return None
        self.last_update_s = now

        xy_m, distance_3d_m = self.distances()
        in_range = self.recommended_min_m <= xy_m <= self.recommended_max_m
        status = "候选区间内" if in_range else "候选区间外"
        background = "#176b36" if in_range else "#9b2525"
        text = (
            f"底座 → 抽屉把手\n"
            f"水平距离：{xy_m:.3f} m\n"
            f"三维距离：{distance_3d_m:.3f} m\n"
            f"候选范围：{self.recommended_min_m:.2f}–{self.recommended_max_m:.2f} m\n"
            f"状态：{status}"
        )
        self.root.configure(bg=background)
        self.label.configure(text=text, bg=background)
        self.root.update_idletasks()
        self.root.update()
        return xy_m, distance_3d_m

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass


def install_distance_hud(
    env: Any,
    target_site_name: str | None = None,
    recommended_min_m: float = 0.50,
    recommended_max_m: float = 0.65,
    refresh_hz: float = 20.0,
) -> DesktopBaseHandleDistanceHUD:
    """Install the HUD by wrapping this environment's reset, step, and render."""
    hud = DesktopBaseHandleDistanceHUD(
        env=env,
        target_site_name=target_site_name,
        recommended_min_m=recommended_min_m,
        recommended_max_m=recommended_max_m,
        refresh_hz=refresh_hz,
    )
    original_reset = env.reset
    original_step = env.step
    original_render = env.render
    original_close = env.close

    def reset_with_distance_hud(*args: Any, **kwargs: Any) -> Any:
        result = original_reset(*args, **kwargs)
        if target_site_name is None:
            hud.target_site_name = find_drawer_handle_site(hud.base_env)
        hud.update(force=True)
        return result

    def step_with_distance_hud(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        hud.update()
        return result

    def render_with_distance_hud(*args: Any, **kwargs: Any) -> Any:
        result = original_render(*args, **kwargs)
        hud.update()
        return result

    def close_with_distance_hud(*args: Any, **kwargs: Any) -> Any:
        hud.close()
        return original_close(*args, **kwargs)

    env.reset = reset_with_distance_hud
    env.step = step_with_distance_hud
    env.render = render_with_distance_hud
    env.close = close_with_distance_hud
    hud.update(force=True)
    return hud
