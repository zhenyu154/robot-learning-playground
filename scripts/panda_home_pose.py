"""Project-local Panda home-pose utilities.

The installed Gym-HIL package keeps its home joint vector private to the
environment instance. These helpers configure that instance without editing
site-packages or changing the Day5-Day10 defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


def load_home_pose(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if "home_position" not in payload:
        raise ValueError(f"Home-pose file has no home_position: {path}")
    home = np.asarray(payload["home_position"], dtype=np.float64)
    if home.shape != (7,) or not np.isfinite(home).all():
        raise ValueError("home_position must contain seven finite joint values")
    return home


def configure_panda_home_pose(env, home_position: np.ndarray) -> np.ndarray:
    """Set an environment instance's reset/controller home pose and return TCP XYZ."""
    base = env.unwrapped
    if not hasattr(base, "_home_position") or not hasattr(base, "_panda_dof_ids"):
        raise TypeError("Expected a Gym-HIL Panda environment with a home pose")

    home = np.asarray(home_position, dtype=np.float64)
    if home.shape != (7,) or not np.isfinite(home).all():
        raise ValueError("home_position must have shape (7,) and finite values")

    # Validate against MuJoCo joint limits before changing the live instance.
    for value, joint_id in zip(home, base._panda_dof_ids, strict=True):
        if base.model.jnt_limited[joint_id]:
            low, high = base.model.jnt_range[joint_id]
            if not low <= value <= high:
                raise ValueError(
                    f"Home joint {joint_id}={value:.6f} is outside [{low:.6f}, {high:.6f}]"
                )

    base._home_position = home.copy()
    # Keep qpos, mocap target, and the operational-space controller reference
    # synchronized. The next reset will repeat this through reset_robot().
    base.reset_robot()
    mujoco.mj_forward(base.model, base.data)
    tcp_xyz = np.asarray(base.data.sensor("2f85/pinch_pos").data, dtype=np.float64).copy()

    current = env
    while current is not None:
        viewer = getattr(current, "_viewer", None)
        if viewer is not None and hasattr(viewer, "sync"):
            viewer.sync()
            break
        current = getattr(current, "env", None)
    return tcp_xyz


def configure_min_tcp_z(env, min_tcp_z: float) -> float:
    """Raise the local Cartesian lower Z bound without editing Gym-HIL source."""
    if not np.isfinite(min_tcp_z) or min_tcp_z < 0:
        raise ValueError("min_tcp_z must be finite and non-negative")
    base = env.unwrapped
    if not hasattr(base, "_cartesian_bounds"):
        raise TypeError("Expected a Gym-HIL environment with Cartesian bounds")
    bounds = np.asarray(base._cartesian_bounds, dtype=np.float64).copy()
    if min_tcp_z >= bounds[1, 2]:
        raise ValueError(f"min_tcp_z must be below the upper Z bound {bounds[1, 2]:.6f}")
    bounds[0, 2] = float(min_tcp_z)
    base._cartesian_bounds = bounds
    return float(bounds[0, 2])
