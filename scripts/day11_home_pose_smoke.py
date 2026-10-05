"""Smoke-test a project-local Panda home pose and small X motions."""

from __future__ import annotations

import argparse
from pathlib import Path

import gym_hil  # noqa: F401
import mujoco
import numpy as np
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv

from panda_home_pose import configure_panda_home_pose, load_home_pose


def tcp(env) -> np.ndarray:
    return env.data.sensor("2f85/pinch_pos").data.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home-pose", type=Path, required=True)
    parser.add_argument("--delta-x", type=float, default=0.02)
    args = parser.parse_args()
    if args.delta_x <= 0:
        raise ValueError("--delta-x must be positive")

    env = PandaPickCubeGymEnv(
        render_mode="rgb_array", image_obs=False, random_block_position=False
    )
    try:
        home = load_home_pose(args.home_pose)
        initial = configure_panda_home_pose(env, home)
        print("Home pose:", np.round(home, 6).tolist())
        print("Initial TCP XYZ:", np.round(initial, 6).tolist())

        env.reset()
        reset_tcp = tcp(env)
        print("Reset TCP XYZ:", np.round(reset_tcp, 6).tolist())

        env.unwrapped.apply_action([args.delta_x, 0, 0, 0, 0, 0, 0])
        plus_tcp = tcp(env)
        env.reset()
        env.unwrapped.apply_action([-args.delta_x, 0, 0, 0, 0, 0, 0])
        minus_tcp = tcp(env)
        mujoco.mj_forward(env.model, env.data)

        print("After +X motion TCP XYZ:", np.round(plus_tcp, 6).tolist())
        print("After -X motion TCP XYZ:", np.round(minus_tcp, 6).tolist())
        if not (plus_tcp[0] > reset_tcp[0] and minus_tcp[0] < reset_tcp[0]):
            raise RuntimeError("The smoke test did not observe both X motion directions")
        print("Home-pose/X-motion smoke: PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
