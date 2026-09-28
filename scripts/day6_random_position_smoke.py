"""Smoke-test the underlying Gym-HIL Panda random-position option.

This script does not train, record demonstrations, or modify the environment.
It resets the base Panda task several times and prints the sampled cube XY
position. It is intentionally separate from the LeRobot recording pipeline so
we can validate the environment before integrating the randomization into a
registered task.
"""

from __future__ import annotations

import argparse

import gym_hil  # noqa: F401: registers Gym-HIL assets
import numpy as np
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # The current Gym-HIL implementation samples with numpy.random directly.
    # Seed it here so this smoke test is repeatable.
    np.random.seed(args.seed)

    env = PandaPickCubeGymEnv(
        render_mode="rgb_array",
        image_obs=True,
        random_block_position=True,
    )

    try:
        print("Environment:", type(env).__name__)
        print("Random block position: True")
        print("Sampling resets:", args.resets)

        positions = []
        for index in range(args.resets):
            _, info = env.reset()
            del info
            position = env.data.sensor("block_pos").data.copy()
            positions.append(position)
            print(
                f"reset={index + 1}: "
                f"cube_xyz=({position[0]:.4f}, "
                f"{position[1]:.4f}, "
                f"{position[2]:.4f})"
            )

        xy = np.asarray(positions)[:, :2]
        print("unique XY positions:", len({tuple(row.round(6)) for row in xy}))
        print("x range:", float(xy[:, 0].min()), "to", float(xy[:, 0].max()))
        print("y range:", float(xy[:, 1].min()), "to", float(xy[:, 1].max()))

    finally:
        env.close()


if __name__ == "__main__":
    main()
