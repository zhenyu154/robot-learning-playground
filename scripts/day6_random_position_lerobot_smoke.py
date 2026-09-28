"""Smoke-test the Day6 random-position task through LeRobot's env factory."""

from __future__ import annotations

import argparse

import numpy as np
from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import make_robot_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Gym-HIL currently samples block positions via NumPy's global RNG.
    # Seed it so this environment integration check is reproducible.
    np.random.seed(args.seed)

    cfg = HILSerlRobotEnvConfig(
        name="gym_hil",
        task="PandaPickCubeKeyboardRandom-v0",
        fps=10,
        robot=None,
        teleop=None,
        processor=HILSerlProcessorConfig(),
    )
    env, _ = make_robot_env(cfg)

    try:
        positions = []
        print("Task:", cfg.task)
        print("Seed:", args.seed)
        for index in range(args.resets):
            env.reset()
            base_env = env.unwrapped
            position = base_env.data.sensor("block_pos").data.copy()
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
        print("negative-y resets:", int((xy[:, 1] < 0).sum()))
        print("positive-y resets:", int((xy[:, 1] > 0).sum()))
        if not ((xy[:, 1] < 0).any() and (xy[:, 1] > 0).any()):
            raise RuntimeError(
                "This smoke test did not sample both sides of the Y axis. "
                "Increase --resets before recording a spatial dataset."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
