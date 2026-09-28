"""Record Day6 demonstrations at a reproducible cube-position schedule."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import mujoco

from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import (
    DatasetConfig,
    GymManipulatorConfig,
    control_loop,
    make_processors,
    make_robot_env,
)


class ScheduledBlockPositionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, positions: list[dict[str, float]]):
        super().__init__(env)
        self.positions = positions
        self.reset_count = 0
        self.position_history: list[dict[str, float]] = []

    def reset(self, **kwargs):
        if self.reset_count >= len(self.positions):
            raise RuntimeError("Position schedule exhausted before recording finished")

        observation, info = self.env.reset(**kwargs)
        position = self.positions[self.reset_count]
        base = self.unwrapped
        base.data.jnt("block").qpos[:3] = (position["x"], position["y"], base._block_z)
        mujoco.mj_forward(base.model, base.data)
        base._z_init = float(base.data.sensor("block_pos").data[2])
        base._z_success = base._z_init + 0.1
        # Recompute the returned observation after moving the block.
        observation = base._compute_observation()

        self.position_history.append(
            {
                "episode_index": self.reset_count,
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(base._block_z),
            }
        )
        self.reset_count += 1
        return observation, info




def set_viewer_robot_visibility(env: gym.Env, hide_robot_arm: bool) -> None:
    """Hide only Panda arm visual meshes in the human viewer.

    The gripper visual meshes stay visible. The model's camera renderers keep
    their default geom groups, so this is a viewer-only convenience and does
    not alter recorded front/wrist observations.
    """
    if not hide_robot_arm:
        return

    base = env.unwrapped
    model = base.model
    hidden_group = 5
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 2:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is not None and body_name.startswith("link"):
            model.geom_group[geom_id] = hidden_group

    current = env
    while current is not None:
        viewer = getattr(current, "_viewer", None)
        if viewer is not None and hasattr(viewer, "opt"):
            viewer.opt.geomgroup[2] = 1  # keep gripper visual meshes
            viewer.opt.geomgroup[hidden_group] = 0  # hide Panda arm visuals
            # With the arm hidden, use a stricter top-down view so the
            # gripper and cube are easy to align.
            viewer.cam.elevation = -80.0
            viewer.cam.distance = 1.05
            viewer.sync()
            print("Viewer mode: Panda arm visuals hidden; gripper remains visible")
            return
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find the passive viewer handle")


def set_keyboard_input_step_size(env: gym.Env, xy_step_size: float, z_step_size: float) -> None:
    """Adjust keyboard controller deltas without changing the robot EE scale."""
    current = env
    while current is not None:
        controller = getattr(current, "controller", None)
        if controller is not None and hasattr(controller, "x_step_size"):
            controller.x_step_size = xy_step_size
            controller.y_step_size = xy_step_size
            controller.z_step_size = z_step_size
            return
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find the Gym-HIL keyboard controller wrapper")

def load_schedule(path: Path) -> tuple[str, list[dict[str, float]]]:
    data = json.loads(path.read_text())
    split = str(data["split"])
    positions = data["positions"]
    if not positions:
        raise ValueError("Position schedule is empty")
    for index, position in enumerate(positions):
        if not (0.30 <= position["x"] <= 0.50):
            raise ValueError(f"Position {index} has unsafe x={position['x']}")
        if not (-0.15 <= position["y"] <= 0.15):
            raise ValueError(f"Position {index} has unsafe y={position['y']}")
    return split, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-description", default="pick_cube_scheduled_position")
    parser.add_argument(
        "--xy-step-size",
        type=float,
        default=0.25,
        help="Keyboard X/Y delta in normalized EE-action units (0.25 = 6.25 mm/tick).",
    )
    parser.add_argument(
        "--z-step-size",
        type=float,
        default=0.50,
        help="Keyboard Z delta in normalized EE-action units (0.50 = 12.5 mm/tick).",
    )
    parser.add_argument(
        "--hide-robot-arm",
        action="store_true",
        help=(
            "Hide only Panda arm visual meshes in the passive viewer; the "
            "gripper remains visible and recorded camera observations are unchanged."
        ),
    )
    args = parser.parse_args()
    if not 0.05 <= args.xy_step_size <= 1.0:
        raise ValueError("--xy-step-size must be between 0.05 and 1.0")
    if not 0.05 <= args.z_step_size <= 1.0:
        raise ValueError("--z-step-size must be between 0.05 and 1.0")

    split, positions = load_schedule(args.schedule)
    if args.root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset root: {args.root}")

    env_cfg = HILSerlRobotEnvConfig(
        name="gym_hil",
        task="PandaPickCubeKeyboardRandomViewerLong-v0",
        fps=10,
        robot=None,
        teleop=None,
        processor=HILSerlProcessorConfig(),
    )
    env, teleop_device = make_robot_env(env_cfg)
    set_viewer_robot_visibility(env, args.hide_robot_arm)
    set_keyboard_input_step_size(env, args.xy_step_size, args.z_step_size)
    env = ScheduledBlockPositionWrapper(env, positions)
    env_processor, action_processor = make_processors(
        env=env,
        teleop_device=teleop_device,
        cfg=env_cfg,
        device="cpu",
    )
    cfg = GymManipulatorConfig(
        env=env_cfg,
        dataset=DatasetConfig(
            repo_id=args.repo_id,
            root=str(args.root),
            task=args.task_description,
            num_episodes_to_record=len(positions),
            push_to_hub=False,
        ),
        mode="record",
        device="cpu",
    )

    print("Position split:", split)
    print("Episodes:", len(positions))
    print("Schedule:", args.schedule)
    print("Dataset root:", args.root)
    print("Keyboard XY input step size:", args.xy_step_size)
    print("Keyboard Z input step size:", args.z_step_size)
    print("Physical XY step per held-key tick:", args.xy_step_size * 0.025, "m")
    print("Physical Z step per held-key tick:", args.z_step_size * 0.025, "m")
    print("Hide Panda arm visuals:", args.hide_robot_arm)
    try:
        control_loop(env, env_processor, action_processor, teleop_device, cfg)
    finally:
        # Write metadata before viewer teardown. A known GLFW assertion can
        # occur during process shutdown, so the schedule evidence must already
        # be durable when the viewer is closed.
        if args.root.exists():
            shutil.copy2(args.schedule, args.root / "position_schedule.json")
            history_path = args.root / "position_history.json"
            history_path.write_text(json.dumps(env.position_history, indent=2) + "\n")
            protocol_path = args.root / "collection_protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "xy_step_size": args.xy_step_size,
                        "z_step_size": args.z_step_size,
                        "physical_xy_step_m": args.xy_step_size * 0.025,
                        "physical_z_step_m": args.z_step_size * 0.025,
                        "fps": 10,
                        "gripper_close_protocol": "10-12 frames",
                    },
                    indent=2,
                )
                + "\n"
            )
            print("Position history saved to:", history_path)
            print("Collection protocol saved to:", protocol_path)
        else:
            print("Dataset root was not created; no metadata was written.")
        env.close()


if __name__ == "__main__":
    main()
