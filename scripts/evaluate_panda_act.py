import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import torch
from PIL import Image, ImageDraw

from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.policies.act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.utils.robot_utils import precise_sleep


class ScheduledBlockPositionWrapper(gym.Wrapper):
    """Set the cube position from a JSON schedule at every reset."""

    def __init__(self, env: gym.Env, positions: list[dict[str, float]]):
        super().__init__(env)
        self.positions = positions
        self.reset_count = 0
        self.current_position = None

    def reset(self, **kwargs):
        if self.reset_count >= len(self.positions):
            raise RuntimeError("Position schedule exhausted")
        observation, info = self.env.reset(**kwargs)
        position = self.positions[self.reset_count]
        base = self.unwrapped
        base.data.jnt("block").qpos[:3] = (position["x"], position["y"], base._block_z)
        mujoco.mj_forward(base.model, base.data)
        base._z_init = float(base.data.sensor("block_pos").data[2])
        base._z_success = base._z_init + 0.1
        observation = base._compute_observation()
        self.current_position = {
            "episode_index": self.reset_count,
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(base._block_z),
        }
        self.reset_count += 1
        return observation, info


def load_position_schedule(path: Path):
    data = json.loads(path.read_text())
    positions = data["positions"]
    if not positions:
        raise ValueError(f"Empty position schedule: {path}")
    return positions


def observation_to_pil(observation: dict, camera: str) -> Image.Image:
    """Convert one processed observation camera to a PIL image."""
    key = f"observation.images.{camera}"
    image = observation[key].detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    image = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image, mode="RGB")


def make_gif_frame(observation: dict, camera: str, step: int) -> Image.Image:
    front = observation_to_pil(observation, "front")
    wrist = observation_to_pil(observation, "wrist")
    if camera == "front":
        frame = front
    elif camera == "wrist":
        frame = wrist
    else:
        frame = Image.new("RGB", (front.width + wrist.width, front.height + 20), "white")
        frame.paste(front, (0, 20))
        frame.paste(wrist, (front.width, 20))
        draw = ImageDraw.Draw(frame)
        draw.text((4, 3), "front", fill="black")
        draw.text((front.width + 4, 3), "wrist", fill="black")

    # A small step label makes the GIF interpretable without obscuring the scene.
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 72, 16), fill=(255, 255, 255))
    draw.text((4, 2), f"step {step}", fill="black")
    return frame.convert("P", palette=Image.Palette.ADAPTIVE)


def save_gif(frames: list[Image.Image], output: Path, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot save an empty GIF")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        optimize=False,
    )
    print("GIF saved to:", output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu"),
        default="auto",
        help=(
            "Inference device. 'auto' follows the checkpoint device when available; "
            "explicitly choose cpu or xpu to override it."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained_model directory.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Hard per-episode safety limit, even if the environment omits TimeLimit.",
    )

    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help=(
            "Optional evaluation-only override for ACT's action queue length. "
            "Use 1 or 5 to test tighter closed-loop feedback without retraining."
        ),
    )

    parser.add_argument(
        "--debug-geometry",
        action="store_true",
        help="Print cube, TCP, and gripper state at logged rollout steps.",
    )

    parser.add_argument(
        "--gif-output",
        type=Path,
        default=None,
        help=(
            "Optional GIF path. Records the first episode from the front and "
            "wrist observation cameras; use --episodes 1."
        ),
    )

    parser.add_argument(
        "--gif-camera",
        choices=("front", "wrist", "side-by-side"),
        default="side-by-side",
        help="Camera layout for --gif-output.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="PandaPickCubeKeyboard-v0",
        help=(
            "Gym-HIL task name. Use PandaPickCubeKeyboardRandomLong-v0 for Day6 "
            "random-position evaluation with a 150-step horizon."
        ),
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--position-schedule",
        type=Path,
        default=None,
        help="Optional JSON schedule with a positions list for Day6 evaluation.",
    )

    parser.add_argument(
        "--gripper-mode",
        choices=("policy", "close-at-step", "close-on-upward-motion"),
        default="policy",
        help=(
            "How to execute the gripper channel. 'policy' uses the ACT output. "
            "'close-at-step' injects a fixed diagnostic close command. "
            "'close-on-upward-motion' injects a close when learned Z motion "
            "changes from descending to ascending."
        ),
    )

    parser.add_argument(
        "--close-step",
        type=int,
        default=60,
        help="1-indexed step for --gripper-mode=close-at-step.",
    )

    parser.add_argument(
        "--close-duration",
        type=int,
        default=6,
        help="Number of consecutive steps to inject gripper=2.",
    )

    parser.add_argument(
        "--lift-threshold",
        type=float,
        default=0.0,
        help="Z threshold for --gripper-mode=close-on-upward-motion.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be a positive integer")

    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    requested_device = args.device

    print("=" * 60)
    print("Loading ACT policy")
    print("=" * 60)
    print("Checkpoint:", checkpoint)

    policy = ACTPolicy.from_pretrained(checkpoint)

    if args.n_action_steps is not None:
        if args.n_action_steps <= 0:
            raise ValueError("--n-action-steps must be positive")
        policy.config.n_action_steps = args.n_action_steps
        print("Evaluation-only n_action_steps override:", args.n_action_steps)

    if requested_device == "auto":
        # Prefer the checkpoint's configured device, but safely fall back to CPU
        # if an XPU checkpoint is opened outside an XPU-enabled environment.
        selected_device = policy.config.device or "cpu"
        if selected_device.startswith("xpu") and not torch.xpu.is_available():
            selected_device = "cpu"
    else:
        selected_device = requested_device
        if selected_device == "xpu" and not torch.xpu.is_available():
            raise RuntimeError("--device=xpu was requested, but torch.xpu.is_available() is False")

    device = torch.device(selected_device)
    print("Inference device:", device)
    # The saved preprocessing pipeline also reads policy.config.device.
    policy.config.device = device.type
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print("Policy:", policy.__class__.__name__)
    print("Device:", device)
    print("Chunk size:", policy.config.chunk_size)
    print("n_action_steps:", policy.config.n_action_steps)
    print("Gripper mode:", args.gripper_mode)
    if args.gripper_mode == "close-at-step":
        print("Close step:", args.close_step)
        print("Close duration:", args.close_duration)

    # Same simulator/task used for demonstration collection.
    env_cfg = HILSerlRobotEnvConfig(
        name="gym_hil",
        task=args.task,
        fps=args.fps,
        robot=None,
        teleop=None,
        processor=HILSerlProcessorConfig(),
    )

    env, teleop_device = make_robot_env(env_cfg)

    scheduled_env = None
    if args.position_schedule is not None:
        positions = load_position_schedule(args.position_schedule)
        if args.episodes > len(positions):
            raise ValueError(
                f"Requested {args.episodes} episodes but schedule has only {len(positions)} positions"
            )
        scheduled_env = ScheduledBlockPositionWrapper(env, positions)
        env = scheduled_env
        print("Position schedule:", args.position_schedule)

    env_processor, action_processor = make_processors(
        env=env,
        teleop_device=teleop_device,
        cfg=env_cfg,
        device=str(device),
    )

    print("\nEnvironment observation space:")
    print(env.observation_space)

    print("\nEnvironment action space:")
    print(env.action_space)

    successes = 0
    gif_frames: list[Image.Image] = []

    if args.gif_output is not None and args.episodes != 1:
        raise ValueError("Use --episodes 1 when recording a GIF")

    try:
        for episode in range(args.episodes):
            print("\n" + "=" * 60)
            print(f"Episode {episode + 1}/{args.episodes}")
            print("=" * 60)

            # VERY IMPORTANT:
            # ACT caches an action chunk internally.
            # Clear it whenever a new episode starts.
            policy.reset()

            transition = reset_and_build_transition(
                env,
                env_processor,
                action_processor,
            )
            if scheduled_env is not None:
                print("Scheduled cube position:", scheduled_env.current_position)

            if args.gif_output is not None:
                gif_frames.append(
                    make_gif_frame(
                        transition[TransitionKey.OBSERVATION],
                        args.gif_camera,
                        step=0,
                    )
                )

            episode_reward = 0.0
            step = 0
            episode_start = time.perf_counter()
            previous_z_action = None
            close_steps_remaining = 0
            diagnostic_close_started = False
            max_gripper_action = float("-inf")
            min_z_action = float("inf")
            max_z_action = float("-inf")

            while True:
                step_start = time.perf_counter()

                # Keep only the observation features expected by ACT.
                observation = {
                    key: value
                    for key, value in transition[
                        TransitionKey.OBSERVATION
                    ].items()
                    if key in policy.config.input_features
                }

                # Policy-specific preprocessing:
                # normalization, device placement, batching, etc.
                processed_observation = (
                    preprocessor.process_observation(observation)
                )

                # No gradients during rollout.
                with torch.inference_mode():
                    action = policy.select_action(
                        batch=processed_observation
                    )

                # Undo action normalization.
                action = postprocessor.process_action(action)

                # Optional diagnostic overrides. These do not change the
                # learned policy; they test whether the learned XYZ motion
                # can succeed when a gripper close event is supplied.
                z_action = float(action.reshape(-1)[2].item())
                if previous_z_action is not None:
                    upward_transition = (
                        previous_z_action < args.lift_threshold
                        <= z_action
                    )
                else:
                    upward_transition = False

                if (
                    args.gripper_mode == "close-at-step"
                    and step + 1 == args.close_step
                    and not diagnostic_close_started
                ):
                    close_steps_remaining = max(args.close_duration, 1)
                    diagnostic_close_started = True
                    print(
                        f"diagnostic gripper close at step={step + 1} "
                        f"(fixed-step mode)"
                    )

                if (
                    args.gripper_mode == "close-on-upward-motion"
                    and upward_transition
                    and not diagnostic_close_started
                ):
                    close_steps_remaining = max(args.close_duration, 1)
                    diagnostic_close_started = True
                    print(
                        f"diagnostic gripper close at step={step + 1} "
                        f"(z_action={z_action:.3f})"
                    )

                if close_steps_remaining > 0:
                    action = action.clone()
                    action.reshape(-1)[3] = 2.0
                    close_steps_remaining -= 1

                previous_z_action = z_action

                action_flat = action.detach().cpu().float().reshape(-1)
                gripper_action = float(action_flat[3].item())
                max_gripper_action = max(max_gripper_action, gripper_action)
                min_z_action = min(min_z_action, z_action)
                max_z_action = max(max_z_action, z_action)

                # Step simulator using exactly LeRobot's environment
                # action-processing pipeline.
                transition = step_env_and_process_transition(
                    env=env,
                    transition=transition,
                    action=action,
                    env_processor=env_processor,
                    action_processor=action_processor,
                )

                reward = float(
                    transition[TransitionKey.REWARD]
                )

                terminated = bool(
                    transition.get(
                        TransitionKey.DONE,
                        False,
                    )
                )

                truncated = bool(
                    transition.get(
                        TransitionKey.TRUNCATED,
                        False,
                    )
                )

                episode_reward += reward
                step += 1

                if args.gif_output is not None:
                    gif_frames.append(
                        make_gif_frame(
                            transition[TransitionKey.OBSERVATION],
                            args.gif_camera,
                            step=step,
                        )
                    )

                # A second guard protects evaluation if a custom environment
                # was constructed without Gymnasium's TimeLimit wrapper.
                if step >= args.max_steps and not terminated:
                    truncated = True

                action_np = action.detach().cpu().numpy()

                # Don't print every single step forever.
                if step == 1 or step % 10 == 0 or terminated or truncated:
                    print(
                        f"step={step:3d} | "
                        f"action={action_np.round(3)} | "
                        f"reward={reward:.1f} | "
                        f"terminated={terminated} | "
                        f"truncated={truncated}"
                    )
                    if args.debug_geometry:
                        base_env = env.unwrapped
                        cube_xyz = base_env.data.sensor("block_pos").data.copy()
                        tcp_xyz = base_env.data.sensor("2f85/pinch_pos").data.copy()
                        gripper_pos = float(base_env.get_gripper_pose()[0])
                        print(
                            f"  geometry: cube={cube_xyz.round(4)}, "
                            f"tcp={tcp_xyz.round(4)}, "
                            f"gripper_pos={gripper_pos:.1f}"
                        )

                if terminated or truncated:
                    elapsed = time.perf_counter() - episode_start

                    success = reward > 0.0 or episode_reward > 0.0

                    if success:
                        successes += 1

                    print(
                        f"\nEpisode finished: "
                        f"success={success}, "
                        f"steps={step}, "
                        f"reward={episode_reward:.1f}, "
                        f"time={elapsed:.2f}s, "
                        f"max_gripper={max_gripper_action:.3f}, "
                        f"z_range=[{min_z_action:.3f}, {max_z_action:.3f}]"
                    )

                    break

                # Maintain approximately the same 10 Hz control
                # frequency used during demonstration recording.
                elapsed_step = time.perf_counter() - step_start
                precise_sleep(
                    max(
                        1.0 / args.fps - elapsed_step,
                        0.0,
                    )
                )

    finally:
        env.close()

    if args.gif_output is not None:
        save_gif(gif_frames, args.gif_output, args.fps)

    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)

    print(f"Successes: {successes}/{args.episodes}")
    print(
        f"Success rate: "
        f"{successes / args.episodes * 100:.1f}%"
    )


if __name__ == "__main__":
    main()