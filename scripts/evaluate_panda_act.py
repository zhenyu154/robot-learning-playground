import argparse
import time
from pathlib import Path

import torch

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
        "--fps",
        type=int,
        default=10,
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

    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    requested_device = args.device

    print("=" * 60)
    print("Loading ACT policy")
    print("=" * 60)
    print("Checkpoint:", checkpoint)

    policy = ACTPolicy.from_pretrained(checkpoint)

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
        task="PandaPickCubeKeyboard-v0",
        fps=args.fps,
        robot=None,
        teleop=None,
        processor=HILSerlProcessorConfig(),
    )

    env, teleop_device = make_robot_env(env_cfg)

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