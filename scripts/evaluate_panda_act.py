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

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    device = torch.device("cpu")

    print("=" * 60)
    print("Loading ACT policy")
    print("=" * 60)
    print("Checkpoint:", checkpoint)

    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
    )

    print("Policy:", policy.__class__.__name__)
    print("Device:", device)
    print("Chunk size:", policy.config.chunk_size)
    print("n_action_steps:", policy.config.n_action_steps)

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
                        f"time={elapsed:.2f}s"
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