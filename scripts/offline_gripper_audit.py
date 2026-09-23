import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies.act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit ACT gripper predictions on expert observations."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the ACT pretrained_model directory.",
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/wusanggg/robotics/data/panda_pick_cube_day1",
        help="Local LeRobot dataset root.",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="wusanggg/panda_pick_cube_day1",
        help="Dataset repo id used by LeRobot metadata.",
    )

    parser.add_argument(
        "--close-threshold",
        type=float,
        default=1.5,
        help=(
            "Analysis threshold: target or predicted gripper values "
            ">= this value are counted as close."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum number of frames to audit. "
            "0 means audit the entire dataset."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/offline_gripper_audit.csv",
        help="CSV path for frame-level audit results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    device = torch.device("cpu")

    print("=" * 70)
    print("Offline ACT gripper prediction audit")
    print("=" * 70)
    print("Checkpoint:", checkpoint)
    print("Dataset root:", args.dataset_root)
    print("Close threshold:", args.close_threshold)

    # Load the local dataset.
    #
    # download_videos=False means:
    # do not try to download anything from the internet.
    # The local video files already exist in this dataset.
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        download_videos=False,
    )

    print("\nDataset:")
    print("  frames:", len(dataset))
    print("  fps:", dataset.fps)
    print("  features:", list(dataset.features))

    # Load the trained ACT policy and its saved preprocessing pipelines.
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
    )

    print("\nPolicy:")
    print("  type:", policy.__class__.__name__)
    print("  chunk_size:", policy.config.chunk_size)
    print("  n_action_steps:", policy.config.n_action_steps)
    print("  input features:", list(policy.config.input_features))

    total_frames = len(dataset)
    if args.limit > 0:
        total_frames = min(total_frames, args.limit)

    rows = []

    for index in range(total_frames):
        sample = dataset[index]

        # The target action recorded by the human demonstrator.
        target_action = sample["action"].detach().cpu().float().reshape(-1)

        # Keep exactly the observation features expected by ACT.
        observation = {
            key: sample[key]
            for key in policy.config.input_features
        }

        # IMPORTANT:
        # select_action() normally uses ACT's internal action queue.
        # We reset it so this frame is evaluated independently.
        policy.reset()

        processed_observation = preprocessor.process_observation(
            observation
        )

        with torch.inference_mode():
            predicted_action = policy.select_action(
                batch=processed_observation
            )

        # Convert normalized policy output back to the original action units.
        predicted_action = postprocessor.process_action(
            predicted_action
        )

        predicted_action = (
            predicted_action.detach().cpu().float().reshape(-1)
        )

        episode_index = int(sample["episode_index"].item())
        frame_index = int(sample["frame_index"].item())

        target_gripper = float(target_action[3].item())
        predicted_gripper = float(predicted_action[3].item())

        target_is_close = target_gripper >= args.close_threshold
        predicted_is_close = predicted_gripper >= args.close_threshold

        rows.append(
            {
                "dataset_index": index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "target_gripper": target_gripper,
                "predicted_gripper": predicted_gripper,
                "target_is_close": int(target_is_close),
                "predicted_is_close": int(predicted_is_close),
                "target_delta_x": float(target_action[0].item()),
                "target_delta_y": float(target_action[1].item()),
                "target_delta_z": float(target_action[2].item()),
                "predicted_delta_x": float(predicted_action[0].item()),
                "predicted_delta_y": float(predicted_action[1].item()),
                "predicted_delta_z": float(predicted_action[2].item()),
            }
        )

        if (index + 1) % 50 == 0 or index == total_frames - 1:
            print(f"Audited {index + 1}/{total_frames} frames")

    # Save frame-level results.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Convert important columns to NumPy arrays for summary statistics.
    target_gripper = np.array(
        [row["target_gripper"] for row in rows],
        dtype=np.float32,
    )
    predicted_gripper = np.array(
        [row["predicted_gripper"] for row in rows],
        dtype=np.float32,
    )

    target_is_close = target_gripper >= args.close_threshold
    predicted_is_close = predicted_gripper >= args.close_threshold

    close_rows = target_is_close
    hold_rows = ~target_is_close

    print("\n" + "=" * 70)
    print("Audit summary")
    print("=" * 70)

    print("Results saved to:", output_path)
    print("Audited frames:", len(rows))

    print("\nTarget label distribution:")
    print("  target hold frames:", int(hold_rows.sum()))
    print("  target close frames:", int(close_rows.sum()))

    print("\nGripper prediction statistics:")
    print(
        "  predicted gripper mean:",
        f"{predicted_gripper.mean():.4f}",
    )
    print(
        "  predicted gripper min:",
        f"{predicted_gripper.min():.4f}",
    )
    print(
        "  predicted gripper max:",
        f"{predicted_gripper.max():.4f}",
    )

    if close_rows.any():
        close_prediction = predicted_gripper[close_rows]
        close_hit_rate = predicted_is_close[close_rows].mean()

        print("\nOn target close frames:")
        print(
            "  target close frame count:",
            int(close_rows.sum()),
        )
        print(
            "  predicted gripper mean:",
            f"{close_prediction.mean():.4f}",
        )
        print(
            "  predicted gripper min:",
            f"{close_prediction.min():.4f}",
        )
        print(
            "  predicted gripper max:",
            f"{close_prediction.max():.4f}",
        )
        print(
            "  close hit rate:",
            f"{close_hit_rate * 100:.1f}%",
        )

    if hold_rows.any():
        false_close_rate = predicted_is_close[hold_rows].mean()

        print("\nOn target hold frames:")
        print(
            "  target hold frame count:",
            int(hold_rows.sum()),
        )
        print(
            "  false close rate:",
            f"{false_close_rate * 100:.1f}%",
        )

    # Print all target-close rows for direct inspection.
    print("\nFrame-level predictions on target close frames:")
    for row in rows:
        if row["target_is_close"]:
            print(
                f"  ep={row['episode_index']:2d} "
                f"frame={row['frame_index']:3d} "
                f"target_g={row['target_gripper']:.3f} "
                f"pred_g={row['predicted_gripper']:.3f}"
            )


if __name__ == "__main__":
    main()