"""Save representative frames from selected Day6 episodes for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from lerobot.datasets import LeRobotDataset


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a LeRobot CHW float image in [0, 1] to a PIL RGB image."""
    image = tensor.detach().cpu().float().clamp(0, 1)
    image = (image * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image, mode="RGB")


def make_contact_sheet(samples: list[tuple[str, Image.Image]], cell_width: int = 256) -> Image.Image:
    cell_height = cell_width + 28
    columns = len(samples)
    sheet = Image.new("RGB", (columns * cell_width, cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(samples):
        image = image.resize((cell_width, cell_width))
        x = index * cell_width
        sheet.paste(image, (x, 0))
        draw.text((x + 4, cell_width + 5), label, fill="black")
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/wusanggg/robotics/data/panda_pick_cube_day6_spatial_v1"),
    )
    parser.add_argument(
        "--repo-id",
        default="wusanggg/panda_pick_cube_day6_spatial_v1",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0, 5, 10, 15],
        help="Episode indices to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/day6_visual_check"),
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        download_videos=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.episodes)
    episode_frames: dict[int, list[tuple[int, dict]]] = {ep: [] for ep in selected}

    for index in range(len(dataset)):
        sample = dataset[index]
        episode = int(sample["episode_index"].item())
        if episode in selected:
            frame = int(sample["frame_index"].item())
            episode_frames[episode].append((frame, sample))

    for episode in args.episodes:
        frames = episode_frames.get(episode, [])
        if not frames:
            raise ValueError(f"Episode {episode} was not found in the dataset")

        close_indices = [
            index for index, (_, sample) in enumerate(frames)
            if float(sample["action"][3].item()) >= 1.5
        ]
        reward_indices = [
            index for index, (_, sample) in enumerate(frames)
            if float(sample["next.reward"].item()) > 0
        ]

        chosen_indices = {
            0: "start",
            len(frames) // 2: "middle",
            len(frames) - 1: "end",
        }
        if close_indices:
            chosen_indices[close_indices[0]] = "close_start"
            chosen_indices[close_indices[len(close_indices) // 2]] = "close_middle"
        if reward_indices:
            chosen_indices[reward_indices[-1]] = "success"

        # Keep one copy per frame index, even if multiple labels identify it.
        chosen: dict[int, list[str]] = {}
        for index, label in chosen_indices.items():
            chosen.setdefault(index, []).append(label)

        front_samples = []
        wrist_samples = []
        for index, labels in sorted(chosen.items()):
            frame_index, sample = frames[index]
            label = "+".join(labels)
            gripper = float(sample["action"][3].item())
            z_action = float(sample["action"][2].item())
            prefix = f"ep{episode:02d}_frame{frame_index:03d}_{label}_g{gripper:.2f}_z{z_action:.3f}"

            front = tensor_to_image(sample["observation.images.front"])
            wrist = tensor_to_image(sample["observation.images.wrist"])
            front.save(args.output_dir / f"{prefix}_front.png")
            wrist.save(args.output_dir / f"{prefix}_wrist.png")
            front_samples.append((label, front))
            wrist_samples.append((label, wrist))

        make_contact_sheet(front_samples).save(
            args.output_dir / f"episode_{episode:02d}_front_contact_sheet.png"
        )
        make_contact_sheet(wrist_samples).save(
            args.output_dir / f"episode_{episode:02d}_wrist_contact_sheet.png"
        )

        print(
            f"episode={episode}: frames={len(frames)}, "
            f"close_frames={len(close_indices)}, "
            f"success_frames={len(reward_indices)}"
        )

    print("Saved visual QA images to:", args.output_dir)


if __name__ == "__main__":
    main()
