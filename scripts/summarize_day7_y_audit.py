"""Summarize offline ACT delta_y predictions by the scheduled Y position."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize_group(rows: list[dict[str, str]], sign_threshold: float) -> dict[str, float | int]:
    target = np.asarray([float(row["target_delta_y"]) for row in rows])
    predicted = np.asarray([float(row["predicted_delta_y"]) for row in rows])
    active = np.abs(target) > 1e-6
    predicted_active = np.abs(predicted) >= sign_threshold
    correct_sign = active & predicted_active & (np.sign(target) == np.sign(predicted))
    neutral = ~active
    false_motion = neutral & predicted_active
    return {
        "frames": len(rows),
        "target_active": int(active.sum()),
        "sign_correct": int(correct_sign.sum()),
        "sign_accuracy": float(correct_sign.sum() / active.sum()) if active.any() else 0.0,
        "target_y_mean_active": float(target[active].mean()) if active.any() else 0.0,
        "pred_y_mean_active": float(predicted[active].mean()) if active.any() else 0.0,
        "pred_abs_mean_active": float(np.abs(predicted[active]).mean()) if active.any() else 0.0,
        "neutral_frames": int(neutral.sum()),
        "false_motion": int(false_motion.sum()),
        "neutral_false_motion_rate": float(false_motion.sum() / neutral.sum()) if neutral.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("results/offline_gripper_audit_day7_y_axis_xpu_2000.csv"),
    )
    parser.add_argument(
        "--position-history",
        type=Path,
        default=Path("/home/wusanggg/robotics/data/panda_pick_cube_day7_y_axis_v1/position_history.json"),
    )
    parser.add_argument(
        "--sign-threshold",
        type=float,
        default=0.05,
        help="Predicted |delta_y| must meet this magnitude to count as a directional prediction.",
    )
    args = parser.parse_args()
    if args.sign_threshold <= 0:
        raise ValueError("--sign-threshold must be positive")

    history = json.loads(args.position_history.read_text())
    with args.audit_csv.open(newline="") as file:
        frame_rows = list(csv.DictReader(file))

    by_episode: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        by_episode[int(row["episode_index"])].append(row)
    if len(history) != len(by_episode):
        raise ValueError(
            f"Position-history episodes ({len(history)}) do not match audit CSV "
            f"episodes ({len(by_episode)})"
        )

    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    print("Audit CSV:", args.audit_csv)
    print("Position history:", args.position_history)
    print("Prediction magnitude threshold:", args.sign_threshold)
    print()

    for episode, position in enumerate(history):
        rows = by_episode.get(episode, [])
        if not rows:
            raise ValueError(f"No audit rows for episode {episode}")
        y_position = round(float(position["y"]), 6)
        grouped[y_position].extend(rows)
        stats = summarize_group(rows, args.sign_threshold)
        print(
            f"ep={episode:02d} y={y_position:+.3f} "
            f"active_y={stats['target_active']:2d} "
            f"sign_accuracy={stats['sign_accuracy']:.1%} "
            f"target_mean={stats['target_y_mean_active']:+.3f} "
            f"pred_mean={stats['pred_y_mean_active']:+.3f} "
            f"neutral_false_motion={stats['neutral_false_motion_rate']:.1%}"
        )

    print("\nSummary by scheduled Y position")
    for y_position in sorted(grouped):
        stats = summarize_group(grouped[y_position], args.sign_threshold)
        print(
            f"y={y_position:+.3f}: frames={stats['frames']} "
            f"active={stats['target_active']} "
            f"sign_accuracy={stats['sign_accuracy']:.1%} "
            f"target_mean={stats['target_y_mean_active']:+.3f} "
            f"pred_mean={stats['pred_y_mean_active']:+.3f} "
            f"pred_abs_mean={stats['pred_abs_mean_active']:.3f} "
            f"neutral_false_motion={stats['neutral_false_motion_rate']:.1%}"
        )

    all_rows = [row for group_rows in grouped.values() for row in group_rows]
    total = summarize_group(all_rows, args.sign_threshold)
    print(
        f"\nOverall: Y sign accuracy={total['sign_accuracy']:.1%} "
        f"({total['sign_correct']}/{total['target_active']} active frames), "
        f"neutral false-motion={total['neutral_false_motion_rate']:.1%} "
        f"({total['false_motion']}/{total['neutral_frames']} neutral frames)"
    )


if __name__ == "__main__":
    main()
