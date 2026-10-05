"""Summarize offline ACT X/Y predictions for the Day12 2D dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def axis_stats(rows: list[dict[str, str]], target_key: str, pred_key: str, threshold: float) -> dict[str, float | int]:
    target = np.asarray([float(row[target_key]) for row in rows])
    predicted = np.asarray([float(row[pred_key]) for row in rows])
    active = np.abs(target) > 1e-6
    pred_active = np.abs(predicted) >= threshold
    correct = active & pred_active & (np.sign(target) == np.sign(predicted))
    neutral = ~active
    false_motion = neutral & pred_active
    return {
        "frames": len(rows),
        "active": int(active.sum()),
        "sign_correct": int(correct.sum()),
        "sign_accuracy": float(correct.sum() / active.sum()) if active.any() else 0.0,
        "target_mean_active": float(target[active].mean()) if active.any() else 0.0,
        "pred_mean_active": float(predicted[active].mean()) if active.any() else 0.0,
        "pred_abs_mean_active": float(np.abs(predicted[active]).mean()) if active.any() else 0.0,
        "neutral_frames": int(neutral.sum()),
        "false_motion": int(false_motion.sum()),
        "neutral_false_motion_rate": float(false_motion.sum() / neutral.sum()) if neutral.any() else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--position-history", type=Path, required=True)
    parser.add_argument("--motion-threshold", type=float, default=0.05)
    parser.add_argument("--close-threshold", type=float, default=1.5)
    args = parser.parse_args()
    if args.motion_threshold <= 0 or args.close_threshold <= 0:
        raise ValueError("thresholds must be positive")

    history = json.loads(args.position_history.read_text())
    with args.audit_csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    by_episode: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_index"])].append(row)
    if len(history) != len(by_episode):
        raise ValueError("Position history episode count does not match audit CSV")

    grouped: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for episode, position in enumerate(history):
        episode_rows = by_episode.get(episode, [])
        if not episode_rows:
            raise ValueError(f"No rows for episode {episode}")
        grouped[(round(float(position["x"]), 6), round(float(position["y"]), 6))].extend(episode_rows)

    print("Audit CSV:", args.audit_csv)
    print("Position history:", args.position_history)
    print("Motion threshold:", args.motion_threshold)
    print("Close threshold:", args.close_threshold)
    print()

    for position in sorted(grouped):
        rows_at_position = grouped[position]
        x = axis_stats(rows_at_position, "target_delta_x", "predicted_delta_x", args.motion_threshold)
        y = axis_stats(rows_at_position, "target_delta_y", "predicted_delta_y", args.motion_threshold)
        close_rows = [row for row in rows_at_position if float(row["target_gripper"]) >= args.close_threshold]
        hold_rows = [row for row in rows_at_position if float(row["target_gripper"]) < args.close_threshold]
        close_hit = sum(float(row["predicted_gripper"]) >= args.close_threshold for row in close_rows)
        false_close = sum(float(row["predicted_gripper"]) >= args.close_threshold for row in hold_rows)
        print(
            f"pos=({position[0]:+.3f},{position[1]:+.3f}) frames={len(rows_at_position)} "
            f"X_hit={x['sign_accuracy']:.1%} X_pred={x['pred_mean_active']:+.3f} "
            f"Y_hit={y['sign_accuracy']:.1%} Y_pred={y['pred_mean_active']:+.3f} "
            f"X_neutral_false={x['neutral_false_motion_rate']:.1%} "
            f"Y_neutral_false={y['neutral_false_motion_rate']:.1%} "
            f"close_hit={close_hit / len(close_rows):.1%} "
            f"false_close={false_close / len(hold_rows):.1%}"
        )

    all_x = axis_stats(rows, "target_delta_x", "predicted_delta_x", args.motion_threshold)
    all_y = axis_stats(rows, "target_delta_y", "predicted_delta_y", args.motion_threshold)
    close_rows = [row for row in rows if float(row["target_gripper"]) >= args.close_threshold]
    hold_rows = [row for row in rows if float(row["target_gripper"]) < args.close_threshold]
    close_hit = sum(float(row["predicted_gripper"]) >= args.close_threshold for row in close_rows)
    false_close = sum(float(row["predicted_gripper"]) >= args.close_threshold for row in hold_rows)
    print("\nOverall")
    print(f"X sign accuracy: {all_x['sign_accuracy']:.1%} ({all_x['sign_correct']}/{all_x['active']})")
    print(f"Y sign accuracy: {all_y['sign_accuracy']:.1%} ({all_y['sign_correct']}/{all_y['active']})")
    print(f"X neutral false-motion: {all_x['neutral_false_motion_rate']:.1%}")
    print(f"Y neutral false-motion: {all_y['neutral_false_motion_rate']:.1%}")
    print(f"Gripper close hit rate: {close_hit / len(close_rows):.1%}")
    print(f"Gripper hold false-close rate: {false_close / len(hold_rows):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
