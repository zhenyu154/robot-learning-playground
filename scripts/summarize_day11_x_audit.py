"""Summarize offline ACT delta_x predictions by scheduled X position."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HOME_X = 0.40


def summarize(rows: list[dict[str, str]], threshold: float, expected_sign: float) -> dict[str, float | int]:
    target = np.asarray([float(row["target_delta_x"]) for row in rows])
    predicted = np.asarray([float(row["predicted_delta_x"]) for row in rows])
    active = np.abs(target) > 1e-6
    predicted_active = np.abs(predicted) >= threshold
    correct = active & predicted_active & (np.sign(target) == np.sign(predicted))
    neutral = ~active
    false_motion = neutral & predicted_active
    return {
        "frames": len(rows),
        "target_active": int(active.sum()),
        "sign_correct": int(correct.sum()),
        "sign_accuracy": float(correct.sum() / active.sum()) if active.any() else 0.0,
        "target_mean_active": float(target[active].mean()) if active.any() else 0.0,
        "pred_mean_active": float(predicted[active].mean()) if active.any() else 0.0,
        "pred_abs_mean_active": float(np.abs(predicted[active]).mean()) if active.any() else 0.0,
        "neutral_frames": int(neutral.sum()),
        "false_motion": int(false_motion.sum()),
        "neutral_false_motion_rate": float(false_motion.sum() / neutral.sum()) if neutral.any() else 0.0,
        "expected_sign": expected_sign,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--position-history", type=Path, required=True)
    parser.add_argument("--sign-threshold", type=float, default=0.05)
    parser.add_argument("--home-x", type=float, default=HOME_X)
    args = parser.parse_args()
    if args.sign_threshold <= 0:
        raise ValueError("--sign-threshold must be positive")

    history = json.loads(args.position_history.read_text())
    with args.audit_csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    by_episode: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_index"])].append(row)
    if len(history) != len(by_episode):
        raise ValueError("Position history episode count does not match audit CSV")

    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    print("Audit CSV:", args.audit_csv)
    print("Position history:", args.position_history)
    print("Prediction magnitude threshold:", args.sign_threshold)
    print("Home X reference:", args.home_x)
    print()

    for episode, position in enumerate(history):
        episode_rows = by_episode.get(episode, [])
        if not episode_rows:
            raise ValueError(f"No audit rows for episode {episode}")
        x_position = round(float(position["x"]), 6)
        grouped[x_position].extend(episode_rows)
        expected_sign = float(np.sign(x_position - args.home_x))
        stats = summarize(episode_rows, args.sign_threshold, expected_sign)
        print(
            f"ep={episode:02d} x={x_position:+.3f} "
            f"active_x={stats['target_active']:2d} "
            f"sign_accuracy={stats['sign_accuracy']:.1%} "
            f"target_mean={stats['target_mean_active']:+.3f} "
            f"pred_mean={stats['pred_mean_active']:+.3f} "
            f"neutral_false_motion={stats['neutral_false_motion_rate']:.1%}"
        )

    print("\nSummary by scheduled X position")
    for x_position in sorted(grouped):
        expected_sign = float(np.sign(x_position - args.home_x))
        stats = summarize(grouped[x_position], args.sign_threshold, expected_sign)
        print(
            f"x={x_position:+.3f}: frames={stats['frames']} "
            f"active={stats['target_active']} "
            f"sign_accuracy={stats['sign_accuracy']:.1%} "
            f"target_mean={stats['target_mean_active']:+.3f} "
            f"pred_mean={stats['pred_mean_active']:+.3f} "
            f"pred_abs_mean={stats['pred_abs_mean_active']:.3f} "
            f"neutral_false_motion={stats['neutral_false_motion_rate']:.1%}"
        )

    all_rows = [row for group_rows in grouped.values() for row in group_rows]
    total = summarize(all_rows, args.sign_threshold, 0.0)
    print(
        f"\nOverall: X sign accuracy={total['sign_accuracy']:.1%} "
        f"({total['sign_correct']}/{total['target_active']} active frames), "
        f"neutral false-motion={total['neutral_false_motion_rate']:.1%} "
        f"({total['false_motion']}/{total['neutral_frames']} neutral frames)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
