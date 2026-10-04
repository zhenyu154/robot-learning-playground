"""QA a Day10 four-position Y-axis dataset before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


EXPECTED_Y_VALUES = (-0.10, -0.05, 0.05, 0.10)


def close_enough(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes-per-position", type=int, default=5)
    parser.add_argument("--min-active-y-frames", type=int, default=4)
    parser.add_argument("--expected-y-values", type=float, nargs="+", default=list(EXPECTED_Y_VALUES))
    args = parser.parse_args()

    if args.episodes_per_position <= 0:
        raise ValueError("--episodes-per-position must be positive")
    if args.min_active_y_frames <= 0:
        raise ValueError("--min-active-y-frames must be positive")

    root = args.dataset_root
    schedule_path = root / "position_schedule.json"
    history_path = root / "position_history.json"
    info_path = root / "meta/info.json"
    parquet_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    required = (schedule_path, history_path, info_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required dataset metadata missing: {missing}")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    info = json.loads(info_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    history = json.loads(history_path.read_text())
    positions = schedule.get("positions", [])
    if len(positions) != len(history):
        raise ValueError(f"Schedule/history mismatch: {len(positions)} vs {len(history)}")

    expected_y = tuple(sorted({round(float(value), 6) for value in args.expected_y_values}))
    schedule_xy = [(round(float(p["x"]), 6), round(float(p["y"]), 6)) for p in positions]
    history_xy = [(round(float(p["x"]), 6), round(float(p["y"]), 6)) for p in history]
    if schedule_xy != history_xy:
        raise ValueError("position_history.json does not match position_schedule.json")

    data = pq.read_table(
        parquet_files,
        columns=["action", "episode_index", "next.reward", "next.done"],
    ).to_pandas()
    actions = np.stack(data["action"].to_numpy()).astype(np.float32)
    episodes = data["episode_index"].to_numpy(dtype=np.int64)

    counts = Counter(round(float(p["y"]), 6) for p in positions)
    print("Dataset root:", root)
    print("Split:", schedule.get("split"))
    print("FPS:", info.get("fps"))
    print("Episodes:", int(data["episode_index"].nunique()))
    print("Frames:", len(data))
    print("Y schedule values:", sorted(counts))
    print("Episodes per Y:", dict(sorted(counts.items())))
    print("Reward-positive frames:", int((data["next.reward"] > 0).sum()))
    print("Done frames:", int(data["next.done"].sum()))
    print("Front videos:", len(list(root.glob("videos/observation.images.front/**/*.mp4"))))
    print("Wrist videos:", len(list(root.glob("videos/observation.images.wrist/**/*.mp4"))))
    print()

    errors: list[str] = []
    if len(positions) != len(data["episode_index"].unique()):
        errors.append("schedule episode count does not match recorded episode count")
    if sorted(counts) != list(expected_y):
        errors.append(f"expected Y values {list(expected_y)}, found {sorted(counts)}")
    for y in expected_y:
        if counts[y] != args.episodes_per_position:
            errors.append(f"Y={y:+.3f} has {counts[y]} episodes, expected {args.episodes_per_position}")
    if len({x for x, _ in schedule_xy}) != 1 or not close_enough(schedule_xy[0][0], 0.48):
        errors.append("X must be fixed at 0.48")
    if any(abs(y) <= 1e-6 for _, y in schedule_xy):
        errors.append("training schedule must not contain y=0.00")

    active_total = 0
    correct_total = 0
    failed_episodes: list[int] = []
    episode_indices = set(int(value) for value in data["episode_index"].unique())
    for episode, position in enumerate(positions):
        mask = episodes == episode
        if not mask.any():
            errors.append(f"episode {episode} has no frames")
            continue
        action = actions[mask]
        y_action = action[:, 1]
        active = np.abs(y_action) > 1e-6
        expected_sign = np.sign(float(position["y"]))
        correct = active & (np.sign(y_action) == expected_sign)
        active_count = int(active.sum())
        correct_count = int(correct.sum())
        active_total += active_count
        correct_total += correct_count
        x_active = int((np.abs(action[:, 0]) > 1e-6).sum())
        close_count = int((action[:, 3] >= 1.5).sum())
        reward = float(data.loc[mask, "next.reward"].sum())
        done_count = int(data.loc[mask, "next.done"].sum())
        if reward <= 0 or done_count == 0:
            failed_episodes.append(episode)
        sign_accuracy = correct_count / active_count if active_count else 0.0
        print(
            f"ep={episode:02d} pos=({position['x']:.2f},{position['y']:+.2f}) "
            f"frames={int(mask.sum()):3d} y_active={active_count:2d} "
            f"y_sign_match={sign_accuracy:.0%} x_active={x_active:2d} "
            f"close={close_count:2d} reward={reward:.1f} done={done_count}"
        )
        if active_count < args.min_active_y_frames:
            errors.append(f"episode {episode} has only {active_count} active Y frames")
        if active_count and sign_accuracy < 0.8:
            errors.append(f"episode {episode} Y sign agreement is {sign_accuracy:.0%} (<80%)")
        if x_active > max(args.min_active_y_frames, active_count // 2):
            errors.append(f"episode {episode} has {x_active} active X frames; expected fixed X")

    if failed_episodes:
        errors.append(f"failed demonstrations: {failed_episodes}")
    if len(episode_indices) != len(positions):
        errors.append("recorded episode indices are incomplete")

    total_accuracy = correct_total / active_total if active_total else 0.0
    print()
    print("Total active Y frames:", active_total)
    print("Overall Y sign agreement:", f"{total_accuracy:.1%}")
    print("Failed demonstrations:", failed_episodes)
    print("Manual video check required: confirm Panda arm is visible in front/wrist videos.")

    if errors:
        print("\nQA: FAIL")
        for error in errors:
            print("-", error)
        return 1

    print("\nQA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
