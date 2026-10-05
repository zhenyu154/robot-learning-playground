"""QA a Day11 fixed-Y, intermediate-distance X-axis dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EXPECTED_X_VALUES = (0.32, 0.36, 0.44, 0.48)
HOME_X = 0.40


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes-per-position", type=int, default=5)
    parser.add_argument("--min-active-x-frames", type=int, default=4)
    parser.add_argument("--expected-x-values", type=float, nargs="+", default=list(EXPECTED_X_VALUES))
    args = parser.parse_args()
    if args.episodes_per_position <= 0 or args.min_active_x_frames <= 0:
        raise ValueError("episode count and minimum active frames must be positive")

    root = args.dataset_root
    schedule_path, history_path, info_path = (root / n for n in ("position_schedule.json", "position_history.json", "meta/info.json"))
    parquet_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    for path in (schedule_path, history_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not parquet_files:
        raise FileNotFoundError(root / "data")

    info = json.loads(info_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    history = json.loads(history_path.read_text())
    positions = schedule.get("positions", [])
    if len(positions) != len(history):
        raise ValueError("Schedule/history length mismatch")

    schedule_xy = [(round(float(p["x"]), 6), round(float(p["y"]), 6)) for p in positions]
    history_xy = [(round(float(p["x"]), 6), round(float(p["y"]), 6)) for p in history]
    if schedule_xy != history_xy:
        raise ValueError("position history does not match schedule")

    data = pq.read_table(parquet_files, columns=["action", "episode_index", "next.reward", "next.done"]).to_pandas()
    actions = np.stack(data["action"].to_numpy()).astype(np.float32)
    episodes = data["episode_index"].to_numpy(dtype=np.int64)
    expected_x = tuple(sorted(round(float(x), 6) for x in args.expected_x_values))
    counts = Counter(round(float(p["x"]), 6) for p in positions)

    print("Dataset root:", root)
    print("Split:", schedule.get("split"))
    print("FPS:", info.get("fps"))
    print("Episodes:", int(data["episode_index"].nunique()))
    print("Frames:", len(data))
    print("X schedule values:", sorted(counts))
    print("Episodes per X:", dict(sorted(counts.items())))
    print("Reward-positive frames:", int((data["next.reward"] > 0).sum()))
    print("Done frames:", int(data["next.done"].sum()))
    print("Front videos:", len(list(root.glob("videos/observation.images.front/**/*.mp4"))))
    print("Wrist videos:", len(list(root.glob("videos/observation.images.wrist/**/*.mp4"))))
    print()

    errors: list[str] = []
    if sorted(counts) != list(expected_x):
        errors.append(f"expected X values {list(expected_x)}, found {sorted(counts)}")
    for x in expected_x:
        if counts[x] != args.episodes_per_position:
            errors.append(f"X={x:.3f} has {counts[x]} episodes, expected {args.episodes_per_position}")
    if any(abs(y) > 1e-6 for _, y in schedule_xy):
        errors.append("Y must remain fixed at 0.00")
    if len({round(x, 6) for x, _ in schedule_xy}) != len(set(schedule_xy)):
        errors.append("duplicate schedule entries are not allowed")

    active_total = correct_total = 0
    failed: list[int] = []
    for episode, position in enumerate(positions):
        mask = episodes == episode
        if not mask.any():
            errors.append(f"episode {episode} has no frames")
            continue
        action = actions[mask]
        x_action = action[:, 0]
        active = np.abs(x_action) > 1e-6
        expected_sign = np.sign(float(position["x"]) - HOME_X)
        correct = active & (np.sign(x_action) == expected_sign)
        active_count, correct_count = int(active.sum()), int(correct.sum())
        x_accuracy = correct_count / active_count if active_count else 0.0
        y_active = int((np.abs(action[:, 1]) > 1e-6).sum())
        close_count = int((action[:, 3] >= 1.5).sum())
        reward = float(data.loc[mask, "next.reward"].sum())
        done = int(data.loc[mask, "next.done"].sum())
        if reward <= 0 or done == 0:
            failed.append(episode)
        active_total += active_count
        correct_total += correct_count
        print(
            f"ep={episode:02d} pos=({position['x']:.2f},{position['y']:+.2f}) "
            f"frames={int(mask.sum()):3d} x_active={active_count:2d} "
            f"x_sign_match={x_accuracy:.0%} y_active={y_active:2d} "
            f"close={close_count:2d} reward={reward:.1f} done={done}"
        )
        if active_count < args.min_active_x_frames:
            errors.append(f"episode {episode} has only {active_count} active X frames")
        if active_count and x_accuracy < 0.8:
            errors.append(f"episode {episode} X sign agreement is {x_accuracy:.0%} (<80%)")
        if y_active > max(args.min_active_x_frames, active_count // 2):
            errors.append(f"episode {episode} has {y_active} active Y frames; expected fixed Y")

    if failed:
        errors.append(f"failed demonstrations: {failed}")
    total_accuracy = correct_total / active_total if active_total else 0.0
    print()
    print("Total active X frames:", active_total)
    print("Overall X sign agreement:", f"{total_accuracy:.1%}")
    print("Failed demonstrations:", failed)
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
