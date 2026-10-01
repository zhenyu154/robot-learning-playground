"""QA the Day7 Y-axis isolation dataset and action-sign supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/wusanggg/robotics/data/panda_pick_cube_day7_y_axis_v1"),
    )
    parser.add_argument("--min-active-y-frames", type=int, default=8)
    args = parser.parse_args()

    root = args.dataset_root
    schedule_path = root / "position_schedule.json"
    history_path = root / "position_history.json"
    info_path = root / "meta/info.json"
    parquet_files = sorted(root.glob("data/chunk-*/file-*.parquet"))

    for required in (schedule_path, history_path, info_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required Day7 QA file missing: {required}")
    if not parquet_files:
        raise FileNotFoundError(f"No frame parquet found under {root / 'data'}")

    info = json.loads(info_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    history = json.loads(history_path.read_text())
    positions = schedule["positions"]
    if len(positions) != len(history):
        raise ValueError(
            f"Schedule/history length mismatch: {len(positions)} vs {len(history)}"
        )

    data = pq.read_table(
        parquet_files,
        columns=["action", "episode_index", "next.reward", "next.done"],
    ).to_pandas()
    actions = np.stack(data["action"].to_numpy()).astype(np.float32)
    episodes = data["episode_index"].to_numpy(dtype=np.int64)

    print("Dataset root:", root)
    print("Split:", schedule.get("split"))
    print("FPS:", info.get("fps"))
    print("Episodes:", int(data["episode_index"].nunique()))
    print("Frames:", len(data))
    print("Y schedule values:", sorted({float(p["y"]) for p in positions}))
    print("Reward-positive frames:", int((data["next.reward"] > 0).sum()))
    print("Done frames:", int(data["next.done"].sum()))
    print()

    active_total = 0
    correct_total = 0
    failed_episodes: list[int] = []
    errors: list[str] = []

    for episode, position in enumerate(positions):
        mask = episodes == episode
        if not mask.any():
            errors.append(f"episode {episode} has no frames")
            continue

        action = actions[mask]
        y = action[:, 1]
        active = np.abs(y) > 1e-6
        expected_sign = np.sign(float(position["y"]))
        correct = active & (np.sign(y) == expected_sign)
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
            errors.append(
                f"episode {episode} has only {active_count} active Y frames "
                f"(minimum {args.min_active_y_frames})"
            )
        if active_count and sign_accuracy < 0.8:
            errors.append(f"episode {episode} Y sign agreement is {sign_accuracy:.0%} (<80%)")
        if x_active > max(args.min_active_y_frames, active_count // 2):
            errors.append(f"episode {episode} has {x_active} active X frames; expected fixed X")

    total_sign_accuracy = correct_total / active_total if active_total else 0.0
    print()
    print("Total active Y frames:", active_total)
    print("Overall Y sign agreement:", f"{total_sign_accuracy:.1%}")
    print("Failed demonstrations:", failed_episodes)

    y_values = {round(float(p["y"]), 6) for p in positions}
    if not any(y < 0 for y in y_values) or not any(y > 0 for y in y_values):
        errors.append("schedule must include negative and positive Y positions")
    if len(y_values) != 2:
        errors.append(f"expected exactly two Y positions, found {sorted(y_values)}")
    if len({round(float(p["x"]), 6) for p in positions}) != 1:
        errors.append("X position is not fixed across the schedule")
    if failed_episodes:
        errors.append(f"{len(failed_episodes)} demonstrations did not record success")

    if errors:
        print("\nQA: FAIL")
        for error in errors:
            print("-", error)
        return 1

    print("\nQA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
