"""QA a staged Day14 2D pick-and-lift dataset.

The audit checks the temporal structure that Day13 failed to execute reliably:
XY approach -> neutral hover -> descent -> stationary close -> positive-Z lift.
It uses only recorded action labels and episode metadata; videos still require
manual inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EXPECTED = ((0.32, -0.075), (0.32, 0.075), (0.48, -0.075), (0.48, 0.075))
EPS = 1e-6


def _count_consecutive(mask: np.ndarray, start: int, stop: int | None = None) -> int:
    """Count true entries from start until the first false entry."""
    end = len(mask) if stop is None else min(stop, len(mask))
    count = 0
    for index in range(start, end):
        if not bool(mask[index]):
            break
        count += 1
    return count


def _episode_metrics(action: np.ndarray) -> dict[str, int | float | bool]:
    x_active = np.abs(action[:, 0]) > EPS
    y_active = np.abs(action[:, 1]) > EPS
    z_active = np.abs(action[:, 2]) > EPS
    xy_active = x_active | y_active
    descend = action[:, 2] < -EPS
    lift = action[:, 2] > EPS
    close = action[:, 3] >= 1.5
    neutral_hover = (~xy_active) & (~z_active) & (~close)
    close_stationary = close & (~xy_active) & (~z_active)

    first_descend = int(np.flatnonzero(descend)[0]) if descend.any() else -1
    xy_before_descend = np.flatnonzero(xy_active & (np.arange(len(action)) < max(first_descend, 0)))
    last_xy_before_descend = int(xy_before_descend[-1]) if xy_before_descend.size else -1
    if first_descend >= 0 and last_xy_before_descend >= 0:
        hover_count = int(neutral_hover[last_xy_before_descend + 1:first_descend].sum())
    else:
        hover_count = 0

    close_after_descend = np.flatnonzero(close & (np.arange(len(action)) > first_descend)) if first_descend >= 0 else np.array([], dtype=int)
    first_close = int(close_after_descend[0]) if close_after_descend.size else -1
    lift_after_close = np.flatnonzero(lift & (np.arange(len(action)) > first_close)) if first_close >= 0 else np.array([], dtype=int)
    first_lift = int(lift_after_close[0]) if lift_after_close.size else -1

    if first_close >= 0 and first_lift >= 0:
        close_hold_count = int(close_stationary[first_close:first_lift].sum())
    else:
        close_hold_count = 0
    lift_count = int(lift[first_lift:].sum()) if first_lift >= 0 else 0
    xy_during_lift = int(xy_active[first_lift:].sum()) if first_lift >= 0 else 0
    close_before_descend = int(close[:first_descend].sum()) if first_descend >= 0 else int(close.sum())

    return {
        "frames": len(action),
        "x_active": int(x_active.sum()),
        "y_active": int(y_active.sum()),
        "xy_overlap": int((x_active & y_active).sum()),
        "first_descend": first_descend,
        "last_xy_before_descend": last_xy_before_descend,
        "hover": hover_count,
        "close": int(close.sum()),
        "close_before_descend": close_before_descend,
        "first_close": first_close,
        "close_hold": close_hold_count,
        "first_lift": first_lift,
        "lift": lift_count,
        "xy_during_lift": xy_during_lift,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes-per-position", type=int, default=5)
    parser.add_argument("--min-active-frames", type=int, default=4)
    parser.add_argument("--min-hover-frames", type=int, default=3)
    parser.add_argument("--min-close-frames", type=int, default=8)
    parser.add_argument("--min-close-hold-frames", type=int, default=3)
    parser.add_argument("--min-lift-frames", type=int, default=8)
    parser.add_argument("--max-xy-during-lift", type=int, default=1)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.episodes_per_position, args.min_active_frames, args.min_hover_frames, args.min_close_frames, args.min_close_hold_frames, args.min_lift_frames)):
        raise ValueError("minimum counts must be positive")
    if args.max_xy_during_lift < 0:
        raise ValueError("--max-xy-during-lift must be non-negative")

    root = args.dataset_root
    schedule = json.loads((root / "position_schedule.json").read_text())
    history = json.loads((root / "position_history.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(root / "data")
    if len(schedule.get("positions", [])) != len(history):
        raise ValueError("schedule/history mismatch")

    data = pq.read_table(files, columns=["action", "episode_index", "next.reward", "next.done"]).to_pandas()
    actions = np.stack(data["action"].to_numpy()).astype(np.float32)
    episodes = data["episode_index"].to_numpy(dtype=np.int64)
    positions = [(round(float(p["x"]), 6), round(float(p["y"]), 6)) for p in schedule["positions"]]
    history_positions = [(round(float(p["x"]), 6), round(float(p["y"]), 6)) for p in history]
    counts = Counter(positions)
    errors: list[str] = []

    print("Dataset root:", root)
    print("Split:", schedule.get("split"))
    print("FPS:", info.get("fps"))
    print("Episodes:", int(data["episode_index"].nunique()))
    print("Frames:", len(data))
    print("Position counts:", dict(sorted(counts.items())))
    print("Reward-positive frames:", int((data["next.reward"] > 0).sum()))
    print("Done frames:", int(data["next.done"].sum()))
    print("Front videos:", len(list(root.glob("videos/observation.images.front/**/*.mp4"))))
    print("Wrist videos:", len(list(root.glob("videos/observation.images.wrist/**/*.mp4"))))
    print()

    if positions != history_positions:
        errors.append("position history does not match position schedule")
    if sorted(counts) != sorted(EXPECTED):
        errors.append(f"expected positions {list(EXPECTED)}, found {sorted(counts)}")
    for position in EXPECTED:
        if counts[position] != args.episodes_per_position:
            errors.append(f"{position} has {counts[position]} episodes")

    failed: list[int] = []
    total_close = total_lift = 0
    for episode, position in enumerate(positions):
        mask = episodes == episode
        if not mask.any():
            errors.append(f"episode {episode} has no frames")
            continue
        action = actions[mask]
        metrics = _episode_metrics(action)
        reward = float(data.loc[mask, "next.reward"].sum())
        done = int(data.loc[mask, "next.done"].sum())
        if reward <= 0 or done == 0:
            failed.append(episode)

        total_close += int(metrics["close"])
        total_lift += int(metrics["lift"])
        print(
            f"ep={episode:02d} pos=({position[0]:.3f},{position[1]:+.3f}) "
            f"frames={metrics['frames']:3d} x_active={metrics['x_active']:2d} "
            f"y_active={metrics['y_active']:2d} xy_overlap={metrics['xy_overlap']:2d} "
            f"hover={metrics['hover']:2d} descend={metrics['first_descend']:3d} "
            f"close={metrics['close']:2d} close_hold={metrics['close_hold']:2d} "
            f"lift={metrics['lift']:2d} xy_during_lift={metrics['xy_during_lift']:2d} "
            f"reward={reward:.1f} done={done}"
        )

        if int(metrics["x_active"]) < args.min_active_frames or int(metrics["y_active"]) < args.min_active_frames:
            errors.append(f"episode {episode} has insufficient X/Y active frames")
        if int(metrics["xy_overlap"]) > 0:
            errors.append(f"episode {episode} has simultaneous X/Y activity")
        if int(metrics["first_descend"]) < 0:
            errors.append(f"episode {episode} has no descent")
        if int(metrics["hover"]) < args.min_hover_frames:
            errors.append(f"episode {episode} has only {metrics['hover']} neutral hover frames")
        if int(metrics["close_before_descend"]) > 0:
            errors.append(f"episode {episode} closes before descent")
        if int(metrics["close"]) < args.min_close_frames:
            errors.append(f"episode {episode} has only {metrics['close']} close frames")
        if int(metrics["close_hold"]) < args.min_close_hold_frames:
            errors.append(f"episode {episode} has only {metrics['close_hold']} stationary closed frames before lift")
        if int(metrics["first_lift"]) < 0:
            errors.append(f"episode {episode} has no positive-Z lift after closing")
        if int(metrics["lift"]) < args.min_lift_frames:
            errors.append(f"episode {episode} has only {metrics['lift']} lift frames")
        if int(metrics["xy_during_lift"]) > args.max_xy_during_lift:
            errors.append(f"episode {episode} has {metrics['xy_during_lift']} XY frames during lift")

    if failed:
        errors.append(f"failed demonstrations: {failed}")
    print()
    print("Total close frames:", total_close)
    print("Total positive-Z lift frames:", total_lift)
    print("Failed demonstrations:", failed)
    print("Manual video check required: confirm arm visibility, hover, grasp, and lift phases.")
    if errors:
        print("\nQA: FAIL")
        for error in errors:
            print("-", error)
        return 1
    print("\nQA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
