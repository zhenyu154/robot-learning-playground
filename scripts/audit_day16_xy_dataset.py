"""QA Day16's eight-position staged pick-and-lift demonstrations.

Verify episode/position integrity and action phase ORDER, not just total label
counts. This is label QA: manual front/wrist video inspection is still required
to confirm visibility, grasp quality, and actual cube lift. Gripper=1 is a hold
command, not a command to open the fingers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

EXPECTED_POSITIONS = tuple(
    (x, y) for x in (0.32, 0.36, 0.44, 0.48) for y in (-0.075, 0.075)
)
EPSILON = 1e-6
# Centered home pose used in Day11--16, not the default x≈0.49 pose.
HOME_X = 0.401855


@dataclass(frozen=True)
class Limits:
    min_active_frames: int = 4
    min_hover_frames: int = 3
    min_close_frames: int = 6
    min_close_hold_frames: int = 3
    min_lift_frames: int = 8
    max_xy_during_lift: int = 1

    def __post_init__(self):
        counts = (
            self.min_active_frames, self.min_hover_frames, self.min_close_frames,
            self.min_close_hold_frames, self.min_lift_frames,
        )
        if any(value <= 0 for value in counts):
            raise ValueError('minimum frame counts must be positive')
        if self.max_xy_during_lift < 0:
            raise ValueError('maximum lateral frames during lift must be non-negative')


def first_frame(mask: np.ndarray) -> int:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else -1


def longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for active in mask:
        current = current + 1 if active else 0
        longest = max(longest, current)
    return longest


def trailing_run(mask: np.ndarray) -> int:
    count = 0
    for active in mask[::-1]:
        if not active:
            break
        count += 1
    return count


def inspect_episode(
    actions: np.ndarray, position: tuple[float, float], limits: Limits = Limits(),
) -> tuple[dict[str, int | float], list[str]]:
    """Return measured phases and errors for one chronologically ordered episode."""
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != 4 or not len(a):
        raise ValueError('expected a nonempty [frames, 4] action array')
    if not np.isfinite(a).all():
        raise ValueError('action array contains NaN or infinite values')
    if np.any(np.abs(a[:, :3]) > 1 + EPSILON) or np.any(a[:, 3] < -EPSILON) or np.any(a[:, 3] > 2 + EPSILON):
        raise ValueError('recorded action outside the normalized environment bounds')

    ax, ay, az = (np.abs(a[:, channel]) > EPSILON for channel in range(3))
    xy = ax | ay
    down, up = a[:, 2] < -EPSILON, a[:, 2] > EPSILON
    close = a[:, 3] >= 1.5
    hover = ~xy & ~az & ~close
    stationary_close = close & ~xy & ~az
    descend_frame = first_frame(down)
    close_frame = first_frame(close)
    lift_frame = first_frame(up)
    x_frames, y_frames = np.flatnonzero(ax), np.flatnonzero(ay)
    x_sign = float(np.mean(np.sign(a[ax, 0]) == np.sign(position[0] - HOME_X))) if ax.any() else 0.0
    y_sign = float(np.mean(np.sign(a[ay, 1]) == np.sign(position[1]))) if ay.any() else 0.0

    hover_frames = trailing_run(hover[:descend_frame]) if descend_frame >= 0 else 0
    ordered_close = close_frame >= 0 and lift_frame > close_frame
    close_before_lift = int(close[close_frame:lift_frame].sum()) if ordered_close else 0
    close_hold = longest_run(stationary_close[close_frame:lift_frame]) if ordered_close else 0
    lift_frames = int(up[lift_frame:].sum()) if lift_frame >= 0 else 0
    lift_run = longest_run(up[lift_frame:]) if lift_frame >= 0 else 0
    lateral_lift = int(xy[lift_frame:].sum()) if lift_frame >= 0 else 0

    metrics = {
        'frames': len(a), 'x_active': int(ax.sum()), 'y_active': int(ay.sum()),
        'x_sign': x_sign, 'y_sign': y_sign, 'xy_overlap': int((ax & ay).sum()),
        'hover': hover_frames, 'descend_frame': descend_frame,
        'descent': int(down.sum()), 'close_frame': close_frame,
        'close': int(close.sum()), 'close_before_lift': close_before_lift,
        'close_hold': close_hold, 'lift_frame': lift_frame, 'lift': lift_frames,
        'lift_run': lift_run, 'xy_during_lift': lateral_lift,
    }
    errors = []
    for axis, count, agreement in (('X', int(ax.sum()), x_sign), ('Y', int(ay.sum()), y_sign)):
        if count < limits.min_active_frames:
            errors.append(f'only {count} active {axis} frames (minimum {limits.min_active_frames})')
        if count and agreement < 0.8:
            errors.append(f'{axis} sign agreement is {agreement:.1%} (<80%)')
    if (ax & ay).any():
        errors.append('simultaneous X/Y actions violate the sequential approach protocol')
    if len(x_frames) and len(y_frames) and x_frames[-1] >= y_frames[0]:
        errors.append('X movement does not finish before Y movement begins')
    if descend_frame < 0:
        errors.append('no negative-Z descent')
    else:
        if xy[descend_frame:].any() and (
            xy[descend_frame:lift_frame].any() if lift_frame > descend_frame else True
        ):
            errors.append('lateral approach/correction continues after descent begins, before lift')
        if hover_frames < limits.min_hover_frames:
            errors.append(f'only {hover_frames} consecutive neutral hover frames immediately before descent')
    if close_frame < 0:
        errors.append('no close command')
    elif descend_frame < 0 or close_frame <= descend_frame:
        errors.append('close command starts before descent')
    elif down[close_frame:].any():
        errors.append('descent continues after the close phase starts')
    if lift_frame < 0:
        errors.append('no positive-Z lift')
    elif close_frame < 0 or lift_frame <= close_frame:
        errors.append('positive-Z motion starts before the close phase')
    else:
        if down[lift_frame:].any():
            errors.append('negative-Z actions occur after lift begins')
        if close_before_lift < limits.min_close_frames:
            errors.append(f'only {close_before_lift} close-command frames before lift (minimum {limits.min_close_frames})')
        if close_hold < limits.min_close_hold_frames:
            errors.append(f'only {close_hold} consecutive stationary close-command frames before lift')
    if close_frame >= 0 and np.any(a[close_frame:, 3] < 0.5):
        errors.append('open-gripper command occurs after closing; hold=1 is allowed during lift')
    if lift_run < limits.min_lift_frames:
        errors.append(f'only {lift_run} consecutive positive-Z lift frames (minimum {limits.min_lift_frames})')
    if lateral_lift > limits.max_xy_during_lift:
        errors.append(f'{lateral_lift} lateral-action frames during lift')
    return metrics, errors


def position_key(position: dict) -> tuple[float, float]:
    return (round(float(position['x']), 6), round(float(position['y']), 6))


def main(
    argv: list[str] | None = None,
    *,
    expected_counts: dict[tuple[float, float], int] | None = None,
    description: str = __doc__,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--episodes-per-position', type=int, default=5)
    parser.add_argument('--min-active-frames', type=int, default=4)
    parser.add_argument('--min-hover-frames', type=int, default=3)
    parser.add_argument('--min-close-frames', type=int, default=6)
    parser.add_argument('--min-close-hold-frames', type=int, default=3)
    parser.add_argument('--min-lift-frames', type=int, default=8)
    parser.add_argument('--max-xy-during-lift', type=int, default=1)
    args = parser.parse_args(argv)
    if args.episodes_per_position <= 0:
        parser.error('--episodes-per-position must be positive')
    limits = Limits(
        args.min_active_frames, args.min_hover_frames, args.min_close_frames,
        args.min_close_hold_frames, args.min_lift_frames, args.max_xy_during_lift,
    )
    # No torch/ACT/video decoding/GUI is needed for this audit.
    import pyarrow.parquet as pq

    root = args.dataset_root
    schedule = json.loads((root / 'position_schedule.json').read_text())
    history = json.loads((root / 'position_history.json').read_text())
    info = json.loads((root / 'meta/info.json').read_text())
    files = sorted(root.glob('data/chunk-*/file-*.parquet'))
    if not files:
        raise FileNotFoundError(root / 'data')
    positions = [position_key(p) for p in schedule['positions']]
    history_positions = [position_key(p) for p in history]
    errors = []
    if positions != history_positions:
        errors.append('position history does not match schedule')
    if any(p.get('episode_index') != i for i, p in enumerate(history)):
        errors.append('position history episode indices are not sequential')
    counts = Counter(positions)
    required = expected_counts if expected_counts is not None else {
        pos: args.episodes_per_position for pos in EXPECTED_POSITIONS
    }
    if not required or any(count <= 0 for count in required.values()):
        raise ValueError('expected position counts must be positive')
    if set(counts) != set(required):
        errors.append(f'expected positions {sorted(required)}, found {sorted(counts)}')
    for pos, count in required.items():
        if counts[pos] != count:
            errors.append(f'{pos} has {counts[pos]} scheduled episodes; expected {count}')
    if info.get('fps') != 10:
        errors.append('expected 10 FPS')
    feature = info.get('features', {}).get('action', {})
    if feature.get('shape') != [4] or feature.get('names') != ['delta_x', 'delta_y', 'delta_z', 'gripper']:
        errors.append('action feature must use [delta_x, delta_y, delta_z, gripper]')

    data = pq.read_table(files, columns=['action', 'episode_index', 'frame_index', 'next.reward', 'next.done']).to_pandas()
    data = data.sort_values(['episode_index', 'frame_index'])
    actual_ids = set(data['episode_index'].astype(int))
    if actual_ids != set(range(len(positions))):
        errors.append('recorded episode indices do not match schedule')
    if info.get('total_frames') != len(data) or info.get('total_episodes') != len(actual_ids):
        errors.append('metadata frame/episode counts do not match recorded data')
    video_counts = {
        camera: len(list(root.glob(f'videos/observation.images.{camera}/**/*.mp4')))
        for camera in ('front', 'wrist')
    }
    for camera, count in video_counts.items():
        if not count:
            errors.append(f'no {camera} video files')

    print('Dataset root:', root)
    print('Split:', schedule.get('split'))
    print('FPS:', info.get('fps'))
    print('Episodes:', len(actual_ids))
    print('Frames:', len(data))
    print('Position counts:', dict(sorted(counts.items())))
    print('Reward-positive frames:', int((data['next.reward'] > 0).sum()))
    print('Done frames:', int(data['next.done'].sum()))
    print('Front videos:', video_counts['front'])
    print('Wrist videos:', video_counts['wrist'])
    print('Phase indices below are zero-based recorded frame indices.\n')
    failed = []
    for episode, position in enumerate(positions):
        rows = data[data['episode_index'] == episode]
        if rows.empty:
            errors.append(f'episode {episode}: no frames')
            continue
        if not np.array_equal(rows['frame_index'].to_numpy(), np.arange(len(rows))):
            errors.append(f'episode {episode}: missing or duplicate frame indices')
        reward = float(rows['next.reward'].sum())
        done = int(rows['next.done'].sum())
        if reward <= 0 or done <= 0:
            failed.append(episode)
            errors.append(f'episode {episode}: missing success reward/done flag')
        try:
            m, episode_errors = inspect_episode(np.stack(rows['action'].to_numpy()), position, limits)
        except ValueError as exc:
            errors.append(f'episode {episode}: {exc}')
            continue
        print(
            f"ep={episode:02d} pos=({position[0]:.3f},{position[1]:+.3f}) "
            f"frames={m['frames']:3d} x_active={m['x_active']:2d} x_sign={m['x_sign']:.0%} "
            f"y_active={m['y_active']:2d} y_sign={m['y_sign']:.0%} "
            f"hover={m['hover']:2d} descend_at={m['descend_frame']:3d} "
            f"close_before_lift={m['close_before_lift']:2d} close_hold={m['close_hold']:2d} "
            f"lift_at={m['lift_frame']:3d} lift={m['lift']:2d} lift_run={m['lift_run']:2d} "
            f"xy_during_lift={m['xy_during_lift']:2d} reward={reward:.1f} done={done}"
        )
        errors.extend(f'episode {episode}: {error}' for error in episode_errors)
    print('\nFailed demonstrations:', failed)
    print('Manual video check required: visible arm, completed XY approach, stable grasp, and actual cube lift.')
    if errors:
        print('\nQA: FAIL')
        for error in errors:
            print('-', error)
        return 1
    print('\nQA: PASS (action labels/metadata; manual video check still required)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
