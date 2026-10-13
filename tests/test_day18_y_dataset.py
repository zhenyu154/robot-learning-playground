"""Day18 collection/split checks and synthetic shorter-Y phase QA."""

from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_day16_xy_dataset import inspect_episode
from scripts.audit_day18_y_dataset import ADDED_POSITIONS, main

ROOT = Path(__file__).resolve().parents[1]


def short_y_episode(position):
    x_sign = -1 if position[0] < 0.401855 else 1
    y_sign = -1 if position[1] < 0 else 1
    # Shorter Y motion, same amplitudes and stage ordering as the old protocol.
    return np.asarray(
        [[0, 0, 0, 1]] * 2
        + [[x_sign * 0.25, 0, 0, 1]] * 7
        + [[0, y_sign * 0.25, 0, 1]] * 6
        + [[0, 0, 0, 1]] * 4
        + [[0, 0, -0.5, 1]] * 5
        + [[0, 0, 0, 2]] * 6
        + [[0, 0, 0, 1]] * 2
        + [[0, 0, 0.5, 1]] * 10,
        dtype=np.float32,
    )


def write_dataset(root, positions):
    (root / 'meta').mkdir()
    (root / 'data/chunk-000').mkdir(parents=True)
    for camera in ('front', 'wrist'):
        directory = root / f'videos/observation.images.{camera}/chunk-000'
        directory.mkdir(parents=True)
        # The phase audit checks presence only; real video decoding is separate.
        (directory / 'file-000.mp4').touch()
    history = [{'episode_index': i, 'x': x, 'y': y, 'z': 0.02} for i, (x, y) in enumerate(positions)]
    (root / 'position_history.json').write_text(json.dumps(history))
    (root / 'position_schedule.json').write_text(json.dumps({
        'split': 'smoke', 'positions': [{'x': x, 'y': y} for x, y in positions],
    }))
    rows = []
    for episode, position in enumerate(positions):
        actions = short_y_episode(position)
        rows.extend({
            'episode_index': episode, 'frame_index': j, 'action': action.tolist(),
            'next.reward': float(j == len(actions) - 1),
            'next.done': bool(j == len(actions) - 1),
        } for j, action in enumerate(actions))
    pq.write_table(pa.Table.from_pylist(rows), root / 'data/chunk-000/file-000.parquet')
    (root / 'meta/info.json').write_text(json.dumps({
        'fps': 10, 'total_episodes': len(positions), 'total_frames': len(rows),
        'features': {'action': {'shape': [4], 'names': ['delta_x', 'delta_y', 'delta_z', 'gripper']}},
    }))


class ScheduleTests(unittest.TestCase):
    def keys(self, name):
        data = json.loads((ROOT / 'configs' / name).read_text())
        return data, [(p['x'], p['y']) for p in data['positions']]

    def test_smoke_covers_exactly_four_added_positions(self):
        data, positions = self.keys('day18_y_intermediate_smoke_positions.json')
        self.assertEqual(data['split'], 'smoke')
        self.assertEqual(Counter(positions), Counter(ADDED_POSITIONS))

    def test_training_twenty_episodes_and_balanced_blocks(self):
        data, positions = self.keys('day18_y_intermediate_train_positions.json')
        self.assertEqual(data['split'], 'train')
        self.assertEqual(len(positions), 20)
        self.assertEqual(Counter(positions), Counter({p: 5 for p in ADDED_POSITIONS}))
        for index in range(0, 20, 4):
            self.assertEqual(set(positions[index:index + 4]), set(ADDED_POSITIONS))

    def test_added_positions_are_labeled_seen(self):
        data, positions = self.keys('day18_y_intermediate_seen_eval_positions.json')
        self.assertEqual(data['split'], 'seen_eval_added_positions')
        self.assertEqual(Counter(positions), Counter({p: 3 for p in ADDED_POSITIONS}))

    def test_development_and_reserved_splits_do_not_overlap_training(self):
        old_data, old = self.keys('day16_xy_intermediate_train_positions.json')
        del old_data
        data, development = self.keys('day18_xy_development_positions.json')
        reserved_data, reserved = self.keys('day16_xy_reserved_test_positions.json')
        self.assertEqual(data['split'], 'development_interpolation')
        self.assertEqual(reserved_data['split'], 'reserved_test')
        self.assertEqual(len(development), 12)
        self.assertEqual(Counter(development), Counter({(x, y): 3 for x in (0.36, 0.44) for y in (-0.05625, 0.05625)}))
        train = set(old) | set(ADDED_POSITIONS)
        self.assertTrue(train.isdisjoint(development))
        self.assertTrue(train.isdisjoint(reserved))
        self.assertTrue(set(development).isdisjoint(reserved))


class AuditTests(unittest.TestCase):
    def test_valid_shorter_y_phases_and_hold_command(self):
        for position in ADDED_POSITIONS:
            with self.subTest(position=position):
                metrics, errors = inspect_episode(short_y_episode(position), position)
                self.assertEqual(errors, [])
                self.assertEqual(metrics['y_active'], 6)
                self.assertEqual(metrics['hover'], 4)
                self.assertEqual(metrics['close_before_lift'], 6)
                self.assertEqual(metrics['lift_run'], 10)

    def test_wrong_sign_and_lateral_overlap_are_rejected(self):
        position = ADDED_POSITIONS[0]
        actions = short_y_episode(position)
        actions[actions[:, 1] != 0, 1] *= -1
        errors = inspect_episode(actions, position)[1]
        self.assertTrue(any('Y sign agreement' in error for error in errors))
        actions = short_y_episode(position)
        actions[3, 1] = -0.25
        self.assertTrue(any('simultaneous' in error for error in inspect_episode(actions, position)[1]))

    def test_end_to_end_smoke_and_training_counts(self):
        for repetitions in (1, 5):
            with self.subTest(repetitions=repetitions), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_dataset(root, list(ADDED_POSITIONS) * repetitions)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['--dataset-root', str(root), '--episodes-per-position', str(repetitions)]), 0)
                    self.assertEqual(main(['--dataset-root', str(root), '--episodes-per-position', str(repetitions + 1)]), 1)

    def test_old_y_magnitude_is_not_accepted_as_new_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, [(x, np.sign(y) * 0.075) for x, y in ADDED_POSITIONS])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--dataset-root', str(root), '--episodes-per-position', '1']), 1)


if __name__ == '__main__':
    unittest.main()
