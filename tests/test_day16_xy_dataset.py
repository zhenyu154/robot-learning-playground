"""Synthetic Day16 phase-order and schedule tests; no GUI/training/videos."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_day16_xy_dataset import EXPECTED_POSITIONS, Limits, inspect_episode, main

ROOT = Path(__file__).resolve().parents[1]


def episode(position=(0.32, -0.075)):
    x_sign = -1 if position[0] < 0.401855 else 1
    y_sign = -1 if position[1] < 0 else 1
    return np.array(
        [[0, 0, 0, 1]] * 2
        + [[x_sign * 0.25, 0, 0, 1]] * 6
        + [[0, y_sign * 0.25, 0, 1]] * 6
        + [[0, 0, 0, 1]] * 4
        + [[0, 0, -0.5, 1]] * 5
        + [[0, 0, 0, 2]] * 6
        + [[0, 0, 0, 1]] * 2
        + [[0, 0, 0.5, 1]] * 10,
        dtype=np.float32,
    )


class PhaseTests(unittest.TestCase):
    def test_valid_stages_for_all_positions(self):
        for position in EXPECTED_POSITIONS:
            with self.subTest(position=position):
                m, errors = inspect_episode(episode(position), position)
                self.assertEqual(errors, [])
                self.assertEqual(m['hover'], 4)
                self.assertEqual(m['close_before_lift'], 6)
                self.assertEqual(m['close_hold'], 6)
                self.assertEqual(m['lift_run'], 10)

    def test_six_close_frames_are_accepted(self):
        self.assertEqual(inspect_episode(episode(), (0.32, -0.075))[1], [])

    def test_missing_lift_is_rejected(self):
        a = episode()
        a[-10:, 2] = 0
        self.assertIn('no positive-Z lift', inspect_episode(a, (0.32, -0.075))[1])

    def test_early_descent_is_rejected(self):
        a = episode()
        a[10, 2] = -0.5
        errors = inspect_episode(a, (0.32, -0.075))[1]
        self.assertTrue(any('after descent begins' in e for e in errors))
        self.assertTrue(any('hover' in e for e in errors))

    def test_lift_before_close_is_rejected(self):
        a = episode()
        a[20, 2] = 0.5
        self.assertIn('positive-Z motion starts before the close phase', inspect_episode(a, (0.32, -0.075))[1])

    def test_close_counts_after_lift_do_not_satisfy_pre_lift_requirement(self):
        a = episode()
        a[24:29, 3] = 1
        a[-10:, 3] = 2
        errors = inspect_episode(a, (0.32, -0.075))[1]
        self.assertTrue(any('only 1 close-command frames before lift' in e for e in errors))

    def test_fragmented_hover_is_not_counted_as_contiguous(self):
        a = episode()
        a[16, 3] = 2
        m, errors = inspect_episode(a, (0.32, -0.075))
        self.assertEqual(m['hover'], 1)
        self.assertTrue(any('hover' in e for e in errors))

    def test_fragmented_lift_is_not_long_enough(self):
        a = episode()
        a[-6, 2] = 0
        errors = inspect_episode(a, (0.32, -0.075))[1]
        self.assertTrue(any('consecutive positive-Z' in e for e in errors))

    def test_simultaneous_xy_is_rejected(self):
        a = episode()
        a[3, 1] = -0.25
        self.assertTrue(any('simultaneous' in e for e in inspect_episode(a, (0.32, -0.075))[1]))

    def test_open_after_close_is_rejected_but_hold_is_allowed(self):
        a = episode()
        a[-5, 3] = 0
        self.assertTrue(any('open-gripper command' in e for e in inspect_episode(a, (0.32, -0.075))[1]))

    def test_invalid_arrays_and_limits(self):
        for a in (np.empty((0, 4)), np.zeros((3, 5)), np.full((3, 4), np.nan)):
            with self.assertRaises(ValueError):
                inspect_episode(a, (0.32, -0.075))
        with self.assertRaises(ValueError):
            Limits(min_close_frames=0)


class ScheduleTests(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / 'configs' / name).read_text())

    def keys(self, data):
        return [(p['x'], p['y']) for p in data['positions']]

    def test_recording_counts_and_balanced_blocks(self):
        smoke = self.keys(self.load('day16_xy_intermediate_smoke_positions.json'))
        train = self.keys(self.load('day16_xy_intermediate_train_positions.json'))
        self.assertEqual(set(smoke), set(EXPECTED_POSITIONS))
        self.assertEqual(len(smoke), 8)
        self.assertEqual(len(train), 40)
        for p in EXPECTED_POSITIONS:
            self.assertEqual(train.count(p), 5)
        for start in range(0, 40, 8):
            self.assertEqual(set(train[start:start + 8]), set(EXPECTED_POSITIONS))

    def test_evaluation_counts_and_split_separation(self):
        seen = self.keys(self.load('day16_xy_intermediate_seen_eval_positions.json'))
        development = self.load('day16_xy_development_positions.json')
        final = self.load('day16_xy_reserved_test_positions.json')
        self.assertEqual(len(seen), 24)
        self.assertEqual(len(self.keys(development)), 12)
        self.assertEqual(len(self.keys(final)), 12)
        for p in EXPECTED_POSITIONS:
            self.assertEqual(seen.count(p), 3)
        train = set(EXPECTED_POSITIONS)
        self.assertTrue(train.isdisjoint(self.keys(development)))
        self.assertTrue(train.isdisjoint(self.keys(final)))
        self.assertTrue(set(self.keys(development)).isdisjoint(self.keys(final)))
        self.assertEqual(development['split'], 'development_interpolation')
        self.assertEqual(final['split'], 'reserved_test')


class DatasetTests(unittest.TestCase):
    def test_end_to_end_synthetic_smoke_and_corrupt_history(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'meta').mkdir()
            (root / 'data/chunk-000').mkdir(parents=True)
            for camera in ('front', 'wrist'):
                d = root / f'videos/observation.images.{camera}/chunk-000'
                d.mkdir(parents=True)
                # The audit only checks file presence; decoding is manual QA.
                (d / 'file-000.mp4').touch()
            positions = [{'x': x, 'y': y} for x, y in EXPECTED_POSITIONS]
            history = [{'episode_index': i, **p} for i, p in enumerate(positions)]
            (root / 'position_schedule.json').write_text(json.dumps({'split': 'smoke', 'positions': positions}))
            (root / 'position_history.json').write_text(json.dumps(history))
            rows = []
            for i, pos in enumerate(EXPECTED_POSITIONS):
                actions = episode(pos)
                rows.extend({
                    'action': a.tolist(), 'episode_index': i, 'frame_index': j,
                    'next.reward': float(j == len(actions) - 1),
                    'next.done': bool(j == len(actions) - 1),
                } for j, a in enumerate(actions))
            pq.write_table(pa.Table.from_pylist(rows), root / 'data/chunk-000/file-000.parquet')
            (root / 'meta/info.json').write_text(json.dumps({
                'fps': 10, 'total_frames': len(rows), 'total_episodes': 8,
                'features': {'action': {'shape': [4], 'names': ['delta_x', 'delta_y', 'delta_z', 'gripper']}},
            }))
            args = ['--dataset-root', str(root), '--episodes-per-position', '1']
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
                history[0]['x'] = 0.48
                (root / 'position_history.json').write_text(json.dumps(history))
                self.assertEqual(main(args), 1)


if __name__ == '__main__':
    unittest.main()
