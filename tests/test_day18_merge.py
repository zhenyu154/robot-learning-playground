"""Day18 mixture/provenance/task remapping tests, no GUI or policy training."""

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

from scripts.audit_day16_xy_dataset import EXPECTED_POSITIONS
from scripts.audit_day18_merged_dataset import EXPECTED_COUNTS, main as audit_merged
from scripts.audit_day18_y_dataset import ADDED_POSITIONS
from scripts.build_day17_dataset import fingerprint, verify_labels_and_stats
from scripts.build_day18_dataset import main, make_plan, verify_task_mapping


def write_dataset(root, positions, tasks):
    (root / 'meta').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    history, rows = [], []
    for ep, (x, y) in enumerate(positions):
        actions = np.array([[0, 0, 0, 1]] * 2
                           + [[-0.25 if x < 0.401855 else 0.25, 0, 0, 1]] * 6
                           + [[0, -0.25 if y < 0 else 0.25, 0, 1]] * 6
                           + [[0, 0, 0, 1]] * 4 + [[0, 0, -0.5, 1]] * 5
                           + [[0, 0, 0, 2]] * 6 + [[0, 0, 0.5, 1]] * 10, dtype=np.float32)
        history.append({'episode_index': ep, 'x': x, 'y': y, 'z': 0.02})
        for frame, action in enumerate(actions):
            rows.append({'episode_index': ep, 'frame_index': frame, 'index': len(rows),
                         'timestamp': frame / 10, 'action': action.tolist(),
                         'observation.state': [float(ep)] * 18,
                         'next.reward': float(frame == len(actions) - 1),
                         'next.done': bool(frame == len(actions) - 1),
                         'task_index': ep % len(tasks)})
    pq.write_table(pa.Table.from_pylist(rows), root / 'data/chunk-000/file-000.parquet')
    (root / 'meta/info.json').write_text(json.dumps({
        'codebase_version': 'v3.0', 'fps': 10, 'total_frames': len(rows),
        'total_episodes': len(history), 'robot_type': None,
        'features': {'action': {'shape': [4], 'dtype': 'float32', 'names': ['delta_x', 'delta_y', 'delta_z', 'gripper']}},
    }))
    (root / 'position_schedule.json').write_text(json.dumps({'split': 'train', 'positions': [{'x': x, 'y': y} for x, y in positions]}))
    (root / 'position_history.json').write_text(json.dumps(history))
    (root / 'collection_protocol.json').write_text(json.dumps({
        'fps': 10, 'xy_step_size': 0.25, 'z_step_size': 0.5,
        'physical_xy_step_m': 0.00625, 'physical_z_step_m': 0.0125,
        'home_pose': 'configs/day11_home_x_centered.json', 'min_tcp_z': 0.008,
    }))
    pq.write_table(pa.Table.from_pylist([{'task_index': i, 'task': task} for i, task in enumerate(tasks)]), root / 'meta/tasks.parquet')
    for camera in ('front', 'wrist'):
        directory = root / f'videos/observation.images.{camera}/chunk-000'
        directory.mkdir(parents=True)
        (directory / 'file-000.mp4').touch()
    return rows, history


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.old, self.added, self.output = (self.base / p for p in ('old', 'added', 'merged'))
        self.old_positions = [p for p in EXPECTED_POSITIONS if p[0] in (0.32, 0.48)] * 10 + [p for p in EXPECTED_POSITIONS if p[0] in (0.36, 0.44)] * 5
        self.old_rows, history = write_dataset(self.old, self.old_positions, ['day14-task', 'day16-task'])
        self.new_rows, _ = write_dataset(self.added, list(ADDED_POSITIONS) * 5, ['day18-task'])
        provenance = {'stage': 'verified_complete', 'episodes': 60, 'frames': len(self.old_rows), 'episode_provenance': []}
        for ep, pos in enumerate(history):
            provenance['episode_provenance'].append({**pos, 'frames': len(self.old_rows) // 60,
                'source': 'day14' if ep < 40 else 'day16', 'source_episode_index': ep if ep < 40 else (ep - 40)})
        (self.old / 'dataset_provenance.json').write_text(json.dumps(provenance))

    def plan(self):
        return make_plan(self.old, self.added, self.output, 'test/day18')

    def merged_fixture(self):
        plan, sources = self.plan()
        rows, _ = write_dataset(self.output, self.old_positions + list(ADDED_POSITIONS) * 5,
                                ['day14-task', 'day16-task', 'day18-task'])
        for ep in range(80):
            for row in rows:
                if row['episode_index'] == ep:
                    row['task_index'] = ep % 2 if ep < 60 else 2
        pq.write_table(pa.Table.from_pylist(rows), self.output / 'data/chunk-000/file-000.parquet')
        # New sources' state values use their own episode IDs; preserve them.
        for row in rows:
            if row['episode_index'] >= 60:
                row['observation.state'] = [float(row['episode_index'] - 60)] * 18
        pq.write_table(pa.Table.from_pylist(rows), self.output / 'data/chunk-000/file-000.parquet')
        stats = {}
        for key in ('action', 'observation.state'):
            values = np.stack([p[key] for p in rows])
            stats[key] = {name: value.tolist() for name, value in (
                ('mean', values.mean(0)), ('std', values.std(0)), ('min', values.min(0)), ('max', values.max(0)),
            )}
            stats[key]['count'] = [len(rows)]
        (self.output / 'meta/stats.json').write_text(json.dumps(stats))
        return plan, sources

    def test_preserves_all_old_and_all_added_episodes_with_ancestor_mapping(self):
        plan, _ = self.plan()
        self.assertEqual(plan['episodes'], 80)
        self.assertEqual(plan['frames'], len(self.old_rows) + len(self.new_rows))
        rows = plan['episode_provenance']
        self.assertEqual(Counter((p['x'], p['y']) for p in rows), Counter(EXPECTED_COUNTS))
        self.assertEqual([p['source_episode_index'] for p in rows[:60]], list(range(60)))
        self.assertEqual([p['source_episode_index'] for p in rows[60:]], list(range(20)))
        self.assertEqual(rows[40]['original_source'], 'day16')
        self.assertEqual(rows[40]['original_source_episode_index'], 0)
        self.assertEqual(rows[60]['source'], 'day18')

    def test_dry_run_is_read_only(self):
        before = fingerprint(self.base)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--existing-root', str(self.old), '--added-root', str(self.added), '--output-root', str(self.output)]), 0)
        self.assertEqual(fingerprint(self.base), before)
        self.assertFalse(self.output.exists())

    def test_existing_destination_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_missing_parent_provenance_rejected(self):
        (self.old / 'dataset_provenance.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()

    def test_incorrect_ancestor_episode_is_rejected(self):
        file = self.old / 'dataset_provenance.json'
        d = json.loads(file.read_text())
        d['episode_provenance'][0]['x'] = 0.48
        file.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'ancestor provenance'):
            self.plan()

    def test_wrong_collection_speed_is_rejected(self):
        p = self.added / 'collection_protocol.json'
        d = json.loads(p.read_text()); d['xy_step_size'] = 0.5
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'collection setting'):
            self.plan()

    def test_merged_eighty_episode_audit_stats_and_task_strings(self):
        plan, sources = self.merged_fixture()
        self.assertTrue(verify_labels_and_stats(self.output, plan, sources)['exact_label_state_timestamp_preservation'])
        self.assertTrue(verify_task_mapping(self.output, plan, sources)['task_strings_preserved'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit_merged(['--dataset-root', str(self.output)]), 0)

    def test_wrong_task_mapping_is_rejected(self):
        plan, sources = self.merged_fixture()
        f = self.output / 'data/chunk-000/file-000.parquet'
        rows = pq.read_table(f).to_pylist()
        rows[-1]['task_index'] = 0
        pq.write_table(pa.Table.from_pylist(rows), f)
        with self.assertRaisesRegex(ValueError, 'task label changed'):
            verify_task_mapping(self.output, plan, sources)


if __name__ == '__main__':
    unittest.main()
