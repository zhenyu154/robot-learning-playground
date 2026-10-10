"""Day17 planning/safety/label verification tests without training or videos."""

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
from scripts.audit_day17_xy_dataset import main as audit17
from scripts.build_day17_dataset import (
    ENDPOINTS, EXPECTED_COUNTS, check_output_path, fingerprint, main,
    make_plan, verify_labels_and_stats,
)


def actions(position):
    x = -0.25 if position[0] < 0.401855 else 0.25
    y = -0.25 if position[1] < 0 else 0.25
    return np.array([[0, 0, 0, 1]] * 2 + [[x, 0, 0, 1]] * 6
                    + [[0, y, 0, 1]] * 6 + [[0, 0, 0, 1]] * 4
                    + [[0, 0, -0.5, 1]] * 5 + [[0, 0, 0, 2]] * 6
                    + [[0, 0, 0.5, 1]] * 10, dtype=np.float32)


def write_source(root, label):
    positions = list(ENDPOINTS) * 10 if label == 'day14' else list(EXPECTED_POSITIONS) * 5
    rows = []
    history = []
    for i, position in enumerate(positions):
        a = actions(position)
        history.append({'episode_index': i, 'x': position[0], 'y': position[1], 'z': 0.02})
        for j, action in enumerate(a):
            rows.append({'episode_index': i, 'frame_index': j, 'index': len(rows),
                         'timestamp': j / 10, 'action': action.tolist(),
                         'observation.state': [float(i)] * 18,
                         'next.reward': float(j == len(a) - 1), 'next.done': bool(j == len(a) - 1)})
    write_base(root, rows, history)


def write_base(root, rows, history):
    (root / 'meta').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root / 'data/chunk-000/file-000.parquet')
    feature = {'shape': [4], 'names': ['delta_x', 'delta_y', 'delta_z', 'gripper'], 'dtype': 'float32'}
    (root / 'meta/info.json').write_text(json.dumps({
        'codebase_version': 'v3.0', 'fps': 10, 'total_episodes': len(history),
        'total_frames': len(rows), 'robot_type': None, 'features': {'action': feature},
    }))
    (root / 'position_history.json').write_text(json.dumps(history))
    (root / 'position_schedule.json').write_text(json.dumps({
        'split': 'train', 'positions': [{'x': p['x'], 'y': p['y']} for p in history],
    }))
    (root / 'collection_protocol.json').write_text(json.dumps({
        'fps': 10, 'xy_step_size': 0.25, 'z_step_size': 0.5,
        'physical_xy_step_m': 0.00625, 'physical_z_step_m': 0.0125,
    }))


class Day17Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.a, self.b, self.out = (self.base / name for name in ('day14', 'day16', 'merged'))
        write_source(self.a, 'day14')
        write_source(self.b, 'day16')

    def plan(self):
        return make_plan(self.a, self.b, self.out, 'test/day17')

    def merged_fixture(self, plan, sources):
        data = []
        history = []
        by_label = {s['label']: s for s in sources}
        for entry in plan['episode_provenance']:
            src = by_label[entry['source']]['data']
            ep = src[src.episode_index == entry['source_episode_index']]
            for row in ep.to_dict('records'):
                row['episode_index'] = entry['episode_index']
                row['index'] = len(data)
                # PyArrow expects nested lists, not numpy arrays in a dict list.
                for column in ('action', 'observation.state'):
                    row[column] = row[column].tolist()
                data.append(row)
            history.append({k: entry[k] for k in ('episode_index', 'x', 'y', 'z')})
        write_base(self.out, data, history)
        stats = {}
        for key in ('action', 'observation.state'):
            a = np.stack([r[key] for r in data])
            stats[key] = {name: value.tolist() for name, value in (
                ('mean', a.mean(0)), ('std', a.std(0)), ('min', a.min(0)), ('max', a.max(0)),
            )}
            stats[key]['count'] = [len(data)]
        (self.out / 'meta/stats.json').write_text(json.dumps(stats))
        for camera in ('front', 'wrist'):
            d = self.out / f'videos/observation.images.{camera}/chunk-000'
            d.mkdir(parents=True)
            (d / 'file-000.mp4').touch()

    def test_plan_exact_sources_counts_and_order(self):
        plan, _ = self.plan()
        self.assertEqual(plan['episodes'], 60)
        self.assertEqual(plan['frames'], 60 * len(actions(ENDPOINTS[0])))
        provenance = plan['episode_provenance']
        self.assertEqual(Counter((p['x'], p['y']) for p in provenance), Counter(EXPECTED_COUNTS))
        self.assertEqual([p['source_episode_index'] for p in provenance[:40]], list(range(40)))
        self.assertTrue(all(p['source'] == 'day14' for p in provenance[:40]))
        self.assertTrue(all(p['source'] == 'day16' and p['x'] in (0.36, 0.44) for p in provenance[40:]))
        self.assertEqual(len({(p['source'], p['source_episode_index']) for p in provenance}), 60)

    def test_dry_run_has_no_writes(self):
        before = fingerprint(self.base)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--endpoint-root', str(self.a), '--intermediate-root', str(self.b),
                                   '--output-root', str(self.out)]), 0)
        self.assertEqual(fingerprint(self.base), before)
        self.assertFalse(self.out.exists())

    def test_output_must_not_exist_or_overlap_sources(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()
        with self.assertRaises(ValueError):
            check_output_path(self.a / 'nested_output', (self.a, self.b))

    def test_mismatched_schema_is_rejected(self):
        p = self.b / 'meta/info.json'
        d = json.loads(p.read_text())
        d['features']['action']['names'][0] = 'bad_name'
        p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            self.plan()

    def test_mismatched_protocol_is_rejected(self):
        p = self.b / 'collection_protocol.json'
        d = json.loads(p.read_text())
        d['z_step_size'] = 0.3
        p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            self.plan()

    def test_wrong_position_history_is_rejected(self):
        p = self.b / 'position_history.json'
        d = json.loads(p.read_text())
        d[0]['x'] = 0.44
        p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            self.plan()

    def test_merged_labels_statistics_and_phase_audit(self):
        plan, sources = self.plan()
        self.merged_fixture(plan, sources)
        result = verify_labels_and_stats(self.out, plan, sources)
        self.assertTrue(result['normalization_stats_match_merged_rows'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit17(['--dataset-root', str(self.out)]), 0)

    def test_changed_action_in_merge_is_rejected(self):
        plan, sources = self.plan()
        self.merged_fixture(plan, sources)
        file = self.out / 'data/chunk-000/file-000.parquet'
        rows = pq.read_table(file).to_pylist()
        rows[-1]['action'][2] = 0.25
        pq.write_table(pa.Table.from_pylist(rows), file)
        with self.assertRaisesRegex(ValueError, 'changed action'):
            verify_labels_and_stats(self.out, plan, sources)

    def test_stats_copied_from_wrong_dataset_are_rejected(self):
        plan, sources = self.plan()
        self.merged_fixture(plan, sources)
        p = self.out / 'meta/stats.json'
        d = json.loads(p.read_text())
        d['action']['mean'][0] += 0.1
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'statistics mismatch'):
            verify_labels_and_stats(self.out, plan, sources)


if __name__ == '__main__':
    unittest.main()
