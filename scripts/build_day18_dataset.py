"""Append all 20 shorter-Y demonstrations to ALL 60 verified Day17 episodes.

Default is a read-only dry run. --execute builds in a temporary sibling root
using the local LeRobot merge API, copies video shards without concatenation,
verifies source hashes, exact labels/states/task mappings and sampled images,
and publishes to a NEW root only after merged QA passes. No training, network
fetch, source edits, or uploads are performed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .audit_day16_xy_dataset import EXPECTED_POSITIONS, inspect_episode, position_key
    from .audit_day18_merged_dataset import EXPECTED_COUNTS, main as audit_merged
    from .audit_day18_y_dataset import ADDED_POSITIONS
    from .build_day17_dataset import (
        PROTOCOL_KEYS, check_output_path, fingerprint, read_source,
        verify_labels_and_stats, verify_videos,
    )
else:
    from audit_day16_xy_dataset import EXPECTED_POSITIONS, inspect_episode, position_key
    from audit_day18_merged_dataset import EXPECTED_COUNTS, main as audit_merged
    from audit_day18_y_dataset import ADDED_POSITIONS
    from build_day17_dataset import (
        PROTOCOL_KEYS, check_output_path, fingerprint, read_source,
        verify_labels_and_stats, verify_videos,
    )

SOURCE_REPO_IDS = {
    'day17': 'wusanggg/panda_pick_cube_day17_xy_preserved_endpoints_v1',
    'day18': 'wusanggg/panda_pick_cube_day18_y_intermediate_v1',
}


def make_plan(existing_root: Path, added_root: Path, output: Path, repo_id: str):
    check_output_path(output, (existing_root, added_root))
    if existing_root.resolve() == added_root.resolve():
        raise ValueError('source roots must differ')
    sources = [read_source(existing_root, 'day17', expected_episodes=60),
               read_source(added_root, 'day18', expected_episodes=20)]
    a, b = sources
    for key in ('features', 'fps', 'codebase_version', 'robot_type'):
        if a['info'].get(key) != b['info'].get(key):
            raise ValueError(f'incompatible source {key}')
    for key in PROTOCOL_KEYS:
        if key not in a['protocol'] or a['protocol'][key] != b['protocol'].get(key):
            raise ValueError(f'incompatible/missing collection setting: {key}')
    if a['protocol'].get('home_pose') != 'configs/day11_home_x_centered.json' or a['protocol'].get('min_tcp_z') != 0.008:
        raise ValueError('unexpected Day17 home pose/Z bound')
    for source, counts in ((a, {p: EXPECTED_COUNTS[p] for p in EXPECTED_POSITIONS}),
                           (b, {p: 5 for p in ADDED_POSITIONS})):
        if Counter(position_key(p) for p in source['history']) != Counter(counts):
            raise ValueError(f"{source['label']}: wrong source position counts")
    lineage_file = existing_root / 'dataset_provenance.json'
    lineage = json.loads(lineage_file.read_text())
    if lineage.get('stage') != 'verified_complete' or lineage.get('episodes') != 60 or lineage.get('frames') != a['info']['total_frames']:
        raise ValueError('Day17 source lacks completed build provenance')
    ancestors = lineage.get('episode_provenance', [])
    if len(ancestors) != 60:
        raise ValueError('Day17 provenance does not cover all 60 episodes')
    for i, (entry, position) in enumerate(zip(ancestors, a['history'], strict=True)):
        if entry.get('episode_index') != i or position_key(entry) != position_key(position):
            raise ValueError('Day17 ancestor provenance disagrees with position history')
        if entry.get('frames') != int((a['data'].episode_index == i).sum()):
            raise ValueError('Day17 ancestor provenance disagrees with episode length')
    rows = []
    for source in sources:
        ids = list(range(source['info']['total_episodes']))
        source['selected_episode_indices'] = ids
        for source_ep in ids:
            position = source['history'][source_ep]
            data = source['data'][source['data'].episode_index == source_ep]
            _, errors = inspect_episode(np.stack(data.action), position_key(position))
            if errors:
                raise ValueError(f"{source['label']} episode {source_ep} failed phase QA: {errors}")
            entry = {
                'episode_index': len(rows), 'source': source['label'],
                'source_episode_index': source_ep, 'frames': len(data),
                'x': float(position['x']), 'y': float(position['y']), 'z': float(position['z']),
            }
            if source['label'] == 'day17':
                ancestor = ancestors[source_ep]
                entry['original_source'] = ancestor['source']
                entry['original_source_episode_index'] = ancestor['source_episode_index']
            rows.append(entry)
    counts = Counter((p['x'], p['y']) for p in rows)
    if len(rows) != 80 or counts != Counter(EXPECTED_COUNTS):
        raise ValueError('planned output is not the required 80-episode mixture')
    plan = {
        'builder_version': 1, 'repo_id': repo_id, 'output_root': str(output.resolve()),
        'episodes': 80, 'frames': sum(p['frames'] for p in rows), 'stage': 'dry_run',
        'position_counts': [{'x': x, 'y': y, 'episodes': n} for (x, y), n in sorted(counts.items())],
        'sources': [{'label': s['label'], 'repo_id': SOURCE_REPO_IDS[s['label']],
                     'root': str(s['root'].resolve()), 'selected_episode_indices': s['selected_episode_indices']}
                    for s in sources],
        'day17_ancestor_provenance_sha256': hashlib.sha256(lineage_file.read_bytes()).hexdigest(),
        'episode_provenance': rows,
        'order': 'all Day17 episodes become 0--59; all Day18 new episodes become 60--79, source order retained',
        'video_handling': 'merge all episodes; copy source video shards with no concatenation or new re-encoding',
        'split_note': 'added Y=±0.0375 combinations are now seen; development Y=±0.05625 and reserved test excluded',
    }
    return plan, sources


def verify_task_mapping(output_root: Path, plan: dict, sources: list[dict]):
    """Task IDs may change on merge; task STRINGS must still match source rows."""
    import pyarrow.parquet as pq
    if __package__:
        from .build_day17_dataset import read_data
    else:
        from build_day17_dataset import read_data

    def tasks(root):
        rows = pq.read_table(root / 'meta/tasks.parquet').to_pylist()
        mapping = {int(p['task_index']): p['task'] for p in rows}
        if len(mapping) != len(rows):
            raise ValueError('duplicate task IDs')
        return mapping

    output_tasks = tasks(output_root)
    data = read_data(output_root)
    originals = {s['label']: (s['data'], tasks(s['root'])) for s in sources}
    for p in plan['episode_provenance']:
        source_data, source_tasks = originals[p['source']]
        ids = source_data[source_data.episode_index == p['source_episode_index']].task_index
        actual_ids = data[data.episode_index == p['episode_index']].task_index
        try:
            expected = [source_tasks[int(i)] for i in ids]
            actual = [output_tasks[int(i)] for i in actual_ids]
        except KeyError as exc:
            raise ValueError('data refers to unknown task ID') from exc
        if actual != expected:
            raise ValueError(f"task label changed in output episode {p['episode_index']}")
    return {'task_strings_preserved': True, 'output_tasks': list(output_tasks.values())}


def execute(plan: dict, sources: list[dict]):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets

    output = Path(plan['output_root'])
    roots = tuple(s['root'] for s in sources)
    check_output_path(output, roots)
    before = {s['label']: fingerprint(s['root']) for s in sources}
    for source in plan['sources']:
        source['file_sha256'] = before[source['label']]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.building-', dir=output.parent) as directory:
        built = Path(directory) / 'merged'
        datasets = [LeRobotDataset(SOURCE_REPO_IDS[s['label']], root=s['root'],
                                   download_videos=False, video_backend='pyav') for s in sources]
        merge_datasets(datasets, output_repo_id=plan['repo_id'], output_dir=built,
                       concatenate_videos=False, concatenate_data=False)
        rows = plan['episode_provenance']
        (built / 'position_schedule.json').write_text(json.dumps({
            'split': 'train', 'description': 'All 60 Day17 episodes retained, 20 shorter-Y episodes appended',
            'positions': [{'x': p['x'], 'y': p['y']} for p in rows],
        }, indent=2) + '\n')
        (built / 'position_history.json').write_text(json.dumps([
            {key: p[key] for key in ('episode_index', 'x', 'y', 'z')} for p in rows
        ], indent=2) + '\n')
        protocol = {key: sources[0]['protocol'][key] for key in PROTOCOL_KEYS}
        protocol.update({
            'construction': 'merge all Day17 and full new Day18 recording, no new re-encoding',
            'home_pose': 'configs/day11_home_x_centered.json', 'min_tcp_z': 0.008,
            'home_pose_note': 'Day17 evidence and user recording command, not inferred from video merge',
            'gripper_close_protocol': 'measured minimum 6 before lift; see phase QA',
            'source_collection_protocols': {s['label']: s['protocol'] for s in sources},
        })
        (built / 'collection_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        verification = verify_labels_and_stats(built, plan, sources)
        verification.update(verify_task_mapping(built, plan, sources))
        video = verify_videos(built, plan, sources, exact_source_labels=('day17', 'day18'))
        video['all_source_samples_exact'] = True
        video['new_reencoding'] = False
        # These historical keys describe Day17's selective re-encode. This build
        # copies all source shards, so they would misdescribe its scope.
        video.pop('endpoint_samples_exact', None)
        video.pop('intermediate_reencode_mae_limit', None)
        verification['videos'] = video
        if audit_merged(['--dataset-root', str(built)]) != 0:
            raise RuntimeError('merged Day18 phase/metadata QA failed')
        if before != {s['label']: fingerprint(s['root']) for s in sources}:
            raise RuntimeError('source data changed; refusing to publish')
        verification['source_files_unchanged'] = True
        plan['stage'] = 'verified_complete'
        plan['verification'] = verification
        (built / 'dataset_provenance.json').write_text(json.dumps(plan, indent=2) + '\n')
        check_output_path(output, roots)
        built.rename(output)
    return plan


def main(argv=None):
    base = Path.home() / 'robotics/data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-root', type=Path, default=base / 'panda_pick_cube_day17_xy_preserved_endpoints_v1')
    parser.add_argument('--added-root', type=Path, default=base / 'panda_pick_cube_day18_y_intermediate_v1')
    parser.add_argument('--output-root', type=Path, default=base / 'panda_pick_cube_day18_xy_expanded_v1')
    parser.add_argument('--repo-id', default='wusanggg/panda_pick_cube_day18_xy_expanded_v1')
    parser.add_argument('--execute', action='store_true', help='Build and verify a new root; default is read-only preview.')
    args = parser.parse_args(argv)
    plan, sources = make_plan(args.existing_root.resolve(), args.added_root.resolve(), args.output_root.resolve(), args.repo_id)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print('\nDRY RUN: no files written; add --execute to build.')
        return 0
    plan = execute(plan, sources)
    print('\nDataset created:', plan['output_root'])
    print('Episodes:', plan['episodes'], 'Frames:', plan['frames'])
    print('Source hashes unchanged; labels, normalization stats, tasks, and 480 sampled camera frames verified.')
    print('Camera review sheet:', Path(plan['output_root']) / 'verification/source_vs_merged.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
