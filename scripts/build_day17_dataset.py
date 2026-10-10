"""Build Day17 from Day14 endpoints plus only Day16 intermediate-X episodes.

Default: read-only dry run. --execute builds in a temporary sibling directory,
uses local LeRobot split/merge tools, verifies labels/stats/video samples, then
publishes to a NEW output root. Both source roots must remain untouched. This
script neither downloads/publishes datasets nor trains a policy.
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
else:
    from audit_day16_xy_dataset import EXPECTED_POSITIONS, inspect_episode, position_key

ENDPOINTS = tuple(p for p in EXPECTED_POSITIONS if p[0] in (0.32, 0.48))
INTERMEDIATES = tuple(p for p in EXPECTED_POSITIONS if p[0] in (0.36, 0.44))
EXPECTED_COUNTS = {p: 10 if p in ENDPOINTS else 5 for p in EXPECTED_POSITIONS}
CAMERAS = ('observation.images.front', 'observation.images.wrist')
PROTOCOL_KEYS = ('fps', 'xy_step_size', 'z_step_size', 'physical_xy_step_m', 'physical_z_step_m')


def read_json(path: Path):
    return json.loads(path.read_text())


def read_data(root: Path):
    import pyarrow.parquet as pq

    files = sorted(root.glob('data/chunk-*/file-*.parquet'))
    if not files:
        raise FileNotFoundError(root / 'data')
    return pq.read_table(files).to_pandas().sort_values(['episode_index', 'frame_index']).reset_index(drop=True)


def read_source(root: Path, label: str) -> dict:
    info = read_json(root / 'meta/info.json')
    history = read_json(root / 'position_history.json')
    schedule = read_json(root / 'position_schedule.json')
    protocol = read_json(root / 'collection_protocol.json')
    feature = info.get('features', {}).get('action', {})
    if feature.get('shape') != [4] or feature.get('names') != ['delta_x', 'delta_y', 'delta_z', 'gripper']:
        raise ValueError(f'{label}: unexpected action schema')
    if not str(info.get('codebase_version', '')).startswith('v3.'):
        raise ValueError(f'{label}: expected a LeRobot v3 dataset')
    if info.get('fps') != 10 or protocol.get('fps') != 10:
        raise ValueError(f'{label}: expected 10 FPS')
    if info.get('total_episodes') != 40 or len(history) != 40:
        raise ValueError(f'{label}: expected all 40 source episodes')
    if [position_key(p) for p in history] != [position_key(p) for p in schedule['positions']]:
        raise ValueError(f'{label}: schedule/history mismatch')
    if any(p.get('episode_index') != i for i, p in enumerate(history)):
        raise ValueError(f'{label}: nonsequential history indices')
    if any(not np.isclose(float(p.get('z', 0)), 0.02) for p in history):
        raise ValueError(f'{label}: unexpected cube Z position')
    data = read_data(root)
    if len(data) != info['total_frames'] or set(data.episode_index) != set(range(40)):
        raise ValueError(f'{label}: frame/episode metadata mismatch')
    for ep, rows in data.groupby('episode_index'):
        if not np.array_equal(rows.frame_index.to_numpy(), np.arange(len(rows))):
            raise ValueError(f'{label}: noncontiguous frame indices in episode {ep}')
        if rows['next.reward'].sum() <= 0 or not rows['next.done'].any():
            raise ValueError(f'{label}: failed demonstration {ep}')
    return {'root': root, 'label': label, 'info': info, 'history': history,
            'protocol': protocol, 'data': data}


def check_output_path(output: Path, sources: tuple[Path, ...]):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    for source in sources:
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('output root must be separate from both source datasets')


def make_plan(endpoint_root: Path, intermediate_root: Path, output: Path, repo_id: str) -> tuple[dict, list[dict]]:
    check_output_path(output, (endpoint_root, intermediate_root))
    if endpoint_root.resolve() == intermediate_root.resolve():
        raise ValueError('sources must be different datasets')
    sources = [read_source(endpoint_root, 'day14'), read_source(intermediate_root, 'day16')]
    a, b = sources
    for key in ('features', 'fps', 'codebase_version', 'robot_type'):
        if a['info'].get(key) != b['info'].get(key):
            raise ValueError(f'incompatible source {key}')
    for key in PROTOCOL_KEYS:
        if key not in a['protocol'] or a['protocol'][key] != b['protocol'].get(key):
            raise ValueError(f'incompatible/missing collection protocol: {key}')
    endpoint_counts = Counter(position_key(p) for p in a['history'])
    intermediate_counts = Counter(position_key(p) for p in b['history'])
    if endpoint_counts != Counter({p: 10 for p in ENDPOINTS}):
        raise ValueError('Day14 must contain ten demonstrations at each endpoint corner')
    if intermediate_counts != Counter({p: 5 for p in EXPECTED_POSITIONS}):
        raise ValueError('Day16 must contain five demonstrations at each of the eight positions')
    selected = [list(range(40)), [i for i, p in enumerate(b['history']) if position_key(p) in INTERMEDIATES]]
    rows = []
    for source, ids in zip(sources, selected, strict=True):
        source['selected_episode_indices'] = ids
        for source_ep in ids:
            position = source['history'][source_ep]
            ep_data = source['data'][source['data'].episode_index == source_ep]
            _, errors = inspect_episode(np.stack(ep_data.action), position_key(position))
            if errors:
                raise ValueError(f"{source['label']} episode {source_ep} failed phase QA: {errors}")
            rows.append({
                'episode_index': len(rows), 'source': source['label'],
                'source_episode_index': source_ep, 'frames': len(ep_data),
                'x': float(position['x']), 'y': float(position['y']), 'z': float(position['z']),
            })
    counts = Counter((p['x'], p['y']) for p in rows)
    if counts != Counter(EXPECTED_COUNTS) or len(rows) != 60:
        raise ValueError('planned mixture is not the intended 60-episode composition')
    plan = {
        'builder_version': 1, 'repo_id': repo_id, 'output_root': str(output.resolve()),
        'episodes': len(rows), 'frames': sum(p['frames'] for p in rows),
        'position_counts': [{'x': x, 'y': y, 'episodes': n} for (x, y), n in sorted(counts.items())],
        'sources': [{'label': s['label'], 'root': str(s['root'].resolve()),
                     'selected_episode_indices': s['selected_episode_indices']} for s in sources],
        'episode_provenance': rows,
        'order': 'all Day14 episodes followed by selected Day16 episodes; source order preserved',
        'video_handling': 'LeRobot selection may re-encode the Day16 subset; endpoint videos copied; no cross-source concatenation',
        'stage': 'dry_run',
    }
    return plan, sources


def fingerprint(root: Path) -> dict[str, str]:
    """Hash source files to detect changes, including camera videos and metadata."""
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
            result[str(path.relative_to(root))] = sha.hexdigest()
    return result


def verify_labels_and_stats(root: Path, plan: dict, sources: list[dict]):
    data = read_data(root)
    info = read_json(root / 'meta/info.json')
    if info['total_episodes'] != plan['episodes'] or len(data) != plan['frames'] or info['total_frames'] != len(data):
        raise ValueError('merged frame/episode counts differ from plan')
    if set(data.episode_index) != set(range(60)) or not np.array_equal(data['index'], np.arange(len(data))):
        raise ValueError('merged global/episode indices are not contiguous')
    by_source = {s['label']: s for s in sources}
    unchanged = ('action', 'observation.state', 'next.reward', 'next.done', 'timestamp', 'frame_index')
    for p in plan['episode_provenance']:
        dst = data[data.episode_index == p['episode_index']]
        src_data = by_source[p['source']]['data']
        src = src_data[src_data.episode_index == p['source_episode_index']]
        if len(dst) != p['frames']:
            raise ValueError(f"length mismatch for output episode {p['episode_index']}")
        for column in unchanged:
            if not np.array_equal(np.stack(dst[column]), np.stack(src[column])):
                raise ValueError(f"changed {column} values for output episode {p['episode_index']}")
    stats = read_json(root / 'meta/stats.json')
    # Check normalization features against the actual merged rows, not a copied
    # full-Day16 stats file that includes excluded endpoint episodes.
    for key in ('action', 'observation.state'):
        values = np.stack(data[key]).astype(np.float64)
        for name, expected in (('mean', values.mean(0)), ('std', values.std(0)),
                               ('min', values.min(0)), ('max', values.max(0))):
            if not np.allclose(np.asarray(stats[key][name]).reshape(-1), expected, rtol=1e-4, atol=1e-5):
                raise ValueError(f'merged normalization statistics mismatch: {key}/{name}')
        if not np.all(np.asarray(stats[key]['count']) == len(data)):
            raise ValueError(f'merged normalization statistics have wrong count: {key}')
    return {'exact_label_state_timestamp_preservation': True, 'normalization_stats_match_merged_rows': True}


def sampled_images(root: Path, episode_ids: list[int]) -> dict:
    """Decode first/middle/last camera frames using v3 video timestamps."""
    import av
    import pyarrow.parquet as pq

    info = read_json(root / 'meta/info.json')
    metadata = pq.read_table(sorted(root.glob('meta/episodes/chunk-*/file-*.parquet'))).to_pandas()
    files = {}
    for ep in episode_ids:
        rows = metadata[metadata.episode_index == ep]
        if len(rows) != 1:
            raise ValueError(f'video metadata does not identify exactly one episode {ep}')
        row = rows.iloc[0]
        length = int(row['length'])
        for camera in CAMERAS:
            start = float(row[f'videos/{camera}/from_timestamp'])
            stop = float(row[f'videos/{camera}/to_timestamp'])
            if not np.isclose((stop - start) * info['fps'], length, atol=1e-3):
                raise ValueError('video interval length does not match episode frame count')
            path = root / info['video_path'].format(
                video_key=camera, chunk_index=int(row[f'videos/{camera}/chunk_index']),
                file_index=int(row[f'videos/{camera}/file_index']),
            )
            targets = files.setdefault(path, {})
            for offset in sorted({0, length // 2, length - 1}):
                targets.setdefault(round(start * info['fps']) + offset, []).append((ep, camera, offset))
    images = {}
    for path, targets in files.items():
        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in targets:
                    if frame.pts is None or abs(float(frame.pts * frame.time_base) * info['fps'] - frame_index) > 0.1:
                        raise ValueError(f'nonuniform video timestamp near frame {frame_index}: {path}')
                    rgb = frame.to_ndarray(format='rgb24')
                    for key in targets[frame_index]:
                        images[key] = rgb
                if frame_index >= max(targets):
                    break
        expected = sum(len(v) for v in targets.values())
        actual = sum(key in images for keys in targets.values() for key in keys)
        if actual != expected:
            raise ValueError(f'missing sampled video frames: {path}')
    return images


def verify_videos(root: Path, plan: dict, sources: list[dict]) -> dict:
    merged = sampled_images(root, list(range(60)))
    originals = {s['label']: sampled_images(s['root'], s['selected_episode_indices']) for s in sources}
    errors = []
    comparisons = []
    for p in plan['episode_provenance']:
        for camera in CAMERAS:
            for offset in sorted({0, p['frames'] // 2, p['frames'] - 1}):
                src = originals[p['source']][(p['source_episode_index'], camera, offset)]
                dst = merged[(p['episode_index'], camera, offset)]
                if src.shape != dst.shape:
                    raise ValueError('merged camera frame dimensions changed')
                mae = float(np.abs(src.astype(np.float32) - dst).mean())
                if p['source'] == 'day14' and mae != 0:
                    raise ValueError('endpoint camera frames are not exact copies')
                if mae > 8:
                    errors.append((p['episode_index'], camera, offset, mae))
                comparisons.append(mae)
    if errors:
        raise ValueError(f'sampled source/output video mismatch: {errors}')
    # A compact contact sheet for a HUMAN comparison: middle/lift frames from
    # both sources, front and wrist, original on the left, merged on the right.
    from PIL import Image, ImageDraw

    selected = [0, 1, 20, 21, 40, 41, 50, 51]
    sheet = Image.new('RGB', (620, len(selected) * 156), 'white')
    draw = ImageDraw.Draw(sheet)
    for row_index, ep in enumerate(selected):
        p = plan['episode_provenance'][ep]
        offset = p['frames'] - 1
        y = row_index * 156
        draw.text((4, y + 2), f"ep={ep} {p['source']}:{p['source_episode_index']} ({p['x']},{p['y']}) source | merged", fill='black')
        for j, camera in enumerate(CAMERAS):
            src = originals[p['source']][(p['source_episode_index'], camera, offset)]
            dst = merged[(ep, camera, offset)]
            sheet.paste(Image.fromarray(src).resize((128, 128)), (4 + j * 304, y + 23))
            sheet.paste(Image.fromarray(dst).resize((128, 128)), (148 + j * 304, y + 23))
    (root / 'verification').mkdir()
    sheet.save(root / 'verification/source_vs_merged.png')
    return {'sampled_frames': len(comparisons), 'max_rgb_mae_0_to_255': max(comparisons),
            'mean_rgb_mae_0_to_255': float(np.mean(comparisons)),
            'endpoint_samples_exact': True, 'intermediate_reencode_mae_limit': 8,
            'contact_sheet': 'verification/source_vs_merged.png',
            'scope': 'first/middle/last frames of all 60 episodes, both cameras; not exhaustive video equality'}


def execute(plan: dict, sources: list[dict]) -> dict:
    # Fail closed: the installed tools must use local data only, not attempt Hub
    # downloads if a source is missing. No policy construction/XPU work occurs.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets, split_dataset

    output = Path(plan['output_root'])
    check_output_path(output, tuple(s['root'] for s in sources))
    before = {s['label']: fingerprint(s['root']) for s in sources}
    source_info = plan['sources']
    for s in source_info:
        s['file_sha256'] = before[s['label']]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.building-', dir=output.parent) as directory:
        workspace = Path(directory)
        endpoints = LeRobotDataset('wusanggg/panda_pick_cube_day14_xy_staged_v1', root=sources[0]['root'],
                                   download_videos=False, video_backend='pyav')
        intermediate_all = LeRobotDataset('wusanggg/panda_pick_cube_day16_xy_intermediate_v1', root=sources[1]['root'],
                                          download_videos=False, video_backend='pyav')
        subset = split_dataset(intermediate_all, {'intermediate': sources[1]['selected_episode_indices']},
                               output_dir=workspace / 'subset')['intermediate']
        built_root = workspace / 'merged'
        merge_datasets([endpoints, subset], output_repo_id=plan['repo_id'], output_dir=built_root,
                       concatenate_videos=False, concatenate_data=False)
        rows = plan['episode_provenance']
        (built_root / 'position_schedule.json').write_text(json.dumps({
            'split': 'train', 'description': 'Day14 endpoints retained; only Day16 intermediate-X episodes added',
            'positions': [{'x': p['x'], 'y': p['y']} for p in rows],
        }, indent=2) + '\n')
        (built_root / 'position_history.json').write_text(json.dumps([
            {key: p[key] for key in ('episode_index', 'x', 'y', 'z')} for p in rows
        ], indent=2) + '\n')
        protocol = {key: sources[0]['protocol'][key] for key in PROTOCOL_KEYS}
        protocol.update({
            'construction': 'merged existing recordings; no new teleoperation',
            'gripper_close_protocol': 'actual minimum 6 close-command frames; see per-source QA',
            'source_collection_protocols': {s['label']: s['protocol'] for s in sources},
            'home_pose': 'configs/day11_home_x_centered.json', 'min_tcp_z': 0.008,
            'home_pose_note': 'recording setup documented in project notes; not inferred from a merge',
        })
        (built_root / 'collection_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        verification = verify_labels_and_stats(built_root, plan, sources)
        verification['videos'] = verify_videos(built_root, plan, sources)
        if __package__:
            from .audit_day17_xy_dataset import main as audit
        else:
            from audit_day17_xy_dataset import main as audit
        if audit(['--dataset-root', str(built_root)]) != 0:
            raise RuntimeError('Day17 merged dataset failed phase/metadata QA')
        after = {s['label']: fingerprint(s['root']) for s in sources}
        if before != after:
            raise RuntimeError('source files changed during dataset construction; refusing to publish')
        verification['source_files_unchanged'] = True
        plan['stage'] = 'verified_complete'
        plan['verification'] = verification
        (built_root / 'dataset_provenance.json').write_text(json.dumps(plan, indent=2) + '\n')
        # No previous output is replaced. Source videos and labels are only read.
        check_output_path(output, tuple(s['root'] for s in sources))
        built_root.rename(output)
    return plan


def main(argv=None) -> int:
    base = Path.home() / 'robotics/data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-root', type=Path, default=base / 'panda_pick_cube_day14_xy_staged_v1')
    parser.add_argument('--intermediate-root', type=Path, default=base / 'panda_pick_cube_day16_xy_intermediate_v1')
    parser.add_argument('--output-root', type=Path, default=base / 'panda_pick_cube_day17_xy_preserved_endpoints_v1')
    parser.add_argument('--repo-id', default='wusanggg/panda_pick_cube_day17_xy_preserved_endpoints_v1')
    parser.add_argument('--execute', action='store_true', help='Create and verify the new dataset; without this flag, print a read-only plan.')
    args = parser.parse_args(argv)
    plan, sources = make_plan(args.endpoint_root.resolve(), args.intermediate_root.resolve(), args.output_root.resolve(), args.repo_id)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print('\nDRY RUN: no files written; add --execute to build the new dataset.')
        return 0
    plan = execute(plan, sources)
    print('\nDataset created:', plan['output_root'])
    print('Episodes:', plan['episodes'], 'Frames:', plan['frames'])
    print('Source files unchanged; exact actions/states preserved; sampled camera frames verified.')
    print('Manual review image:', Path(plan['output_root']) / 'verification/source_vs_merged.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
