"""Compare existing Day18 expert approaches and saved policy evidence on CPU.

No policy inference, simulation, video decoding, or dataset changes. Frame
indices in demonstrations are zero-based; terminal-log steps are one-based.
The offline CSV contains first-action predictions with a reset queue on each
expert observation, so it cannot reveal the online chunk tail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

if __package__:
    from .audit_day16_xy_dataset import inspect_episode
    from .summarize_day12_xy_audit import axis_stats
else:
    from audit_day16_xy_dataset import inspect_episode
    from summarize_day12_xy_audit import axis_stats


PROJECT = Path(__file__).resolve().parents[1]
POSITIONS = ((0.36, -0.0375), (0.36, 0.0375))
AUDIT_NAME = 'offline_gripper_audit_day18_xy_expanded_xy2_z2_liftz4_closew4_xpu_14000.csv'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_y_burst_and_corrections(actions):
    """Separate the initial contiguous Y-label run from later Y-label runs.

    'Correction' describes timing in the labels; it does not prove an operator's
    intent or that every later movement was physically necessary.
    """
    frames = np.flatnonzero(np.abs(actions[:, 1]) > 1e-6)
    if not len(frames):
        raise ValueError('expected at least one active Y frame')
    gap = np.flatnonzero(np.diff(frames) > 1)
    split = int(gap[0] + 1) if len(gap) else len(frames)
    return frames[:split], frames[split:]


def audit_y(rows):
    if not rows:
        return {'frames': 0, 'hits': 0, 'hit_rate': None, 'predicted_y': []}
    target = np.array([float(r['target_delta_y']) for r in rows])
    pred = np.array([float(r['predicted_delta_y']) for r in rows])
    hit = (np.sign(pred) == np.sign(target)) & (np.abs(pred) >= 0.05)
    return {'frames': len(rows), 'hits': int(hit.sum()),
            'hit_rate': float(hit.mean()), 'predicted_y': pred.tolist()}


def read_sparse_log(path):
    """Read printed samples only; never treat these as a full action trace."""
    episodes, current, pending = {}, None, None
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'Episode (\d+)/(\d+)', line)
        if match:
            current = {'episode_one_based': int(match[1]), 'samples': []}
            episodes[int(match[1])] = current
            pending = None
        match = re.search(r'step=\s*(\d+) \| action=\[\[(.*?)\]\]', line)
        if match and current is not None:
            action = np.fromstring(match[2], sep=' ')
            if action.shape != (4,):
                raise ValueError('unexpected logged action shape')
            pending = {'step_one_based': int(match[1]), 'action': action.tolist()}
        match = re.search(r'geometry: cube=\[(.*?)\], tcp=\[(.*?)\], gripper_pos=(.*)', line)
        if match and pending is not None:
            cube, tcp = (np.fromstring(match[i], sep=' ') for i in (1, 2))
            if cube.shape != (3,) or tcp.shape != (3,):
                raise ValueError('unexpected logged geometry shape')
            pending.update(cube_xyz=cube.tolist(), tcp_xyz=tcp.tolist(),
                           gripper_control_target=float(match[3]))
            current['samples'].append(pending)
            pending = None
    return episodes


def analyze(root, audit_path):
    info = json.loads((root / 'meta/info.json').read_text())
    history = json.loads((root / 'position_history.json').read_text())
    paths = sorted(root.glob('data/chunk-*/file-*.parquet'))
    columns = ['episode_index', 'frame_index', 'action', 'observation.state', 'next.reward', 'next.done']
    data = pq.read_table(paths, columns=columns).to_pandas()
    if len(data) != info['total_frames'] or len(history) != info['total_episodes']:
        raise ValueError('dataset counts do not match metadata')
    with audit_path.open(newline='') as stream:
        audit = list(csv.DictReader(stream))
    if len(audit) != len(data):
        raise ValueError('audit and dataset frame counts differ')
    audit_by_frame = {(int(r['episode_index']), int(r['frame_index'])): r for r in audit}
    if len(audit_by_frame) != len(data):
        raise ValueError('duplicate audit episode/frame keys')
    groups = []
    for position in POSITIONS:
        selected = [p for p in history if (p['x'], p['y']) == position]
        if len(selected) != 5:
            raise ValueError(f'expected five demonstrations at {position}')
        experts, position_audit, initial_rows, correction_rows = [], [], [], []
        for entry in selected:
            ep = entry['episode_index']
            rows = data[data.episode_index == ep].sort_values('frame_index')
            if not np.array_equal(rows.frame_index.to_numpy(), np.arange(len(rows))):
                raise ValueError('noncontiguous expert frame indices')
            actions = np.stack(rows.action)
            states = np.stack(rows['observation.state'])
            if states.shape != (len(rows), 18):
                raise ValueError('expected the documented 18-value robot state')
            # Verified against installed gym_hil/mujoco_gym_env.py:get_robot_state.
            tcp = states[:, -3:]
            phases, errors = inspect_episode(actions, position)
            if errors or rows['next.reward'].max() < 1 or not rows['next.done'].any():
                raise ValueError(f'expert {ep} failed phase/reward/done checks: {errors}')
            initial, corrections = first_y_burst_and_corrections(actions)
            ep_audit = [audit_by_frame[(ep, i)] for i in range(len(rows))]
            for i, r in enumerate(ep_audit):
                target = np.array([float(r[f'target_delta_{axis}']) for axis in 'xyz']
                                  + [float(r['target_gripper'])])
                if not np.allclose(target, actions[i], atol=1e-7, rtol=0):
                    raise ValueError(f'audit labels differ from dataset at {ep}/{i}')
            initial_audit = [ep_audit[int(i)] for i in initial]
            correction_audit = [ep_audit[int(i)] for i in corrections]
            initial_rows.extend(initial_audit)
            correction_rows.extend(correction_audit)
            position_audit.extend(ep_audit)
            down = phases['descend_frame']
            experts.append({
                'episode_index': ep, 'phases_zero_based': phases,
                'initial_y_frames': initial.tolist(), 'later_y_frames': corrections.tolist(),
                'hover_frames_after_last_y': down - int(np.flatnonzero(abs(actions[:, 1]) > 1e-6)[-1]) - 1,
                'tcp_at_first_descent_m': tcp[down].tolist(),
                'tcp_xy_minus_scheduled_position_at_descent_mm': ((tcp[down, :2] - position) * 1000).tolist(),
                'initial_y_offline': audit_y(initial_audit),
                'later_y_offline': audit_y(correction_audit),
            })
        close = [r for r in position_audit if float(r['target_gripper']) >= 1.5]
        groups.append({
            'position': list(position), 'expert_episodes': experts,
            'offline_y': axis_stats(position_audit, 'target_delta_y', 'predicted_delta_y', 0.05),
            'offline_initial_y': audit_y(initial_rows), 'offline_later_y': audit_y(correction_rows),
            'offline_close': {'frames': len(close), 'hits': sum(float(r['predicted_gripper']) >= 1.5 for r in close)},
        })
    summary_path = PROJECT / 'results/day18_summary.json'
    summary = json.loads(summary_path.read_text())
    policy, logs = {}, []
    for name in ('added_shorter_y_seen_n5', 'added_shorter_y_seen_n10'):
        run = summary['evaluations'][name]
        path = PROJECT / run['log']
        logs.append(path)
        samples = read_sparse_log(path)
        selected = []
        for ep in run['episode_results']:
            if (ep['x'], ep['y']) in POSITIONS:
                observed = samples[ep['episode']]['samples']
                if not observed or observed[-1]['step_one_based'] != ep['steps']:
                    raise ValueError('log/summary final-step mismatch')
                selected.append({**ep, 'printed_samples_only': observed})
        if len(selected) != 6:
            raise ValueError('expected six relevant policy episodes in each old run')
        policy[name] = {'log': run['log'], 'n_action_steps': run['n_action_steps'],
                        'checkpoint': run['checkpoint'], 'max_steps': run['max_steps'],
                        'episode_results': selected}
    inputs = [root / 'meta/info.json', root / 'position_history.json', *paths,
              audit_path, summary_path, *logs, Path(__file__).resolve(),
              PROJECT / 'scripts/audit_day16_xy_dataset.py',
              PROJECT / 'scripts/summarize_day12_xy_audit.py']
    return {'status': 'existing_evidence_analyzed_new_trace_pending',
            'analysis_date': '2026-10-15', 'experiment_stage': 'Day19',
            'dataset_root': str(root), 'expert_groups': groups, 'saved_policy_runs': policy,
            'definitions': {
                'tcp_state': 'last three observation.state values; measured pinch_pos in meters',
                'expert_timing': 'zero-based observation/action frame indices; snapshot precedes that frame action',
                'policy_geometry': 'rounded, sparse post-step terminal samples; not a full trace',
                'later_y': 'active Y labels after a gap of at least one inactive frame following the first Y burst',
                'offline_hit': 'correct Y sign and absolute prediction >=0.05; reset action queue per expert frame',
            },
            'limitations': ['In-sample first-action predictions do not show the online full chunk.',
                            'Missed later Y labels occur on both sides; they alone do not explain the asymmetry.',
                            'Scheduled cube coordinates are not per-frame measured cube poses.',
                            'This CPU analysis does not run or certify new policy rollouts.'],
            'input_sha256': {str(p): sha256(p) for p in inputs}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--audit-csv', type=Path, default=PROJECT / 'results' / AUDIT_NAME)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    result = analyze(args.dataset_root, args.audit_csv)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for group in result['expert_groups']:
        print(group['position'], 'initial Y:', group['offline_initial_y']['hits'], '/',
              group['offline_initial_y']['frames'], 'later Y:', group['offline_later_y']['hits'], '/',
              group['offline_later_y']['frames'])
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
