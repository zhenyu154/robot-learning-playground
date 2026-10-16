"""Validate and summarize passive policy-only ACT traces before first close.

No inference or simulator is run. Action numbers are one-based. Target-motion
residuals use the tested Gym-HIL scale of 0.025 m per XYZ action unit; target Z
clipping is expected during successful descents and is not a pipeline defect.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .panda_action_prefix import ActionPrefixSchedule
else:
    from panda_action_prefix import ActionPrefixSchedule


def first_action(actions, channel, predicate):
    return next((i + 1 for i, a in enumerate(actions) if predicate(a[channel])), None)


def summarize(path):
    groups, metadata, counts = {}, None, Counter()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            counts[row['event']] += 1
            if row['event'] == 'metadata':
                if metadata is not None:
                    raise ValueError('duplicate metadata')
                metadata = row
                continue
            ep = groups.setdefault(row['episode_index'], {'predictions': [], 'steps': [], 'resets': [], 'ends': []})
            key = {'prediction': 'predictions', 'step': 'steps', 'reset': 'resets', 'end': 'ends'}[row['event']]
            ep[key].append(row)
    if metadata is None or not metadata['policy_only_cli_settings']:
        raise ValueError('expected policy-only trace metadata')
    n, chunk = metadata['n_action_steps'], metadata['chunk_size']
    if not 0 < n <= chunk:
        raise ValueError('invalid action prefix/chunk size')
    schedule = None
    if 'execution_prefix_schedule' in metadata:
        schedule = ActionPrefixSchedule.from_metadata(metadata['execution_prefix_schedule'])
        schedule.validate_chunk_size(chunk)
        if schedule.initial_n_action_steps != n:
            raise ValueError('initial trace prefix differs from declared schedule')
    if sorted(groups) != list(range(len(groups))):
        raise ValueError('noncontiguous episode indices')
    episodes = []
    for index, ep in sorted(groups.items()):
        if len(ep['resets']) != 1 or len(ep['ends']) != 1:
            raise ValueError(f'episode {index}: expected one reset and one end')
        steps, end = ep['steps'], ep['ends'][0]
        if [r['step'] for r in steps] != list(range(1, end['steps'] + 1)):
            raise ValueError(f'episode {index}: missing/duplicate step records')
        if not steps or not (steps[-1]['terminated'] or steps[-1]['truncated']):
            raise ValueError(f'episode {index}: missing terminal transition')
        if bool(end['success']) != any(r['reward'] > 0 for r in steps):
            raise ValueError(f'episode {index}: success/reward mismatch')
        if not np.isclose(end['reward'], sum(r['reward'] for r in steps)):
            raise ValueError(f'episode {index}: episode reward mismatch')
        predictions = ep['predictions']
        expected_step = 1
        for pred in predictions:
            if pred['step'] != expected_step or pred['step'] > len(steps):
                raise ValueError(f'episode {index}: unexpected replanning schedule')
            prefix = schedule.steps_at(pred['step']) if schedule is not None else n
            if pred['queue_executed_prefix_length'] != prefix:
                raise ValueError('chunk prefix differs from declared execution schedule')
            expected_step += prefix
        if expected_step <= len(steps):
            raise ValueError(f'episode {index}: missing chunk prediction')
        position = ep['resets'][0]['scheduled_position']
        sign = float(np.sign(position['y']))
        first_close = next((r['step'] for r in steps if r['requested_action'][3] >= 1.5), None)
        chunks = []
        for expected_id, pred in enumerate(predictions, 1):
            prefix = pred['queue_executed_prefix_length']
            a = np.asarray(pred['actions'])
            if a.shape != (chunk, 4) or not np.isfinite(a).all():
                raise ValueError('unexpected full-chunk shape or nonfinite action')
            if pred['chunk_id'] != expected_id:
                raise ValueError('chunk ID/prefix mismatch')
            for offset, step in enumerate(steps[pred['step'] - 1:pred['step'] - 1 + prefix]):
                if step['chunk_id'] != pred['chunk_id'] or step['chunk_offset'] != offset:
                    raise ValueError('step queue accounting mismatch')
                if not np.allclose(step['policy_action'], a[offset], atol=1e-6, rtol=1e-5):
                    raise ValueError('executed policy action differs from captured chunk')
                if not np.allclose(step['policy_action'], step['requested_action'], atol=1e-6, rtol=0):
                    raise ValueError('requested action differs from policy action')
            if first_close is not None and pred['step'] >= first_close:
                continue
            down = first_action(a, 2, lambda v: v <= -0.1)
            y = first_action(a, 1, lambda v: sign * v >= 0.05)
            chunks.append({
                'prediction_step': pred['step'], 'tcp_before_m': pred['state_before_prediction']['tcp_xyz'],
                'queue_executed_prefix_length': prefix,
                'first_correct_direction_y_action': y,
                'max_correct_direction_y_full_chunk': float(np.max(sign * a[:, 1])),
                'first_descent_action': down, 'descent_only_in_unexecuted_tail': down is not None and down > prefix,
                'first_close_action': first_action(a, 3, lambda v: v >= 1.5),
                'executed_prefix_y_sum': float(a[:prefix, 1].sum()),
                'full_chunk_y_sum': float(a[:, 1].sum()),
                'executed_prefix_min_z': float(a[:prefix, 2].min()),
            })
        actions = np.asarray([r['requested_action'] for r in steps])
        deltas = np.asarray([np.asarray(r['after']['tcp_target_xyz']) - r['before']['tcp_target_xyz'] for r in steps])
        residual = np.max(np.abs(deltas - 0.025 * actions[:, :3]), axis=0)
        last = steps[-1]['after']
        episodes.append({
            'episode_index': index, 'position': position, 'success': end['success'], 'steps': end['steps'],
            'first_executed_descent_step': next((r['step'] for r in steps if r['requested_action'][2] <= -0.1), None),
            'first_executed_close_step': first_close,
            'final_tcp_m': last['tcp_xyz'], 'final_tcp_target_m': last['tcp_target_xyz'],
            'final_cube_m': last['cube_xyz'],
            'final_cube_clearance_m': last['cube_bottom_clearance_m'],
            'final_tracking_error_mm': (1000 * (np.asarray(last['tcp_xyz']) - last['tcp_target_xyz'])).tolist(),
            'max_step_target_motion_minus_scaled_command_abs_m': residual.tolist(),
            'total_requested_y': float(actions[:, 1].sum()),
            'total_target_y_motion_m': float(deltas[:, 1].sum()),
            'preclose_chunks': chunks,
        })
    schedule = metadata.get('position_schedule')
    schedule_path = Path(schedule) if schedule else None
    if schedule_path is not None and not schedule_path.is_file():
        schedule_path = Path(__file__).resolve().parents[1] / schedule
    schedule_verified = False
    if schedule_path is not None and schedule_path.is_file():
        positions = json.loads(schedule_path.read_text())['positions']
        if len(positions) != len(episodes) or any(
            any(e['position'][axis] != p[axis] for axis in ('x', 'y'))
            for e, p in zip(episodes, positions)
        ):
            raise ValueError('trace episode positions/count differ from saved schedule')
        schedule_verified = True
    return {
        'trace': str(path), 'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'summarizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metadata': metadata, 'event_counts': dict(counts), 'complete_episodes': len(episodes),
        'schedule_count_and_positions_verified': schedule_verified,
        'successes': sum(e['success'] for e in episodes), 'episodes': episodes,
        'definitions': {'action_number': 'one-based within each predicted chunk',
                        'descent': 'raw Z action <= -0.1', 'correct_direction_y': 'signed Y action >= 0.05',
                        'close': 'raw gripper action >= 1.5',
                        'preclose_chunks': 'predictions made before the first executed close command',
                        'target_residual': 'per-step target delta minus 0.025 * raw XYZ action; Z clipping can make this nonzero'},
        'limitations': ['Trace validation does not make process exit successful.',
                        'Raw policy actions and mocap target changes are distinct from measured TCP motion.',
                        'Tail predictions are discarded; their execution and task outcome remain untested.',
                        'Changing the prefix also changes approach trajectories; it is not a same-state causal intervention.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    result = summarize(args.trace)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"Complete episodes: {result['complete_episodes']}; successes: {result['successes']}")
    for e in result['episodes']:
        print('episode', e['episode_index'], 'Y', e['position']['y'], 'steps', e['steps'],
              'descent', e['first_executed_descent_step'], 'close', e['first_executed_close_step'])
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
