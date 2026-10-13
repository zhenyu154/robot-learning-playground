"""Summarize passive JSONL grasp diagnostics; thresholds are analysis only.

This describes associations, not proof of cue dependence or grasp stability.
Lift commands later in a chunk are evidence to inspect, not proof that changing
queue length would solve the task.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(path: Path, positive_z_threshold=0.1, clearance_threshold=0.001):
    episodes = defaultdict(lambda: {'steps': [], 'predictions': [], 'end': None})
    metadata = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            event = row['event']
            if event == 'metadata':
                metadata = row
            elif event == 'step':
                episodes[row['episode_index']]['steps'].append(row)
            elif event == 'prediction':
                episodes[row['episode_index']]['predictions'].append(row)
            elif event == 'end':
                episodes[row['episode_index']]['end'] = row
    if metadata is None:
        raise ValueError('missing trace metadata')
    results = []
    for ep, data in sorted(episodes.items()):
        steps = data['steps']
        closing = next((r for r in steps if r['requested_action'][3] >= 1.5), None)
        close_step = closing['step'] if closing else None
        later = [r for r in steps if close_step is not None and r['step'] >= close_step]
        lift = next((r for r in later if r['requested_action'][2] >= positive_z_threshold), None)
        clear = next((r for r in later if r['after']['cube_bottom_clearance_m'] > clearance_threshold
                      and not r['after']['contact_flags']['cube_table_active']), None)
        deferred = []
        for row in data['predictions']:
            if close_step is None or row['step'] < close_step:
                continue
            z = [a[2] for a in row['actions']]
            prefix = row['queue_executed_prefix_length']
            first_positive = next((i for i, value in enumerate(z) if value >= positive_z_threshold), None)
            if first_positive is not None and first_positive >= prefix:
                deferred.append({'step': row['step'], 'chunk_id': row['chunk_id'],
                                 'first_positive_z_offset_zero_based': first_positive,
                                 'executed_prefix_max_z': max(z[:prefix]), 'full_chunk_max_z': max(z)})
        row = {'episode_index': ep, 'complete': data['end'] is not None,
               'success': data['end']['success'] if data['end'] else None,
               'steps': data['end']['steps'] if data['end'] else len(steps),
               'first_close_command_step': close_step,
               'first_post_close_positive_z_command_step': lift['step'] if lift else None,
               'first_post_close_clearance_without_table_contact_step': clear['step'] if clear else None,
               'maximum_cube_bottom_clearance_m': max((r['after']['cube_bottom_clearance_m'] for r in steps), default=None),
               'maximum_post_close_z_command': max((r['requested_action'][2] for r in later), default=None),
               'post_close_steps_with_active_table_contact': sum(r['after']['contact_flags']['cube_table_active'] for r in later),
               'post_close_steps_with_both_finger_contacts': sum(r['after']['contact_flags']['cube_left_active']
                                                                and r['after']['contact_flags']['cube_right_active'] for r in later),
               'chunks_with_positive_z_only_beyond_executed_prefix': deferred}
        results.append(row)
    return {'trace': str(path), 'checkpoint': metadata.get('checkpoint'),
            'n_action_steps': metadata['n_action_steps'], 'max_steps': metadata.get('max_steps'),
            'positive_z_action_threshold': positive_z_threshold,
            'cube_bottom_clearance_threshold_m': clearance_threshold,
            'episodes': results,
            'scope': 'passive association/queue/contact diagnostic, not evidence of causal dependence on sway',
            'notes': ['first/late lift is an ACTION threshold, not a measure of physical cube motion',
                      'contact counts and actuator target are not proof of a stable grasp',
                      'existing final benchmark is not replaced by these repeated-position traced trials']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--positive-z-threshold', type=float, default=0.1)
    parser.add_argument('--clearance-threshold', type=float, default=0.001)
    args = parser.parse_args()
    if args.positive_z_threshold <= 0 or args.clearance_threshold <= 0:
        parser.error('analysis thresholds must be positive')
    summary = summarize(args.trace, args.positive_z_threshold, args.clearance_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        file.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for row in summary['episodes']:
        print(f"ep={row['episode_index']:02d} success={row['success']} steps={row['steps']} "
              f"close_at={row['first_close_command_step']} "
              f"positive_z_at={row['first_post_close_positive_z_command_step']} "
              f"table_clear_at={row['first_post_close_clearance_without_table_contact_step']} "
              f"late_chunk_lift_count={len(row['chunks_with_positive_z_only_beyond_executed_prefix'])}")
    print('Detailed summary:', args.output)


if __name__ == '__main__':
    main()
