"""Capped prefixes preserve the 20-step phase and real ACT queue semantics."""

import copy
from collections import deque
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.panda_action_prefix import ActionPrefixSchedule
from scripts.panda_capped_action_prefix import CappedActionPrefixSchedule
from scripts.panda_grasp_trace import GraspTrace
from scripts import summarize_approach_trace as legacy_summary
from scripts.summarize_capped_approach_trace import summarize


spec = importlib.util.spec_from_file_location(
    '_capped_prefix_fixture', Path(__file__).with_name('test_panda_grasp_trace.py'),
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class CappedPrefixTests(unittest.TestCase):
    def test_invalid_settings_and_kind_rejected(self):
        for args in ((0, 20, 21), (7, 0, 21), (7, 20, 1), (7.5, 20, 21)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                CappedActionPrefixSchedule(*args)
        with self.assertRaises(ValueError):
            CappedActionPrefixSchedule(7, 51, 21).validate_chunk_size(50)
        with self.assertRaises(ValueError):
            CappedActionPrefixSchedule.from_metadata({'kind': 'unknown'})
        restored = CappedActionPrefixSchedule.from_metadata(ActionPrefixSchedule(10, 20, 21).to_metadata())
        self.assertIs(type(restored), ActionPrefixSchedule)

    def test_never_discards_pending_actions_at_boundary(self):
        schedule = CappedActionPrefixSchedule(7, 20, 21)
        queue = deque(['pending'], maxlen=7)
        p = SimpleNamespace(config=SimpleNamespace(n_action_steps=7), _action_queue=queue)
        with self.assertRaisesRegex(RuntimeError, 'unexecuted queued actions'):
            schedule.before_step(p, 15)
        self.assertEqual(list(queue), ['pending'])
        self.assertEqual(p.config.n_action_steps, 7)

    def test_unexpected_queue_depletion_rejected(self):
        p = SimpleNamespace(config=SimpleNamespace(n_action_steps=7), _action_queue=deque([], maxlen=7))
        with self.assertRaisesRegex(RuntimeError, 'outside the declared prediction boundary'):
            CappedActionPrefixSchedule(7, 20, 21).before_step(p, 2)

    def test_seven_seven_six_then_twenty_with_real_act_and_trace(self):
        p = fixture.policy()
        with torch.no_grad():
            p.model.values[0, :, 0] = torch.arange(50) * 0.001
        schedule = CappedActionPrefixSchedule(7, 20, 21)
        schedule.reset_episode(p)
        obs = {'observation.state': torch.zeros(1, 18)}
        post = SimpleNamespace(process_action=lambda a: a.clone())
        offsets = list(range(7)) * 2 + list(range(6)) + list(range(20)) + list(range(5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = GraspTrace(path, p, post, fixture.env(), {
                'checkpoint': 'cpu-test', 'max_steps': 45, 'policy_only_cli_settings': True,
                'execution_prefix_schedule': schedule.to_metadata(),
            })
            try:
                for episode in range(2):
                    schedule.reset_episode(p)
                    self.assertEqual(p._action_queue.maxlen, 7)
                    trace.reset(episode, {'x': 0.44, 'y': 0.0375})
                    for step in range(1, 46):
                        changed = schedule.before_step(p, step)
                        self.assertEqual(changed, step in (15, 21))
                        trace.before_step(step)
                        action = p.select_action(obs)
                        trace.after_prediction(action)
                        self.assertAlmostEqual(float(action[0, 0]), offsets[step - 1] * 0.001, places=6)
                        trace.after_step(action, action, 0, False, step == 45)
                    trace.end_episode(False, 45, 0)
            finally:
                trace.close()
            self.assertEqual(p.model.calls, 10)  # 1,8,15,21,41 in each episode
            self.assertEqual(len(p.model._forward_hooks), 0)
            factory = legacy_summary.ActionPrefixSchedule
            with self.assertRaisesRegex(ValueError, 'unsupported execution-prefix schedule kind'):
                legacy_summary.summarize(path)
            result = summarize(path)
            self.assertIs(legacy_summary.ActionPrefixSchedule, factory)
            self.assertEqual(result['complete_episodes'], 2)
            for e in result['episodes']:
                self.assertEqual([r['prediction_step'] for r in e['preclose_chunks']], [1, 8, 15, 21, 41])
                self.assertEqual([r['queue_executed_prefix_length'] for r in e['preclose_chunks']], [7, 7, 6, 20, 20])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            next(r for r in rows if r['event'] == 'prediction' and r['step'] == 15)['queue_executed_prefix_length'] = 7
            path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            with self.assertRaisesRegex(ValueError, 'declared execution schedule'):
                summarize(path)
            self.assertIs(legacy_summary.ActionPrefixSchedule, factory)

    def test_aligned_five_and_ten_profiles_keep_original_actions(self):
        obs = {'observation.state': torch.zeros(1, 18)}
        for initial in (5, 10):
            with self.subTest(initial=initial):
                original = fixture.policy()
                with torch.no_grad():
                    original.model.values[0, :, 0] = torch.arange(50) * 0.001
                capped = copy.deepcopy(original)
                old_schedule = ActionPrefixSchedule(initial, 20, 21)
                new_schedule = CappedActionPrefixSchedule(initial, 20, 21)
                old_schedule.reset_episode(original)
                new_schedule.reset_episode(capped)
                for step in range(1, 46):
                    old_schedule.before_step(original, step)
                    new_schedule.before_step(capped, step)
                    torch.testing.assert_close(original.select_action(obs), capped.select_action(obs), rtol=0, atol=0)
                self.assertEqual(original.model.calls, capped.model.calls)


if __name__ == '__main__':
    unittest.main()
