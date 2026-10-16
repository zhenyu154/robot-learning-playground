"""Guard the declared prefix boundary before any GUI/model rollout is run."""

from collections import deque
from types import SimpleNamespace
import unittest

from scripts.panda_action_prefix import ActionPrefixSchedule


class PrefixScheduleTests(unittest.TestCase):
    def test_rejects_switch_inside_initial_chunk(self):
        for step in (1, 10, 20, 22):
            with self.subTest(step=step), self.assertRaises(ValueError):
                ActionPrefixSchedule(10, 20, step)

    def test_rejects_unsupported_policy_settings(self):
        schedule = ActionPrefixSchedule(10, 20, 21)
        for chunk, ensemble in ((15, None), (50, 0.01)):
            with self.subTest(chunk=chunk, ensemble=ensemble), self.assertRaises(ValueError):
                schedule.validate_policy(SimpleNamespace(config=SimpleNamespace(
                    chunk_size=chunk, temporal_ensemble_coeff=ensemble,
                )))
        for initial, later in ((0, 20), (10, 0)):
            with self.subTest(initial=initial, later=later), self.assertRaises(ValueError):
                ActionPrefixSchedule(initial, later, 21)

    def test_never_discards_remaining_actions_on_switch(self):
        schedule = ActionPrefixSchedule(10, 20, 21)
        queue = deque(['pending'], maxlen=10)
        policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=10), _action_queue=queue)
        with self.assertRaisesRegex(RuntimeError, 'unexecuted queued actions'):
            schedule.before_step(policy, 21)
        self.assertEqual(list(queue), ['pending'])
        self.assertEqual(policy.config.n_action_steps, 10)


if __name__ == '__main__':
    unittest.main()
