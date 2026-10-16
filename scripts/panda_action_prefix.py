"""An explicit, fixed-time ACT execution-prefix schedule for diagnostics.

This uses only the episode control-step counter. It does not inspect images,
cube coordinates, contacts or action values, and does not change model outputs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionPrefixSchedule:
    initial_n_action_steps: int
    later_n_action_steps: int
    switch_step: int

    def __post_init__(self):
        if self.initial_n_action_steps <= 0 or self.later_n_action_steps <= 0:
            raise ValueError('execution prefixes must be positive')
        if self.switch_step <= 1:
            raise ValueError('prefix switch step must be greater than 1')
        if (self.switch_step - 1) % self.initial_n_action_steps:
            raise ValueError('prefix switch step must be an empty-queue boundary')

    def validate_chunk_size(self, chunk_size):
        if max(self.initial_n_action_steps, self.later_n_action_steps) > chunk_size:
            raise ValueError('execution prefix cannot exceed chunk_size')

    def validate_policy(self, policy):
        self.validate_chunk_size(policy.config.chunk_size)
        if policy.config.temporal_ensemble_coeff is not None:
            raise ValueError('fixed-time prefix scheduling requires the ordinary ACT queue')

    def steps_at(self, step):
        if step <= 0:
            raise ValueError('control steps are one-based and positive')
        return self.initial_n_action_steps if step < self.switch_step else self.later_n_action_steps

    def reset_episode(self, policy):
        self.validate_policy(policy)
        policy.config.n_action_steps = self.initial_n_action_steps
        policy.reset()

    def before_step(self, policy, step):
        """Resize an EMPTY queue at the declared step; never discard queued actions."""
        desired = self.steps_at(step)
        if policy.config.n_action_steps == desired:
            return False
        if step != self.switch_step:
            raise RuntimeError('prefix transition missed its declared switch step')
        if len(policy._action_queue):
            raise RuntimeError('refusing to change prefix with unexecuted queued actions')
        policy.config.n_action_steps = desired
        # ACT reset reconstructs deque(maxlen=...), preventing silent truncation
        # to the old capacity. Temporal ensembling is explicitly unsupported.
        policy.reset()
        if policy._action_queue.maxlen != desired:
            raise RuntimeError('ACT queue capacity did not follow the new prefix')
        return True

    def to_metadata(self):
        return {'kind': 'fixed_time',
                'initial_n_action_steps': self.initial_n_action_steps,
                'later_n_action_steps': self.later_n_action_steps,
                'switch_step': self.switch_step}

    @classmethod
    def from_metadata(cls, value):
        if value['kind'] != 'fixed_time':
            raise ValueError('unsupported execution-prefix schedule kind')
        return cls(value['initial_n_action_steps'], value['later_n_action_steps'], value['switch_step'])
