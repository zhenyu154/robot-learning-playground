"""Opt-in initial-prefix cap that preserves the declared switch time.

For initial=7, later=20, switch=21, execute prefixes 7,7,6,20,20,...
The final initial prefix is shortened before prediction/queue filling, so no
selected queued action is dropped. The frozen Day19 v1 scheduler is unchanged.
"""

from dataclasses import dataclass

if __package__:
    from .panda_action_prefix import ActionPrefixSchedule
else:
    from panda_action_prefix import ActionPrefixSchedule


@dataclass(frozen=True)
class CappedActionPrefixSchedule(ActionPrefixSchedule):
    def __post_init__(self):
        values = (self.initial_n_action_steps, self.later_n_action_steps, self.switch_step)
        if any(type(value) is not int for value in values):
            raise ValueError('prefixes and switch step must be integers')
        if min(self.initial_n_action_steps, self.later_n_action_steps) <= 0:
            raise ValueError('execution prefixes must be positive')
        if self.switch_step <= 1:
            raise ValueError('prefix switch step must be greater than 1')

    def steps_at(self, step):
        """Selected prefix length at a prediction step, not at every action."""
        if type(step) is not int or step <= 0:
            raise ValueError('prediction step must be a positive integer')
        if step >= self.switch_step:
            return self.later_n_action_steps
        return min(self.initial_n_action_steps, self.switch_step - step)

    def before_step(self, policy, step):
        if type(step) is not int or step <= 0:
            raise ValueError('control steps are one-based positive integers')
        if step < self.switch_step:
            start = 1 + ((step - 1) // self.initial_n_action_steps) * self.initial_n_action_steps
        else:
            start = self.switch_step + ((step - self.switch_step) // self.later_n_action_steps) * self.later_n_action_steps
        desired = self.steps_at(start)
        queue = policy._action_queue
        if step != start:
            if not queue:
                raise RuntimeError('queue depleted outside the declared prediction boundary')
            if policy.config.n_action_steps != desired or queue.maxlen != desired:
                raise RuntimeError('active queue differs from the declared capped schedule')
            return False
        if queue:
            raise RuntimeError('refusing to discard unexecuted queued actions at a prediction boundary')
        if policy.config.n_action_steps == desired:
            if queue.maxlen != desired:
                raise RuntimeError('ACT queue capacity differs from configured prefix')
            return False
        policy.config.n_action_steps = desired
        policy.reset()
        if policy._action_queue.maxlen != desired:
            raise RuntimeError('ACT queue capacity did not follow the capped prefix')
        return True

    def to_metadata(self):
        value = super().to_metadata()
        value.update(kind='fixed_time_capped_initial', initial_boundary='shorten_last_prefix_before_switch')
        return value

    @classmethod
    def from_metadata(cls, value):
        if value['kind'] == 'fixed_time':
            return ActionPrefixSchedule.from_metadata(value)
        if value['kind'] != 'fixed_time_capped_initial':
            raise ValueError('unsupported execution-prefix schedule kind')
        return cls(value['initial_n_action_steps'], value['later_n_action_steps'], value['switch_step'])
