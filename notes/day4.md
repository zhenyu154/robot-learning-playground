# Day 4: Controlled Longer-Training Experiment

## Starting point

Day 3 established the fixed-position ACT baseline:

- 10 demonstration episodes
- 680 frames at 10 FPS
- ACT checkpoint: 500 training steps
- policy-only rollout: 0/10 success
- assisted gripper rollout: 5/5 success
- offline gripper close hit rate: 0/51

The offline audit showed that the 500-step policy predicts the hold
command near `1.0` even on all 51 expert frames whose target gripper
action is `2.0`.

## Day 4 question

Does the policy fail because 500 training steps are insufficient, or does
the current data/objective make the sparse gripper-close event difficult to
learn even with more optimization?

## Controlled experiment

Keep fixed:

- dataset
- ACT policy type
- random seed
- batch size
- device
- chunk_size
- n_action_steps
- preprocessing and action representation

Change only:

- training steps: `500 -> 2000`
- output directory: `act_panda_longer_v1`

Training script:

```text
scripts/day4_act_longer.sh
```

## Planned evaluation order

1. Verify the training script syntax.
2. Run the 2000-step training.
3. Locate the final checkpoint.
4. Run offline gripper audit first.
5. Compare the 2000-step audit with the 500-step baseline.
6. Run online policy-only evaluation only after the audit result is known.

## Decision rule

- If offline close hit rate improves substantially, 500 steps was likely
  underfitting the gripper event. Continue with online evaluation.
- If offline close hit rate remains near 0%, longer training alone is
  probably not enough. The next experiment should change the data or the
  gripper objective, not blindly increase steps again.

## Results (2026-09-25)

### Training

- Final checkpoint:
  `outputs/act_panda_longer_v1/checkpoints/002000/pretrained_model`
- Training steps: `2000`
- Training runtime: approximately 27 minutes 47 seconds
- Final total loss: `1.112`
- Final L1 loss: `0.321`
- Final KL loss: `0.079`

Compared with the 500-step baseline, the offline loss decreased substantially.

### Offline gripper audit

- Audited frames: `680`
- Target hold frames: `629`
- Target close frames: `51`
- Predicted gripper mean on target close frames: `1.1760`
- Predicted gripper maximum: `1.2845`
- Close hit rate with threshold `1.5`: `0/51`
- False close rate on target hold frames: `0.0%`

The 2000-step checkpoint learned a stronger but still weak and smoothed
close signal. It did not produce a sufficiently strong close command under
the audit threshold.

### Online evaluation

- Policy-only: failed in the first diagnostic episode; gripper stayed near
  `1.0` and the task did not terminate successfully.
- Fixed-step assisted gripper:
  - close injected at step `60`
  - close duration: `6` steps
  - successes: `5/5`
  - success rate: `100%`
  - all five episodes finished at step `66`
  - all five had `max_gripper=2.000`
  - all five had `z_range=[-0.029, 0.982]`

### Interpretation

Increasing training from 500 to 2000 steps improved the offline loss and
created a weak gripper-close signal, but it did not make the policy emit a
sufficiently strong close command during autonomous rollout.

When the close event was externally supplied at step 60, the policy
consistently produced the subsequent lift action and completed the task in
5/5 episodes. This suggests that the learned lift behavior is conditioned
on the gripper reaching a closed state. The main remaining bottleneck is
the autonomous gripper-close event, not the learned post-grasp lift
trajectory.

This assisted result is diagnostic and is not an end-to-end ACT success
rate.
