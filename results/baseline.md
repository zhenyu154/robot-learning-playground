# ACT Fixed-Position Pick-Cube Baseline

Date: 2026-09-22

## Environment

- Task: PandaPickCubeKeyboard-v0
- Simulator: MuJoCo through Gym-HIL
- Control frequency: 10 FPS
- Cube position: fixed
- Maximum episode length: 100 steps

## Dataset

- Demonstration episodes: 10
- Total frames: 680
- Observation state: 18 dimensions
- Cameras:
  - front: 128 x 128 RGB
  - wrist: 128 x 128 RGB
- Action:
  - delta_x
  - delta_y
  - delta_z
  - gripper

## Policy

- Policy: ACT
- Training steps: 500
- Checkpoint: 000500
- Vision backbone: ResNet18
- Parameters: approximately 52M
- chunk_size: 50
- n_action_steps: 10
- Device: CPU

## Evaluation results

| Controller | Episodes | Successes | Success rate |
|---|---:|---:|---:|
| ACT policy-only | 5 | 0 | 0% |
| ACT policy-only | 10 | 0 | 0% |
| ACT + assisted gripper close | 5 | 5 | 100% |

## Policy-only behavior

Every policy-only episode:

- reached the maximum 100 steps;
- received reward 0.0;
- ended by truncation;
- showed a similar descend-then-lift trajectory;
- did not produce an effective gripper close event.

## Assisted behavior

The assisted controller supplied a gripper close command
when the learned Z motion changed from descending to ascending.

The cube was successfully lifted in 5/5 episodes.

## Interpretation

The current ACT policy learned a plausible XYZ trajectory
for the fixed-position task, but did not reliably predict the
gripper close event.

The assisted result suggests that the main learned XYZ motion
is sufficient for the current fixed-position task. The primary
failure is currently associated with gripper action prediction,
rather than checkpoint loading, environment setup, or the main
XYZ action pipeline.

## Limitations

- The cube position is fixed.
- The dataset contains only 10 demonstrations.
- The policy-only sample size is 10 episodes.
- These results do not establish visual generalization.
- The assisted result is not an end-to-end ACT success rate.

## Offline gripper audit

The checkpoint was evaluated on all 680 expert observations.

- Target hold frames: 629
- Target close frames: 51
- Predicted gripper range over all frames: 0.9926 to 1.0100
- Mean prediction on target close frames: 1.0061
- Close hit rate: 0.0%
- False close rate on target hold frames: 0.0%

The policy predicts the hold command for essentially every
expert observation, including frames where the demonstrator
issued a close command.