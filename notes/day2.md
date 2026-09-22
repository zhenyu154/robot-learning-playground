# Day 2: From Demonstrations to ACT Policy

## Dataset

- Task: Panda pick cube
- Episodes: 10
- Frames: 680
- FPS: 10
- Observation:
  - front image: 3 x 128 x 128
  - wrist image: 3 x 128 x 128
  - state: 18-dimensional
- Action:
  - delta_x
  - delta_y
  - delta_z
  - gripper

## Policy

- Policy: ACT
- Vision backbone: ResNet18
- Parameters: approximately 52M
- chunk_size: 50
- n_action_steps: 10
- Device: CPU

## Training

- Training steps: 500
- Batch size: 8
- Runtime: approximately 4 minutes
- Checkpoint load test: PASS

## Training loss

- step 25:
  - l1_loss: 0.509
  - kl_loss: 1.782
  - total loss: 18.328
- step 100:
  - l1_loss: 0.424
  - kl_loss: 0.364
  - total loss: 4.063
- step 250:
  - l1_loss: 0.393
  - kl_loss: 0.249
  - total loss: 2.883
- step 500:
  - l1_loss: 0.377
  - kl_loss: 0.191
  - total loss: 2.289

## Rollout evaluation

| Mode | Episodes | Successes | Success rate |
|---|---:|---:|---:|
| policy-only | 5 | 0 | 0% |
| assisted gripper close | 5 | 5 | 100% |

## Main observation

The policy learned a plausible descend-then-lift XYZ trajectory,
but did not reliably predict the gripper close event.

The assisted controller successfully completed the task,
which suggests that the main remaining issue is gripper action prediction.

## Conclusion

The dataset-to-policy-to-checkpoint-to-rollout pipeline works.
The current ACT policy is not yet a complete end-to-end pick policy.