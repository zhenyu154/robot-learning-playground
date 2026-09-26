# Day 5: Improve Gripper Supervision

## Starting point

The validated CPU and Intel XPU pipelines show the same behavior:

- 500-step policy-only rollout: no autonomous gripper close.
- Assisted gripper rollout: successful pick and lift.
- 500-step offline gripper audit: `0/51` close hits.
- 2000-step training: stronger but still weak close signal; online policy-only still failed.
- Fixed-step assisted XPU rollout: success, confirming the XPU online pipeline.

## Day 5 research question

Does making the gripper-close supervision more persistent and temporally clear
help ACT learn the close event?

## Controlled variable

Keep fixed:

- PandaPickCubeKeyboard-v0
- fixed cube position
- 10 FPS
- ACT policy and action representation
- `chunk_size=50`
- `n_action_steps=10`
- batch size 8
- random seed 1000
- XPU training backend for the first comparison

Change only the demonstration supervision:

- collect a new 10-episode dataset;
- keep the gripper stationary during the approach;
- at the grasp height, hold the close command for approximately 10--12 frames;
- keep the end effector spatially still while closing;
- after the close hold, lift while maintaining the closed state.

The new dataset should be stored separately as:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day5_gripper_v1
```

## Planned experiment

1. Collect 10 clean successful demonstrations.
2. Verify the dataset before training:
   - 10 episodes;
   - close command duration approximately 10--12 frames per episode;
   - no malformed or failed episodes;
   - front and wrist videos readable.
3. Train a 500-step ACT policy on XPU.
4. Run offline gripper audit and compare with the XPU 500-step baseline.
5. Run one online policy-only episode, then five if the chain is stable.
6. Record whether the new supervision improves close prediction and task success.

## Hypothesis

Longer, more consistent close supervision should increase the separation between
predicted gripper values on hold frames and close frames, and may allow the
policy to produce a strong enough online close command.

## Important limitation

This is a gripper-learning experiment, not a spatial-generalization experiment.
The cube remains fixed so that a change in success can be attributed primarily
to the gripper supervision rather than to simultaneous X/Y generalization.

## Data collection QA (2026-09-26)

The new dataset was created successfully and is readable by `LeRobotDataset`.

- Dataset root:
  `/home/wusanggg/robotics/data/panda_pick_cube_day5_gripper_v1`
- Episodes: `10`
- Total frames: `740`
- FPS: `10`
- Reward-positive episodes: `10/10`
- Done-positive episodes: `10/10`
- Action schema: `(4,)` with `delta_x`, `delta_y`, `delta_z`, `gripper`
- Front and wrist videos: present and decodable through the PyAV fallback

Gripper action distribution:

- Hold (`gripper=1.0`): `592` frames
- Close (`gripper=2.0`): `148` frames
- Close fraction: `20.0%`

The Day1 baseline had 51 close frames out of 680 (`7.5%`). The new dataset
therefore contains approximately 2.7 times as much close supervision.

Per-episode close runs were approximately 11--19 frames (`1.1--1.9 s` at
10 FPS). Episodes 0 and 2 contain two close runs separated by a short hold
interval; their total close counts are still 13 frames each. This is a minor
consistency issue, not a dataset validity failure.

Status: dataset QA passed; ready for Day5 500-step XPU training.

## Training QA (2026-09-26)

The Day 5 XPU training completed and produced:

```text
outputs/act_panda_day5_gripper_xpu_v1/checkpoints/000500/pretrained_model
```

Checkpoint metadata confirms:

- training step: `500`
- device: `xpu`
- dataset: `panda_pick_cube_day5_gripper_v1`
- batch size: `8`
- seed: `1000`
- `chunk_size`: `50`
- `n_action_steps`: `10`

The terminal loss log was not retained in the project, so final loss values
will not be recorded here. This does not block the next required evaluation:
the full offline gripper audit.

## Training result (2026-09-26)

The 500-step XPU training completed successfully on the Day5 dataset.

- Checkpoint:
  `outputs/act_panda_day5_gripper_xpu_v1/checkpoints/000500/pretrained_model`
- Training steps: `500`
- Runtime: approximately `1 minute 12 seconds`
- Final total loss: `2.253`
- Final L1 loss: `0.398`
- Final KL loss: `0.186`
- Final reported epoch: `5.41`
- Checkpoint save: passed

Offline gripper audit is the next step; the training loss alone does not
indicate whether the autonomous close event was learned.

## 500-step audit result (2026-09-26)

The 500-step XPU checkpoint was audited on all 740 frames:

- Target hold frames: `592`
- Target close frames: `148`
- Mean prediction on target close frames: `1.0142`
- Maximum prediction on target close frames: `1.0378`
- Close hit rate with threshold `1.5`: `0/148`
- False close rate on target hold frames: `0.0%`

Compared with the old XPU 500-step baseline (`close-frame mean 1.0001`,
maximum `1.0056`, close hit rate `0/51`), the new checkpoint shows only a
small numerical increase. It still collapses to the hold command near `1.0`.
The longer-close dataset alone did not produce a usable close prediction at
500 steps.

## Next controlled experiment

Keep the Day5 dataset and all model settings fixed, and increase training
from `500` to `2000` XPU steps. The script is:

```text
scripts/day5_act_gripper_longer_xpu.sh
```

This tests whether the richer close labels need more optimization before we
change the objective or model representation.

## 2000-step offline audit result (2026-09-26)

The 2000-step XPU checkpoint was audited on the same 740-frame Day5 dataset:

- Target hold frames: `592`
- Target close frames: `148`
- Overall predicted gripper mean: `1.1197`
- Mean on target close frames: `1.6175`
- Range on target close frames: `0.9793` to `2.1467`
- Close hit rate at analysis threshold `1.5`: `56.1%` (`83/148` frames)
- False close rate on target hold frames: `0.0%` (`0/592` frames)

This is a substantial improvement over the Day5 500-step checkpoint, whose
close-frame mean was `1.0142`, maximum was `1.0378`, and close hit rate was
`0/148`. The Day1 XPU 500-step baseline was also near the hold command on
close frames (`mean 1.0001`, maximum `1.0056`).

The predictions form contiguous close-like bursts within episodes, rather
than isolated frame hits. The first frame of many target-close runs is still
predicted near the hold command, followed by several strong close predictions;
this suggests a possible onset delay. Frame-level hits are temporally
correlated and must not be interpreted as a 56.1% episode-success probability.

Some predicted values exceed the demonstrated action-space maximum `2.0`
(maximum `2.1467`). The Gym-HIL Panda gripper internally saturates its
resulting actuator position to its valid range, so this should produce a
maximum-close increment in simulation, but it is still an out-of-range policy
prediction worth watching in rollout logs.

### Interpretation

The hypothesis that longer close supervision plus more optimization helps is
supported by both offline and online evaluation: the model predicts strong
close actions on many expert close observations while keeping all expert hold
observations below the analysis threshold, and it succeeds policy-only in
`5/5` fixed-position rollouts.

## First policy-only online rollout (2026-09-26)

The Day5 2000-step XPU policy completed the task without any external
assistant:

- success: `1/1`
- steps: `55`
- reward: `1.0`
- maximum commanded gripper value: `1.251`
- Z action range: `[-0.047, 1.038]`

The policy did not need to output exactly `gripper=2.0` in a single step.
Repeated partial close commands around the grasp phase were sufficient for
the simulated gripper to close, after which the policy produced a strong lift
command (`z=1.038`). This is the first complete policy-only success in the
project.

## Five-episode policy-only evaluation (2026-09-26)

The same Day5 2000-step XPU checkpoint was evaluated for five policy-only
episodes with no gripper assistance:

- successes: `5/5`
- success rate: `100%`
- every episode terminated successfully at step `55`
- reward per episode: `1.0`
- `max_gripper`: `1.251` in every episode
- `z_range`: `[-0.047, 1.038]` in every episode

The identical trajectories show highly repeatable behavior in the current
fixed-position environment. This is the first stable end-to-end autonomous
success result in the project. It does not yet establish spatial
generalization because the cube position and simulator reset are fixed.

## Day 5 conclusion

Longer and more persistent gripper-close demonstrations, combined with 2000
optimization steps, changed the policy from hold-command collapse to a usable
sequence of partial close commands followed by a learned lift. The fixed
position task reached `5/5` policy-only success.
