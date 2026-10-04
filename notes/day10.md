# Day10: Intermediate Y Supervision and Timing-Standardized Demonstrations

**Experiment date:** October 3, 2026
**Closeout date:** October 4, 2026

## Research question

Can intermediate Y-position demonstrations and a more consistent close-to-lift
temporal protocol improve continuous Y-axis stopping and local interpolation?

Day8's weighted ACT policy solved the trained endpoint positions but failed at
one nonzero interior position:

```text
trained endpoints y=-0.10/+0.10: 6/6
unseen y=-0.05: 3/3
unseen y=+0.05: 0/3
```

Day9 diagnostics showed that manually stopping Y near the target, then closing
and lifting, made both intermediate positions physically solvable (`2/2`). This
localized the main positive-side failure to lateral stopping/calibration, while
also exposing an unreliable learned close-to-lift transition.

## Common controls

Both Day10 datasets used:

- fixed X: `0.48`;
- Y positions: `-0.10`, `-0.05`, `+0.05`, `+0.10`;
- five demonstrations per position, 20 episodes total;
- visible Panda arm in saved front/wrist observations;
- XY keyboard step size `0.25` and Z step size `0.50`;
- 10 FPS, same MuJoCo/Gym-HIL task and gripper protocol;
- ACT Y-active loss weight `4`;
- batch size `8`, seed `1000`, 2,000 updates;
- evaluation `n_action_steps=5`, policy-only unless explicitly labeled as a
  diagnostic intervention.

Once these four positions became training positions, they were no longer held
out. Fresh held-out positions were selected as `y=-0.075` and `y=+0.075`.

## Day10 v1: intermediate positions, original timing

Dataset:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day10_intermediate_v1
```

QA passed:

- 20 episodes, 1,791 frames;
- 5 episodes per Y position;
- 20/20 reward-positive and done-positive episodes;
- Y sign agreement `100%`;
- X active frames `0` in every episode.

Temporal audit found close-to-lift gaps ranging from about **4 to 15 frames**
(mean approximately **9 frames**). At 10 FPS this is roughly `0.4--1.5 s`,
which is inconsistent for a short manipulation stage.

Training completed at:

```text
outputs/act_panda_day10_intermediate_w4_xpu_v1/checkpoints/002000/pretrained_model
```

Final training metrics:

```text
loss: 1.138
weighted l1_loss: 0.338
unweighted l1_loss: 0.312
kld_loss: 0.080
epochs: 8.93
```

Offline audit:

| Metric | Day10 v1 |
|---|---:|
| Active-Y sign-and-magnitude hit, threshold 0.05 | 84.6% |
| Neutral Y false motion | 12.5% |
| Gripper close hit, threshold 1.5 | 67.8% |
| Hold false-close | 0.7% |

Policy-only seen evaluation:

| Position | Result |
|---|---:|
| `y=-0.10` | 3/3 |
| `y=-0.05` | 0/3 |
| `y=+0.05` | 0/3 |
| `y=+0.10` | 0/3 |
| **Total** | **3/12** |

The model often reached good Y alignment at intermediate positions but stayed
near the table after closing instead of lifting.

### Day10 v1 diagnostics

These are **not policy-only results**:

- fixed close + fixed lift: **4/4**;
- fixed close + policy lift: **1/4**;
- policy gripper + fixed lift: **4/4**.

Therefore all four positions were physically graspable and liftable. The main
v1 policy-only problem was the learned close-to-lift transition under closed-
loop distribution shift.

Evidence:

```text
results/day10_y_intermediate_w4_n5_seen_eval.log
results/day10_fixed_close_lift_seen_diagnostic.log
results/day10_fixed_close_policy_lift_diagnostic.log
results/day10_policy_gripper_fixed_lift_diagnostic.log
results/offline_gripper_audit_day10_y_intermediate_w4_xpu_2000.csv
```

## Timing-v2 dataset

The collection protocol was repeated with a more consistent manipulation
sequence:

1. approach laterally;
2. descend to grasp height;
3. close for approximately 10--12 frames;
4. begin lifting immediately after the close hold;
5. keep the gripper closed during the lift.

Dataset:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day10_intermediate_timing_v2
```

QA passed:

- 20 episodes, 1,456 frames;
- 5 episodes per Y position;
- 20/20 reward-positive and done-positive episodes;
- Y sign agreement `100%`;
- X active frames `0` in every episode;
- close-to-lift gaps: **2--8 frames**, mean **4.75 frames**;
- close counts: approximately 8--12 frames per episode.

The timing-v2 gap is substantially tighter than v1's approximate 4--15-frame
range. The frame count is also lower than v1, so the same 2,000 update count
corresponds to more dataset passes; this is a limitation of the comparison.

## Timing-v2 training and offline audit

Checkpoint:

```text
outputs/act_panda_day10_intermediate_timing_w4_xpu_v1/checkpoints/002000/pretrained_model
```

Final training metrics:

```text
loss: 1.132
weighted l1_loss: 0.329
unweighted l1_loss: 0.308
kld_loss: 0.080
epochs: 10.99
```

Offline audit:

| Metric | Timing-v2 |
|---|---:|
| Active-Y sign-and-magnitude hit, threshold 0.05 | 88.3% |
| Neutral Y false motion | 16.2% |
| Gripper close hit, threshold 1.5 | 78.0% |
| Hold false-close | 5.6% |

Compared with Day10 v1, timing-v2 improved active-Y and close prediction, but
also increased neutral Y motion and hold false-close rate. Offline metrics are
diagnostic and do not establish closed-loop task success.

Evidence:

```text
results/offline_gripper_audit_day10_intermediate_timing_w4_xpu_2000.csv
```

## Timing-v2 policy-only results

### Seen positions

All 12 seen-position episodes succeeded:

| Position | Result |
|---|---:|
| `y=-0.10` | 3/3 |
| `y=-0.05` | 3/3 |
| `y=+0.05` | 3/3 |
| `y=+0.10` | 3/3 |
| **Total** | **12/12** |

### Held-out interpolation

Held-out schedule:

```text
y=-0.075: 3 episodes
y=+0.075: 3 episodes
```

Result:

```text
6/6 policy-only successes
```

This is the strongest spatial-control result in the project so far. It provides
evidence of local one-dimensional interpolation after adding intermediate
positions and standardizing the close-to-lift transition.

Evidence:

```text
results/day10_intermediate_timing_w4_n5_seen_eval.log
results/day10_intermediate_timing_w4_n5_heldout_eval.log
```

## Conclusions

1. Day10 v1's intermediate positions alone did not solve policy-only control:
   `3/12`.
2. Day10 timing-v2 achieved `12/12` on all four training positions and `6/6`
   on fresh held-out `y=±0.075` positions.
3. The result strongly suggests that **temporal consistency in the
   close-to-lift transition** was important, although the timing-v2 dataset also
   had fewer frames and therefore more effective dataset passes at 2,000
   updates. A matched-pass replication would be needed to isolate the causal
   contribution of timing alone.
4. The experiment demonstrates local 1D Y-axis interpolation at fixed X. It
   does not establish broad 2D spatial generalization, multi-seed robustness,
   randomized object generalization, or real-robot transfer.
5. Do not mix the diagnostic intervention results with the policy-only results.

## Recommended next direction

Day10's planned objective is complete. The next reasonable experiment is either:

- an X-axis isolation task using the same controlled methodology; or
- a carefully designed 2D position dataset after preserving Day10 as the current
  one-dimensional spatial baseline.

Before changing the task, close out Day10 by committing the schedules, QA script,
audit CSVs, timing QA evidence, training/evaluation logs, and this note.
