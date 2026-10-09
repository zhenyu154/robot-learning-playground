# Day16: Intermediate-X Coverage with Fixed Lift-Specific Weighting

**Experiment date:** October 8, 2026
**Status:** Setup prepared; smoke collection and training results pending.

## Question and controls

Does adding intermediate-X demonstrations improve positioning and the full
pick-and-lift sequence while keeping Day15's weights fixed?

Use X/Y active weight 2, ordinary active-Z weight 2, positive-Z weight 4, and
close-command weight 4. Positive-Z weighting is a raw action-sign mask, not an
explicit post-close stage detector. Preserve seed 1000, batch size 8, chunk size
50, 10 FPS, visible-arm cameras, the Day11 centered home pose, and min TCP Z
0.008. Evaluation uses five-step replanning and a 100-step primary horizon.
Training starts fresh; choose the update count after QA to target about 15
frame-sample dataset passes. Save only the final checkpoint.

## Collection

Training positions: X in {0.32, 0.36, 0.44, 0.48}, Y in {-0.075, +0.075}.

- Smoke: eight episodes, one per position, separate from training data.
- Training: 40 episodes, five per position, deterministically shuffled in five
  balanced blocks of eight (schedule-generation seed 2026).
- This preserves the total episode budget, but halves demonstrations at each
  endpoint corner compared with Day14. Monitor endpoint regressions separately;
  a new recording is not a pure causal isolation of spatial coverage.

Record X -> Y -> brief neutral hover -> descent -> close/stabilize -> lift.
Keep the Day14 timing protocol; do not insert additional long waits. The QA
requires at least three contiguous pre-descent neutral frames, six close-command
frames before lift, three contiguous stationary close-command frames, and eight
consecutive positive-Z lift frames. Do not descend during lateral approach.
Gripper command 1 means hold, not open; it is allowed during lifting. The cube
must remain physically held, verified in front/wrist videos.

Smoke dataset: `panda_pick_cube_day16_xy_intermediate_smoke`.
Training dataset: `panda_pick_cube_day16_xy_intermediate_v1`.
Use new directories; preserve all Day14/15 data and final model checkpoints.

## Evaluation split discipline

1. `configs/day16_xy_intermediate_seen_eval_positions.json`: 24 episodes, three
   repeats per training position; report endpoints and intermediate X separately.
2. `configs/day16_xy_development_positions.json`: repeat Day15's 1/12 benchmark
   at X={0.36,0.44}, Y={-0.0375,+0.0375}. These exact combinations are not in
   the new training set, but their failures informed Day16 design. This is a
   development benchmark, NOT an untouched final test. X is now a trained value;
   this primarily tests new Y values at known X values.
3. `configs/day16_xy_reserved_test_positions.json`: predeclared fresh 2D
   combinations X={0.34,0.46}, Y={-0.05,+0.05}, three repeats each. No exact
   combination appeared in the existing project schedules during setup. Do not
   record or tune on them. Run only after freezing the candidate; if the results
   subsequently drive changes, describe them as development data thereafter.

All primary evaluations are policy-only. Diagnostics and longer-horizon trials
must be reported separately. Single-seed repeated resets are limited evidence,
not a broad generalization guarantee.

## Baseline correction

The actual Day15 liftZ4 seen log reports endpoint counts 3/3, 3/3, 1/3, 2/3
in ascending X/Y order (total 9/12). The previous Day15 note reversed the two
far-X counts; the log is the source of truth. The previous held-out result was
1/12. An apparent 8/12 -> 9/12 gain is one extra success, not evidence of a
statistically established optimum.

## Full dataset QA

The full Day16 dataset passed the phase-aware QA and manual video inspection:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day16_xy_intermediate_v1
```

- 40 episodes, 3,875 frames;
- 5 demonstrations per each of 8 positions;
- 40/40 reward-positive and done-positive episodes;
- X/Y sign agreement `100%`;
- neutral hover, descent, close/stabilization, and lift phases present;
- no lateral action during lift.

The smoke dataset also passed at all eight positions. The QA tool is:

```text
scripts/audit_day16_xy_dataset.py
```

The phase-aware audit requires ordered stages rather than merely counting
positive actions. It also accepts the Day14/Day16 minimum of six close-command
frames, because the recordings consistently include stationary close frames and
at least eight consecutive lift frames.

## Training

The same best Day15 loss was kept fixed:

```text
X weight = 2
Y weight = 2
ordinary Z weight = 2
positive-Z lift weight = 4
close weight = 4
```

The 3,875-frame dataset was trained for 7,200 updates, approximately 14.86
passes, matching Day15's exposure. Only the final checkpoint was saved.

Checkpoint:

```text
outputs/act_panda_day16_xy_intermediate_xy2_z2_liftz4_closew4_xpu_7200/checkpoints/007200/pretrained_model
```

Final training metrics:

```text
loss: 0.277
unweighted_l1_loss: 0.195
x_active_mae_norm: 0.402
y_active_mae_norm: 0.560
z_active_mae_norm: 0.176
positive_z_active_mae_norm: 0.082
close_active_mae_norm: 0.235
```

## Offline audit

```text
X sign accuracy: 95.5%
Y sign accuracy: 86.2%
X neutral false motion: 3.0%
Y neutral false motion: 1.3%
Gripper close hit rate: 95.8%
Hold false-close rate: 4.6%
```

Per-position offline predictions remained reasonable for both Y signs, so the
online failure is not explained by missing positive-Y labels in the expert
observations.

Evidence:

```text
results/offline_gripper_audit_day16_xy_intermediate_xy2_z2_liftz4_closew4_xpu_7200.csv
```

## Policy-only seen evaluation

The 24-episode seen schedule used three repeats per each of the eight training
positions. Result:

```text
Successes: 6/24
Success rate: 25.0%
```

By position:

```text
(0.32,-0.075): 1/3
(0.32,+0.075): 0/3
(0.36,-0.075): 2/3
(0.36,+0.075): 0/3
(0.44,-0.075): 3/3
(0.44,+0.075): 0/3
(0.48,-0.075): 0/3
(0.48,+0.075): 0/3
```

All six successful episodes used negative Y. Positive-Y rollouts had `0/12`
successes in this run. Failures included both missing close/descent transitions
and grasp-without-lift transitions.

This is a substantial regression from Day15's endpoint-only result of `9/12`.
Adding intermediate-X positions while reducing endpoint repetition did not
produce reliable seen-position control.

Evidence:

```text
results/day16_xy_intermediate_seen_eval.log
```

## Longer-horizon diagnostic

To distinguish time-limit failures from genuine policy failures, the same Day16
checkpoint was evaluated on the eight smoke positions with a 150-step horizon.
Result:

```text
Successes: 2/8
Success rate: 25.0%
```

Only two episodes succeeded. The additional 50 steps rescued at most a small
number of delayed trajectories; most failures remained stuck before closing or
after grasping without a sufficient lift. Therefore, the Day16 regression is
not primarily caused by the 100-step horizon.

Evidence:

```text
results/day16_seen_150step_diagnostic.log
```

## Development interpolation result

The Day15 development interpolation benchmark remained poor for the Day15
checkpoint (`1/12`), and the Day16 seen-position regression made it inappropriate
to use the reserved test. The reserved schedule remains untouched:

```text
configs/day16_xy_reserved_test_positions.json
```

No reserved-test claim is made for Day16.

## Conclusions and next direction

1. Intermediate-X data alone did not improve the intended closed-loop behavior.
2. Day16's lower training loss and acceptable offline metrics did not translate
   to online success.
3. The 150-step diagnostic shows that extra time is not the main solution.
4. The endpoint-only Day15 model remains the best policy checkpoint so far:
   `9/12` seen and `1/12` on the development interpolation benchmark.
5. A future dataset should preserve endpoint coverage rather than replace half
   of the endpoint repetitions. A proposed next design is 60 fresh episodes:
   10 per endpoint combination and 5 per intermediate-X combination.
6. Do not tune or evaluate the reserved test until a candidate with stable seen
   performance is selected.
