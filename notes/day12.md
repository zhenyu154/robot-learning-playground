# Day12: Sequential 2D X/Y Compositional Control

**Experiment date:** October 5, 2026

## Research question

Can ACT combine the validated one-dimensional X and Y skills in a fixed-home,
2D pick-and-lift task when both coordinates vary?

Day10 established local Y interpolation at fixed X. Day11 established strong
seen X control and partial X interpolation at fixed Y. Day12 combines the two
axes for the first controlled 2D experiment.

## Protocol and controls

- Centered Panda home pose: `configs/day11_home_x_centered.json` (TCP X about
  `0.402`)
- Minimum Cartesian TCP Z bound: `0.008`
- Training corner positions:
  `(0.32,-0.075)`, `(0.32,+0.075)`, `(0.48,-0.075)`, `(0.48,+0.075)`
- Five successful demonstrations per corner, 20 episodes total.
- X/Y actions were sequential in the smoke and training demonstrations:
  `X movement -> Y movement -> descend -> close -> lift`.
- No frame had simultaneous nonzero X and Y labels in the smoke QA. This is
  sequential 2D compositional control, not simultaneous diagonal-action
  supervision.
- Same visible-arm camera setup, 10 FPS, XY step `0.25`, Z step `0.50`, and
  standardized Day10 close-to-lift protocol.
- ACT training: active X weight `4`, active Y weight `4`, close weight `4`,
  batch size `8`, seed `1000`, 2,000 updates, `chunk_size=50`.
- Policy-only evaluation: `n_action_steps=5`, `max_steps=150`.

## Dataset QA

Corrected v2 dataset:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day12_xy_v2
```

- 20 episodes, 2,203 frames;
- five episodes per corner;
- 20/20 reward-positive and done-positive episodes;
- X sign agreement `100%`;
- Y sign agreement `100%`;
- no failed demonstrations;
- manual camera/video check completed.

The first Day12 dataset had an unbalanced corner schedule and was not used as
the final training dataset. The v2 dataset has the intended balanced schedule.

QA script:

```text
scripts/audit_day12_xy_dataset.py
```

## Multi-axis loss implementation

The Day8/Day11 active-axis adapter was generalized to support:

```text
--active-axis=x
--active-axis=y
--active-axis=xy
```

For Day12, valid action errors were weighted as:

```text
active X scalar: weight 4
active Y scalar: weight 4
close-frame gripper scalar: weight 4
other valid scalars: weight 1
padding: weight 0
```

The raw masks are captured before action normalization. The KL term and standard
ACT checkpoint format remain unchanged. The test suite now contains 23 passing
CPU tests, including simultaneous X/Y and close-mask weighting.

## Training

Checkpoint:

```text
outputs/act_panda_day12_xy_weighted_xy4_closew4_xpu_v1/checkpoints/002000/pretrained_model
```

Final metrics:

```text
loss: 1.279
weighted l1_loss: 0.446
unweighted l1_loss: 0.362
kld_loss: 0.083
x_active_mae_norm: 1.826
y_active_mae_norm: 1.019
close_active_mae_norm: 0.349
epochs: 7.26
```

## Offline audit

The audit covered 2,203 frames. It is an expert-observation diagnostic, not a
held-out action-accuracy measurement.

| Metric | Result |
|---|---:|
| Overall X sign-and-magnitude hit, threshold 0.05 | 77.1% |
| Overall Y sign-and-magnitude hit, threshold 0.05 | 80.0% |
| X neutral false motion | 13.2% |
| Y neutral false motion | 9.2% |
| Gripper close hit, threshold 1.5 | 96.4% |
| Gripper hold false-close | 8.9% |

By corner, the predicted signs were correct but the motion hit rates were lower
than Day10's one-axis results. The high close hit rate comes with a higher hold
false-close rate, so online behavior is still required to assess the tradeoff.

Evidence:

```text
results/offline_gripper_audit_day12_xy_weighted_xy4_closew4_xpu_2000.csv
```

## Policy-only seen evaluation

Schedule: three repeats per training corner.

| Corner | Result |
|---|---:|
| `(0.32,-0.075)` | 3/3 |
| `(0.32,+0.075)` | 3/3 |
| `(0.48,-0.075)` | 0/3 |
| `(0.48,+0.075)` | 3/3 |
| **Total** | **9/12** |

The failure is systematic at the positive-X/negative-Y corner. The policy
approaches the corner but does not reliably transition from the combined X/Y
approach into descent, close, and lift. This is a compositional stage-transition
failure, not simply a wrong X or Y sign.

Evidence:

```text
results/day12_xy_weighted_xy4_closew4_n5_seen_eval.log
```

## Failing-corner diagnostic

For `(x=0.48, y=-0.075)`, external intervention froze X/Y after step 31,
forced descent, left the policy gripper active, and supplied a fixed lift.
Result:

```text
1/1 success
```

The cube was genuinely lifted. This proves that the corner is physically
reachable and graspable. It also localizes the policy-only failure to the
learned descent/stage transition in this combined 2D state.

This result is **not** a policy-only benchmark.

Evidence:

```text
configs/day12_xy_failing_corner_diagnostic.json
results/day12_xy_freeze_descend_policy_gripper_fixed_lift.log
```

## Conclusions

1. Day12 achieved partial 2D compositional control: `9/12` policy-only seen
   successes across four corner combinations.
2. Three corners were reliable; `(0.48,-0.075)` failed consistently.
3. The failing corner became successful when descent timing and lateral stopping
   were externally corrected, so it is not physically impossible.
4. The model learned basic X/Y signs, but combining positive X and negative Y
   did not reliably trigger the descent/close/lift stage.
5. Day12 is a meaningful return toward the original generalization goal, but it
   does not establish broad 2D generalization or arbitrary-position robustness.
6. No held-out 2D evaluation was run because seen-corner control was not yet
   fully reliable.

## Next direction

Day12 should be treated as a completed controlled 2D baseline. The next
experiment should focus on the failing compositional stage transition rather
than immediately expanding held-out positions. Possible directions include:

- collect more balanced 2D demonstrations emphasizing the failing sign
  combination `(positive X, negative Y)`;
- standardize and inspect the X/Y-to-descent transition timing;
- add a stage-aware descent/lift supervision experiment;
- or run a targeted 2D recovery dataset after preserving this baseline.

Any new data or intervention position used for tuning must not later be called
an untouched held-out test position.
