# Day13: Z-Weighted 2D Control and Stage-Transition Diagnostics

**Experiment date:** October 6, 2026

## Research question

Does adding explicit active-Z weighting improve Day12's 2D pick-and-lift
behavior, especially the descent and lift stages?

## Motivation

The 4,000-step Day12 model predicted X/Y and gripper actions reasonably well
offline, but achieved `0/12` policy-only successes. Rollouts often reached the
cube and closed the gripper without producing a reliable lift. The project-local
loss adapter previously supported active X/Y weighting and close-frame weighting,
but not active Z weighting.

## Implementation

The active-axis loss adapter now supports:

```text
--active-axis=x
--active-axis=y
--active-axis=z
--active-axis=xy
--active-axis=xyz
```

Day13 used:

```text
active X/Y/Z weight = 4
close-frame weight = 4
```

The KL term and standard ACT checkpoint format were unchanged. The new launcher
is:

```text
scripts/day13_act_xyz_weighted_xpu.sh
```

The CPU correctness suite passed all 24 tests after the extension.

## Training

Dataset: the balanced Day12 v2 dataset with four corner positions and 2,203
frames. Training used 4,000 updates, batch size 8, seed 1000, and the same XPU
recipe as Day12.

Checkpoint:

```text
outputs/act_panda_day13_xy_z_weighted_xy4_z4_closew4_xpu_4000/checkpoints/004000/pretrained_model
```

Final training metrics:

```text
loss: 0.630
l1_loss: 0.374
kld_loss: 0.026
unweighted_l1_loss: 0.309
x_active_mae_norm: 1.659
y_active_mae_norm: 0.889
z_active_mae_norm: 0.186
close_active_mae_norm: 0.319
```

## Offline audit

| Metric | Day13 result |
|---|---:|
| X sign-and-magnitude hit | 77.1% |
| Y sign-and-magnitude hit | 81.6% |
| X neutral false motion | 13.3% |
| Y neutral false motion | 8.2% |
| Active Z direction/magnitude hit | **98.4%** |
| Neutral Z false motion | 19.3% |
| Gripper close hit | **99.0%** |
| Hold false-close | 6.1% |

The model learned the expert Z actions very accurately offline. This did not
translate directly into reliable closed-loop control.

Evidence:

```text
results/offline_gripper_audit_day13_xy_z_weighted_xy4_z4_closew4_xpu_4000.csv
```

## Policy-only evaluation

The same 12-episode seen-corner evaluation was used throughout, with
`n_action_steps=5`, the centered home pose, and minimum TCP Z `0.008`.

```text
Day12 XY-weighted 4K: 0/12
Day13 XYZ-weighted 4K: 2/12
```

Day13 results by corner:

| Corner | Result |
|---|---:|
| `(0.32,-0.075)` | 2/3 |
| `(0.32,+0.075)` | 0/3 |
| `(0.48,-0.075)` | 0/3 |
| `(0.48,+0.075)` | 0/3 |
| **Total** | **2/12** |

A tighter `n_action_steps=1` diagnostic achieved `0/4`, so action chunk length
was not the primary cause.

## Stage-transition diagnostics

At `(0.48,-0.075)`, forcing descent and lateral freezing while retaining the
policy gripper and forcing the lift succeeded `1/1`. The cube was physically
graspable.

A second diagnostic forced descent and lateral freezing but left both gripper
and lift policy-controlled. The policy closed the gripper but did not generate
a meaningful positive-Z lift (`0/1`).

These interventions are not policy-only benchmark results. They localize the
remaining failure to the learned post-grasp transition:

```text
descent -> close -> positive-Z lift
```

## Conclusion

Z weighting improved the policy-only result from `0/12` to `2/12` and produced
excellent active-Z offline metrics, but did not solve closed-loop stage timing.
The next experiment therefore used a new staged dataset with explicit hover,
closed stabilization, and lift phases rather than further increasing the Z
weight.
