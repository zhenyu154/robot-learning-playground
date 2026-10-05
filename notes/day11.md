# Day11: X-Axis Isolation with Centered Home Pose and Close Weighting

**Experiment date:** October 4, 2026
**Closeout date:** October 5, 2026

## Research question

Can the controlled Day10 methodology transfer from Y-axis interpolation to
X-axis interpolation when the Panda starts from a centered TCP X position?

A second question emerged during training: does X-active weighting suppress
rare gripper-close predictions, and can a separate close-frame loss weight
restore them?

## Centered home pose

The default Gym-HIL Panda home TCP was approximately `x=0.490`. Because the
safe cube range is approximately `x=0.30--0.50`, this made a symmetric X-axis
experiment impossible. A project-local IK-adjusted home pose was added:

```text
configs/day11_home_x_centered.json
TCP ≈ [0.402, 0.000, 0.078]
```

The pose preserves the default downward gripper orientation while moving the
TCP near X `0.40`. It is applied only when `--home-pose` is provided; Day5--
Day10 behavior is unchanged. The installed Gym-HIL package is not modified.

The Day11 data also uses a project-local minimum Cartesian Z bound:

```text
--min-tcp-z 0.008
```

This prevents the gripper from descending into an inconsistent table/fingertip
contact configuration. The same home pose and Z bound must be supplied during
online evaluation.

## Dataset and QA

Dataset:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day11_x_axis_v2
```

Training positions:

```text
x ∈ {0.32, 0.36, 0.44, 0.48}
y = 0.00
```

QA passed:

- 20 episodes, 1,536 frames;
- five episodes per X position;
- 20/20 reward-positive and done-positive episodes;
- X sign agreement `100%`;
- Y active frames `0` in every episode;
- close counts approximately 5--9 frames per episode.

The X position schedule and QA script are:

```text
configs/day11_x_axis_train_positions.json
scripts/audit_day11_x_dataset.py
```

## Active-X training

The project-local active-axis adapter was generalized from Day8's Y-only loss.
Day11 uses:

```text
active X weight = 4
close-frame weight = 1
```

Training checkpoint:

```text
outputs/act_panda_day11_x_weighted_w4_z008_xpu_v1/checkpoints/002000/pretrained_model
```

Final training metrics:

```text
loss: 1.128
weighted l1_loss: 0.334
unweighted l1_loss: 0.304
kld_loss: 0.079
x_active_mae_norm: 1.088
x_neutral_mae_norm: 0.184
```

Offline audit:

| Metric | Active-X model |
|---|---:|
| X sign-and-magnitude hit, threshold 0.05 | 85.4% |
| Neutral X false motion | 16.3% |
| Gripper close hit, threshold 1.5 | 0.0% |
| Hold false-close | 0.0% |

The X prediction means had the expected distance ordering:

```text
x=0.32 → -0.233
x=0.36 → -0.190
x=0.44 → +0.175
x=0.48 → +0.206
```

However, gripper predictions stayed near hold (`max≈1.06`). This was a real
close-signal regression, not an evaluator failure. The Day11 dataset had only
130 close frames (`8.5%`), fewer than Day10 timing-v2 (`12.5%`).

## Close-weighted training

A second fresh model reused the exact same Day11 dataset, home pose, seed, and
training recipe, adding only:

```text
close-frame weight = 4
active-X weight = 4
```

Close frames are identified from raw gripper actions (`gripper >= 1.5`) before
normalization. Other scalar action errors remain weight 1. The checkpoint recipe
records both weights.

Checkpoint:

```text
outputs/act_panda_day11_x_closew4_weighted_w4_z008_xpu_v1/checkpoints/002000/pretrained_model
```

Offline audit:

| Metric | Active-X only | Active-X + close weight |
|---|---:|---:|
| X sign-and-magnitude hit | 85.4% | 86.5% |
| Neutral X false motion | 16.3% | 16.1% |
| Gripper close hit | 0.0% | **81.5%** |
| Hold false-close | 0.0% | **0.7%** |

This is strong evidence that close-specific loss weighting recovered the missing
gripper signal without materially damaging X prediction.

## Policy-only evaluation

All online evaluation used:

```text
n_action_steps = 5
home pose = configs/day11_home_x_centered.json
min TCP Z = 0.008
policy gripper
```

### Smoke evaluation

The close-weighted model succeeded at all four X positions:

```text
x=0.32, 0.36, 0.44, 0.48: 4/4
```

### Formal seen evaluation

Three repeats per training position:

```text
x=0.32: 3/3
x=0.36: 3/3
x=0.44: 3/3
x=0.48: 3/3
Total: 12/12
```

### Held-out X interpolation

Held-out positions:

```text
x=0.34, 0.40, 0.46
```

Three repeats per position:

```text
x=0.34: 3/3
x=0.40: 3/3
x=0.46: 0/3
Total: 6/9
```

At `x=0.46`, the policy approached the target but continued toward the
training endpoint near `x=0.48`, then failed to complete the descend/close/lift
sequence. This is asymmetric local interpolation, not a total X-control
failure.

## X-axis diagnostics

A fixed-close/fixed-lift diagnostic succeeded at all four training positions:

```text
4/4
```

The first X-freeze diagnostic at held-out `x=0.46` closed/lifted too early,
while TCP Z was still high, and failed. A delayed close/lift diagnostic then
succeeded:

```text
freeze X after step 10
close at step 50
lift at step 60
x=0.46: 1/1
```

This shows that `x=0.46` is physically graspable. The remaining policy-only
failure is associated with stage timing and interpolation, not basic reach or
object graspability.

These interventions are not policy-only benchmark results.

Evidence:

```text
results/day11_x_closew4_n5_seen_eval.log
results/day11_x_closew4_n5_heldout_eval.log
results/day11_x_fixed_close_lift_diagnostic.log
results/day11_x_freeze_close_lift_x046_diagnostic.log
results/day11_x_freeze_x046_late_close_lift_diagnostic.log
```

## Conclusions

1. Centering the initial TCP made a symmetric X-axis experiment feasible.
2. Active-X loss weighting learned useful bidirectional X motion.
3. X-active weighting alone caused a severe gripper prediction regression.
4. Adding close-frame loss weight 4 recovered gripper prediction (`81.5%` close
   hit) while preserving X performance.
5. The close-weighted model achieved `12/12` policy-only seen successes.
6. It achieved partial held-out X interpolation (`6/9`), with a consistent
   failure at `x=0.46`.
7. A delayed close/lift intervention succeeded at `x=0.46`, showing physical
   feasibility and implicating learned stage timing/interpolation.

The strongest defensible claim is:

> In the fixed-Y, centered-home MuJoCo Panda task, active-X plus close-frame
> loss weighting enabled reliable policy-only control at four training X
> positions and partial local X interpolation at held-out positions.

This does not establish broad 2D spatial generalization, multi-seed robustness,
randomized-object robustness, or real-robot transfer.

## Next direction

Day11 is complete. The next experiment should either:

- isolate a second spatial axis under the same centered-home/timing protocol;
  or
- build a controlled 2D dataset using the validated X and Y methodology.

Do not mix diagnostic interventions with policy-only results. Keep the Day11
v2 dataset/checkpoints as the baseline for any future 2D experiment.
