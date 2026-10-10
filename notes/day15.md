# Day15: Lift-Specific Weighting and 2D Interpolation Evaluation

**Experiment date:** October 7, 2026

## Research questions

1. Can a separate positive-Z/lift loss weight improve the post-grasp lift
   transition without increasing ordinary Z weighting?
2. Does the best seen-corner policy generalize to held-out 2D positions?

## Training and loss changes

The active-axis adapter was extended with axis-specific weights and a
positive-Z lift-specific weight. The Day15 candidate used:

```text
active X weight = 2
active Y weight = 2
ordinary active Z weight = 2
positive-Z lift weight = 4
close-frame weight = 4
```

The positive-Z mask is derived from raw action labels (`delta_z > 0`). This is
a proxy for the staged lift phase because the Day14 demonstrations contain
positive-Z motion during the post-grasp lift segment.

The adapter now supports:

```text
--x-weight
--y-weight
--z-weight
--positive-z-weight
```

The modern training launchers also default to saving only the final checkpoint
(`SAVE_FREQ=$STEPS`) to avoid accumulating hundreds of gigabytes of optimizer
state and intermediate checkpoints.

## Storage cleanup

Before closeout, the local `outputs/` directory was reduced from approximately
224 GiB to approximately 6.2 GiB. Intermediate checkpoints and optimizer/RNG
training states were removed while retaining the final `pretrained_model`
directory for each experiment. These output artifacts are ignored by Git and
were not pushed to GitHub.

## Candidate weight results

The most relevant policy-only results were:

| Configuration | Result |
|---|---:|
| Day14 `XY2/Z2/close4` | Seen `8/12` |
| Day15 `XY2/Z2/liftZ3/close4` | Four-corner screen `2/4` |
| Day15 `XY2/Z2/liftZ4/close4` | Four-corner screen `3/4`; seen `9/12` |
| Day15 `XY2/Z2/liftZ5/close4` | Four-corner screen `2/4` |
| Day15 `XY2/Z3/close4` | Four-corner screen `1/4` |

The lift-specific weight of 4 is the best tested setting. Increasing ordinary Z
weight or lift-specific weight to 5 did not improve closed-loop behavior and
could disrupt earlier descent/stage transitions.

Best current checkpoint:

```text
outputs/act_panda_day15_xy_staged_xy2_z2_liftz4_closew4_xpu_7000/checkpoints/007000/pretrained_model
```

## Day15 seen evaluation

The best Day15 model achieved:

```text
Successes: 9/12
Success rate: 75.0%
```

By endpoint corner:

```text
(0.32,-0.075): 3/3
(0.32,+0.075): 3/3
(0.48,-0.075): 1/3
(0.48,+0.075): 2/3
```

Successful rollouts reached approximately `z_range max=+0.52`. Failures usually
closed the gripper but produced little or no subsequent positive-Z lift.

Evidence:

```text
results/day15_xy_staged_xy2_z2_liftz4_closew4_7000_n5_seen_eval.log
```

## Held-out 2D interpolation

The same best checkpoint was evaluated on untouched positions:

```text
x ∈ {0.36, 0.44}
y ∈ {-0.0375, +0.0375}
```

Result:

```text
Successes: 1/12
Success rate: 8.3%
```

By position:

```text
(0.36,-0.0375): 0/3
(0.36,+0.0375): 0/3
(0.44,-0.0375): 1/3
(0.44,+0.0375): 0/3
```

The `x=0.36` failures often occurred before grasp: the policy stopped near
`x≈0.318` instead of reaching the intermediate target. This is an X-axis
interpolation failure, distinct from the seen-corner grasp-without-lift
failures. The result shows that endpoint-only 2D training does not provide
reliable 2D interpolation.

Evidence:

```text
results/day15_xy_staged_xy2_z2_liftz4_closew4_7000_heldout_2d_eval.log
```

## Conclusions

1. Positive-Z-specific weighting is more useful than globally increasing Z
   weighting. `liftZ4` improved the Day14 seen result from `8/12` to `9/12`.
2. The remaining seen failures are concentrated at the far-X corners and are
   usually post-grasp lift-transition failures.
3. Held-out 2D interpolation remains poor (`1/12`), especially at `x=0.36`,
   where failures often occur before grasp.
4. Further blind scalar weight search is not justified yet. The next experiment
   should add intermediate X positions while preserving the staged protocol and
   the best Day15 weighting.

## Next experiment: Day16

Collect a balanced staged dataset at:

```text
x ∈ {0.32, 0.36, 0.44, 0.48}
y ∈ {-0.075, +0.075}
```

with five demonstrations per combination (40 episodes total). Keep the exact
sequence:

```text
XY approach -> neutral hover -> descent -> close -> closed stabilization -> lift
```

Use the current best weighting (`XY2/Z2/liftZ4/close4`) and preserve the
original held-out combinations for evaluation. The Day16 experiment should
change spatial coverage, not loss weighting, so the effect of intermediate-X
data remains interpretable.
