# Day9: Closed-Loop Failure Localization (2026-10-02)

## Starting point

Day8's Y-active loss weighting improved lateral motion substantially. With the
weighted checkpoint and `n_action_steps=5`, policy-only results were:

- trained positions `y=-0.10/+0.10`: `6/6`;
- midpoint `y=0.00`: `5/5`;
- nonzero interior `y=-0.05`: `3/3`;
- nonzero interior `y=+0.05`: `0/3`.

The positive-side failures repeatedly overshot an object at `y=+0.05` toward
approximately `y=+0.10`, closed off-target, and failed to lift. Day9 asks
whether the remaining failure is primarily grasp/lift timing or lateral
stopping/calibration.

## Diagnostic evaluator additions

`scripts/evaluate_panda_act.py` now supports clearly labeled, non-policy-only
diagnostics:

- `--lift-at-step`, `--lift-duration`, `--lift-command`: inject a fixed positive
  Z command for a limited duration;
- `--freeze-y-at-step`: force executed `delta_y=0` from a chosen step onward.

The options validate their ranges, print their activation, and are not part of
policy-only benchmark results. The reported Z range now uses the executed Z
command, so fixed-lift diagnostics are represented accurately. The script
passes Python syntax validation.

## Diagnostic 1: fixed close only

Configuration: weighted Day8 checkpoint, `n_action_steps=5`, positions
`y=-0.05/+0.05`, fixed close at step 30 for 10 steps, policy Z output.

Result: **0/2**.

- Negative side was approximately aligned at close time, but the policy kept
  producing downward/small negative Z commands and did not lift the cube.
- Positive side was already far beyond the `+0.05` target and failed before a
  valid grasp could form.

Log: `results/day9_weighted_n5_fixed_gripper_nonzero_diagnostic.log`.

## Diagnostic 2: fixed close plus fixed lift

Configuration: same as above, plus a positive Z command of `0.5` from step 40
for 15 steps.

Result: **1/2**.

### Negative side

At step 30:

```text
cube_y = -0.0624
tcp_y  = -0.0615
```

The cube was laterally aligned. After the fixed lift, at step 47:

```text
cube_z = 0.1235
tcp_z  = 0.1210
```

The cube followed the TCP and the environment returned success.

### Positive side

At step 30:

```text
cube_y = +0.0581
tcp_y  = +0.0998
```

The TCP was about 42 mm beyond the target. The forced lift raised the TCP, but
the cube stayed near the table (`cube_z` about `0.018--0.020`), proving that a
lift command cannot rescue a grasp that was formed off-target.

Log: `results/day9_fixed_gripper_and_lift_nonzero_diagnostic.log`.

## Diagnostic 3: freeze Y, fixed close, and fixed lift

Configuration: same weighted checkpoint and `n_action_steps=5`; force
`delta_y=0` from step 11, close at step 30 for 10 steps, and lift with Z `0.5`
from step 40 for 15 steps.

Result: **2/2**.

### Negative side

At step 30:

```text
cube_y = -0.0525
tcp_y  = -0.0527
```

At step 50:

```text
cube_z = 0.1281
tcp_z  = 0.1255
```

### Positive side

At step 30:

```text
cube_y = +0.0473
tcp_y  = +0.0473
```

At step 50:

```text
cube_z = 0.1278
tcp_z  = 0.1254
```

Both cubes followed the TCP upward. This is strong causal evidence that the
positive-side policy-only failure is primarily a lateral stopping/calibration
problem. It is not evidence of a new policy-only success because Y, gripper,
and lift actions were externally intervened on.

Log: `results/day9_freeze_y_close_lift_diagnostic.log`.

## Conclusions

1. The weighted model can physically grasp and lift at both intermediate target
   positions when lateral stopping is corrected externally.
2. The positive `y=+0.05` policy-only failure is not explained by an inability
   to close or lift. The policy continues toward the trained positive endpoint
   near `+0.10` instead of stopping at the intermediate target.
3. The negative-side off-distribution state also benefits from an explicit
   close-to-lift transition; learned lift timing remains unreliable outside
   demonstrated trajectories.
4. `n_action_steps=1` reduced positive-side overshoot but did not solve the
   task (`1/2` diagnostic). Arbitrarily sweeping queue lengths is not the next
   research step.
5. Day9 diagnostic success must remain separate from policy-only results:
   `freeze-Y + fixed close + fixed lift = 2/2`.

## Next experiment: intermediate-distance supervision

Do not add `y=+0.05` to the old Day8 held-out claim. The next training dataset
should deliberately include intermediate distances, for example:

```text
y in {-0.10, -0.05, +0.05, +0.10}
```

with balanced successful demonstrations and the same visible-camera setup,
keyboard step sizes, gripper protocol, weighted loss (`4`), and evaluation
replanning (`n_action_steps=5`). Once `+/-0.05` are used for training, new
held-out positions such as `+/-0.075` should be used for a fresh generalization
claim. Keep the Day8 checkpoint and results unchanged as the baseline.

## Reproduction command for the final Day9 diagnostic

```bash
python scripts/evaluate_panda_act.py \\
  --device xpu \\
  --checkpoint outputs/act_panda_day8_y_weighted_w4_xpu_v1/checkpoints/002000/pretrained_model \\
  --task PandaPickCubeKeyboardRandomLong-v0 \\
  --position-schedule configs/day8_y_axis_nonzero_interpolation_positions.json \\
  --episodes 2 --max-steps 150 --n-action-steps 5 \\
  --gripper-mode close-at-step --close-step 30 --close-duration 10 \\
  --freeze-y-at-step 11 --lift-at-step 40 --lift-duration 15 \\
  --lift-command 0.5 --debug-geometry
```

This command is an intervention diagnostic, not a policy-only evaluation.
Datasets and checkpoints remain local and are not part of the repository.
