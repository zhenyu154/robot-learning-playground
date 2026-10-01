# Day7: Y-Axis Isolation

## Research question and protocol

Can ACT infer the sign and useful magnitude of Y movement from front/wrist
images and proprioception when X is held fixed?

- Fixed X: `0.48`; training Y positions: `-0.10`, `+0.10`.
- 20 successful demonstrations, 10 per side, interleaved.
- Keyboard steps: XY `0.25` (6.25 mm/tick), Z `0.50` (12.5 mm/tick).
- Gripper close hold: approximately 10--12 frames, then lift.
- ACT/XPU: batch size 8, `chunk_size=50`, `n_action_steps=10`, seed 1000,
  2,000 optimizer updates.
- Seen evaluation: 6 scheduled rollouts, 3 per trained Y position. The
  interpolation schedule uses `x=0.48, y=0.00`; it is within the training
  range, not extrapolation.

## Camera visibility issue and correction

Reviewing saved videos on 2026-09-30 showed that the Panda arm was missing from
recorded observations in the original Day7 dataset `panda_pick_cube_day7_y_axis_v1`
and in some Day6 datasets. Arm hiding had been intended to affect only the
operator's viewer, but the helper reassigned `model.geom_group`, which also
affected camera observations. This introduced a visual train/evaluation
mismatch, so the original Day7 policy-only failures are confounded and are not
a clean test of Y-axis learning under the intended visual setup.

The `--hide-robot-arm` option and the model-geometry mutation were removed from
`scripts/day6_record_scheduled.py`. A fresh smoke recording was checked in both
saved camera streams before collecting the corrected dataset. The operator can
still move the interactive viewer camera with the mouse; the recorded front
and wrist cameras remain the policy inputs.

## Corrected v2 dataset QA

Dataset (local, not included in this repository):
`/home/wusanggg/robotics/data/panda_pick_cube_day7_y_axis_v2`

- QA: **PASS**; 20 episodes, 2,037 frames, all 20 reward-positive and done.
- 10 demonstrations at each scheduled Y position; X remained fixed.
- 366 active Y frames; expert-label sign agreement: **94.3%**.
- No failed demonstrations. Close labels: generally 10--17 frames per episode.
- Front/wrist videos were checked and showed the arm as intended.

Run QA with:

```bash
python scripts/audit_day7_y_dataset.py \
  --dataset-root /path/to/panda_pick_cube_day7_y_axis_v2
```

## Training

Corrected v2 run:

- Dataset frames: `2037`; batch size: `8`.
- 2,000 updates; logged `epch=7.85`; runtime about 4m57s.
- Final total loss `1.118`, L1 loss `0.307`, KL loss `0.081`.
- Checkpoint:
  `outputs/act_panda_day7_y_axis_xpu_v2/checkpoints/002000/pretrained_model`

The v2 run can be repeated with the launcher overrides:

```bash
DATASET_REPO_ID=wusanggg/panda_pick_cube_day7_y_axis_v2 \
DATASET_ROOT=/home/wusanggg/robotics/data/panda_pick_cube_day7_y_axis_v2 \
OUTPUT_DIR="$PWD/outputs/act_panda_day7_y_axis_xpu_v2" \
JOB_NAME=act_panda_day7_y_axis_xpu_v2 \
bash scripts/day7_act_y_axis_xpu.sh
```

The original arm-hidden v1 checkpoint and v2 checkpoint are both retained
locally for provenance; checkpoints and datasets are not committed.

## Offline audits

Frame-level prediction CSVs are committed under `results/`.

Corrected v2 gripper audit (`2037` frames):

- Target hold: `1788`; target close: `249`.
- Close hit rate: `66.7%`; hold false-close rate: `0.0%`.
- This is conservative gripper behavior: fewer false closes, but about one
  third of target close frames are missed.

Corrected v2 Y audit uses a prediction magnitude threshold of `0.05`. A frame
counts as a correct active-Y prediction only if the target is active, the
predicted magnitude reaches that threshold, and the predicted sign matches:

- Overall: `258/366 = 70.5%`; neutral false-motion: `7/1671 = 0.4%`.
- `y=-0.10`: `56.7%` sign-and-magnitude hit rate; target mean `-0.225`,
  predicted mean `-0.046`.
- `y=+0.10`: `83.9%`; target mean `+0.218`, predicted mean `+0.070`.

Thus the corrected policy has measurable directional signal with little neutral
motion, but the prediction magnitude remains well below the demonstrated
commands and is weaker for negative Y. This metric is not a rollout success
rate. The original v1 audit had no predictions reaching the same `0.05`
magnitude threshold; analysis at a lower threshold found some correct polarity,
but outputs were near zero.

## Online evaluation (2026-10-01)

Corrected v2, policy-only, seen positions, `n_action_steps=10`:

- Result: **0/6**.
- Geometry logs show TCP Y moving only about 1--2 cm toward cubes placed at
  `y=+/-0.10`; the remaining lateral error was around 9 cm.
- The policy descended close to the table and closed the gripper while still
  misaligned; it did not lift the cube.

A two-episode evaluation-only diagnostic with `n_action_steps=1` also failed
(**0/2**). It changed the behavior rather than fixing it: the arm descended to
its lower position early, Y motion stayed under about 1 cm, and the gripper
remained open. This is an ablation, not the trained benchmark setting. It
suggests the policy does not recover reliably after its own actions take it
away from demonstrated states (closed-loop distribution shift).

The corrected v2 interpolation schedule was **not** evaluated. The original
arm-hidden v1 interpolation failures should not be used as clean evidence.

## Conclusions and next experiment

1. Correcting camera visibility improved the offline Y signal, but did not
   solve closed-loop spatial control. The camera issue was a real confound, not
   the only source of failure.
2. The principal observed failure is insufficient lateral Y motion followed
   by premature descent; with `n_action_steps=10`, the gripper closes off-target.
3. The v2 data labels pass QA, but only about 18% of frames have active Y
   labels. A plausible next hypothesis is that the ACT regression objective and
   sampling dilute sparse movement commands; this should be tested rather than
   assumed.
4. Keep `n_action_steps=10` as the benchmark setting. Do not run interpolation
   until seen-position control improves. A next training experiment should
   change one factor related to Y-active window sampling/loss weighting and
   track both active-Y hit rate and neutral false-motion.

## Reproduction assets in this repository

- Position schedules: `configs/day7_y_axis_*_positions.json`
- Dataset QA: `scripts/audit_day7_y_dataset.py`
- Offline summaries: `scripts/offline_gripper_audit.py`,
  `scripts/summarize_day7_y_audit.py`
- XPU training launcher: `scripts/day7_act_y_axis_xpu.sh`
- Frame-level audit CSVs: `results/offline_gripper_audit_day7_y_axis_xpu*2000.csv`
