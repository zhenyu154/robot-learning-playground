# Day18: Intermediate-Y Supervision, Action-Chunk Postponement, and Reserved Test

**Experiment date:** October 12, 2026

**Status:** Complete: data collection/merge, training, offline audit, inference
comparison, passive grasp diagnostics, and reserved evaluation recorded.

## Outcome

The selected Day18 checkpoint with ten-action execution completed **12/12**
policy-only rollouts at four predeclared, untrained X/Y combinations. All
completed in **76--87 steps** within a 150-step run. The checkpoint, processors,
and evaluation schedule match hashes recorded before the test.

This is evidence of **local 2D positional interpolation**, not broad robustness.
The added TRAINING position `(0.36,-0.0375)` remains `0/3`, and longer execution
can reduce approach-alignment margins. These limitations remain part of the
result, even though the reserved test succeeded.

## Research questions and controls

1. Does adding shorter-Y demonstrations improve stopping at new Y distances
   without replacing established demonstrations?
2. Why does the policy sometimes close but delay or never initiate lift?
3. Can executing more of the already-predicted chunk reach a postponed lift
   while preserving positioning?

The training weights were fixed: active X/Y/Z `2`, positive-Z `4`, close `4`.
Positive-Z weighting is a raw action-sign mask, not an explicit post-close stage
controller. All benchmark actions were policy-controlled: no forced close,
forced lift, lateral freeze, action multiplier, or oracle cube/contact input.

## Collection and dataset construction

The four Day17 development combinations became NEW TRAINING positions:

```text
(0.36,-0.0375), (0.36,+0.0375),
(0.44,-0.0375), (0.44,+0.0375)
```

- Smoke: 4 episodes / 404 frames, separate from training, QA passed.
- New source: 20 successful episodes / 1,902 frames, five per combination, QA
  passed. X/Y signs were correct, close-before-lift counts were 6--10, and all
  episodes had ten lift frames with no lateral action during lift.
- Recording retained the centered Day11 home, min TCP Z `0.008`, 10 FPS,
  XY input `0.25`, Z input `0.50`, visible-arm cameras, and sequential
  `X -> Y -> hover -> descend -> close/stabilize -> lift` protocol.
- Every existing Day17 episode was retained; nothing was removed to make room.

| Source | Episodes | Frames |
|---|---:|---:|
| Day17 preserved-endpoint mixture | 60 | 5,635 |
| New Day18 shorter-Y source | 20 | 1,902 |
| **Merged Day18 dataset** | **80** | **7,537** |

Local dataset:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day18_xy_expanded_v1
```

The final mixture has twelve training combinations. Day17 episodes remain
0--59; new episodes become 60--79. Day17's original Day14/Day16 source IDs are
preserved as ancestor provenance.

Implementation:

- `scripts/audit_day18_y_dataset.py`: source-only 4/20-episode phase QA.
- `scripts/audit_day18_merged_dataset.py`: full 80-episode count/phase QA.
- `scripts/build_day18_dataset.py`: read-only preview by default; `--execute`
  builds in a temporary directory, verifies, and publishes to a NEW root only
  after checks pass. It uses local LeRobot merging without video concatenation.
- Shared Day17 build helpers accept different source/output episode counts and
  exact-copy source labels while keeping Day17 defaults and tests intact.

Verification passed:

- 80/80 merged episodes have successful reward/done and ordered phases.
- Exact action/state/timestamp/frame/reward/done preservation; contiguous
  global and episode indices; correct remapped task STRINGS for all episodes.
- Action/state means, standard deviations, minima, maxima, and counts agree
  with the merged rows; source files match before/after hashes.
- 480 first/middle/last camera samples across all episodes match source images
  exactly. This merge introduced no new re-encoding; the retained Day17 source
  still contains its earlier Day16-subset re-encode.
- LeRobot loads both cameras at the first/last frames and the source boundary
  at global frames 5,634/5,635. Images are finite 3×128×128 tensors.
- Assistant inspected source/merged contact-sheet samples. The user reported
  smoke/full-source QA pass; a separate full-source video-review confirmation
  was not recorded. Sampled images cannot certify every grasp in every frame.

The source datasets/checkpoints remain local. Build provenance is copied to
`results/day18_dataset_provenance.json`; image statistics are aggregated source
per-episode stats rather than fresh statistics over every decoded pixel.

## Training

Checkpoint:

```text
outputs/act_panda_day18_xy_expanded_xy2_z2_liftz4_closew4_xpu_14000/checkpoints/014000/pretrained_model
```

Fresh XPU training, seed `1000`, batch `8`, chunk `50`, ordinary training queue
`10`, learning rate `1e-5`, and unchanged loss weights. Only checkpoint `014000`
was saved. Dataset metadata/statistics hashes match the saved recipe.

```text
Day17: 10500 × 8 / 5635 ≈ 14.91 nominal frame-sample passes
Day18: 14000 × 8 / 7537 ≈ 14.86 nominal frame-sample passes
```

| Final logged metric | Day18 |
|---|---:|
| Loss | 0.211 |
| Weighted L1 | 0.188 |
| Unweighted L1 | 0.182 |
| KL loss | 0.002 |
| Active X normalized MAE | 0.576 |
| Active Y normalized MAE | 0.737 |
| Active Z normalized MAE | 0.154 |
| Positive-Z normalized MAE | 0.071 |
| Close-command normalized MAE | 0.197 |

Training errors use changed dataset distributions/normalizers; lower loss,
partly from a smaller KL contribution, is not online success evidence.

## Offline audit

The first action was audited independently on all 7,537 expert observations
with the queue reset per frame. This is IN-SAMPLE diagnosis, not held-out action
accuracy. Motion hits require correct sign AND magnitude >=0.05; gripper-close
hits use >=1.5.

| Metric | Result |
|---|---:|
| X hit | 96.1% (773/804) |
| Y hit | 87.7% (700/798) |
| X neutral false motion | 5.1% |
| Y neutral false motion | 1.7% |
| Close hit | 96.7% |
| Hold false-close | 3.4% |

There were 639 close-command and 6,898 non-close frames. New shorter-Y X/Y hits
were weaker than those for old positions; for `(0.36,+0.0375)`, Y hit was 66.7%.
A missed hit may indicate a small prediction rather than an incorrect sign.
The online failure pattern was different, underscoring this audit's limits.

## Evaluation splits and frozen baseline

- Original seen: eight combinations at X={0.32,0.36,0.44,0.48}, Y=±0.075,
  three repeats each (`configs/day16_xy_intermediate_seen_eval_positions.json`).
- Added seen: four shorter-Y combinations, three repeats each
  (`configs/day18_y_intermediate_seen_eval_positions.json`). They are now
  training positions and must NOT be called held-out.
- Development: X={0.36,0.44}, Y=±0.05625, predeclared before Day18 training
  (`configs/day18_xy_development_positions.json`). Exact combinations excluded
  from training; outcomes used for candidate/inference selection.
- Reserved final test: X={0.34,0.46}, Y=±0.05, predeclared on Day16 and first
  run AFTER selecting/fixing the Day18 candidate.

The frozen Day17 baseline on the new development schedule was **7/12** with
five-action execution and 150 steps, six successes by step 100. Breakdown:
3/3,3/3,1/3,0/3 in ascending X/Y order. All failures were at X=0.44, including
misalignment, arm lift without cube elevation, and missing descent. This differs
from the earlier Day17 3/12 benchmark at Y=±0.0375; do not equate the schedules.

All runs below use the same Day18 checkpoint, centered home, min TCP Z 0.008,
10 FPS, and the 150-step Long task. The ordinary task has a separate 100-step
TimeLimit; setting `--max-steps 150` alone does not extend it.

## Five-action baseline versus ten-action candidate

| Evaluation | n=5 total | n=5 by step 100 | n=10 total | n=10 by step 100 |
|---|---:|---:|---:|---:|
| Original seen | 20/24 | 19/24 | **23/24** | **23/24** |
| Added shorter-Y seen | 6/12 | 5/12 | **9/12** | 3/12 |
| Development interpolation | 11/12 | 10/12 | **12/12** | **12/12** |

These are separate reset trajectories, not paired identical-state interventions.
Completion by 100 is extracted WITHIN 150-step runs, not a newly repeated
100-step benchmark. Do not combine seen/development scores into one
"generalization rate."

The user reported a second n=5 development run also 11/12, but only one log is
retained. Per-position and timing claims refer to that retained log; no fully
verified 22/24 aggregate is claimed.

Known behaviors:

- n=5 added failures are consistently `(0.36,-0.0375)` and `(0.44,+0.0375)`,
  three failures each, BEFORE meaningful descent/close. These are approach
  stalls, not missing post-grasp lifts.
- n=10 recovers `(0.44,+0.0375)` to 3/3. `(0.36,-0.0375)` remains 0/3: X near
  0.365, Y stops near -0.024, and TCP drifts upward rather than finishing the
  lateral approach. This remains visible despite reserved success.
- Six n=10 added-position successes need steps 101--128. Longer execution is
  not uniformly faster or geometrically more accurate.
- The single n=10 original failure is `(0.36,+0.075)`, with user-observed cube
  tilt during closing. Logged geometry is consistent with irregular contact,
  but does not establish secure grasp or that forced lift would succeed.
- At `(0.36,+0.075)`, n=10 TCP X≈0.3486 at step 40 versus n=5 X≈0.3575 for
  target X=0.36. The longer prefix narrows alignment margin in that approach.

## Passive trace: repeated postponement of lift

The user noticed slight post-grasp suspension/sway often accompanying eventual
lift. Competing explanations include contact geometry/controller settling,
observation dependence, and a postponed action sequence. No causal dependence
on irrelevant noise is established.

Read-only dataset analysis found a pre-lift TCP rise around closing in all 80
expert episodes: median ≈8.8 mm, range ≈6.9--10.0 mm. This is TCP sensor motion,
not measured cube suspension. `get_gripper_pose`/logged `gripper_pos=255` is
ACTUATOR TARGET, not actual finger position or grasp force.

Instrumentation added:

- `--trace-output`: read-only forward hook on the normal ACT model call.
  Copies the full 50-action prediction and unnormalizes a separate clone;
  verifies queued actions agree with the captured prefix. No extra inference.
- `scripts/panda_grasp_trace.py`: before/after TCP and target, actual eight
  finger hinge positions/velocities, cube orientation and tilt-aware bottom
  clearance, table/finger contacts and MuJoCo solver-force estimates.
- Diagnostic cube/contact measurements are NEVER fed to ACT. They are not
  binary grasp-success certificates; pad distance is between body origins.
- `scripts/summarize_grasp_trace.py`: close/upward/clearance timing and chunks
  predicting positive Z only beyond the executed prefix. Analysis thresholds
  are Z>=0.1 and clearance>1 mm without an active table contact.
- Full JSONL traces remain local/ignored; compact summaries and conventional
  terminal logs are committed. Trace creation refuses overwrite.

Six policy-only traced repeats at existing development position
`(0.44,-0.05625)` (`configs/day18_grasp_transition_diagnostic_positions.json`):

```text
n=5: 5/6 successes; one success delayed to step 109; one failure at step 150
n=10: 6/6 successes; completion in steps 80--89
```

The n=5 failed episode first closes at step 47. On ALL 20 predictions from
steps 51--146, meaningful positive Z first occurs at action **7--10**, beyond
the **five** actions executed before replanning. Later commands reach ≈+0.52,
while actual post-close Z never exceeds ≈+0.0136. Each replan discards the old
unexecuted tail and places lifting later again. This is direct evidence of
repeated postponement in one stalled episode, not proof all failures share it.

| Prediction step | First upward action number | Planned absolute step | Max executed-prefix Z |
|---|---:|---:|---:|
| 51 | 10 | 60 | 0.0080 |
| 56 | 7 | 62 | 0.0108 |
| 101 | 9 | 109 | 0.0101 |
| 146 | 9 | 154 | 0.0108 |

The n=5 long successful episode has eight deferred-lift chunks before escaping.
For all five successful n=5 episodes, measured table clearance occurs AT OR
AFTER the first thresholded upward command. Tiny settling/sway may influence
observations, but suspension is not established as a necessary prior trigger.

In five n=10 trials, the step-71 chunk places upward action at zero-based offset
8; it is now executed at step 79, followed by clearance at step 80. This supports
using longer execution to reach the existing predicted lift. Approach also
changes, so n=10 is not a pure intervention at an identical post-grasp state.
Table and gripper contact can still matter when stronger actions are executed.
Tracing overhead and contact-model assumptions limit causal conclusions.

## Reserved test after candidate freeze

Before the test, `results/day18_frozen_candidate.json` recorded model/config/
processor and schedule hashes, weights, task/horizon, prefix, and known failures.
That file remains unchanged as HISTORICAL pre-test evidence, even though its
stored `evaluated=false` is now outdated. Current outcome is in the separate
reserved summary and experiment summary. All frozen hashes match after testing.

| Predeclared reserved configuration | Result |
|---|---:|
| (0.34,-0.05) | 3/3 |
| (0.34,+0.05) | 3/3 |
| (0.46,-0.05) | 3/3 |
| (0.46,+0.05) | 3/3 |
| **Total** | **12/12** |

All finish in steps 76--87 within the 150-step run. Both X and Y coordinates
are absent from the current training values, and exact combinations are absent
from development. No model/prefix/threshold adjustment was made in response to
this test result; no action was externally supplied.

Claim scope: local 2D positional interpolation across four predeclared simulator
configurations, one trained seed, three repeats each. Not a population-wide
100% guarantee, arbitrary-position coverage, varied-object/camera/home/dynamics
robustness, or real-robot transfer. The known seen failure remains 0/3.

The reserved schedule is NOW observed, not untouched. Preserve its historical
final-test result; if outcomes inform future tuning, reclassify it as development
for that future candidate and reserve new test positions before training.

## Verification at closeout

71 CPU tests passed. Trace tests exercise a real ACT queue and saved-style
unnormalizer, confirming identical action selection and model-forward counts,
hook cleanup/overwrite protection, read-only MuJoCo measurement, tilt-aware
clearance, and trace interpretation. Real Panda XML sensor/joint names were
checked without GUI or policy rollout. Build/QA tests cover earlier Day16/17
behavior, source/provenance preservation, and remapped tasks. Final hash and
source-integrity checks are recorded alongside structured summaries.

Datasets/checkpoints are ignored and not pushed. `HANDOFF.md` remains local;
raw traces remain ignored. No additional training, rollout, deployment, or
external message is part of closeout.

## Reproduction setup and next work

With local sources/environment, preview a fresh destination before rebuilding:

```bash
conda activate lerobot-xpu
cd /home/wusanggg/robotics/robot-learning-playground

python scripts/build_day18_dataset.py \
  --output-root /home/wusanggg/robotics/data/panda_pick_cube_day18_xy_expanded_v2 \
  --repo-id wusanggg/panda_pick_cube_day18_xy_expanded_v2

# Add --execute to the same command to build that NEW destination.
python scripts/audit_day18_merged_dataset.py \
  --dataset-root /home/wusanggg/robotics/data/panda_pick_cube_day18_xy_expanded_v1
```

The next requested work can preserve this checkpoint/settings, capture a spatial
policy-only demo, and investigate the incomplete approach at the known training
failure or narrowed alignment margins. Do not launch another Z-weight search
for lateral errors. Any later selected candidate needs fresh reserved positions.
No demo, new collection, adaptive-prefix controller, or new training is started.

## Evidence

```text
results/day18_summary.json
results/day18_dataset_summary.json
results/day18_dataset_provenance.json
results/day18_dataset_build.log
results/day18_dataset_dry_run.log
results/day18_dataset_loading_check.log
results/day18_merged_dataset_qa.log
results/day18_y_intermediate_smoke_qa.log
results/day18_y_intermediate_train_qa.log
results/day18_loss_recipe.json
results/day18_training_tail.log
results/offline_gripper_audit_day18_xy_expanded_xy2_z2_liftz4_closew4_xpu_14000.csv
results/day18_offline_xy_summary.log
results/day18_day17_baseline_development_150step.log
results/day18_day17_baseline_summary.json
results/day18_original_seen_150step_eval.log
results/day18_added_shorter_y_seen_150step_eval.log
results/day18_development_interpolation_150step_eval.log
results/day18_grasp_transition_diagnostic_v1.log
results/day18_grasp_trace_summary_v1.json
results/day18_grasp_trace_analysis_v1.json
results/day18_grasp_transition_n10_v1.log
results/day18_grasp_trace_n10_summary_v1.json
results/day18_grasp_prefix_comparison.json
results/day18_original_seen_n10_150step_eval.log
results/day18_original_seen_prefix_comparison.json
results/day18_added_shorter_y_seen_n10_150step_eval.log
results/day18_added_seen_prefix_comparison.json
results/day18_development_interpolation_n10_150step_eval.log
results/day18_development_n10_summary.json
results/day18_frozen_candidate.json
results/day18_reserved_test_n10_150step_eval.log
results/day18_reserved_test_summary.json
```
