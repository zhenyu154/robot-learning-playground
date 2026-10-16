# Day19: Execution-Prefix Tradeoffs and Missing Lateral Recovery

**Experiment date:** October 15, 2026

**Status:** Complete: existing-data analysis, execution-prefix ablations,
passive traces, training/development validation, frozen reserved evaluation,
and post-test failure diagnosis recorded. No new demonstrations or training.

## Outcome

Reusing the frozen Day18 checkpoint, executing ten actions per prediction
through step 20 and twenty from step 21 achieved **24/24 original-training,
12/12 added-training, and 12/12 development successes**. All finished by
step 100 within the fixed 150-step horizon. After freezing that candidate,
a fresh four-position reserved evaluation scored **9/12**. The failing
configuration `(0.45,+0.0625)` overshot in X and failed all three grasps.

Shorter initial prefixes recovered that observed overshoot case but introduced
training-position failures. Initial-five and capped initial-seven each scored
**9/12** on the added-training suite, failing at opposite Y signs for X=0.44.
Neither follow-up was selected as a replacement. The next investigation is
lateral recovery and approach-to-descent progression, rather than another
unbounded prefix sweep. A newly declared v2 reserved schedule remains unused.

Day18's historical **12/12** final result remains intact. Its reserved
positions differ from Day19's, so the two rates are not a matched comparison.
Perfect training/development scores are not pooled into generalization claims.

## Research questions and controls

1. Does an approach stall occur because the needed movement is absent from
   predictions, or because it lies beyond the executed prefix?
2. Can longer commitment recover stage progression without overshooting?
3. Does a fixed early/late schedule retain both approach precision and lift?
4. What remains when changing initial cadence moves the failure elsewhere?

| Setting | Unchanged Day19 value |
|---|---|
| Checkpoint | Day18 ACT, 14,000 updates; seed 1000 |
| Dataset | 80 episodes / 7,537 frames; no collection or edits today |
| Training weights | Active X/Y/Z 2, positive-Z 4, close 4 |
| Observations | Front/wrist RGB and 18-value robot state |
| Predicted chunk | 50 actions; temporal ensembling disabled |
| Action | Learned delta-X/Y/Z and gripper, without scaling or forced events |
| Home / minimum TCP Z | Day11 centered home / 0.008 m |
| Task / horizon / FPS | `PandaPickCubeKeyboardRandomViewerLong-v0` / 150 / 10 |
| Varied factor | Number of predicted actions executed before replanning |

Checkpoint and processors:

```text
outputs/act_panda_day18_xy_expanded_xy2_z2_liftz4_closew4_xpu_14000/checkpoints/014000/pretrained_model
```

All action values come from ACT. Scheduled variants use only the episode step
counter; cube/contact measurements are passive diagnostics, never policy inputs.
The scheduler is a fixed-time inference ablation, not an alignment detector.
At 10 FPS, prefixes of 5,10,20 mean nominal prediction intervals of 0.5,1,2 s.
Changing the prefix also changes the trajectory and later model observations;
comparisons are not same-state causal interventions.

## Existing evidence and fixed-prefix diagnostics

The initial read-only analysis compared five expert episodes at each of
`(0.36,±0.0375)` with saved offline predictions and earlier sparse rollouts.
All experts reach the intended Y region. Saved first-action predictions detect
the negative-side initial Y burst in 24/24 labeled frames but later Y labels
in 0/6; the mirror has 18/23 initial hits and 0/4 later hits. These expert-state
first-action measurements cannot reveal online full-chunk tails or alone
explain the asymmetric failure.

| Check | Fixed 10 | Fixed 20 |
|---|---:|---:|
| Six diagnostic trials at `(0.36,±0.0375)` | 3/6 | **6/6** |
| Original eight training positions, 24 trials | 23/24, Day18 reference | **12/24** |

The fixed-10 negative-side failures reach 150 steps without success. In all
**33 failed-episode replans at steps 41–141**, descent begins at chunk actions
14–24, beyond the ten executed actions. Later predicted Y corrections are
weak or absent. Target motion follows the expected command scaling; selected
queued actions are not being dropped by the action pipeline.

Fixed-20 recovers both diagnostic positions, completing in steps 94–99, but
overshoots the inner X training positions. Original-training results are 0/12
at X=0.36/0.44 and 12/12 at X=0.32/0.48. Representative inner-position X offsets
at step 40 are about 32–48 mm. Fixed-20 is therefore rejected as a replacement.

Evidence: [existing-data analysis](../results/day19_approach_baseline_analysis_v1.json),
[fixed-10 trace summary](../results/day19_approach_n10_v1_summary.json),
[fixed-20 trace summary](../results/day19_approach_n20_v1_summary.json),
[prefix comparison](../results/day19_prefix_comparison.json), and
[original-training fixed-20 result](../results/day19_original_seen_n20_summary.json).

## Ten then twenty actions: training and development validation

The first scheduled rule executes prefixes 10,10,20,20,... at prediction steps
1,11,21,41,... . The switch occurs on an empty queue, preserving all selected
queued actions. The six-trial diagnostic alternates the conflicting positions
`(0.36,-0.075)` and `(0.36,-0.0375)` and succeeds 6/6 in steps 82–86.

| Suite | Positions | Fixed-10 reference | 10 then 20 | New completion steps |
|---|---|---:|---:|---|
| Original training | X=0.32,0.36,0.44,0.48; Y=±0.075 | 23/24 | **24/24** | 73–86 |
| Added training | X=0.36,0.44; Y=±0.0375 | 9/12 | **12/12** | 70–98 |
| Development | X=0.36,0.44; Y=±0.05625 | 12/12 | **12/12** | 81–88 |

The reference row values come from preserved Day18 runs at these same suites.
Completion by step 100 is derived from the 150-step runs, not a separate
100-step experiment. No model, loss, data or camera changes occurred.

Evidence: [scheduled diagnostic](../results/day19_scheduled_prefix_outcome_v1.json),
[original training](../results/day19_original_seen_scheduled_summary.json),
[added training](../results/day19_added_seen_scheduled_summary.json), and
[development](../results/day19_development_scheduled_summary.json).

## Frozen candidate and fresh reserved result

Before development selection, four new configurations were declared in
[the v1 declaration](../results/day19_reserved_test_declaration.json).
After the successful training/development checks, the
[candidate manifest](../results/day19_frozen_candidate.json) recorded model,
processor, implementation, home and schedule hashes before the reserved run.

| Day19 reserved v1 configuration | Result | Completion / timeout steps |
|---|---:|---|
| `(0.35,-0.0625)` | 3/3 | 85,87,85 |
| `(0.35,+0.0625)` | 3/3 | 77,77,77 |
| `(0.45,-0.0625)` | 3/3 | 74,74,74 |
| `(0.45,+0.0625)` | **0/3** | 150,150,150 |
| **Total** | **9/12** | Successful episodes 74–87 |

The failure starts early: at step 20 TCP X is about 0.4693 versus cube 0.45,
already 19.3 mm beyond the target while the schedule still uses initial-ten.
By step 40 the representative XY offset is approximately (+20.1,+10.8) mm.
The arm descends, closes and moves upward, but the cube remains at table height.
This is a misaligned grasp, not simply absent lift commands. Sparse log samples
do not establish contact forces or grasp stability.

All frozen hashes matched after the test and at closeout. The manifest's
`evaluated=false` remains historical pre-test evidence; the completed outcome
is in [the reserved result](../results/day19_reserved_test_summary.json).
The observed positions may inform diagnosis, but subsequent recovery there
cannot replace this historical 9/12 result. Day18's reserved positions were
`(0.34,±0.05)` and `(0.46,±0.05)`, so its 12/12 is not a matched baseline here.

## Post-test diagnostics: shorter initial execution

A fixed-10 follow-up at `(0.45,+0.0625)` scores **1/3**. All three still have
about 19.27 mm of initial X overtravel; the single success does not restore
centering. Two failures show weak or deferred cube elevation after closing.

Initial-five through step 20, followed by 20 from step 21, succeeds **6/6** at
`(0.45,±0.0625)`. At the positive-Y failure position, step 20 X error decreases
to about 6.37 mm. It also retains original training 24/24, then fails **9/12**
on added training: `(0.44,+0.0375)` becomes 0/3. One repeat never meaningfully
descends/closes; two grasp and lift too late to reach task success by 150.
Those failures are untraced, so repeated discarded-tail postponement at that
specific position is not established. They remain failures at the agreed cap.

Capped initial-seven tests one intermediate cadence while retaining the same
20-step initial phase. Because 20 is not divisible by 7, the actual prefixes are:

```text
Prediction at step 1:  execute 7 actions, steps 1–7.
Prediction at step 8:  execute 7 actions, steps 8–14.
Prediction at step 15: execute 6 actions, steps 15–20.
Prediction at step 21 and thereafter: execute 20 actions per prediction.
```

The last initial prefix is shortened before queuing; no selected action is
discarded at the switch. This is not a constant-seven schedule.
Its two-position diagnostic at `(0.44,+0.0375)` and `(0.45,+0.0625)` passes
**6/6**, in steps 76–85. At the former overshoot point, step 20 X error is now
about **0.54 mm**. The training point still has about 14.6 mm of Y offset when
descent begins, despite successful lifts. Success alone does not establish
precise centering or a large grasp margin.

| Added training configuration | Initial 10 then 20 | Initial 5 then 20 | Capped initial 7 then 20 | Initial 7 steps |
|---|---:|---:|---:|---|
| `(0.36,-0.0375)` | 3/3 | 3/3 | 3/3 | 87,87,87 |
| `(0.36,+0.0375)` | 3/3 | 3/3 | 3/3 | 86,86,86 |
| `(0.44,-0.0375)` | 3/3 | 3/3 | **0/3** | 150,150,150 |
| `(0.44,+0.0375)` | 3/3 | **0/3** | 3/3 | 83,83,83 |
| **Total** | **12/12** | **9/12** | **9/12** | |

The initial-seven regression is at the negative-Y mirror of the initial-five
failure. Original-training and development tests for initial-seven were not
run. Development was not run for initial-five. Neither follow-up was frozen
or evaluated on a fresh reserved set.

Evidence: [fixed-10 observed-failure diagnostic](../results/day19_observed_failure_fixed_n10_outcome.json),
[initial-five diagnostic](../results/day19_early_feedback_outcome_v1.json),
[initial-five original training](../results/day19_original_seen_early5_summary.json),
[initial-five added training](../results/day19_added_seen_early5_summary.json),
[initial-seven diagnostic](../results/day19_intermediate_prefix_outcome_v1.json), and
[initial-seven added training](../results/day19_added_seen_initial7_outcome_v1.json).

## What the final failure trace and saved audit establish

All three initial-seven failures at `(0.44,-0.0375)` are approach stalls.
TCP X is 0.426196 at step 20: **13.804 mm short** of cube X=0.44. At step 40
TCP XY is approximately (0.426193,-0.002130). None executes a meaningful descent
(Z<=-0.1) or close action (gripper>=1.5), and the cube remains at table height.

Two observations separate early positioning from later transition delay:

- The step 8 prediction's remaining X tail sums to only 0.0233 action units,
  about **0.58 mm** of target displacement at the measured 0.025 m/action scale.
  It does not contain the missing 14 mm X travel. The step 15 prediction gives
  one substantial positive-X action, then nearly stops. Across all 21 full
  later predictions at steps 21–141 in the three failures, no X action exceeds 0.05.
- Those same 21 later predictions put descent at actions 22–37, beyond the
  executed 20. Full-chunk/action accounting passes, and target-motion residuals
  during these stalls are numerically negligible. The policy lacks substantial
  X correction while repeatedly postponing descent; selected queued actions
  are not being overwritten by the execution pipeline.

The existing ten mirrored expert demonstrations and saved Day18 offline audit
provide a read-only recovery check. All experts succeed and reach X≈0.439355
before descent, within 0.65 mm of target. First-action X prediction hits on
initial positive-X label frames are 25/26 at negative Y and 25/25 at positive Y;
hits on later positive-X label frames are **0/4** and **0/5** respectively.
The initial burst is the first contiguous active-X label run; later labels
describe timing, not operator intent. The hit threshold is positive X>=0.05.

These expert-state first-action predictions differ from online 50-action chunks
at the failed trajectory. They do not prove the cause or that adding recovery
data will solve it. They motivate a targeted recovery-data or phase-dependent
inference experiment, with a separate protocol before execution.

Evidence: [complete initial-seven trace summary](../results/day19_added_seen_initial7_capped_then20_step21_v1_summary.json)
and [recovery analysis](../results/day19_initial7_failure_recovery_analysis_v1.json).

## Implementation and operational notes

- `scripts/panda_action_prefix.py` validates aligned fixed-time schedules,
  resets each episode, and changes capacity only on an empty queue. Rebuilding
  the ACT deque is necessary: changing only the config can silently retain an
  old maximum length and select the wrong part of a longer prefix.
- `scripts/evaluate_panda_act.py` adds paired optional flags
  `--later-n-action-steps` and `--prefix-switch-step`; default fixed-prefix
  evaluation remains available. `scripts/panda_grasp_trace.py` records the
  actual per-prediction prefix and checks complete consumption before replanning.
- `scripts/panda_capped_action_prefix.py` and
  `scripts/evaluate_panda_capped_prefix.py` provide an opt-in capped adapter,
  preserving the source bytes frozen for the earlier final test. The adapter
  restores the process-local scheduler binding on exit.
- `scripts/summarize_approach_trace.py` checks full chunks, executed prefixes,
  policy/requested actions, contiguous steps, terminal records and schedule.
  Capped metadata requires `scripts/summarize_capped_approach_trace.py`.
- CPU tests exercise real ACT queue behavior with synthetic outputs, episode
  resets, cap 7+7+6, aligned 5/10 compatibility, corruption rejection and hook
  cleanup. They do not certify learned-policy success or XPU availability.

The first fixed-10 diagnostic ended with a GLFW TLS assertion after printing
its 3/6 summary. Six complete episode records and all 695 step records were
validated before accepting the evidence; the process crash remains in its log.
No rerun was needed solely to replace a complete recorded result.

The first capped launch failed before any episode because its evaluator file
was absent; its 153-byte error log is preserved. A reused output name then
correctly triggered the overwrite guard. The completed six-episode capped run
uses `results/day19_intermediate_prefix_recipe_v3.json` and v2 output names.
The unused v2 preparation recipe has six new-source hashes that differ from
the current implementation; it is retained as superseded preparation history,
not a valid recipe for a completed run. The v1 and v3 capped Python source
hashes match the current files. Do not rewrite the historical recipes to hide
this preparation error.

Shell launchers used during diagnosis are retained for provenance. Subsequent
user-facing commands will be explicit Python invocations with visible flags,
as requested. No new training, collection or GUI rollout was launched during
closeout.

## Verification at closeout

**80 CPU tests passed** using the tested `lerobot-xpu` interpreter. Saved raw
traces were revalidated against their summaries; terminal logs were checked
against episode counts, scheduled positions, outcomes and switch messages.
All six checkpoint/processor hashes and frozen-candidate/source hashes match.
The superseded unused-v2 recipe exception is listed separately in
[closeout verification](../results/day19_closeout_verification.json).

The Day18 baseline, historical manifests and all observed failures are
preserved. Models, datasets, high-volume raw traces and the local handoff are
excluded from the public commit. Public evidence includes schedules, code,
tests, terminal logs, compact summaries, declarations and recipes.

## Explicit reproduction commands

These are reference commands for the completed experiments, not instructions
to start another run at closeout. Local data/checkpoints and the environment
in [ENVIRONMENT.md](../ENVIRONMENT.md) are required. Use unused output names.
The Long task and the 150-step cap are both needed.

Example: reproduce the frozen 10→20 rule on the original training suite:

```bash
conda activate lerobot-xpu
cd /home/wusanggg/robotics/robot-learning-playground
set -o pipefail
RUN_TAG=day19_reproduce_original_10_then20_v1
DAY18_MODEL="$PWD/outputs/act_panda_day18_xy_expanded_xy2_z2_liftz4_closew4_xpu_14000/checkpoints/014000/pretrained_model"
LOG="results/${RUN_TAG}_eval.log"

if [[ -e "$LOG" ]]; then
  echo "Output exists; choose a new RUN_TAG before running."
else
  python scripts/evaluate_panda_act.py \
    --checkpoint "$DAY18_MODEL" --device xpu \
    --task PandaPickCubeKeyboardRandomViewerLong-v0 \
    --episodes 24 --max-steps 150 --fps 10 \
    --n-action-steps 10 --later-n-action-steps 20 --prefix-switch-step 21 \
    --gripper-mode policy \
    --home-pose configs/day11_home_x_centered.json --min-tcp-z 0.008 \
    --position-schedule configs/day16_xy_intermediate_seen_eval_positions.json \
    --debug-geometry \
    2>&1 | tee "$LOG"
fi
```

Example: reproduce the capped initial-seven added-training diagnostic:

```bash
conda activate lerobot-xpu
cd /home/wusanggg/robotics/robot-learning-playground
set -o pipefail
DAY18_MODEL="$PWD/outputs/act_panda_day18_xy_expanded_xy2_z2_liftz4_closew4_xpu_14000/checkpoints/014000/pretrained_model"
RUN_TAG=day19_reproduce_added_initial7_v1
TRACE="results/${RUN_TAG}_grasp_trace_v1.jsonl"
LOG="results/${RUN_TAG}_eval.log"

if [[ -e "$TRACE" || -e "$LOG" ]]; then
  echo "Output exists; choose a new RUN_TAG before running."
else
  python scripts/evaluate_panda_capped_prefix.py \
    --checkpoint "$DAY18_MODEL" --device xpu \
    --task PandaPickCubeKeyboardRandomViewerLong-v0 \
    --episodes 12 --max-steps 150 --fps 10 \
    --n-action-steps 7 --later-n-action-steps 20 --prefix-switch-step 21 \
    --gripper-mode policy \
    --home-pose configs/day11_home_x_centered.json --min-tcp-z 0.008 \
    --position-schedule configs/day18_y_intermediate_seen_eval_positions.json \
    --debug-geometry --trace-output "$TRACE" \
    2>&1 | tee "$LOG"
fi
```

The capped entry point is required for 7+7+6; the strict scheduler rejects an
unaligned n=7/switch-at-21 pair. To validate an existing local trace without a new
rollout, choose a fresh summary output:

```bash
python scripts/summarize_capped_approach_trace.py \
  --trace results/day19_added_seen_initial7_capped_then20_step21_v1_grasp_trace_v1.jsonl \
  --output results/day19_added_seen_initial7_trace_recheck_v1.json
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Next work

Preserve the checkpoint and all final-test records. Define a bounded recovery
experiment that addresses early X undertravel/overtravel and later correction,
then compare it on observed training/development configurations. No recovery
collection, new training, or phase-dependent policy was started during closeout. Use a new dataset/output root if collection is agreed; retain
all existing demonstrations rather than replacing endpoint coverage.

Reserved v2 positions `(0.38,±0.04375)` and `(0.42,±0.04375)` were declared
before broader follow-up selection and remain **unrun**. Freeze any future
candidate after development selection and before evaluating them. Do not use
reserved outcomes as the next tuning loop or compare different reserved sets
as a matched baseline.

## Evidence

The main index is [results/day19_summary.json](../results/day19_summary.json).
It separates the frozen 10→20 evaluations, initial-five follow-up, capped
initial-seven diagnostic/regression, and future reserved status. Individual
JSON summaries link retained logs and record hashes of raw local inputs.
Raw traces are not distributed, so independent replay requires the local
checkpoint/environment and a new run; CPU queue tests use synthetic outputs.
