# Day17: Preserve Day14 Endpoint Demonstrations and Add Intermediate X

**Experiment date:** October 9, 2026
**Status:** Complete: dataset build, training, offline audit, seen evaluation,
longer-horizon evaluation, and development diagnostics recorded.

## Question

Does retaining the exact 40 endpoint demonstrations that supported Day15 and
adding 20 intermediate-X demonstrations improve control across all eight
positions? Reduced endpoint repetition is a hypothesis for Day16's regression,
not an established causal explanation. Loss weights were kept unchanged.

## Mixture

New local dataset:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day17_xy_preserved_endpoints_v1
```

| Source | Selection | Episodes | Frames |
|---|---|---:|---:|
| Day14 staged v1 | All endpoint demonstrations | 40 | 3,765 |
| Day16 intermediate v1 | Only X=0.36/0.44, Y=±0.075 | 20 | 1,870 |
| **Total** | Eight combinations | **60** | **5,635** |

Endpoint combinations have 10 demonstrations each. Intermediate-X combinations
have 5 each. Day16 endpoint demonstrations are NOT included. Both original
roots remain unchanged. The new dataset follows source episode order: Day14
episodes become 0--39; selected Day16 episodes become 40--59. Source ID mappings
are retained in `dataset_provenance.json` within the dataset root.

## Implementation

- `scripts/build_day17_dataset.py`: default read-only preview; `--execute` uses
  local LeRobot split/merge utilities in a temporary directory, validates the
  result, and publishes to a new root only after checks pass.
- `scripts/audit_day17_xy_dataset.py`: phase-aware QA with 10 endpoint and 5
  intermediate episodes per combination. It reuses Day16 checks with an explicit
  count map. Day16's existing CLI/default counts are unchanged.
- `tests/test_day17_dataset.py`: composition, dry-run no-write behavior, source
  compatibility, overwrite protection, unchanged labels, and stats correctness.

No upstream LeRobot files were modified. Source selection can re-encode a
Day16 video file because it contains both kept and excluded episodes. Merging
keeps source video shards separate and copies Day14 camera files, avoiding a
second re-encode or cross-source video concatenation.

## Verification

- 50 CPU tests passed, including all earlier weighted-loss and Day16 QA tests.
- 60/60 demonstrations passed phase/order, reward, and done QA.
- Actions, robot states, per-episode timestamps, frame indices, and reward/done
  labels match their selected source rows exactly.
- Output episode/global indices are contiguous; counts are 60 episodes and
  5,635 frames.
- Action/state means, standard deviations, minima, maxima, and counts were
  checked against the actual merged data. Per-camera image stats are aggregated
  by upstream from source episode stats, not freshly recomputed over re-encoded
  pixels; small compression differences remain a limitation.
- 360 source/output camera-frame comparisons: first, middle, and last frames
  of each episode, both cameras. All Day14 samples match exactly. Maximum
  sampled RGB mean-absolute error was about 1.997 on the 0--255 scale; overall
  mean was about 0.383. This is sampled alignment verification, not exhaustive
  pixel equality.
- Full file-hash snapshots confirm both source datasets remained unchanged.
- The final dataset was opened through LeRobot and frames 0, 3,764, 3,765, and
  5,634 loaded successfully with both 3×128×128 observation images. This includes
  the Day14-to-Day16 boundary.
- The contact sheet was inspected locally; sampled source/merged views align.
  Manual picture/video review was completed successfully before training.

Local contact sheet:

```text
<dataset-root>/verification/source_vs_merged.png
```

## Rebuild and audit

The builder refuses to overwrite an existing root. The v1 dataset is already
built; use a new output root for any fresh preview or rebuild:

```bash
conda activate lerobot-xpu
cd /home/wusanggg/robotics/robot-learning-playground

# Read-only preview of a NEW destination.
python scripts/build_day17_dataset.py \
  --output-root /home/wusanggg/robotics/data/panda_pick_cube_day17_xy_preserved_endpoints_v2 \
  --repo-id wusanggg/panda_pick_cube_day17_xy_preserved_endpoints_v2

# To build that destination, rerun the same command with --execute.
python scripts/audit_day17_xy_dataset.py \
  --dataset-root /home/wusanggg/robotics/data/panda_pick_cube_day17_xy_preserved_endpoints_v1
```

The source datasets and local environment are required; they are not published
with the repository. The build performs CPU video selection/verification, not
policy training. It makes no Hub upload.

## Training

Checkpoint:

```text
outputs/act_panda_day17_xy_preserved_endpoints_xy2_z2_liftz4_closew4_xpu_10500/checkpoints/010500/pretrained_model
```

Training started fresh with seed 1000, batch size 8, chunk size 50, learning
rate 1e-5, active X/Y/Z weights 2, positive-Z weight 4, and close weight 4.
Only the final checkpoint was saved. The upstream log abbreviates the final
step as `10K`; the actual update count and checkpoint are **10,500**.

```text
10500 × 8 / 5635 ≈ 14.91 nominal frame-sample dataset passes
Day15: 7000 × 8 / 3765 ≈ 14.87
Day16: 7200 × 8 / 3875 ≈ 14.86
```

These are approximate sampling exposures, not guaranteed shuffled epochs or a
count of independently observed action-chunk labels.

| Final logged metric | Day17 |
|---|---:|
| Total loss | 0.230 |
| Weighted L1 | 0.190 |
| Unweighted L1 | 0.185 |
| KL loss | 0.004 |
| Active X normalized MAE | 0.331 |
| Active Y normalized MAE | 0.624 |
| Active Z normalized MAE | 0.164 |
| Positive-Z normalized MAE | 0.080 |
| Close-command normalized MAE | 0.236 |

The lower total loss partly reflects the smaller KL contribution. Logged
training errors use different dataset distributions/normalizers across days
and are not a like-for-like test of policy performance.

A reproducible local launch, using existing scripts:

```bash
DATASET_REPO_ID=wusanggg/panda_pick_cube_day17_xy_preserved_endpoints_v1 \
DATASET_ROOT=/home/wusanggg/robotics/data/panda_pick_cube_day17_xy_preserved_endpoints_v1 \
ACTIVE_WEIGHT=2 X_WEIGHT=2 Y_WEIGHT=2 Z_WEIGHT=2 \
POSITIVE_Z_WEIGHT=4 CLOSE_WEIGHT=4 \
STEPS=10500 SAVE_FREQ=10500 LOG_FREQ=100 PROGRESS_MINITERS=50 SAVE_LOG=1 \
OUTPUT_DIR="$PWD/outputs/act_panda_day17_xy_preserved_endpoints_replication" \
JOB_NAME=act_panda_day17_xy_preserved_endpoints_replication \
bash scripts/day15_act_xyz_axis_weighted_xpu.sh
```

## Offline audit

All 5,635 expert frames were audited. The policy queue was reset for each frame,
so these are first-action, in-sample diagnostics, not closed-loop success or
held-out action accuracy. Motion hits require the correct sign and absolute
prediction magnitude at least 0.05; close hits use gripper command >=1.5.

| Metric | Result |
|---|---:|
| X sign-and-magnitude hit | 96.9% (652/673) |
| Y sign-and-magnitude hit | 91.4% (627/686) |
| X neutral false motion | 3.0% |
| Y neutral false motion | 3.8% |
| Close-command hit | 97.3% |
| Hold false-close | 4.4% |
| Descent hit | 98.0% |
| Positive-Z lift hit | 100.0% |
| Neutral Z false motion | 11.3% |

There were 483 close-command frames and 5,152 non-close frames. High offline
lift accuracy did not eliminate online timing or stopping failures.

## Seen-position evaluation: original 100-step benchmark

The unchanged eight-position schedule uses three repeats per position,
five-step replanning, centered home pose, min TCP Z 0.008, and policy-controlled
X/Y/Z/gripper actions. No forced actions or lateral freezes were used.

```text
100-step seen evaluation: 16/24 (66.7%)
Day16 on the same schedule/horizon: 6/24 (25.0%)
```

| Position | Day16, 100 steps | Day17, 100 steps | Day17, 150 steps |
|---|---:|---:|---:|
| (0.32,-0.075) | 1/3 | 0/3 | 3/3 |
| (0.32,+0.075) | 0/3 | 3/3 | 3/3 |
| (0.36,-0.075) | 2/3 | 3/3 | 3/3 |
| (0.36,+0.075) | 0/3 | 3/3 | 3/3 |
| (0.44,-0.075) | 3/3 | 3/3 | 3/3 |
| (0.44,+0.075) | 0/3 | 2/3 | 2/3 |
| (0.48,-0.075) | 0/3 | 1/3 | 3/3 |
| (0.48,+0.075) | 0/3 | 1/3 | 3/3 |
| **Total** | **6/24** | **16/24** | **23/24** |

Day17 recovered positive-Y control to 9/12 at 100 steps. Intermediate-X
positions reached 11/12, versus Day16's 5/12. Endpoint control was still 5/12,
which is below Day15's 9/12 endpoint result at the same 100-step horizon. Thus,
endpoint preservation did not fully restore speed/reliability at the original
horizon.

The eight failures all reached close-command states, but geometry distinguishes
weak/absent upward motion from late physical elevation. In one failure, the
cube was rising at step 100 with Z≈0.0853, below the success threshold. A strong
instantaneous Z command alone is not evidence of a securely lifted cube.

## Extended-horizon seen evaluation

A NEW 24-episode run used the 150-step Long task and the same checkpoint,
positions, and five-step replanning. All action channels stayed policy-controlled.

```text
150-step seen evaluation: 23/24 (95.8%)
Successes by step 100 WITHIN this run: 15/24
Additional successes after step 100: 8/24 (steps 105--143)
```

The endpoint group completed 12/12 and the intermediate-X group 11/12. The one
failure at (0.44,+0.075) stayed near the table with little positive-Z action
through step 150. Delayed lift initiation is therefore important, but not the
only failure mode.

This does NOT replace the original 16/24 result. The Long task changes the
horizon (and environment construction/viewer path), and the runs are not paired
continuations of the same simulator trajectories. Do not describe the gain as
seven particular original failures being rescued. The eight successes after
step 100 within the new run directly show that latency matters.

## Development interpolation

With the checkpoint and weights fixed, the prior development positions were
run at 150 steps and five-step replanning:

```text
X={0.36,0.44}, Y={-0.0375,+0.0375}
Result: 3/12 (25.0%)
Successes by step 100: 2/12
One additional success at step 113
```

| Development position | Result |
|---|---:|
| (0.36,-0.0375) | 1/3 |
| (0.36,+0.0375) | 2/3 |
| (0.44,-0.0375) | 0/3 |
| (0.44,+0.0375) | 0/3 |

These exact combinations are absent from training, but earlier failures informed
experiment design. This is development evaluation, NOT an independent final
generalization test. The Day15 1/12 benchmark used 100 steps, so it must not be
directly equated with the Day17 3/12 total at 150 steps.

Lateral overshoot was evident BEFORE cube contact. For example, the target Y
of -0.0375 was passed as TCP Y reached -0.0695 or -0.0847. Several trajectories
stopped near the training magnitude |Y|=0.075 rather than the nearer cube. One
failed rollout executed a large upward command and raised the TCP, while the
cube remained near the table: this was not a missing-lift-command failure.
These observations suggest weak position-dependent stopping at new Y magnitudes,
but do not establish a particular representation or fixed-duration memorization.

## One-step replanning diagnostic

The same development positions were screened once each with `n_action_steps=1`
and a 150-step horizon. Result:

```text
0/4
```

Overshoot persisted; some trials also developed unstable X motion or wrong-sign
Y actions. One trial went to Y≈-0.111 for a target of -0.0375. Another moved
negative Y although the cube was positive. More frequent replanning was not a
solution in this small screen. It does not rule out every chunking effect, but
`n_action_steps=1` is not adopted. This override did not modify the saved model.

## Conclusions and next experiment

1. Preserving the Day14 endpoint demonstrations while adding intermediate-X
   data produced stronger seen control than Day16: 16/24 at 100 steps.
2. With 150 steps, Day17 completed 23/24 seen rollouts; eight successes required
   more than 100 steps. Reliable completion and fast completion remain distinct.
3. This is not broad 2D generalization: development interpolation remains 3/12
   at 150 steps, with lateral overshoot and occasional wrong stopping/direction.
4. Do not continue Z-weight or replanning sweeps based solely on these lateral
   errors. Keep the current dataset/checkpoint and return to five-step replanning.
5. The proposed next experiment is four smoke demonstrations, then five
   demonstrations per intermediate-Y development combination (20 additional
   episodes). Append them to ALL 60 Day17 episodes, aiming for an 80-episode
   dataset rather than diluting previously learned positions again.
6. Once those four combinations are in training, success there is no longer
   generalization. Select a separate development benchmark before training.
7. `configs/day16_xy_reserved_test_positions.json` remains unrecorded,
   unevaluated, and unused for tuning. It is reserved for a frozen candidate.
8. The proposed data expansion is a hypothesis, not a guaranteed fix. Single
   seed, small repeats, new normalization statistics, increased total updates,
   and video re-encoding limit causal/statistical claims.

## Evidence

```text
results/day17_dataset_dry_run.log
results/day17_dataset_build.log
results/day17_dataset_loading_check.log
results/day17_dataset_summary.json
results/day17_dataset_provenance.json
results/day17_merged_dataset_qa.log
results/day17_loss_recipe.json
results/day17_training_tail.log
results/offline_gripper_audit_day17_xy_preserved_endpoints_xy2_z2_liftz4_closew4_xpu_10500.csv
results/day17_offline_xy_summary.log
results/day17_xy_preserved_endpoints_seen_eval.log
results/day17_seen_150step_diagnostic.log
results/day17_development_interpolation_150step.log
results/day17_development_n1_four_position_diagnostic.log
results/day17_summary.json
```
