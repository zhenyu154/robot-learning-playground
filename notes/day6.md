# Day 6: Spatial Generalization

## Goal

Test whether ACT can adapt its X/Y motion to cube positions that differ from
the fixed position used in Day1 and Day5, while preserving the improved Day5
gripper supervision.

## Research question

Does the policy learn a visual/spatial controller, or does it memorize a fixed
approach trajectory?

## Keep fixed

- Day5 longer-close demonstration protocol
- Panda task and cube geometry
- 10 FPS
- ACT action representation
- `chunk_size=50`
- `n_action_steps=10`
- XPU backend
- fixed-position Day5 policy as the historical baseline

## Change

- cube X/Y position
- training demonstrations include multiple positions

## Evaluation design

Separate positions into:

- seen positions: positions represented in training demonstrations
- unseen positions: held-out positions not represented in training

Report them separately. A single aggregate success rate can hide poor spatial
generalization.

## Planned stages

1. Validate a random/scheduled-position environment with a short smoke test.
2. Record a small train dataset using multiple discrete cube positions.
3. QA the dataset and preserve position metadata.
4. Train an ACT policy with the Day5 gripper protocol.
5. Evaluate policy-only on seen and unseen positions separately.
6. Compare X/Y actions and failure modes against the fixed-position baseline.

## Important engineering note

The underlying Gym-HIL `PandaPickCubeGymEnv` supports
`random_block_position=True`, but the currently registered
`PandaPickCubeKeyboard-v0` task does not expose that option. Day6 therefore
starts with an environment integration/smoke-test step rather than immediately
recording demonstrations.


## Initial LeRobot integration smoke test (2026-09-26)

The project-local task alias successfully reached the underlying random-position
Panda environment. Five resets produced five unique positions, all within the
configured sampling bounds. The five sampled Y coordinates happened to be
positive, so this verifies task wiring but does not yet verify coverage on both
sides of the Y axis. Use the reproducible 20-reset smoke test before recording.

## LeRobot task-alias smoke test (2026-09-26)

Twenty resets through LeRobot's `make_robot_env` task alias produced 20 unique
positions:

- X range: `0.3061` to `0.4953`
- Y range: `-0.1359` to `0.1463`
- Y<0: `10` resets
- Y>0: `10` resets

The random-position wiring and workspace coverage passed. After printing the
summary, process teardown emitted a GLFW TLS assertion and aborted. This is a
viewer-cleanup issue occurring after sampling, not a sampling/position error;
a short recording smoke should confirm that the regular recording lifecycle
finalizes cleanly before the full dataset is collected.

## Next integration step

A scheduled recorder now supports explicit train-position manifests and saves a
copy of the schedule plus `position_history.json` beside the dataset. Before
recording all 20 training episodes, run the one-episode schedule smoke using
`configs/day6_schedule_smoke.json`. This validates that the scheduled position,
keyboard viewer, dataset writer, and position metadata agree.

## Day6 training-data collection QA (2026-09-27)

The scheduled 20-episode training dataset was collected successfully.

- Dataset root:
  `/home/wusanggg/robotics/data/panda_pick_cube_day6_spatial_v1`
- Episodes: `20`
- Total frames: `1568`
- FPS: `10`
- Reward-positive episodes: `20/20`
- Done-positive episodes: `20/20`
- Hold actions (`gripper=1.0`): `1309`
- Close actions (`gripper=2.0`): `259`
- Close fraction: approximately `16.5%`
- Per-episode close runs: `9--16` frames
- Front and wrist videos: present
- `position_schedule.json`: present
- `position_history.json`: present

The recorded position history matches the four-position training schedule:

- episodes 0--4: `(0.34, -0.10)`
- episodes 5--9: `(0.46, -0.10)`
- episodes 10--14: `(0.34, 0.10)`
- episodes 15--19: `(0.46, 0.10)`

All 20 demonstrations were successful. The dataset is ready for visual
inspection and Day6 training after the metadata/video QA is complete.

## Visual QA (2026-09-27)

Representative front and wrist frames from episodes 0, 5, 10, and 15 were
exported to `results/day6_visual_check/` and visually inspected. The images
show the expected four cube positions, valid approach/alignment, sustained
close phases, and successful lifted end states.

The inspection script emitted the known TorchCodec load warning and then used
PyAV successfully. All four selected episodes reported one success frame and
the expected close-frame counts. No dataset or image-decoding error was found.

## Day6 spatial training result (2026-09-27)

The 2,000-step XPU ACT training completed successfully on the 20-episode
four-position dataset.

- Checkpoint:
  `outputs/act_panda_day6_spatial_xpu_v1/checkpoints/002000/pretrained_model`
- Training steps: `2000`
- Runtime: approximately `4 minutes 54 seconds`
- Final total loss: `1.170`
- Final L1 loss: `0.383`
- Final KL loss: `0.079`
- Final reported epoch: `10.20`
- Checkpoint save: passed

The next required check is an offline audit on the Day6 dataset, followed by
seen-position and held-out-position online evaluation.

## Day6 offline gripper audit (2026-09-27)

The 2,000-step XPU checkpoint was audited on all 1,568 Day6 training frames:

- Target hold frames: `1309`
- Target close frames: `259`
- Overall predicted gripper mean: `1.2380`
- Mean on target close frames: `1.8217`
- Close hit rate at threshold `1.5`: `81.1%` (`210/259` frames)
- Hold false-close rate at threshold `1.5`: `11.1%` (`145/1309` frames)

Compared with Day5 (`56.1%` close hit rate and `0%` hold false-close rate),
Day6 learned a stronger close signal but introduced apparent hold/close
confusion. Frame timing analysis shows the false-close frames are not random:
for every episode, most false-close runs occur immediately before the labeled
close interval, and in the first five episodes some continue immediately after
it. This suggests temporal boundary mismatch/anticipation and persistence,
not necessarily arbitrary gripper hallucination.

The raw frame-wise threshold metrics therefore need to be complemented by
online evaluation and interval-level timing analysis. The next experiment is
seen-position policy-only evaluation, followed by held-out-position evaluation.

## Dense-XY follow-up dataset QA (2026-09-28)

A second 20-episode collection used separate keyboard step sizes:

- XY input step: `0.25` -> `6.25 mm` per 10 FPS tick
- Z input step: `0.50` -> `12.5 mm` per tick

The dataset was saved at:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day6_dense_xy_v1
```

QA results:

- Total frames: `1607`
- Episodes: `20`
- Reward-positive episodes: `20/20`
- Done-positive episodes: `20/20`
- Nonzero X actions: `282` (about `17.5%` of frames)
- Nonzero Y actions: `306` (about `19.0%` of frames)
- Hold actions: `1331`
- Close actions: `276` (about `17.2%` of frames)
- Front and wrist videos: present
- Position schedule/history/protocol: present

Every episode succeeded. The X/Y supervision is substantially denser than
in the original Day6 dataset. The X action is mostly negative because the
robot's home X position is to the positive side of the scheduled cube X
positions; the closer `x=0.46` positions consequently require fewer X pulses
than `x=0.34`. The Y actions include both `-0.25` and `+0.25`, matching the
two Y sides in the schedule.

The previous draft of this note incorrectly described an unsuccessful episode
11 and an exclusion list. The current dataset QA shows `20/20` successful
episodes, so no episode should be excluded from the next training run.

Status: Dense-XY dataset QA passed; ready for a new 2,000-step XPU training
run.

## Dense-XY training and offline audit (2026-09-28)

The Dense-XY 20-episode dataset was trained for 2,000 XPU steps:

- Checkpoint:
  `outputs/act_panda_day6_dense_xy_xpu_v1/checkpoints/002000/pretrained_model`
- Runtime: approximately `4 minutes 51 seconds`
- Final total loss: `1.098`
- Final L1 loss: `0.243`
- Final KL loss: `0.086`
- Checkpoint save: passed

Offline gripper audit on the Dense-XY dataset:

- Hold frames: `1331`
- Close frames: `276`
- Close-frame prediction mean: `1.8273`
- Close hit rate at threshold `1.5`: `80.8%`
- Hold false-close rate: `3.7%`

Compared with the original Day6 spatial checkpoint (`81.1%` close hit,
`11.1%` hold false-close), Dense-XY preserves strong close prediction while
substantially reducing hold/close confusion. The next test is policy-only
online evaluation on seen and unseen position schedules.

## Precision-XY follow-up data collection (2026-09-28)

A new precision-alignment dataset was collected successfully using the
150-step horizon, oblique/top-down human viewer, and separate keyboard step
sizes (`XY=0.25`, `Z=0.50`). At the time, the arm visuals were hidden under the
assumption that this affected only the human viewer. A later review of the
saved dataset videos showed that the arm was also absent from recorded camera
observations. Therefore the Precision-XY visual-policy results are confounded
by a train/evaluation image mismatch; keep the raw dataset for provenance, but
do not treat it as a clean visual-control baseline.

## Precision-XY 2,000-step result (2026-09-28)

The precision-alignment checkpoint was audited on `2373` frames:

- Hold frames: `2084`
- Close frames: `289`
- Close hit rate: `67.8%`
- Hold false-close rate: `0.7%`
- Offline predicted X/Y magnitudes stayed below `0.05`
- Policy-only seen/unseen rollouts all failed in the reported runs

The online logs show the model outputting near-zero X/Y and gripper values
near `1.0` for the entire episode. This is a model underfitting/collapse
signal, not evidence that the precision-alignment data is useless.

The precision dataset has `2373` frames, so with batch size 8, 2,000 updates
are only about `6.7` nominal epochs. Dense-XY had `1607` frames, so 2,000
updates were about `10` epochs. The next controlled experiment keeps the
precision dataset fixed and increases training to `3000` updates, approximately
matching the Dense-XY epoch exposure.


## Precision-XY 3,000-step control result (2026-09-28)

The larger Precision-XY dataset was trained for approximately 10.1 nominal
epochs using 3,000 XPU updates:

- Checkpoint: `outputs/act_panda_day6_precision_xy_longer_xpu_v1/checkpoints/003000/pretrained_model`
- Runtime: approximately `7 minutes 32 seconds`
- Final total loss: `0.757`
- Final L1 loss: `0.306`
- Final KL loss: `0.045`
- Checkpoint save: passed

Offline gripper audit:

- Close-frame mean: `1.8439`
- Close hit rate: `76.1%`
- Hold false-close rate: `0.2%`

Compared with the 2,000-step Precision-XY checkpoint, the extra training
substantially lowered the loss and improved close prediction while keeping hold
false-close very low. The next required test is online seen/unseen evaluation
using the 3,000-step checkpoint.

## Precision-XY 3,000-step online evaluation (2026-09-28)

The 3,000-step Precision-XY checkpoint was evaluated with the 150-step
horizon on the scheduled seen and unseen positions. The reported policy-only
runs all failed:

- Seen positions: `0/8`
- Unseen positions: `0/5`
- Episodes ended by the `150`-step safety limit with reward `0.0`.

The online actions remained near hold for the gripper (`max_gripper` about
`1.04` in the reported runs) and did not produce a successful grasp/lift.
This is consistent with the offline finding that predicted Y actions remained
near zero even after 3,000 updates. More training improved loss and offline
close prediction, but did not solve the online spatial controller.

## Day6 stopping point

Day6 establishes a useful negative result: denser XY supervision improved the
offline X/Y signal relative to the original spatial dataset, but the
Precision-XY variant did not produce reliable closed-loop success. The main
remaining bottleneck is Y-axis spatial control and the offline-to-online state
distribution gap, not gripper learning or XPU execution.

Next session: isolate the Y-axis task with positions sharing one X coordinate
and opposite Y coordinates, then inspect whether the policy can learn the sign
of `delta_y` before attempting another full spatial-generalization run.

## Final Day6 summary (2026-09-28)

### Completed work

- Added reproducible scheduled-position environments and recording support.
- Added safe 100/150-step horizons for custom Gym-HIL task aliases.
- Added seen/unseen position schedules and online evaluation support.
- Added geometry debugging and evaluation-only `n_action_steps` override.
- Validated the random-position environment and recorded 20 successful spatial demonstrations.
- Collected and trained the Dense-XY follow-up dataset.
- Collected and trained the Precision-XY follow-up dataset.

### Final spatial results

| Model | Steps | Seen policy-only | Unseen policy-only | Offline observation |
|---|---:|---:|---:|---|
| Original spatial | 2000 | `0/8` | `1/5` | X/Y sparse; close hit `81.1%`, false close `11.1%` |
| Dense-XY | 2000 | `0/8` | `2/5` | X/Y offline signal improved; close hit `80.8%`, false close `3.7%` |
| Precision-XY | 2000 | `0/8` | `0/5` | Training exposure too low; X/Y collapse online |
| Precision-XY | 3000 | `0/8` | `0/5` | Close hit `76.1%`, false close `0.2%`; Y prediction remained near zero |

The successful unseen cases occurred at positions near the original fixed
centerline and do not establish spatial generalization. The strongest current
conclusion is that gripper learning is solved for the fixed-position task,
while reliable Y-axis closed-loop spatial control remains unsolved.

### Next session

Run a Y-axis isolation experiment with a fixed X coordinate and opposite Y
positions. First verify that demonstrations contain consistent positive and
negative `delta_y` labels, then train and audit a small policy before returning
to the full spatial-generalization benchmark.

### Reproducibility note

Day6 uses a local LeRobot source patch for the custom Gym-HIL task aliases,
viewer setup, explicit episode horizon, and final-episode reset guard. The
patch is stored at:

```text
patches/day6_gym_manipulator.patch
```

Apply it to the local LeRobot checkout before running the Day6 scripts.
