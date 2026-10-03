# ACT for Panda Pick-and-Place

A hands-on robot-learning project built with **LeRobot, ACT, MuJoCo, and Gym-HIL**. The project studies teleoperated demonstrations, gripper-action learning, and spatial generalization for a simulated Franka Panda.

> **Current result:** Day8 Y-active loss weighting with five-step replanning achieved **6/6 policy-only rollouts at two trained Y positions**, versus **0/6** for unweighted ACT at the same replanning setting. Held-out tests succeeded at the midpoint (**5/5**) and negative interior position (**3/3**), but failed at the positive interior position (**0/3**). These small, single-seed tests demonstrate improvement, not reliable general spatial manipulation.

> **Day9 diagnosis:** externally freezing Y near the target and supplying close/lift actions produced **2/2 successful diagnostic lifts** at `y=±0.05`. This localizes the positive-side policy-only failure primarily to lateral stopping/calibration, but the intervention is not a policy-only benchmark.

## Policy demo — fixed-position task

This GIF comes from a real policy-only MuJoCo rollout of the Day5 XPU checkpoint using front and wrist observations. It demonstrates the **fixed-position** task and is not evidence of spatial generalization.

![ACT policy-only Panda pick-and-lift in the fixed-position task](assets/day5_fixed_position_policy_only.gif)

## Results

| Experiment | Policy / evaluation | Result | Interpretation |
|---|---|---:|---|
| Initial fixed-position baseline | CPU ACT, 500 steps; policy-only, 10 episodes | **0/10** | The robot moved, but the policy did not reliably close the gripper. |
| Assisted fixed-position diagnostic | ACT motion + external close event, 5 episodes | **5/5** | Confirmed that supplying the close event was sufficient for the learned approach/lift behavior. |
| Improved fixed-position policy | Day5 XPU ACT, 2,000 steps; policy-only, 5 repeated rollouts | **5/5** | First stable policy-only success on the fixed-position task. The runs used the same deterministic reset/task and are not a broad statistical guarantee. |
| Original Day6 spatial dataset | XPU ACT, 2,000 steps | Seen **0/8**, unseen **1/5** | Randomized positions did not yield reliable closed-loop spatial control. |
| Day6 Dense-XY dataset | XPU ACT, 2,000 steps | Seen **0/8**, unseen **2/5** | Denser XY data improved offline action prediction, but online success remained poor. |
| Day6 Precision-XY dataset | XPU ACT, 3,000 steps | Seen **0/8**, unseen **0/5** | Later found to have arm-hidden camera observations; treat visual-policy result as confounded. |
| Day7 Y-axis, corrected camera | XPU ACT, 2,000 steps | Seen **0/6** | Offline Y-direction signal improved (70.5% sign-and-magnitude hit at threshold 0.05), but predicted motion was too small and the robot descended before lateral alignment. |
| Day8 Y-active loss weighting | Same Day7 v2 data; 2,000 XPU updates; five-step replanning | Seen **6/6**; held-out midpoint **5/5**; interior positions **3/6** | Unweighted ACT with five-step replanning remained **0/6** on seen positions. Nonzero interpolation was asymmetric: negative **3/3**, positive **0/3**. |
| Day9 failure localization | Same Day8 checkpoint; fixed close/lift and optional Y freeze diagnostics | Diagnostic **2/2** with Y freeze + fixed close/lift | Both intermediate positions could be grasped/lifted after external Y stopping correction; this is causal diagnosis, not policy-only success. |

On the Day5 fixed-position data, the offline gripper audit reached **56.1% close-frame hit rate** with **0% hold-frame false-close rate**. Day8 improved active-Y offline sign-and-magnitude hits from **70.5% to 91.5%**, but gripper close hits decreased from **66.7% to 44.2%**. The successful rollouts reinforce that an offline action threshold is not a physical grasp-success criterion. Day7 data-quality caveats and the controlled Day8 results are documented in [`notes/day7.md`](notes/day7.md) and [`notes/day8.md`](notes/day8.md).

## What this project implements

- Teleoperated Panda demonstration collection in Gym-HIL/MuJoCo.
- ACT training from front-camera, wrist-camera, and proprioceptive observations.
- Policy-only and assisted closed-loop rollout evaluation.
- Offline, frame-level action audits for gripper and motion channels.
- Controlled data/training experiments: training duration, longer gripper-close supervision, and denser XY actions.
- Project-local, padding-aware Y-active loss weighting with numerical/gradient tests and loss-recipe provenance in standard ACT checkpoints.
- Explicit evaluator diagnostics for fixed close, fixed lift, and Y-freeze interventions, kept separate from policy-only metrics.
- Intel XPU training/inference validation and CPU/XPU inference comparison.
- Scheduled cube-position recording and separate seen/unseen evaluation schedules.

## Method at a glance

At each control step, ACT receives two RGB views and an 18-dimensional robot state, then predicts a four-dimensional action:

```text
observation = [front image, wrist image, robot state]
action      = [delta_x, delta_y, delta_z, gripper]
```

Training uses `chunk_size=50` and `n_action_steps=10`. Day8's successful evaluation overrides `n_action_steps=5`: predict a chunk, execute its first five actions, then replan. The Day8 intervention weights valid Y-active reconstruction errors by 4, while retaining neutral and other-channel weights of 1 and the upstream KL term. It changes training loss, not inference action magnitude, and does not add a gripper intervention.

## Run the fixed-position experiment locally

Training datasets and model checkpoints are **not included** in this repository. The scripts expect local LeRobot environments and datasets; tested versions and source-revision details are in [`ENVIRONMENT.md`](ENVIRONMENT.md).

### Verify the XPU environment

```bash
conda activate lerobot-xpu
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else '')"
```

### Train the Day5 policy

```bash
bash scripts/day5_act_gripper_longer_xpu.sh
```

This expects a local LeRobot dataset. The training scripts derive `PROJECT_ROOT` from their own location and support these overrides:

- `DATASET_ROOT`: local dataset directory
- `DATASET_REPO_ID`: LeRobot dataset identifier
- `OUTPUT_DIR`: checkpoint/log output directory

For example:

```bash
DATASET_ROOT=/data/my-panda-data \
DATASET_REPO_ID=my-user/my-panda-data \
OUTPUT_DIR="$PWD/outputs/experiment-01" \
bash scripts/day5_act_gripper_longer_xpu.sh
```

Scripts refuse to overwrite existing output directories. The default dataset paths still assume a local `$HOME/robotics/data` layout.

### Evaluate one policy-only episode and optionally save a GIF

```bash
python scripts/evaluate_panda_act.py \
  --device xpu \
  --checkpoint outputs/act_panda_day5_gripper_longer_xpu_v1/checkpoints/002000/pretrained_model \
  --episodes 1 \
  --gripper-mode policy \
  --gif-output assets/my_rollout.gif \
  --gif-camera side-by-side
```

GIF capture requires one episode. It records the actual front/wrist observations from the rollout; it can capture a failure as well as a success, so check the terminal success/reward before presenting the clip as a successful demonstration.

## Day6 spatial and Day7 Y-axis experiments

Datasets and checkpoints are local; schedules, scripts, and summary results are included. The Day6 Gym-HIL aliases require a small patch to the LeRobot source checkout. Tested package versions, source revision, and patch notes are recorded in [`ENVIRONMENT.md`](ENVIRONMENT.md). Experiment details and data-quality caveats are documented in [`notes/day6.md`](notes/day6.md) and [`notes/day7.md`](notes/day7.md).

## Day8 controlled Y-axis improvement

Day8 reuses the corrected Day7 v2 demonstrations: fixed `x=0.48`, ten successful episodes at each of `y=-0.10/+0.10`. It changes only the Y-active loss weighting in training; the evaluation execution-block length is tested separately. The adapter does not edit installed LeRobot source, and saved checkpoints load with the ordinary ACT audit/evaluator.

| Training objective | Evaluation `n_action_steps` | Seen result |
|---|---:|---:|
| Unweighted ACT | 10 | 0/6 |
| Unweighted ACT | 5 | 0/6 |
| Y-active weight 4 | 10 | 0/2 pilot |
| Y-active weight 4 | 5 | 6/6 |

All training uses seed 1000; evaluation repeats essentially deterministic resets at a small set of positions. This is not a multi-seed statistical claim. With the weighted/n5 policy, held-out `y=0.00` succeeded **5/5**, `y=-0.05` succeeded **3/3**, and `y=+0.05` failed **0/3**. A post-failure n1 diagnostic remained **1/2** and is kept separate from the n5 results.

With the tested environment and local v2 data available, check the loss implementation and train using:

```bash
python -m unittest discover -s tests -p 'test_day8*.py' -v
bash scripts/day8_act_y_weighted_xpu.sh
```

Datasets and checkpoints are not bundled. The launcher accepts the dataset/output overrides described above and `Y_ACTIVE_WEIGHT`, `STEPS`, `LOG_FREQ`, and `JOB_NAME`. It preserves a training log alongside the run directory and writes `day8_loss_recipe.json` into each checkpoint. Do not use vanilla `lerobot-train` to resume this custom-loss experiment; resume is intentionally unsupported.

Evaluate the trained positions with the successful setting:

```bash
python scripts/evaluate_panda_act.py \
  --device xpu \
  --checkpoint outputs/act_panda_day8_y_weighted_w4_xpu_v1/checkpoints/002000/pretrained_model \
  --task PandaPickCubeKeyboardRandomLong-v0 \
  --position-schedule configs/day7_y_axis_seen_eval_positions.json \
  --episodes 6 --max-steps 150 --n-action-steps 5 \
  --gripper-mode policy --debug-geometry
```

Full methods and caveats: [`notes/day8.md`](notes/day8.md). Recorded evidence: [`results/day8_summary.json`](results/day8_summary.json), the offline audit CSV, and five rollout logs in `results/`.

Day9 failure-localization methods and logs are documented in [`notes/day9.md`](notes/day9.md), with structured results in [`results/day9_summary.json`](results/day9_summary.json). The diagnostic flags in `scripts/evaluate_panda_act.py` are not enabled by default and must not be used to claim policy-only success.

## Project layout

```text
assets/     Policy rollout GIF used in this README
configs/    Training, recording, and seen/unseen position schedules
notes/      Day-by-day experiment logs and XPU migration record
patches/    Local LeRobot source patch required by the Day6 Gym-HIL integration
ENVIRONMENT.md  Tested CPU/XPU package versions and patch provenance
results/    Baselines and offline audit summaries/CSVs
scripts/    Training, recording, evaluation, and dataset-inspection tools
tests/      CPU correctness tests for the custom weighted ACT training loss
```

## Limitations and next experiment

- The Day5 `5/5` result is for a **fixed cube position** in a deterministic simulator; it is not a broad manipulation success-rate claim.
- Day8 solves two trained Y positions under the tested n5 control setting, but held-out nonzero interpolation remains asymmetric (`-0.05: 3/3`, `+0.05: 0/3`). X, object appearance, home configuration, and the simulator are unchanged; this is not broad spatial generalization.
- All reported Day8 training comparisons use one seed. Repeated rollouts at identical positions primarily demonstrate repeatability, not coverage of a wide test distribution.
- Offline action prediction is diagnostic and does not guarantee closed-loop task success.
- The next investigation should compare successful expert states and failing positive-side states around stopping, closing, and lifting. Positions used for tuning must not continue to be described as untouched held-out tests.
- Day9 diagnostics indicate that adding intermediate-distance demonstrations is the next controlled training experiment; once `y=±0.05` are used for training, new held-out positions such as `y=±0.075` are required for a fresh generalization claim.

## Hardware

The XPU experiments were run with PyTorch `2.10.0+xpu` on an Intel Arc B390. XPU training, checkpoint loading, offline inference, and MuJoCo online inference were validated. See [`notes/xpu_migration.md`](notes/xpu_migration.md) for timing and the non-blocking PyAV video-decoder fallback note.

## License

No license is currently specified. Contact the author before reusing or redistributing this repository's code.
