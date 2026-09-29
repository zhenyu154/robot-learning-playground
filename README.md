# ACT for Panda Pick-and-Place

A hands-on robot-learning project built with **LeRobot, ACT, MuJoCo, and Gym-HIL**. The project studies teleoperated demonstrations, gripper-action learning, and spatial generalization for a simulated Franka Panda.

> **Current result:** an ACT policy trained with improved gripper supervision completed the fixed-position pick-and-lift task in **5/5 repeated policy-only rollouts**. Spatial generalization remains unsolved; Day6 experiments identify unreliable Y-axis closed-loop control as the current bottleneck.

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
| Day6 Precision-XY dataset | XPU ACT, 3,000 steps | Seen **0/8**, unseen **0/5** | More training did not fix online spatial control; Y-axis prediction remained near zero. |

On the Day5 fixed-position data, the offline gripper audit reached **56.1% close-frame hit rate** with **0% hold-frame false-close rate**. For Day6, the principal remaining issue is Y-axis spatial control and the gap between expert-observation predictions and online closed-loop behavior.

## What this project implements

- Teleoperated Panda demonstration collection in Gym-HIL/MuJoCo.
- ACT training from front-camera, wrist-camera, and proprioceptive observations.
- Policy-only and assisted closed-loop rollout evaluation.
- Offline, frame-level action audits for gripper and motion channels.
- Controlled data/training experiments: training duration, longer gripper-close supervision, and denser XY actions.
- Intel XPU training/inference validation and CPU/XPU inference comparison.
- Scheduled cube-position recording and separate seen/unseen evaluation schedules.

## Method at a glance

At each control step, ACT receives two RGB views and an 18-dimensional robot state, then predicts a four-dimensional action:

```text
observation = [front image, wrist image, robot state]
action      = [delta_x, delta_y, delta_z, gripper]
```

The main experiments use `chunk_size=50` and `n_action_steps=10`. The Day5 fixed-position policy learned a usable sequence of partial gripper-closing actions followed by a lift. Day6 increased cube-position diversity and XY supervision, but successful fixed-position grasping did not transfer reliably to off-center positions.

## Reproduce the fixed-position result

Training datasets and model checkpoints are **not included** in this repository. The scripts expect the local LeRobot environments and datasets described in the experiment notes.

### Verify the XPU environment

```bash
conda activate lerobot-xpu
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else '')"
```

### Train the Day5 policy

```bash
bash scripts/day5_act_gripper_longer_xpu.sh
```

This expects the local dataset `panda_pick_cube_day5_gripper_v1`. The script refuses to overwrite an existing output directory.

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

## Day6 spatial experiments

Day6 datasets and checkpoints are local; schedules, scripts, and summary results are included. The local LeRobot checkout needs the compatibility patch at [`patches/day6_gym_manipulator.patch`](patches/day6_gym_manipulator.patch). If the Day6 task aliases and horizon support are not already present in `~/robotics/lerobot/src/lerobot/rl/gym_manipulator.py`, apply the patch from the LeRobot repository root:

```bash
git -C ~/robotics/lerobot apply --check \
  ~/robotics/robot-learning-playground/patches/day6_gym_manipulator.patch
# Only if the check succeeds:
git -C ~/robotics/lerobot apply \
  ~/robotics/robot-learning-playground/patches/day6_gym_manipulator.patch
```

Day6 recording and evaluation details are documented in [`notes/day6.md`](notes/day6.md).

## Project layout

```text
assets/     Policy rollout GIF used in this README
configs/    Training, recording, and seen/unseen position schedules
notes/      Day-by-day experiment logs and XPU migration record
patches/    Local LeRobot source patch required by the Day6 Gym-HIL integration
results/    Baselines and offline audit summaries/CSVs
scripts/    Training, recording, evaluation, and dataset-inspection tools
```

## Limitations and next experiment

- The Day5 `5/5` result is for a **fixed cube position** in a deterministic simulator; it is not a broad manipulation success-rate claim.
- Day6 training used four positions. The current policy did not reliably solve either seen or held-out spatial tasks.
- Offline action prediction is diagnostic and does not guarantee closed-loop task success.
- The next experiment isolates Y-axis control: hold X fixed, train on opposite Y positions, verify consistent positive/negative `delta_y`, then evaluate online before returning to the full spatial benchmark.

## Hardware

The XPU experiments were run with PyTorch `2.10.0+xpu` on an Intel Arc B390. XPU training, checkpoint loading, offline inference, and MuJoCo online inference were validated. See [`notes/xpu_migration.md`](notes/xpu_migration.md) for timing and the non-blocking PyAV video-decoder fallback note.

## License

No license is currently specified. Contact the author before reusing or redistributing this repository's code.
