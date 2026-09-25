# Robot Learning Playground

Hands-on experiments in imitation learning and robotic manipulation.

## Current Goal

Train and evaluate an imitation-learning policy for simulated robotic manipulation using LeRobot and MuJoCo.

## Progress

- [x] Set up LeRobot
- [x] Run Panda manipulation environment
- [x] Collect demonstrations
- [x] Train ACT policy
- [x] Evaluate policy
- [x] Run first controlled experiment

Current baseline:
- ACT learns the main XYZ trajectory.
- Gripper closure is not learned reliably.
- Policy-only success: 0/10 on the 500-step checkpoint.
- Assisted gripper success: 5/5.
- Offline gripper close hit rate: 0/51.

## Intel XPU training (Core Ultra X9 388H)

The original CPU training scripts remain available as a fallback and as the historical baseline.
For XPU work, select the `lerobot-xpu` Conda interpreter in VS Code (the workspace default is configured
to `/home/wusanggg/Downloads/ENTER/envs/lerobot-xpu/bin/python`) and verify it first:

```bash
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else '')"
```

Use the separate scripts so XPU runs never overwrite CPU results:

- `scripts/day2_xpu_smoke.sh` — 5-step device/backend check.
- `scripts/day2_act_mini_xpu.sh` — 500-step XPU baseline.
- `scripts/day4_act_longer_xpu.sh` — 2,000-step XPU counterpart to the completed CPU Day 4 run.

These scripts deliberately refuse to reuse an existing output directory. The first XPU training step may
be much slower due to one-time backend initialization/compilation; judge steady-state speed from later steps.
`step_s` is a training update, not a full dataset epoch. With 680 frames and batch size 8, one nominal
full pass is about 85 updates. Measured timing and migration caveats are recorded in
[`notes/xpu_migration.md`](notes/xpu_migration.md).

### XPU validation status (2026-09-25)

- XPU environment and Intel Arc B390 detection: passed.
- 5-step smoke training and 500-step XPU ACT baseline: passed.
- The same XPU checkpoint produced nearly identical CPU/XPU offline audit results.
- XPU-trained policy-only rollout reproduced the CPU baseline failure mode: the gripper did not
  autonomously close.
- XPU assisted rollout succeeded after injecting the fixed-step gripper close command.
- The XPU training, offline inference, and MuJoCo online inference pipeline is validated.

The remaining research problem is the same on CPU and XPU: learning a sufficiently strong autonomous
gripper-close event from the current demonstrations. The next experiment is Day 5 data/objective work,
not another hardware-debugging round.
