# Intel XPU ACT migration and timing check

Date: 2026-09-24

## Environment verification

- GPU: Intel Arc B390, visible through OpenCL and PyTorch XPU.
- Environment: `lerobot-xpu`, Python 3.12, PyTorch `2.10.0+xpu`.
- `torch.xpu.is_available()`: `True`; device count: `1`.
- Float32 XPU matrix multiplication completed and returned finite output.
- ACT XPU smoke: 5 updates, batch size 2; checkpoint and optimizer state were saved at
  `outputs/act_panda_xpu_smoke_v1/checkpoints/000005/`.
- XPU checkpoint offline audit: two sample frames successfully processed on both XPU and CPU after
  adding runtime-device overrides to the audit/evaluation scripts.

## Correct interpretation of timing

A training `step` is one optimizer update; it is not an epoch. At 680 frames and batch size 8,
roughly 85 updates make one full pass through the dataset. Therefore an update taking about 0.207 s
means roughly 17.6 s per nominal epoch, not less than 0.2 s per epoch.

A separate 25-update ACT run with batch size 8 (matching Day 4) measured:

- First update: `step_s=3.148 s`, `update_s=2.896 s`.
- Remaining 24 updates: mean `step_s=0.2068 s` (range 0.201–0.222 s).
- Remaining updates: mean `update_s` about 0.152 s.
- The 25-step run completed and saved its checkpoint under
  `outputs/act_panda_xpu_bench_25_v1/checkpoints/000025/`.

At that warmed steady-state rate, 2,000 updates project to about 6.9 minutes, versus the completed
CPU Day 4 runtime of about 27 minutes 47 seconds. This is an estimate, not a guaranteed wall-clock
speedup: startup/compilation, thermal/power limits, checkpoint I/O, and data loading can change total
runtime. The first-ever XPU shape/kernel initialization was much slower in the initial smoke run
(about 359 s in the first update); do not treat that one-time cost as normal steady-state performance.

## Migration status

- CPU scripts remain unchanged for reproducibility and rollback.
- XPU counterparts are available:
  - `scripts/day2_xpu_smoke.sh`
  - `scripts/day2_act_mini_xpu.sh`
  - `scripts/day4_act_longer_xpu.sh`
- Each XPU training script validates XPU availability and refuses to overwrite its output directory.
- VS Code workspace default interpreter points to `lerobot-xpu`; it can still be switched back to
  `lerobot` when needed.
- Offline audit and simulator evaluation accept `--device auto|cpu|xpu`; `auto` follows the
  checkpoint's configured device when available.

## 500-step XPU validation (2026-09-25)

The formal XPU baseline checkpoint is:

```text
outputs/act_panda_mini_xpu_v1/checkpoints/000500/pretrained_model
```

The full 680-frame audit was run once with XPU inference and once with CPU inference on the same
checkpoint:

| Inference device | Close-frame mean | Close-frame min | Close-frame max | Close hit rate | Hold false-close rate |
|---|---:|---:|---:|---:|---:|
| XPU | 1.000104 | 0.993726 | 1.005615 | 0/51 | 0/629 |
| CPU | 1.000104 | 0.993726 | 1.005615 | 0/51 | 0/629 |

The tiny numerical differences are immaterial. This establishes cross-device inference parity for
this checkpoint. It does not mean that CPU- and XPU-trained checkpoints must have identical weights;
it means the same checkpoint behaves consistently when inferred on either device.

The XPU-trained policy-only rollout reproduced the CPU baseline failure mode: the robot moved through
the approach trajectory but did not autonomously close the gripper. With a fixed-step external close
at step 60 for six steps, the shown assisted rollout succeeded:

- success: `1/1`
- termination step: `67`
- reward: `1.0`
- action range: `z_range=[-0.050, 0.772]`
- injected maximum gripper action: `2.000`

This confirms that XPU training, offline inference, action processing, and MuJoCo online inference
are functioning. The remaining gripper-close failure is shared by the CPU and XPU experiments and is
not a hardware migration issue.

## Known non-blocking warning

The XPU environment cannot load the installed TorchCodec shared library because of FFmpeg/library
compatibility issues, so LeRobot falls back to PyAV for video decoding. Training, audit, and rollout
all completed successfully with this fallback. It may reduce data-loading performance, but it is not
currently a functional blocker. Avoid changing the working XPU dependencies until a dedicated video
backend cleanup experiment is planned.

## Next step

The XPU migration is complete for the current pipeline. Do not run the 2,000-step XPU counterpart
merely to repeat Day 4 unless a hardware timing comparison is specifically desired. The next research
step is Day 5: improve gripper supervision/data or the gripper objective while keeping the hardware
fixed.
