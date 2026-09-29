# Tested environment snapshot

This document records the environment used for the reported experiments. It
is a tested snapshot, **not a cross-platform lockfile**. In particular, Intel
XPU PyTorch wheels are backend/platform-specific; install the matching Intel
build rather than assuming a generic `pip install torch` will reproduce it.

## OS and hardware

- OS: Ubuntu 24.04.5 LTS
- Python: 3.12.14
- XPU: Intel Arc B390
- XPU PyTorch reports `torch.xpu.is_available() == True`

## Tested Python packages

| Package | CPU environment (`lerobot`) | XPU environment (`lerobot-xpu`) |
|---|---:|---:|
| PyTorch | `2.11.0+cu130` (CPU device used for baseline) | `2.10.0+xpu` |
| torchvision | `0.26.0` | `0.25.0+xpu` |
| LeRobot | `0.6.2` | `0.6.2` |
| gym-hil | `0.1.14` | `0.1.14` |
| Gymnasium | `1.3.0` | `1.3.0` |
| MuJoCo | `3.8.1` | `3.8.1` |
| NumPy | `2.2.6` | `2.2.6` |
| Pillow | `12.3.0` | `12.3.0` |
| PyAV | `15.1.0` | `15.1.0` |

## LeRobot source revision

The local LeRobot source checkout used for these experiments was based on:

```text
d4d94b8e9ceefae20235a9bb5a1f61fab207ddaa
```

Day6 added a local compatibility patch for random-position Gym-HIL task
aliases, viewer configuration, 100/150-step time limits, and finite scheduled
episode shutdown. The patch is included at
[`patches/day6_gym_manipulator.patch`](patches/day6_gym_manipulator.patch).
It is tied to the source context at the revision above; inspect the checkout
before applying it. Do not reset an existing LeRobot checkout with local work
just to match this revision.

## Video decoding

In the tested XPU environment, the installed TorchCodec wheel could not load
its FFmpeg shared libraries. LeRobot fell back to PyAV (`15.1.0`). Dataset
loading, video inspection, training, and rollout were tested with this
fallback. It generates a startup warning but did not block the experiments.

## Reproducibility scope

- Training scripts support `DATASET_ROOT`, `DATASET_REPO_ID`, and `OUTPUT_DIR`
  overrides; otherwise they use workstation-friendly defaults under
  `$HOME/robotics/data` and this repository's `outputs/` directory.
- VS Code does not commit a machine-specific interpreter path. Select the
  `lerobot-xpu` interpreter manually when running XPU commands.
- Demonstration datasets and trained checkpoints are not committed. The public
  repository contains scripts, schedules, result summaries, and selected
  offline audit data; reproducing model results requires obtaining/collecting
  the corresponding local datasets.
- This table is version evidence for the experiments, not a complete package
  lock. A future clean setup should add a platform-specific lock or explicit
  environment export once the XPU installation recipe is finalized.
