## Local gym-hil keyboard patch

My laptop exposes its only Ctrl key through pynput as `Key.ctrl`,
while gym-hil expects `Key.ctrl_l` and `Key.ctrl_r`.

For local teleoperation I remapped:

- O → open gripper
- C → close gripper

File:
gym_hil/wrappers/intervention_utils.py

---

## Dataset Summary

Task:
Panda pick cube

Episodes:
10

FPS:
10

Total frames:
680

Average episode duration:
6.8s

Observation:
- agent position/state: 18-D proprioceptive state
- front camera: 128×128 RGB
- wrist camera: 128×128 RGB

Action:
- dimension: 4
- interpretation: Δx, Δy, Δz, gripper

Success:
10 / 10

## Dataset Quality

Potential issues:
- Fixed cube position
- Z-axis-dominant behavior

## Key observation

The current environment starts the cube at a fixed location,
so most demonstrations require limited X-Y motion.

This dataset is intended as a pipeline baseline rather than
a generalization benchmark.