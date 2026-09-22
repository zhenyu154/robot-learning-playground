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
- Gripper closure is not yet learned reliably.
- Policy-only success: 0/5.
- Assisted gripper success: 5/5.