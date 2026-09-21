### LeRobot config compatibility issue

The current LeRobot main branch parses `env` directly as
`HILSerlRobotEnvConfig`, so the `"type": "gym_manipulator"` field shown
in the IL simulation documentation caused a config validation error.

Fix: removed `env.type` and used `env.name = "gym_hil"`.