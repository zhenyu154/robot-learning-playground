import gymnasium as gym
import gym_hil
import numpy as np


ENV_NAME = "gym_hil/PandaPickCubeKeyboard-v0"

env = gym.make(
    ENV_NAME,
    render_mode="human",
    image_obs=True,
    max_episode_steps=1000,
)

obs, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print()
print("Practice environment started.")
print("You have up to 1000 steps per episode.")
print("Press Ctrl+C in the terminal to stop.")

# Neutral action.
# Keyboard intervention wrapper will override this while
# human intervention is active.
action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

try:
    while True:
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                f"Episode ended: "
                f"terminated={terminated}, truncated={truncated}, reward={reward}"
            )
            obs, info = env.reset()
            print("Environment reset.")

except KeyboardInterrupt:
    print("\nStopping practice.")

finally:
    env.close()
