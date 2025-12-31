import numpy as np
import matplotlib.pyplot as plt

# Load rewards
rewards = np.load("runs/run-ALE/Frostbite-v5-20251221-231248/rewards_1766387302.npy")

# Plot rewards
plt.figure(figsize=(12,6))
plt.plot(rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Frostbite Rewards over Episodes")
plt.legend()
plt.grid(True)
plt.show()
