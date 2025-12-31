import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(rewards, window=100):
    """
    Plot episode rewards and moving average.

    Args:
        rewards (array-like): episode rewards
        window (int): moving average window
    """
    rewards = np.asarray(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", alpha=0.4)

    if len(rewards) >= window:
        moving_avg = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(
            range(window - 1, len(rewards)),
            moving_avg,
            label=f"Moving Avg ({window})",
            linewidth=2,
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_reward_distribution(
    rewards,
    bins=60,
    last_n=None,
    xlim=None,
    title="Reward Distribution (Histogram)",
    density=True,
):
    """Plots a histogram of episode returns to show how performance is distributed over training."""
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    if rewards.size == 0:
        print("No rewards to plot.")
        return

    if last_n is not None:
        rewards = rewards[-last_n:]

    plt.figure(figsize=(10, 5))
    plt.hist(rewards, bins=bins, density=density, alpha=0.85)

    mean = float(np.mean(rewards))
    plt.axvline(mean, linewidth=2, label=f"Mean = {mean:.1f}")

    plt.title(title)
    plt.xlabel("Episode Return")
    plt.ylabel("Density" if density else "Count")

    if xlim is not None:
        plt.xlim(*xlim)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

