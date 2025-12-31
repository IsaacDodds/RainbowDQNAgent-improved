import numpy as np

# Headless-safe plotting for parkin/Hex
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from rainbowdqn_agent import RainbowDQNAgent


def moving_avg(x, w=100):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w, dtype=np.float32) / w, mode="valid")


def save_plot(results, filename="ablations.png", w=100):
    plt.figure()
    for name, rewards in results.items():
        if rewards is None or len(rewards) == 0:
            continue
        plt.plot(moving_avg(rewards, w=w), label=name)

    plt.title(f"Rainbow vs Ablations ({w}-episode moving average)")
    plt.xlabel("Episode (smoothed index)")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def run_one(name, base_cfg, overrides, max_frames, results, progress_every_episodes=20):
    cfg = dict(base_cfg)
    cfg.update(overrides)

    agent = RainbowDQNAgent(cfg)

    def progress_fn(run_name, ep, frame, rewards):
        # update current run curve + redraw combined plot
        results[run_name] = rewards
        save_plot(results, filename="ablations.png", w=100)
        print(f"[PLOT] updated ablations.png at ep={ep} (run={run_name})")

    agent.train(
        run_name=name,
        max_frames=max_frames,
        save_interval=30,
        verbose=True,
        progress_every_episodes=progress_every_episodes,
        progress_fn=progress_fn,
    )

    rewards = agent.reward_history

    # Final update after run completes
    results[name] = list(rewards)
    save_plot(results, filename="ablations.png", w=100)
    print(f"[PLOT] final save for run={name}")

    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rewards


def main():
    base = {
        "env_name": "ALE/Frostbite-v5",

        "batch_size": 32,
        "gamma": 0.99,

        "lr": 6.25e-5,
        "adam_eps": 1.5e-4,

        # C51 support for reward-clipped Atari
        "v_min": -10.0,
        "v_max": 10.0,
        "atom_size": 51,

        "learn_start": 80_000,
        "replay_frequency": 4,

        # IMPORTANT: your code updates target in GRADIENT steps, not env steps
        # target_update_env_steps = 32_000 -> 32_000/4 = 8_000 gradient updates
        "target_update_type": "hard",
        "target_update_freq": 8_000,

        # Frame-only replay lets you scale capacity properly
        "use_frame_replay": True,
        "memory_capacity": 1_000_000,

        "n_step": 3,
        "per_alpha": 0.5,
        "beta_start": 0.4,
        "beta_frames": 5_000_000,

        "use_double": True,
        "use_dueling": True,
        "use_noisy": True,
        "use_per": True,
        "use_nstep": True,

        # Epsilon schedule (used when use_noisy=False). When use_noisy=True this is ignored.
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_frames": 1_000_000,
    }


    MAX_FRAMES_PER_RUN = 2_000_000
    PLOT_EVERY_EPISODES = 20

    experiments = [
        ("full_rainbow", {}),
        ("no_double",  {"use_double": False}),
        ("no_dueling", {"use_dueling": False}),
        ("no_noisy",   {"use_noisy": False}),            # uses epsilon-greedy
        ("no_per",     {"use_per": False}),
        ("no_nstep",   {"use_nstep": False, "n_step": 1}),
    ]

    # results dict is shared and updated during training
    results = {name: [] for (name, _) in experiments}

    for name, overrides in experiments:
        print(f"\n===== RUN: {name} =====")
        run_one(
            name=name,
            base_cfg=base,
            overrides=overrides,
            max_frames=MAX_FRAMES_PER_RUN,
            results=results,
            progress_every_episodes=PLOT_EVERY_EPISODES,
        )

    print("All runs complete. Final plot saved to ablations.png")


if __name__ == "__main__":
    main()



