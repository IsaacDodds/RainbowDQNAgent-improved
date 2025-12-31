from rainbowdqn_agent import RainbowDQNAgent
from rewards import plot_reward_distribution


def main():
    # Strong Rainbow-style defaults for Atari (no RS variants)
    config = {
        "batch_size": 32,
        "gamma": 0.99,
        "tau": 0.005,

        # Optimiser
        "lr": 6.25e-5,
        "adam_eps": 1.5e-4,

        # C51 support for reward-clipped Atari.
        # (Common choice: [-10, 10] for clipped rewards; widen only if you see saturation.)
        "v_min": -10.0,
        "v_max": 10.0,
        "atom_size": 51,

        # Schedules
        "learn_start": 80_000,
        "replay_frequency": 4,

        # Target updates (in *gradient* steps).
        # With replay_frequency=4, 32k env-steps corresponds to 8k gradient updates.
        "target_update_type": "hard",
        "target_update_freq": 8_000,

        # Replay / PER / n-step
        # Frame-only replay allows large capacity; 1e6 frames is ~7GB RAM (uint8 84x84).
        "use_frame_replay": True,
        "memory_capacity": 1_000_000,
        "n_step": 3,
        "per_alpha": 0.5,
        "beta_start": 0.4,
        "beta_frames": 5_000_000,

        # Env
        "env_name": "ALE/Frostbite-v5",
    }

    agent = RainbowDQNAgent(config)

    agent.train(
        run_name="ALE-Frostbite-Rainbow",
        max_frames=10_000_000,
        save_interval=30,          # minutes
        live_plot_enabled=False,
        verbose=True,
    )

    plot_reward_distribution(
        agent.reward_history,
        bins=60,
        last_n=500,
    )


if __name__ == "__main__":
    main()

