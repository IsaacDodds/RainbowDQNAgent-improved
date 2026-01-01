import argparse

from rainbowdqn_agent import RainbowDQNAgent


def build_config(args) -> dict:
    # Canonical Rainbow-style defaults for Atari.
    return {
        "seed": args.seed,

        # Core training
        "batch_size": 32,
        "gamma": 0.99,

        # Optimiser
        "lr": 6.25e-5,
        "adam_eps": 1.5e-4,
        "grad_clip_norm": 10.0,

        # C51 support (standard for reward-clipped Atari)
        "v_min": -10.0,
        "v_max": 10.0,
        "atom_size": 51,

        # Replay / schedules
        "memory_capacity": 1_000_000,
        "learn_start": 80_000,
        "replay_frequency": 4,

        # Target update in env steps (actions). Canonical: 32k steps.
        "target_update_type": "hard",
        "target_update_freq": 32_000,

        # PER + n-step
        "use_per": True,
        "per_alpha": 0.5,
        "beta_start": 0.4,
        "beta_frames": 5_000_000,

        "use_nstep": True,
        "n_step": 3,

        # Other Rainbow components
        "use_double": True,
        "use_dueling": True,
        "use_noisy": True,

        # Atari preprocessing / ALE
        "env_name": args.env,
        "repeat_action_probability": args.repeat_action_probability,  # 0.0 matches canonical Rainbow
        "full_action_space": args.full_action_space,                 # False => minimal action set
        "noop_max": 30,
        "frame_skip": 4,
        "screen_size": 84,
        "history_length": 4,
        "terminal_on_life_loss": True,   # episodic life
        "clip_rewards": True,            # store clipped reward; log raw episode score

        # NoisyNet: resample noise once per replay period (canonical)
        "noise_reset_freq": 4,
    }


def main():
    parser = argparse.ArgumentParser("Rainbow DQN (no RS) - Atari Frostbite")
    parser.add_argument("--env", type=str, default="ALE/Frostbite-v5")
    parser.add_argument("--frames", type=int, default=50_000_000, help="Environment steps to train")
    parser.add_argument("--run_name", type=str, default="frostbite_full_rainbow")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--repeat_action_probability", type=float, default=0.0, help="Sticky actions prob (0.0 matches canonical)")
    parser.add_argument("--full_action_space", action="store_true", help="Use full 18-action ALE space (usually worse)")

    args = parser.parse_args()

    cfg = build_config(args)
    agent = RainbowDQNAgent(cfg)

    agent.train(
        max_frames=args.frames,
        run_name=args.run_name,
        save_interval=30,          # minutes
        live_plot_enabled=False,
        verbose=True,
        print_every_episodes=20,
        progress_every_episodes=20,
    )


if __name__ == "__main__":
    main()
