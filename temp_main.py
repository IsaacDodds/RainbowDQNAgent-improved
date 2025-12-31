import gymnasium as gym
from rainbowlight_agent import DQNAgent

def main():
    config = {
        "batch_size": 32,
        "eps_decay": 2000,
        "env_name": "CartPole-v1",
    }

    env = gym.make(config["env_name"])

    agent = DQNAgent(
        env=env,
        memory_size=10_000,
        batch_size=config["batch_size"],
        target_update=100,
        epsilon_decay=1 / config["eps_decay"],
        seed=777,
    )

    agent.train(num_frames=1_000_000)

    
    # env = gym.make(config["env_name"], render_mode='human')
    # agent.play(episodes=5)

    # env = gym.make(config["env_name"], render_mode="human")

    # agent = DQNAgent(
    #     env=env,
    #     memory_size=10_000,
    #     batch_size=config["batch_size"],
    #     target_update=100,
    #     epsilon_decay=1 / config["eps_decay"],
    #     seed=777,
    # )

    # # Load or train before play if needed
    # # agent.train(num_frames=1_000_000)

    # agent.play(episodes=5)

if __name__ == "__main__":
    main()
