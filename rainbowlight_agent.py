import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os, time
from datetime import datetime

from replay_buffer import ReplayBuffer
from network import Network


class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        gamma: float = 0.99,
        lr: float = 1e-3,
    ):
        self.env = env
        self.seed = seed

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma

        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay

        self.solved = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        self.is_test = False
        self.reward_history = []
        self.run_dir = None

    # -------------------------------------------------
    # Action selection
    # -------------------------------------------------
    def select_action(self, state):
        if not self.is_test and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            q = self.dqn(state_t)

        return q.argmax().item()

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    def train(
        self,
        num_frames: int,
        run_name="default",
        save_interval_min=30,
        verbose=True,
    ):
        self.is_test = False

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join("runs", f"run-{run_name}-{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        if verbose:
            print(f"[RUN] Saving to {self.run_dir}")

        last_save_time = time.time()
        save_interval = save_interval_min * 60

        state, _ = self.env.reset(seed=self.seed)
        episode_reward = 0
        update_cnt = 0

        for frame in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.memory.store(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(self.memory) >= self.batch_size:
                self.update_model()

                self.epsilon = max(
                    self.min_epsilon,
                    self.epsilon
                    - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                )

                update_cnt += 1
                if update_cnt % self.target_update == 0:
                    self.dqn_target.load_state_dict(self.dqn.state_dict())

            if done:
                self.reward_history.append(episode_reward)

                ep = len(self.reward_history)
                avg_100 = np.mean(self.reward_history[-100:])

                if verbose:
                    print(
                        f"Episode {ep:4d} | "
                        f"Reward {episode_reward:8.2f} | "
                        f"Avg(100) {avg_100:7.2f} | "
                        f"Eps {self.epsilon:6.3f}"
                    )

                # ✅ SOLVED CONDITION
                # ✅ SOLVED CONDITION
                if ep >= 100 and avg_100 >= 200.0:
                    print("✅ Solved LunarLander! Saving final model and stopping training.")

                    self.solved = True

                    ts = int(time.time())
                    self.save_model(os.path.join(self.run_dir, f"model_solved_{ts}.pth"))
                    np.save(
                        os.path.join(self.run_dir, f"rewards_solved_{ts}.npy"),
                        np.array(self.reward_history),
                    )

                    break


                episode_reward = 0
                state, _ = self.env.reset(seed=self.seed)


            # -------------------------
            # Auto-save
            # -------------------------
            if time.time() - last_save_time >= save_interval:
                ts = int(time.time())
                self.save_model(os.path.join(self.run_dir, f"model_{ts}.pth"))
                np.save(
                    os.path.join(self.run_dir, f"rewards_{ts}.npy"),
                    np.array(self.reward_history),
                )

                if verbose:
                    print(f"[AUTO-SAVE] Saved at {ts}")

                last_save_time = time.time()

        self.env.close()

    # -------------------------------------------------
    # Gradient step (Double DQN)
    # -------------------------------------------------
    def update_model(self):
        samples = self.memory.sample_batch()

        state = torch.tensor(samples["obs"], dtype=torch.float32, device=self.device)
        next_state = torch.tensor(samples["next_obs"], dtype=torch.float32, device=self.device)
        action = torch.tensor(samples["acts"], dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(samples["rews"], dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.tensor(samples["done"], dtype=torch.float32, device=self.device).unsqueeze(1)

        curr_q = self.dqn(state).gather(1, action)

        next_actions = self.dqn(next_state).argmax(1, keepdim=True)
        next_q = self.dqn_target(next_state).gather(1, next_actions).detach()

        target = reward + self.gamma * next_q * (1 - done)

        loss = F.smooth_l1_loss(curr_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # -------------------------------------------------
    # Save / Load (Rainbow-compatible)
    # -------------------------------------------------
    def save_model(self, path):
        torch.save(
            {
                "dqn": self.dqn.state_dict(),
                "target": self.dqn_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "reward_history": self.reward_history,
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.dqn.load_state_dict(checkpoint["dqn"])
        self.dqn_target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.min_epsilon)
        self.reward_history = checkpoint.get("reward_history", [])

        print(f"[LOAD] Loaded model from {path}")

    # -------------------------------------------------
    # Play
    # -------------------------------------------------
    def play(self, episodes=5):
        self.is_test = True
        self.epsilon = 0.0
        self.dqn.eval()

        for ep in range(episodes):
            state, _ = self.env.reset(seed=self.seed)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                score += reward

            print(f"[PLAY] Episode {ep+1} | Score {score:.2f}")
