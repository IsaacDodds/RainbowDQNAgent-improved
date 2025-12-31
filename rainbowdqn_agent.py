import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import ale_py
try:
    gym.register_envs(ale_py)
except Exception:
    pass

from distributed import C51Distribution
from network import C51NoisyDuelingCNN, C51NoisyDuelingMLP
from replay_buffer import NStepPrioritizedReplayBuffer
from frame_replay_buffer import AtariFramePERNStepReplayBuffer


def _framestack(env, k=4):
    try:
        return FrameStackObservation(env, stack_size=k)
    except TypeError:
        return FrameStackObservation(env, num_stack=k)


class RainbowDQNAgent:
    def __init__(self, config):
        # Core
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.tau = config.get("tau", 0.005)
        self.lr = config["lr"]
        self.env_id = config["env_name"]

        # Rainbow toggles (for ablations)
        self.use_double = config.get("use_double", True)
        self.use_dueling = config.get("use_dueling", True)
        self.use_noisy = config.get("use_noisy", True)
        self.use_per = config.get("use_per", True)
        self.use_nstep = config.get("use_nstep", True)

        # Schedules
        self.learn_start = config.get("learn_start", 80_000)
        self.replay_frequency = config.get("replay_frequency", 4)

        self.beta_start = config.get("beta_start", 0.4)
        self.beta_frames = config.get("beta_frames", 5_000_000)

        # PER / replay / n-step
        self.per_alpha = config.get("per_alpha", 0.5)
        self.n_step = config.get("n_step", 3)
        if not self.use_nstep:
            self.n_step = 1

        # IMPORTANT: stacked obs + next_obs is RAM heavy; keep modest unless you implement frame-only replay
        self.memory_capacity = config.get("memory_capacity", 50_000)

        # Target updates in gradient steps
        self.target_update_type = config.get("target_update_type", "hard")
        self.target_update_freq = config.get("target_update_freq", 32_000)
        self.learn_steps = 0

        # Epsilon-greedy only used when Noisy is OFF
        self.eps_start = config.get("eps_start", 1.0)
        self.eps_end = config.get("eps_end", 0.01)
        self.eps_decay_frames = config.get("eps_decay_frames", 1_000_000)

        # Optimiser eps
        self.adam_eps = config.get("adam_eps", 1.5e-4)

        # Env
        self.is_atari = self.env_id.startswith("ALE/")
        if self.is_atari:
            base_env = gym.make(
                self.env_id,
                frameskip=1,
                repeat_action_probability=0.0,
                full_action_space=False,
            )
            env = AtariPreprocessing(
                base_env,
                noop_max=30,
                frame_skip=4,
                screen_size=84,
                grayscale_obs=True,
                scale_obs=False,             # uint8
                terminal_on_life_loss=True,  # training
            )
            env = _framestack(env, k=4)
        else:
            env = gym.make(self.env_id)

        self.env = env
        self.obs_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # C51
        v_min = config.get("v_min", -100.0)
        v_max = config.get("v_max", 100.0)
        atom_size = config.get("atom_size", 51)

        self.c51 = C51Distribution(v_min=v_min, v_max=v_max, atom_size=atom_size, device=self.device)
        self.atom_size = atom_size

        # Networks
        if self.is_atari:
            self.policy_net = C51NoisyDuelingCNN(
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support,
                use_noisy=self.use_noisy,
                use_dueling=self.use_dueling,
            ).to(self.device)

            self.target_net = C51NoisyDuelingCNN(
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support,
                use_noisy=self.use_noisy,
                use_dueling=self.use_dueling,
            ).to(self.device)
        else:
            obs_dim = self.obs_shape[0]
            self.policy_net = C51NoisyDuelingMLP(
                obs_dim=obs_dim,
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support,
                use_noisy=self.use_noisy,
                use_dueling=self.use_dueling,
            ).to(self.device)

            self.target_net = C51NoisyDuelingMLP(
                obs_dim=obs_dim,
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support,
                use_noisy=self.use_noisy,
                use_dueling=self.use_dueling,
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiser
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, eps=self.adam_eps)

        # Replay
        # Replay
        # Default: use frame-only replay on Atari (lets you use 1,000,000 capacity without insane RAM).
        self.use_frame_replay = bool(config.get("use_frame_replay", self.is_atari))

        if self.is_atari and self.use_frame_replay:
            history = int(self.obs_shape[0])          # should be 4 for (4,84,84)
            frame_shape = tuple(self.obs_shape[1:])   # (84,84)

            self.memory = AtariFramePERNStepReplayBuffer(
                capacity=self.memory_capacity,
                frame_shape=frame_shape,
                history=history,
                n_step=self.n_step,
                gamma=self.gamma,
                alpha=self.per_alpha,
                beta=self.beta_start,
                use_per=self.use_per,
            )
        else:
            self.memory = NStepPrioritizedReplayBuffer(
                capacity=self.memory_capacity,
                obs_shape=self.obs_shape,
                n_step=self.n_step,
                gamma=self.gamma,
                alpha=self.per_alpha,
                beta=self.beta_start,
                use_per=self.use_per,
            )


        self.reward_history = []
        self.run_dir = None
        self.run_name = None
        self.frame = 0

    def _to_torch_state(self, s: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(s, device=self.device).unsqueeze(0)
        if self.is_atari:
            return t.float() / 255.0
        return t.float()

    def _eps(self):
        # linear anneal
        p = min(1.0, self.frame / float(self.eps_decay_frames))
        return self.eps_start + p * (self.eps_end - self.eps_start)

    def select_action(self, state):
        # If noisy: greedy with parameter noise
        if self.use_noisy:
            self.policy_net.reset_noise()
            with torch.no_grad():
                state_t = self._to_torch_state(state)
                dist = self.policy_net(state_t)
                q = self.c51.expectation(dist)
                return q.argmax(1).item()

        # If not noisy: epsilon-greedy
        eps = self._eps()
        if np.random.rand() < eps:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            state_t = self._to_torch_state(state)
            dist = self.policy_net(state_t)
            q = self.c51.expectation(dist)
            return q.argmax(1).item()

    def train(
        self,
        max_frames=5_000_000,
        save_interval=30,
        live_plot_enabled=False,   # kept for compatibility
        verbose=False,
        run_name="default",
        print_every_episodes=20,
        progress_every_episodes=20,
        progress_fn=None,
    ):
        self.run_name = run_name

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join("runs", f"run-{run_name}-{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        last_save_time = time.time()
        save_interval = save_interval * 60

        state, _ = self.env.reset()
        state = np.asarray(state, dtype=np.uint8) if self.is_atari else np.asarray(state, dtype=np.float32)

        episode_reward = 0.0
        frame = 0

        while frame < max_frames:
            self.frame = frame

            action = self.select_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)

            episode_reward += float(reward)
            clipped_reward = float(np.clip(reward, -1.0, 1.0))

            next_state = np.asarray(next_state, dtype=np.uint8) if self.is_atari else np.asarray(next_state, dtype=np.float32)

            frame += 1

            # PER beta anneal
            p = min(1.0, frame / float(self.beta_frames))
            self.memory.beta = self.beta_start + p * (1.0 - self.beta_start)

            self.memory.store(state, action, clipped_reward, next_state, done)

            # Learn scheduling
            if frame >= self.learn_start and len(self.memory) >= self.batch_size:
                if frame % self.replay_frequency == 0:
                    self.train_step()
                    self.learn_steps += 1

                    if self.target_update_type == "soft":
                        self.soft_update()
                    else:  # hard
                        if self.learn_steps % self.target_update_freq == 0:
                            self.hard_update()

            if done:
                self.reward_history.append(episode_reward)
                ep = len(self.reward_history)

                # ---- print every N episodes ----
                if verbose and (print_every_episodes is not None) and (print_every_episodes > 0):
                    if (ep % int(print_every_episodes)) == 0:
                        ts = datetime.now().strftime("%H:%M:%S")
                        avg100 = float(np.mean(self.reward_history[-100:])) if ep > 0 else 0.0
                        if self.use_noisy:
                            extra = ""
                        else:
                            extra = f" eps={self._eps():.3f}"
                        print(f"[{ts}] ep={ep:5d} reward={episode_reward:8.1f} avg100={avg100:8.1f} frame={frame}{extra}")

                # ---- progress callback every N episodes (for live plotting on Hex) ----
                if (progress_fn is not None) and (progress_every_episodes is not None) and (progress_every_episodes > 0):
                    if (ep % int(progress_every_episodes)) == 0:
                        progress_fn(run_name=run_name, ep=ep, frame=frame, rewards=list(self.reward_history))

                episode_reward = 0.0
                state, _ = self.env.reset()
                state = np.asarray(state, dtype=np.uint8) if self.is_atari else np.asarray(state, dtype=np.float32)
            else:
                state = next_state

            # Auto-save
            if time.time() - last_save_time >= save_interval:
                tstamp = int(time.time())
                self.save_model(os.path.join(self.run_dir, f"model_{tstamp}.pth"))
                np.save(
                    os.path.join(self.run_dir, f"rewards_{tstamp}.npy"),
                    np.asarray(self.reward_history, dtype=np.float32),
                )
                last_save_time = time.time()

    def train_step(self):
        (states, actions, rewards, next_states, dones, gammas, idxs, weights) = self.memory.sample(self.batch_size)

        states = torch.as_tensor(states, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)

        if self.is_atari:
            states = states.float() / 255.0
            next_states = next_states.float() / 255.0
        else:
            states = states.float()
            next_states = next_states.float()

        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)
        gammas = torch.as_tensor(gammas, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        if self.use_noisy:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        # Current dist
        dist = self.policy_net(states)                        # [B, A, Z]
        dist_a = dist[torch.arange(states.size(0)), actions]  # [B, Z]

        with torch.no_grad():
            next_dist_target = self.target_net(next_states)   # [B, A, Z]

            if self.use_double:
                next_dist_policy = self.policy_net(next_states)
                next_actions = self.c51.expectation(next_dist_policy).argmax(dim=1)
            else:
                next_actions = self.c51.expectation(next_dist_target).argmax(dim=1)

            next_dist_a = next_dist_target[torch.arange(states.size(0)), next_actions]  # [B, Z]
            target_dist = self.c51.project(next_dist_a, rewards, dones, gammas)

        # Cross-entropy loss
        loss_elem = -(target_dist * (dist_a + 1e-8).log()).sum(dim=1)
        loss = (loss_elem * weights.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update_priorities(idxs, loss_elem.detach().cpu().numpy())

    def hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        with torch.no_grad():
            for t_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                t_param.data.mul_(1.0 - self.tau)
                t_param.data.add_(self.tau * param.data)

    def save_model(self, path):
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "frame": self.frame,
            "reward_history": self.reward_history,
        }
        torch.save(checkpoint, path)

