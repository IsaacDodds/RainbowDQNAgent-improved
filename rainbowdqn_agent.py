import os
import time
import random
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


def _framestack(env, k: int = 4):
    try:
        return FrameStackObservation(env, stack_size=k)
    except TypeError:
        return FrameStackObservation(env, num_stack=k)


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RainbowDQNAgent:
    def __init__(self, config: dict):
        # Seed
        self.seed = config.get("seed", None)
        if self.seed is not None:
            self.seed = int(self.seed)
        _seed_everything(self.seed)

        # GPU perf toggles (do not change algorithm)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Core
        self.batch_size = int(config.get("batch_size", 32))
        self.gamma = float(config.get("gamma", 0.99))

        self.lr = float(config.get("lr", 6.25e-5))
        self.adam_eps = float(config.get("adam_eps", 1.5e-4))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 10.0))

        # Rainbow toggles
        self.use_double = bool(config.get("use_double", True))
        self.use_dueling = bool(config.get("use_dueling", True))
        self.use_noisy = bool(config.get("use_noisy", True))
        self.use_per = bool(config.get("use_per", True))
        self.use_nstep = bool(config.get("use_nstep", True))

        # Schedules
        self.learn_start = int(config.get("learn_start", 80_000))
        self.replay_frequency = int(config.get("replay_frequency", 4))

        self.target_update_type = str(config.get("target_update_type", "hard"))
        self.target_update_freq = int(config.get("target_update_freq", 32_000))  # env steps (actions)

        # PER
        self.per_alpha = float(config.get("per_alpha", 0.5))
        self.beta_start = float(config.get("beta_start", 0.4))
        self.beta_frames = int(config.get("beta_frames", 5_000_000))

        # n-step
        self.n_step = int(config.get("n_step", 3))
        if (not self.use_nstep) or self.n_step < 1:
            self.n_step = 1

        # Replay
        self.memory_capacity = int(config.get("memory_capacity", 1_000_000))

        # Reward handling
        self.clip_rewards = bool(config.get("clip_rewards", True))

        # Noisy resample cadence
        self.noise_reset_freq = int(config.get("noise_reset_freq", self.replay_frequency))

        # Env
        self.env_id = str(config.get("env_name", "ALE/Frostbite-v5"))
        self.is_atari = self.env_id.startswith("ALE/")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.is_atari:
            base_env = gym.make(
                self.env_id,
                frameskip=1,
                repeat_action_probability=float(config.get("repeat_action_probability", 0.0)),
                full_action_space=bool(config.get("full_action_space", False)),
            )
            env = AtariPreprocessing(
                base_env,
                noop_max=int(config.get("noop_max", 30)),
                frame_skip=int(config.get("frame_skip", 4)),
                screen_size=int(config.get("screen_size", 84)),
                grayscale_obs=True,
                scale_obs=False,  # uint8
                terminal_on_life_loss=bool(config.get("terminal_on_life_loss", True)),
            )
            env = _framestack(env, k=int(config.get("history_length", 4)))
        else:
            env = gym.make(self.env_id)

        self.env = env
        self.obs_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        # C51 support
        v_min = float(config.get("v_min", -10.0))
        v_max = float(config.get("v_max", 10.0))
        atom_size = int(config.get("atom_size", 51))

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
            obs_dim = int(self.obs_shape[0])
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

        # IMPORTANT: keep target net in train() so NoisyLinear noise works.
        self.target_net.train()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, eps=self.adam_eps)

        # Replay buffer choice
        self.use_frame_replay = bool(config.get("use_frame_replay", self.is_atari))

        if self.is_atari and self.use_frame_replay:
            history = int(self.obs_shape[0])          # (H,84,84) => H
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

        # Logging
        self.reward_history: list[float] = []
        self.run_dir: str | None = None
        self.frame: int = 0

    def _to_torch_state(self, s: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(s, device=self.device).unsqueeze(0)
        if self.is_atari:
            return t.float().div_(255.0)
        return t.float()

    def select_action(self, state) -> int:
        # NoisyNet exploration: greedy action under noisy weights
        with torch.inference_mode():
            s = self._to_torch_state(state)
            dist = self.policy_net(s)
            q = self.c51.expectation(dist)
            return int(q.argmax(dim=1).item())

    def train(
        self,
        max_frames: int = 50_000_000,
        save_interval: int = 30,
        live_plot_enabled: bool = False,
        verbose: bool = True,
        run_name: str = "default",
        print_every_episodes: int = 20,
        progress_every_episodes: int = 20,
        progress_fn=None,
    ):
        # Run directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join("runs", f"run-{run_name}-{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        save_interval = float(save_interval) * 60.0
        last_save_time = time.time()

        # Reset
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
        else:
            state, _ = self.env.reset()

        state = np.asarray(state, dtype=np.uint8) if self.is_atari else np.asarray(state, dtype=np.float32)

        episode_reward = 0.0
        frame = 0

        self.policy_net.train()

        while frame < int(max_frames):
            self.frame = frame

            # Resample online noise every replay period (canonical)
            if self.use_noisy and (frame % self.noise_reset_freq == 0):
                self.policy_net.reset_noise()

            action = self.select_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)

            # Log raw score; store clipped reward if enabled
            episode_reward += float(reward)
            stored_reward = float(np.clip(reward, -1.0, 1.0)) if self.clip_rewards else float(reward)

            next_state = np.asarray(next_state, dtype=np.uint8) if self.is_atari else np.asarray(next_state, dtype=np.float32)
            frame += 1

            # Anneal PER beta
            p = min(1.0, frame / float(self.beta_frames))
            self.memory.beta = self.beta_start + p * (1.0 - self.beta_start)

            self.memory.store(state, action, stored_reward, next_state, done)

            # Learn
            if frame >= self.learn_start and len(self.memory) >= self.batch_size:
                if frame % self.replay_frequency == 0:
                    self.train_step()

                # Hard target updates in env steps (canonical)
                if self.target_update_type == "hard":
                    if frame % self.target_update_freq == 0:
                        self.hard_update()

            if done:
                self.reward_history.append(episode_reward)
                ep = len(self.reward_history)

                if verbose and (print_every_episodes > 0) and (ep % int(print_every_episodes) == 0):
                    ts = datetime.now().strftime("%H:%M:%S")
                    avg100 = float(np.mean(self.reward_history[-100:])) if ep > 0 else 0.0
                    print(f"[{ts}] ep={ep:5d} reward={episode_reward:8.1f} avg100={avg100:8.1f} frame={frame}")

                if (progress_fn is not None) and (progress_every_episodes > 0) and (ep % int(progress_every_episodes) == 0):
                    progress_fn(run_name=run_name, ep=ep, frame=frame, rewards=list(self.reward_history))

                episode_reward = 0.0
                state, _ = self.env.reset()
                state = np.asarray(state, dtype=np.uint8) if self.is_atari else np.asarray(state, dtype=np.float32)
            else:
                state = next_state

            # Periodic checkpoints
            if time.time() - last_save_time >= save_interval:
                tstamp = int(time.time())
                self.save_model(os.path.join(self.run_dir, f"model_{tstamp}.pth"))
                np.save(os.path.join(self.run_dir, f"rewards_{tstamp}.npy"), np.asarray(self.reward_history, dtype=np.float32))
                last_save_time = time.time()

        # Final save
        self.save_model(os.path.join(self.run_dir, "model_final.pth"))
        np.save(os.path.join(self.run_dir, "rewards_final.npy"), np.asarray(self.reward_history, dtype=np.float32))

    def train_step(self):
        states, actions, returns, next_states, dones, gammas, idxs, weights = self.memory.sample(self.batch_size)

        states = torch.as_tensor(states, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)

        if self.is_atari:
            states = states.float().div_(255.0)
            next_states = next_states.float().div_(255.0)
        else:
            states = states.float()
            next_states = next_states.float()

        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)
        gammas = torch.as_tensor(gammas, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        batch_idx = torch.arange(states.size(0), device=self.device)

        # Online log-probs
        log_ps = self.policy_net(states, log=True)
        log_ps_a = log_ps[batch_idx, actions]

        with torch.no_grad():
            # Sample target noise (canonical)
            if self.use_noisy:
                self.target_net.reset_noise()

            pns = self.target_net(next_states)

            if self.use_double:
                # Select next action with online net
                pns_online = self.policy_net(next_states)
                next_actions = self.c51.expectation(pns_online).argmax(dim=1)
            else:
                next_actions = self.c51.expectation(pns).argmax(dim=1)

            pns_a = pns[batch_idx, next_actions]
            target_dist = self.c51.project(pns_a, returns, dones, gammas)

        # Cross-entropy
        loss_per_sample = -(target_dist * log_ps_a).sum(dim=1)
        loss = (loss_per_sample * weights.squeeze(1)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.memory.update_priorities(idxs, loss_per_sample.detach().cpu().numpy())

    def hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.train()
        for p in self.target_net.parameters():
            p.requires_grad = False

    def save_model(self, path: str):
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "frame": self.frame,
            "reward_history": self.reward_history,
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.frame = int(checkpoint.get("frame", 0))
        self.reward_history = list(checkpoint.get("reward_history", []))

        self.target_net.train()
        for p in self.target_net.parameters():
            p.requires_grad = False

        print(f"[LOAD] Loaded model from {path}")

