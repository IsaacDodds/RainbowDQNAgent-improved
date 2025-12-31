import numpy as np
import random
from collections import deque


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)

    def total_priority(self):
        return self.tree[0]

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left]
                parent = right
        data_index = leaf - self.capacity + 1
        return leaf, self.tree[leaf], data_index


class NStepPrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, n_step=3, gamma=0.99, alpha=0.5, beta=0.4, use_per=True):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.use_per = use_per
        self.alpha = alpha
        self.beta = beta

        # Store images as uint8 (Atari), vectors as float32
        self._obs_dtype = np.uint8 if len(obs_shape) == 3 else np.float32
        self.obs = np.zeros((capacity, *obs_shape), dtype=self._obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=self._obs_dtype)

        self.actions = np.zeros((capacity,), dtype=np.int16)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.discounts = np.zeros((capacity,), dtype=np.float32)

        if self.use_per:
            self.tree = SumTree(capacity)
            self.max_priority = 1.0

        self.n_step = n_step
        self.gamma = gamma
        self.nstep_buffer = deque(maxlen=n_step)

    def __len__(self):
        return self.size

    def get_nstep(self):
        R, next_s, done = self.nstep_buffer[-1][2:]
        discount = self.gamma


        for (_, _, r, ns, d) in reversed(list(self.nstep_buffer)[:-1]):
            R = r + self.gamma * R * (1 - d)
            discount *= self.gamma
            if d:
                next_s = ns
                done = True
                break

        return R, next_s, done, discount

    def _store(self, s, a, Rn, ns, done, g_n):
        idx = self.ptr

        self.obs[idx] = np.asarray(s, dtype=self._obs_dtype)
        self.actions[idx] = a
        self.rewards[idx] = Rn
        self.next_obs[idx] = np.asarray(ns, dtype=self._obs_dtype)
        self.dones[idx] = done
        self.discounts[idx] = g_n

        if self.use_per:
            tree_idx = idx + self.capacity - 1
            self.tree.update(tree_idx, float(self.max_priority))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def store(self, s, a, r, ns, done):
        self.nstep_buffer.append((s, a, r, ns, done))

        if len(self.nstep_buffer) == self.n_step:
            Rn, next_s, dn, g_n = self.get_nstep()
            s0, a0 = self.nstep_buffer[0][:2]
            self._store(s0, a0, Rn, next_s, dn, g_n)

        if done:
            while len(self.nstep_buffer) > 0:
                Rn, next_s, dn, g_n = self.get_nstep()
                s0, a0 = self.nstep_buffer[0][:2]
                self._store(s0, a0, Rn, next_s, dn, g_n)
                self.nstep_buffer.popleft()

    def sample(self, batch_size):
        if not self.use_per:
            idxs = np.random.randint(0, self.size, size=batch_size).astype(np.int32)
            weights = np.ones((batch_size,), dtype=np.float32)
            return (
                self.obs[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_obs[idxs],
                self.dones[idxs],
                self.discounts[idxs],
                idxs,
                weights,
            )

        total = self.tree.total_priority()
        segment = total / batch_size

        idxs = []
        priorities = []

        for i in range(batch_size):
            mass = random.uniform(segment * i, segment * (i + 1))
            _, p, data_idx = self.tree.get_leaf(mass)
            idxs.append(data_idx)
            priorities.append(p)

        idxs = np.asarray(idxs, dtype=np.int32)
        priorities = np.asarray(priorities, dtype=np.float32)

        probs = priorities / max(total, 1e-6)
        probs = np.maximum(probs, 1e-6)

        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()

        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
            self.discounts[idxs],
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        if not self.use_per:
            return

        prios = (np.abs(td_errors) + 1e-6) ** self.alpha
        self.max_priority = max(self.max_priority, float(prios.max(initial=self.max_priority)))

        for idx, p in zip(idxs, prios):
            tree_idx = int(idx) + self.capacity - 1
            self.tree.update(tree_idx, float(p))


