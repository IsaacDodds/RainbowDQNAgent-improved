import numpy as np
import random


class SumTree:
    """
    Standard sum-tree for proportional prioritized replay.
    Stores priorities in a binary tree where each parent is the sum of its children.
    """
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, tree_index: int, priority: float) -> None:
        change = float(priority) - float(self.tree[tree_index])
        self.tree[tree_index] = float(priority)
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v: float):
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
        return leaf, float(self.tree[leaf]), int(data_index)


class AtariFramePERNStepReplayBuffer:
    """
    Rainbow-style Atari replay:
      - stores ONLY the last frame (uint8) of each stacked observation (to save memory),
      - reconstructs stacked states on sampling,
      - supports proportional PER + n-step returns (computed at sampling time).
    """

    def __init__(
        self,
        capacity: int,
        frame_shape=(84, 84),
        history: int = 4,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.5,
        beta: float = 0.4,
        use_per: bool = True,
        eps: float = 1e-6,
    ):
        self.capacity = int(capacity)
        self.frame_shape = tuple(frame_shape)
        self.history = int(history)
        self.n_step = int(n_step)
        self.gamma = float(gamma)

        self.use_per = bool(use_per)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

        # Cyclic storage
        self.ptr = 0
        self.size = 0
        self.full = False

        # Episode timestep counter for boundary-safe stacking
        self.t = 0

        # Storage
        self.frames = np.zeros((self.capacity, *self.frame_shape), dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int16)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        # nonterminal indicates whether the *next* state is nonterminal (canonical Rainbow style)
        self.nonterminals = np.zeros((self.capacity,), dtype=np.bool_)
        self.timesteps = np.zeros((self.capacity,), dtype=np.int32)

        if self.use_per:
            self.tree = SumTree(self.capacity)
            self.max_priority = 1.0

    def __len__(self) -> int:
        return int(self.size)

    def _blank_frame(self):
        return np.zeros(self.frame_shape, dtype=np.uint8)

    def store(self, state, action, reward, next_state=None, done=False):
        """
        Compatibility shim with existing agent code.
        Expects `state` as a stacked Atari observation: [history, 84, 84] uint8.
        Stores ONLY the last frame.
        """
        last_frame = state[-1]
        if last_frame.dtype != np.uint8:
            last_frame = last_frame.astype(np.uint8)

        self.frames[self.ptr] = last_frame
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.nonterminals[self.ptr] = (not bool(done))
        self.timesteps[self.ptr] = int(self.t)

        if self.use_per:
            tree_idx = self.ptr + self.capacity - 1
            self.tree.update(tree_idx, self.max_priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or (self.ptr == 0)
        self.size = self.capacity if self.full else max(self.size, self.ptr)

        self.t = 0 if bool(done) else (self.t + 1)

    def _get_data(self, data_index: int):
        idx = int(data_index) % self.capacity
        return (
            int(self.timesteps[idx]),
            self.frames[idx],
            int(self.actions[idx]),
            float(self.rewards[idx]),
            bool(self.nonterminals[idx]),
        )

    def _get_transition(self, idx: int):
        """
        Returns a list length history + n_step of tuples:
          (timestep, frame, action, reward, nonterminal)
        """
        H = self.history
        N = self.n_step
        blank = (0, self._blank_frame(), 0, 0.0, False)

        trans = [blank] * (H + N)
        trans[H - 1] = self._get_data(idx)

        base = idx - H + 1

        # past frames
        for t in range(H - 2, -1, -1):
            if trans[t + 1][0] == 0:
                trans[t] = blank
            else:
                trans[t] = self._get_data(base + t)

        # future frames
        for t in range(H, H + N):
            if trans[t - 1][4]:
                trans[t] = self._get_data(base + t)
            else:
                trans[t] = blank

        return trans

    def _valid_index(self, idx: int) -> bool:
        if self.size < (self.history + self.n_step):
            return False

        idx = int(idx)

        if not self.full:
            return (idx >= (self.history - 1)) and (idx <= (self.size - self.n_step - 1))

        dist = (self.ptr - idx) % self.capacity
        return (dist > self.n_step) and (dist >= self.history)

    def sample(self, batch_size: int):
        B = int(batch_size)

        if not self.use_per:
            idxs = np.random.randint(0, self.size, size=B).astype(np.int32)
            for i in range(B):
                while not self._valid_index(idxs[i]):
                    idxs[i] = np.random.randint(0, self.size)
            weights = np.ones((B,), dtype=np.float32)
        else:
            total = self.tree.total()
            if total <= 0:
                raise RuntimeError("SumTree total priority is 0; cannot sample.")

            segment = total / B
            idxs = np.empty((B,), dtype=np.int32)
            priorities = np.empty((B,), dtype=np.float32)

            for i in range(B):
                while True:
                    a = segment * i
                    b = segment * (i + 1)
                    v = random.uniform(a, b)
                    leaf, p, data_idx = self.tree.get_leaf(v)
                    if p <= 0:
                        continue
                    if self._valid_index(data_idx):
                        idxs[i] = data_idx
                        priorities[i] = p
                        break

            probs = priorities / total
            cap = self.capacity if self.full else self.size
            weights = (cap * probs) ** (-self.beta)
            weights /= weights.max() + 1e-8
            weights = weights.astype(np.float32)

        H = self.history
        N = self.n_step
        states = np.zeros((B, H, *self.frame_shape), dtype=np.uint8)
        next_states = np.zeros((B, H, *self.frame_shape), dtype=np.uint8)
        actions = np.zeros((B,), dtype=np.int64)
        returns = np.zeros((B,), dtype=np.float32)
        dones = np.zeros((B,), dtype=np.bool_)
        gammas = np.full((B,), (self.gamma ** N), dtype=np.float32)

        for b, idx in enumerate(idxs):
            trans = self._get_transition(int(idx))

            for t in range(H):
                states[b, t] = trans[t][1]
                next_states[b, t] = trans[N + t][1]

            actions[b] = trans[H - 1][2]

            R = 0.0
            for k in range(N):
                R += (self.gamma ** k) * float(trans[H - 1 + k][3])
            returns[b] = R

            nonterminal_n = bool(trans[H + N - 1][4])
            dones[b] = (not nonterminal_n)

        return states, actions, returns, next_states, dones, gammas, idxs, weights

    def update_priorities(self, idxs, td_errors):
        if not self.use_per:
            return
        idxs = np.asarray(idxs, dtype=np.int32)
        td_errors = np.asarray(td_errors, dtype=np.float32)

        prios = (np.abs(td_errors) + self.eps) ** self.alpha
        self.max_priority = max(self.max_priority, float(prios.max(initial=self.max_priority)))

        for idx, p in zip(idxs, prios):
            tree_idx = int(idx) + self.capacity - 1
            self.tree.update(tree_idx, float(p))
