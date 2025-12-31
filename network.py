import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, use_noisy=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.use_noisy = use_noisy

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers for noise
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        limit = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-limit, limit)
        self.bias_mu.data.uniform_(-limit, limit)

        sigma_w = self.std_init / math.sqrt(self.in_features)
        sigma_b = self.std_init / math.sqrt(self.out_features)

        self.weight_sigma.data.fill_(sigma_w)
        self.bias_sigma.data.fill_(sigma_b)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        if not self.use_noisy:
            return
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        # Deterministic in eval() or when use_noisy=False
        if self.training and self.use_noisy:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class C51NoisyDuelingCNN(nn.Module):
    def __init__(self, num_actions, atom_size, support, use_noisy=True, use_dueling=True):
        super().__init__()
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.support = support
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy

        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.feature_size = 7 * 7 * 64

        # If dueling is off, we only need an advantage-like head, but we keep structure simple
        self.advantage_hidden = NoisyLinear(self.feature_size, 512, use_noisy=use_noisy)
        self.advantage = NoisyLinear(512, num_actions * atom_size, use_noisy=use_noisy)

        if use_dueling:
            self.value_hidden = NoisyLinear(self.feature_size, 512, use_noisy=use_noisy)
            self.value = NoisyLinear(512, atom_size, use_noisy=use_noisy)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)

        adv = F.relu(self.advantage_hidden(x))
        adv = self.advantage(adv).view(-1, self.num_actions, self.atom_size)

        if self.use_dueling:
            val = F.relu(self.value_hidden(x))
            val = self.value(val).view(-1, 1, self.atom_size)
            q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            q_atoms = adv

        dist = F.softmax(q_atoms, dim=-1).clamp(min=1e-8)
        return dist

    def reset_noise(self):
        if not self.use_noisy:
            return
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        if self.use_dueling:
            self.value_hidden.reset_noise()
            self.value.reset_noise()


class C51NoisyDuelingMLP(nn.Module):
    def __init__(self, obs_dim, num_actions, atom_size, support, use_noisy=True, use_dueling=True):
        super().__init__()
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.support = support
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy

        self.advantage_hidden = NoisyLinear(obs_dim, 128, use_noisy=use_noisy)
        self.advantage = NoisyLinear(128, num_actions * atom_size, use_noisy=use_noisy)

        if use_dueling:
            self.value_hidden = NoisyLinear(obs_dim, 128, use_noisy=use_noisy)
            self.value = NoisyLinear(128, atom_size, use_noisy=use_noisy)

    def forward(self, x):
        adv = F.relu(self.advantage_hidden(x))
        adv = self.advantage(adv).view(-1, self.num_actions, self.atom_size)

        if self.use_dueling:
            val = F.relu(self.value_hidden(x))
            val = self.value(val).view(-1, 1, self.atom_size)
            q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            q_atoms = adv

        dist = F.softmax(q_atoms, dim=-1).clamp(min=1e-8)
        return dist

    def reset_noise(self):
        if not self.use_noisy:
            return
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        if self.use_dueling:
            self.value_hidden.reset_noise()
            self.value.reset_noise()

