import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Factorised Gaussian NoisyNet layer (Fortunato et al.).
    Deterministic when .eval() or use_noisy=False.
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.1, use_noisy: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = float(std_init)
        self.use_noisy = bool(use_noisy)

        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))

        self.register_buffer("weight_epsilon", torch.empty(self.out_features, self.in_features))
        self.register_buffer("bias_epsilon", torch.empty(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        limit = 1.0 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-limit, limit)
        self.bias_mu.data.uniform_(-limit, limit)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        if not self.use_noisy:
            return
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_noisy:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class C51NoisyDuelingCNN(nn.Module):
    """
    Standard Atari conv torso + (optional) dueling heads + C51 distribution.
    Returns categorical distribution over atoms for each action.
    """
    def __init__(self, num_actions: int, atom_size: int, support: torch.Tensor, use_noisy: bool = True, use_dueling: bool = True):
        super().__init__()
        self.num_actions = int(num_actions)
        self.atom_size = int(atom_size)
        self.support = support
        self.use_dueling = bool(use_dueling)
        self.use_noisy = bool(use_noisy)

        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.feature_size = 7 * 7 * 64

        self.advantage_hidden = NoisyLinear(self.feature_size, 512, use_noisy=self.use_noisy)
        self.advantage = NoisyLinear(512, self.num_actions * self.atom_size, use_noisy=self.use_noisy)

        if self.use_dueling:
            self.value_hidden = NoisyLinear(self.feature_size, 512, use_noisy=self.use_noisy)
            self.value = NoisyLinear(512, self.atom_size, use_noisy=self.use_noisy)

    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
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

        if log:
            return F.log_softmax(q_atoms, dim=-1)

        dist = F.softmax(q_atoms, dim=-1).clamp(min=1e-8)
        return dist

    def reset_noise(self) -> None:
        if not self.use_noisy:
            return
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        if self.use_dueling:
            self.value_hidden.reset_noise()
            self.value.reset_noise()


class C51NoisyDuelingMLP(nn.Module):
    """
    MLP version for non-Atari environments.
    """
    def __init__(self, obs_dim: int, num_actions: int, atom_size: int, support: torch.Tensor, use_noisy: bool = True, use_dueling: bool = True):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.atom_size = int(atom_size)
        self.support = support
        self.use_dueling = bool(use_dueling)
        self.use_noisy = bool(use_noisy)

        self.advantage_hidden = NoisyLinear(self.obs_dim, 128, use_noisy=self.use_noisy)
        self.advantage = NoisyLinear(128, self.num_actions * self.atom_size, use_noisy=self.use_noisy)

        if self.use_dueling:
            self.value_hidden = NoisyLinear(self.obs_dim, 128, use_noisy=self.use_noisy)
            self.value = NoisyLinear(128, self.atom_size, use_noisy=self.use_noisy)

    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
        adv = F.relu(self.advantage_hidden(x))
        adv = self.advantage(adv).view(-1, self.num_actions, self.atom_size)

        if self.use_dueling:
            val = F.relu(self.value_hidden(x))
            val = self.value(val).view(-1, 1, self.atom_size)
            q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            q_atoms = adv

        if log:
            return F.log_softmax(q_atoms, dim=-1)

        dist = F.softmax(q_atoms, dim=-1).clamp(min=1e-8)
        return dist

    def reset_noise(self) -> None:
        if not self.use_noisy:
            return
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        if self.use_dueling:
            self.value_hidden.reset_noise()
            self.value.reset_noise()


