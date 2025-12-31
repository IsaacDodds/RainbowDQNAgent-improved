import torch


class C51Distribution:
    def __init__(self, v_min, v_max, atom_size, device):
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.device = device

        self.support = torch.linspace(v_min, v_max, atom_size, device=device)
        self.delta_z = (v_max - v_min) / (atom_size - 1)

    def expectation(self, dist):
        return (dist * self.support).sum(dim=-1)

    def project(self, next_dist, rewards, dones, gammas):
        B = rewards.size(0)

        tz = rewards + (1 - dones.float()) * gammas * self.support.view(1, -1)
        tz = tz.clamp(self.v_min, self.v_max)

        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = l.clamp(0, self.atom_size - 1)
        u = u.clamp(0, self.atom_size - 1)

        proj = torch.zeros(B, self.atom_size, device=self.device)

        m_l = (u.float() - b)
        m_u = (b - l.float())
        eq = (u == l)
        m_l[eq] = 1.0
        m_u[eq] = 0.0

        proj.scatter_add_(1, l, next_dist * m_l)
        proj.scatter_add_(1, u, next_dist * m_u)

        proj = proj.clamp(1e-8, 1.0)
        proj /= proj.sum(dim=1, keepdim=True)
        return proj

