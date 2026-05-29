"""
Restricted Boltzmann Machine for 2D Ising configurations.

Conventions
-----------
  - Visible units v_j in {-1, +1} (the spins)
  - Hidden units h_i in {0, 1}
  - Energy:     E(v, h) = - v^T b - c^T h - v^T W h
  - Free energy after marginalizing h:
        F(v) = - v^T b - sum_i log(1 + exp(c_i + v^T W[:, i]))

Conditional distributions
-------------------------
  - P(h_i = 1 | v) = sigmoid(c_i + v^T W[:, i])
  - P(v_j = +1 | h) = sigmoid(2 * (b_j + W[j, :] h))
    (factor 2 comes from {-1, +1} vs {0, 1} convention -- if v is in {0, 1}
     this factor disappears, but Ising spins are ±1.)

Training
--------
Contrastive divergence CD-k:
    grad_theta = - d/d theta [ F(v_data) - F(v_neg) ].mean()
where v_neg is the result of k Gibbs steps starting from v_data.

The paper uses Adam, lr = 5e-3, k = 10, ~1000 epochs, ~10000 training configs,
m = n = L^2 hidden units.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IsingRBM(nn.Module):
    """RBM with ±1 visible units and {0, 1} hidden units."""

    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.n_v = int(n_visible)
        self.n_h = int(n_hidden)
        # Hinton's "Practical guide" recommendation for initialization
        self.W = nn.Parameter(0.01 * torch.randn(self.n_v, self.n_h))
        self.b = nn.Parameter(torch.zeros(self.n_v))  # visible bias
        self.c = nn.Parameter(torch.zeros(self.n_h))  # hidden bias

    # ----- forward conditionals -----

    def prob_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        """P(h = 1 | v), shape (B, n_h). v has values in {-1, +1}."""
        return torch.sigmoid(self.c + v @ self.W)

    def sample_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        p = self.prob_h_given_v(v).clamp(1e-6, 1 - 1e-6)
        return (torch.rand_like(p) < p).float()

    def prob_v_plus_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """P(v = +1 | h), shape (B, n_v). Factor of 2 from ±1 convention."""
        return torch.sigmoid(2.0 * (self.b + h @ self.W.T))

    def sample_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        p = self.prob_v_plus_given_h(h).clamp(1e-6, 1 - 1e-6)
        return 2.0 * (torch.rand_like(p) < p).float() - 1.0  # {0,1} -> {-1,+1}

    # ----- Gibbs sampling -----

    def gibbs_step(self, v: torch.Tensor) -> torch.Tensor:
        h = self.sample_h_given_v(v)
        return self.sample_v_given_h(h)

    def gibbs_chain(self, v0: torch.Tensor, k: int) -> torch.Tensor:
        v = v0
        for _ in range(k):
            v = self.gibbs_step(v)
        return v

    # ----- free energy (for CD loss) -----

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """F(v) = -v^T b - sum_i softplus(c_i + v^T W[:, i]). Shape (B,)."""
        wx_b = v @ self.W + self.c  # (B, n_h)
        vbias = v @ self.b  # (B,)
        hidden = F.softplus(wx_b).sum(dim=1)  # (B,)
        return -vbias - hidden

    def cd_loss(self, v_data: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Contrastive divergence loss (lower is better)."""
        with torch.no_grad():
            v_neg = self.gibbs_chain(v_data, k)
        return self.free_energy(v_data).mean() - self.free_energy(v_neg).mean()


def best_device() -> str:
    """Return the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["IsingRBM", "best_device"]