"""
Training loop and configuration generator for IsingRBM.

Paper hyperparameters (Azizi-Pleimling 2021):
  - n_hidden = L^2 = n_visible
  - lr = 5e-3, Adam
  - CD-10
  - ~1000 epochs, ~10000 training configurations
  - Convergence diagnostic: mean absolute value of trainable parameters
    plateaus (log-likelihood is intractable)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .rbm import IsingRBM


@dataclass
class RBMTrainConfig:
    n_epochs: int = 500
    batch_size: int = 100
    lr: float = 5e-3
    cd_k: int = 10
    log_every: int = 20
    device: str = "cpu"


@dataclass
class RBMTrainHistory:
    loss: list[float] = field(default_factory=list)  # mean CD loss per epoch
    theta_abs: list[float] = field(default_factory=list)  # |theta| diagnostic
    walltime: float = 0.0


def _configs_to_tensor(configs: np.ndarray) -> torch.Tensor:
    """(N, L, L) int8 -> (N, L*L) float32 in {-1, +1}."""
    return torch.from_numpy(configs.reshape(len(configs), -1).astype(np.float32))


def train_rbm(
    rbm: IsingRBM,
    train_configs: np.ndarray,
    cfg: RBMTrainConfig,
    log_fn=print,
) -> RBMTrainHistory:
    """
    Train rbm on train_configs (N, L, L) int8 with the given CD-k schedule.

    Returns the training history (loss curve + parameter-magnitude diagnostic).
    """
    device = cfg.device
    rbm = rbm.to(device)
    optim = torch.optim.Adam(rbm.parameters(), lr=cfg.lr)
    v_tensor = _configs_to_tensor(train_configs)
    loader = DataLoader(TensorDataset(v_tensor), batch_size=cfg.batch_size,
                        shuffle=True, drop_last=False)

    history = RBMTrainHistory()
    t0 = time()
    for epoch in range(cfg.n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (v_batch,) in loader:
            v_batch = v_batch.to(device, non_blocking=True)
            loss = rbm.cd_loss(v_batch, k=cfg.cd_k)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        with torch.no_grad():
            theta_abs = float(
                (rbm.W.abs().mean() + rbm.b.abs().mean() + rbm.c.abs().mean()) / 3
            )
        history.loss.append(epoch_loss)
        history.theta_abs.append(theta_abs)
        if epoch % cfg.log_every == 0 or epoch == cfg.n_epochs - 1:
            log_fn(f"  epoch {epoch:4d} | loss={epoch_loss:+.4f} | |theta|={theta_abs:.4f}")

    history.walltime = time() - t0
    return history


@torch.no_grad()
def generate_configs(
    rbm: IsingRBM,
    L: int,
    n_configs: int,
    k_gibbs: int = 1000,
    n_chains: int = 100,
    device: str = "cpu",
    init_from: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate configurations from a trained RBM via parallel Gibbs chains.

    Parameters
    ----------
    init_from : np.ndarray or None
        If None (default), each chain starts from a uniform random +-1
        configuration. This corresponds to the paper's protocol.
        If provided, must have shape (M, L, L) int8 -- training configs.
        Chains are then initialized by randomly sampling from these, which
        helps in cases where Gibbs chains from random init cannot cross
        energy barriers to find ordered phases.

    Note: the paper used k = 10 for generation. We default to k = 1000.

    Returns shape (n_configs, L, L) int8.
    """
    rbm = rbm.to(device).eval()
    out_list: list[np.ndarray] = []
    n_done = 0
    while n_done < n_configs:
        b = min(n_chains, n_configs - n_done)
        if init_from is None:
            v0 = torch.bernoulli(0.5 * torch.ones(b, rbm.n_v, device=device))
            v0 = 2.0 * v0 - 1.0  # {0,1} -> {-1,+1}
        else:
            # Sample b configurations from init_from at random
            idx = np.random.choice(len(init_from), b, replace=True)
            v0_np = init_from[idx].reshape(b, -1).astype(np.float32)
            v0 = torch.from_numpy(v0_np).to(device)
        v = rbm.gibbs_chain(v0, k_gibbs)
        v_np = v.cpu().numpy().astype(np.int8).reshape(b, L, L)
        out_list.append(v_np)
        n_done += b
    return np.concatenate(out_list, axis=0)[:n_configs]


__all__ = ["RBMTrainConfig", "RBMTrainHistory", "train_rbm", "generate_configs"]