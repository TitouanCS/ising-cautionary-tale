"""
Training utilities for the Ising CNN classifier.

The training protocol follows the paper:
  - cross-entropy loss
  - Adam optimizer
  - early stopping when validation accuracy fails to improve for `patience` epochs

The inference helper `evaluate_per_temperature` produces the curve
<p_ordered>(T) that we then analyze for finite-size scaling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import IsingCNN
from .dataset import IsingDataset


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 3  # epochs without val-accuracy improvement
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainResult:
    best_val_acc: float = 0.0
    best_epoch: int = -1
    train_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)


def _accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_cnn(
    model: IsingCNN,
    train_ds: IsingDataset,
    val_ds: IsingDataset,
    cfg: TrainConfig,
    log_fn=print,
) -> TrainResult:
    device = cfg.device
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(device == "cuda"))

    result = TrainResult()
    best_state = None
    stale_epochs = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        val_acc = _accuracy(model, val_loader, device)
        result.train_losses.append(epoch_loss)
        result.val_accs.append(val_acc)
        log_fn(f"  epoch {epoch:3d} | train_loss={epoch_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > result.best_val_acc + 1e-6:
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                log_fn(f"  early stopping at epoch {epoch} "
                       f"(best val_acc={result.best_val_acc:.4f} @ epoch {result.best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return result


@torch.no_grad()
def evaluate_per_temperature(model: IsingCNN, dataset: IsingDataset,
                              device: str, batch_size: int = 256
                              ) -> dict[float, tuple[float, float, int]]:
    """
    Run inference over `dataset` and group results by temperature.

    Returns a dict mapping T -> (mean P(ordered), std P(ordered), n_samples).
    """
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs_all = []
    temps_all = []
    for x, _, T in loader:
        x = x.to(device, non_blocking=True)
        p = model.predict_p_ordered(x).cpu().numpy()
        probs_all.append(p)
        temps_all.append(np.asarray(T))
    probs = np.concatenate(probs_all)
    temps = np.concatenate(temps_all)

    out: dict[float, tuple[float, float, int]] = {}
    for T in np.unique(temps):
        mask = np.abs(temps - T) < 1e-6
        p_T = probs[mask]
        out[float(T)] = (float(p_T.mean()), float(p_T.std(ddof=1) if len(p_T) > 1 else 0.0),
                         int(mask.sum()))
    return out


__all__ = ["TrainConfig", "TrainResult", "train_cnn", "evaluate_per_temperature"]
