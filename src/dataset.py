"""
PyTorch Dataset wrappers for the Ising MC configurations.

Conventions:
  - Label 1 = ordered (T < T_c)
  - Label 0 = disordered (T > T_c)
  - Configurations stored as int8 in {-1, +1}, converted to float on the fly
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))  # exact Onsager value, ~2.2691853


class IsingDataset(Dataset):
    """
    Holds configurations and per-configuration temperatures.

    The label is derived from the temperature relative to T_c. Configurations
    exactly at T_c are excluded by the loader (we never sample exactly there).
    """

    def __init__(self, configs: np.ndarray, temperatures: np.ndarray, T_c: float = T_C):
        assert configs.dtype == np.int8
        assert configs.ndim == 3
        assert len(configs) == len(temperatures)
        self.configs = configs
        self.temperatures = temperatures.astype(np.float32)
        self.labels = (temperatures < T_c).astype(np.int64)

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        x = torch.from_numpy(self.configs[idx]).float().unsqueeze(0)  # (1, L, L)
        y = int(self.labels[idx])
        T = float(self.temperatures[idx])
        return x, y, T


def load_split(npz_path: Path, T_c: float = T_C, seed: int = 0,
               train_frac: float = 0.7, val_frac: float = 0.1):
    """
    Load all configurations from an .npz file and split into train/val/test.

    The .npz must contain:
      - 'configs': (N, L, L) int8
      - 'temperatures': (N,) float32

    Stratification: the split is per-temperature, so each split contains
    roughly the same fraction of configurations from every temperature.
    """
    data = np.load(npz_path)
    configs = data["configs"]
    temperatures = data["temperatures"]

    rng = np.random.default_rng(seed)
    unique_T = np.unique(temperatures)

    train_idx, val_idx, test_idx = [], [], []
    for T in unique_T:
        if abs(T - T_c) < 1e-6:
            continue  # never train on T == T_c exactly
        idx = np.flatnonzero(np.abs(temperatures - T) < 1e-6)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    return (
        IsingDataset(configs[train_idx], temperatures[train_idx], T_c),
        IsingDataset(configs[val_idx], temperatures[val_idx], T_c),
        IsingDataset(configs[test_idx], temperatures[test_idx], T_c),
    )


__all__ = ["IsingDataset", "load_split", "T_C"]
