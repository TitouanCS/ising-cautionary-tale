#!/usr/bin/env python
"""
Train one CNN per lattice size L and save:
  - the trained model weights
  - the per-temperature curve <p_ordered>(T) on the test set

For each L, configurations from all temperatures are loaded, labelled
by (T < T_c), shuffled, split 70/10/20, and used to train and evaluate
the model.

Usage:
    python scripts/02_train_cnn.py --config configs/default.yaml
    python scripts/02_train_cnn.py --config configs/default.yaml --L 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import IsingDataset, T_C  # noqa: E402
from src.model import IsingCNN  # noqa: E402
from src.train import TrainConfig, train_cnn, evaluate_per_temperature  # noqa: E402


def gather_configs_for_L(data_dir: Path, L: int, T_c: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Load all per-(L, T) .npz files and concatenate."""
    files = sorted(data_dir.glob(f"L{L}_T*.npz"))
    if not files:
        raise FileNotFoundError(f"No data for L={L} in {data_dir}")
    configs_list, T_list = [], []
    for f in files:
        d = np.load(f)
        configs_list.append(d["configs"])
        T_val = float(d["T"])
        T_list.append(np.full(len(d["configs"]), T_val, dtype=np.float32))
    configs = np.concatenate(configs_list, axis=0)
    temperatures = np.concatenate(T_list, axis=0)
    return configs, temperatures


def stratified_split(temperatures: np.ndarray, T_c: float, seed: int,
                     train_frac: float, val_frac: float):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for T in np.unique(temperatures):
        if abs(T - T_c) < 1e-6:
            continue
        idx = np.flatnonzero(np.abs(temperatures - T) < 1e-6)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(train_frac * n)
        n_va = int(val_frac * n)
        train_idx.append(idx[:n_tr])
        val_idx.append(idx[n_tr:n_tr + n_va])
        test_idx.append(idx[n_tr + n_va:])
    return (np.concatenate(train_idx),
            np.concatenate(val_idx),
            np.concatenate(test_idx))


def train_one_L(L: int, cfg: dict, data_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training CNN for L = {L} ===")
    configs, temperatures = gather_configs_for_L(data_dir, L, cfg["T_c"])
    print(f"Loaded {len(configs)} configurations across "
          f"{len(np.unique(temperatures))} temperatures.")

    tr_idx, va_idx, te_idx = stratified_split(
        temperatures, T_c=cfg["T_c"], seed=cfg["seed"],
        train_frac=0.7, val_frac=0.1,
    )
    print(f"Split: {len(tr_idx)} train / {len(va_idx)} val / {len(te_idx)} test")

    train_ds = IsingDataset(configs[tr_idx], temperatures[tr_idx], cfg["T_c"])
    val_ds = IsingDataset(configs[va_idx], temperatures[va_idx], cfg["T_c"])
    test_ds = IsingDataset(configs[te_idx], temperatures[te_idx], cfg["T_c"])

    torch.manual_seed(cfg["seed"] + L)
    model = IsingCNN(L=L, circular=cfg["circular_padding"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters")

    tcfg = TrainConfig(
        lr=float(cfg["lr"]),
        batch_size=int(cfg["batch_size"]),
        max_epochs=int(cfg["max_epochs"]),
        patience=int(cfg["patience"]),
    )
    print(f"Training on device: {tcfg.device}")

    result = train_cnn(model, train_ds, val_ds, tcfg)

    # Test set accuracy
    from src.train import _accuracy
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=tcfg.batch_size, shuffle=False)
    test_acc = _accuracy(model, test_loader, tcfg.device)
    print(f"Test accuracy: {test_acc:.4f}")

    # Per-temperature evaluation on test set
    per_T = evaluate_per_temperature(model, test_ds, tcfg.device,
                                      batch_size=tcfg.batch_size)
    Ts = np.array(sorted(per_T.keys()))
    p_means = np.array([per_T[T][0] for T in Ts])
    p_stds = np.array([per_T[T][1] for T in Ts])
    n_per_T = np.array([per_T[T][2] for T in Ts])

    out_npz = results_dir / f"cnn_output_L{L}.npz"
    np.savez(
        out_npz,
        T=Ts,
        p_mean=p_means,
        p_std=p_stds,
        n=n_per_T,
        L=L,
        test_acc=test_acc,
        best_val_acc=result.best_val_acc,
        best_epoch=result.best_epoch,
        train_losses=np.array(result.train_losses),
        val_accs=np.array(result.val_accs),
    )
    print(f"Saved curve to {out_npz}")

    weights_path = results_dir / f"cnn_weights_L{L}.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--L", type=int, default=None,
                    help="If given, train only this lattice size")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    Ls = [args.L] if args.L is not None else cfg["Ls"]

    for L in Ls:
        train_one_L(L, cfg, data_dir, results_dir)


if __name__ == "__main__":
    main()
