#!/usr/bin/env python
"""
Quick smoke test: verify the MC and CNN pipeline on a tiny system (L=8,
few temperatures, ~500 configs). Useful before launching full SLURM jobs.

Runs in ~1 minute on a laptop CPU. Outputs a quick <p>(T) curve so you
can eyeball that the sigmoid is in roughly the right place.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mc import generate_configurations, decorrelation_sweeps, total_energy  # noqa
from src.dataset import IsingDataset, T_C  # noqa
from src.model import IsingCNN  # noqa
from src.train import TrainConfig, train_cnn, evaluate_per_temperature  # noqa


def main():
    L = 8
    Ts = np.array([1.0, 1.5, 2.0, 2.27, 2.5, 3.0, 3.5])
    n_per_T = 500

    print(f"Smoke test: L={L}, {len(Ts)} temperatures, {n_per_T} configs each")
    print(f"T_c (Onsager) = {T_C:.4f}\n")

    all_configs, all_T = [], []
    t0 = time.time()
    for T in Ts:
        decorr = max(20, decorrelation_sweeps(L=L, T=T, base=20))
        configs, energies = generate_configurations(
            L=L, T=T, M0=0.0, n_configs=n_per_T,
            n_thermalize=1000, n_decorrelate=decorr, seed=42,
            return_energies=True,
        )
        all_configs.append(configs)
        all_T.append(np.full(n_per_T, T, dtype=np.float32))
        # Sanity: magnetization conserved
        M = configs.sum(axis=(1, 2))
        assert np.all(M == 0), f"M not conserved at T={T}: {M[:5]}"
        print(f"  T={T:.3f}  <eps>={energies.mean():+.4f}  decorr={decorr}sw")

    configs = np.concatenate(all_configs, axis=0)
    temperatures = np.concatenate(all_T)
    print(f"MC done in {time.time()-t0:.1f}s")

    # Train / val / test split
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(configs))
    n_tr = int(0.7 * len(configs))
    n_va = int(0.1 * len(configs))
    tr = perm[:n_tr]
    va = perm[n_tr:n_tr + n_va]
    te = perm[n_tr + n_va:]
    train_ds = IsingDataset(configs[tr], temperatures[tr])
    val_ds = IsingDataset(configs[va], temperatures[va])
    test_ds = IsingDataset(configs[te], temperatures[te])

    torch.manual_seed(0)
    model = IsingCNN(L=L, circular=True)
    cfg = TrainConfig(max_epochs=30, patience=4, batch_size=64)
    print(f"\nTraining on device: {cfg.device}")
    result = train_cnn(model, train_ds, val_ds, cfg)

    per_T = evaluate_per_temperature(model, test_ds, cfg.device)
    print("\nPer-temperature <p_ordered> on test set:")
    print(f"{'T':>8} {'<p>':>8} {'std':>8} {'n':>6}")
    for T in sorted(per_T):
        p_mean, p_std, n = per_T[T]
        print(f"{T:8.3f} {p_mean:8.3f} {p_std:8.3f} {n:6d}")

    print("\nIf <p> is near 1 below T_c=2.269 and near 0 above, the pipeline works.")


if __name__ == "__main__":
    main()
