#!/usr/bin/env python
"""
RBM smoke test: trains a tiny RBM at one (L, T) on a small batch of MC
configurations, generates samples, and prints observable comparisons.

Use this before running the full RBM experiment to confirm that:
  - torch / numpy / numba are installed and play nicely with each other
  - the RBM training loop runs without errors on your device
  - the generated configurations look qualitatively reasonable

Runs in ~30 seconds on a laptop CPU or in <5 seconds on GPU/MPS.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mc import generate_configurations  # noqa: E402
from src.observables import (  # noqa: E402
    energy_per_config, magnetization_per_config, correlation_function,
)
from src.rbm import IsingRBM, best_device  # noqa: E402
from src.rbm_train import RBMTrainConfig, train_rbm, generate_configs  # noqa: E402


def main():
    L = 8
    T = 2.0  # below T_c, ordered side
    n_mc = 1000
    n_gen = 1000

    device = best_device()
    print(f"Smoke test: L={L}, T={T}, n_mc={n_mc}, n_gen={n_gen}, device={device}\n")

    # 1) Generate MC training data
    t0 = time.time()
    mc_configs, _ = generate_configurations(
        L=L, T=T, M0=0.0,
        n_configs=n_mc, n_thermalize=1000, n_decorrelate=20, seed=0,
    )
    print(f"MC generation: {time.time() - t0:.1f}s")
    print(f"  <eps>_MC  = {energy_per_config(mc_configs).mean():+.4f}")
    print(f"  <|M|>_MC  = {np.abs(magnetization_per_config(mc_configs)).mean():.4f}  "
          f"(should be ~0 since M is conserved)")
    print(f"  C(r)_MC   = {np.array2string(correlation_function(mc_configs, 4), precision=3)}")

    # 2) Train a small RBM
    rbm = IsingRBM(n_visible=L * L, n_hidden=L * L)
    tcfg = RBMTrainConfig(
        n_epochs=100, batch_size=64, lr=5e-3, cd_k=10,
        log_every=20, device=device,
    )
    print(f"\nTraining RBM ({sum(p.numel() for p in rbm.parameters())} parameters)...")
    history = train_rbm(rbm, mc_configs, tcfg)
    print(f"  walltime: {history.walltime:.1f}s")
    print(f"  |theta| went from {history.theta_abs[0]:.4f} -> {history.theta_abs[-1]:.4f}")

    # 3) Generate from RBM
    t0 = time.time()
    rbm_configs = generate_configs(
        rbm, L=L, n_configs=n_gen, k_gibbs=500, n_chains=100, device=device,
    )
    print(f"\nRBM generation: {time.time() - t0:.1f}s")
    print(f"  <eps>_RBM = {energy_per_config(rbm_configs).mean():+.4f}")
    print(f"  <|M|>_RBM = {np.abs(magnetization_per_config(rbm_configs)).mean():.4f}  "
          f"(EXPECT > 0: RBM breaks the M = 0 constraint)")
    print(f"  C(r)_RBM  = {np.array2string(correlation_function(rbm_configs, 4), precision=3)}")

    print("\nIf <|M|>_RBM is noticeably > 0 while <|M|>_MC is 0, "
          "you have already reproduced the main qualitative result of the paper.")


if __name__ == "__main__":
    main()
