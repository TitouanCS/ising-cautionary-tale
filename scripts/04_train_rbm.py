#!/usr/bin/env python
"""
Train one Restricted Boltzmann Machine per (L, T) using MC configurations
as the training set, then sample new configurations from the trained RBM.

Outputs for each (L, T):
    data/results_rbm/L{L}_T{T:.4f}.npz
        - 'rbm_configs': (N_gen, L, L) int8, generated from the trained RBM
        - 'rbm_W', 'rbm_b', 'rbm_c': trained weights
        - 'loss_history', 'theta_abs_history'
        - 'walltime_train', 'walltime_gen'
        - metadata: L, T, n_train, n_gen, k, etc.

Usage:
    # Train all (L, T) sequentially:
    python scripts/04_train_rbm.py --config configs/rbm.yaml

    # Just one (L, T) for testing:
    python scripts/04_train_rbm.py --config configs/rbm.yaml --L 8 --T 2.0

The MC data must already exist in the directory configured by data_dir in the
YAML (run 01_generate_mc.py --config configs/rbm.yaml first).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rbm import IsingRBM, best_device  # noqa: E402
from src.rbm_train import (  # noqa: E402
    RBMTrainConfig, train_rbm, generate_configs,
)


def list_mc_files(data_dir: Path, L: int):
    return sorted(data_dir.glob(f"L{L}_T*.npz"))


def parse_T_from_filename(path: Path) -> float:
    # "L12_T2.2692.npz" -> 2.2692
    stem = path.stem
    return float(stem.split("_T")[1])


def train_one(L: int, T: float, mc_path: Path, cfg: dict, out_dir: Path,
              device: str) -> None:
    out_path = out_dir / f"L{L}_T{T:.4f}.npz"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== L={L}  T={T:.4f}  ===")

    # Load MC training data
    data = np.load(mc_path)
    mc_configs = data["configs"]
    n_avail = len(mc_configs)
    n_train = min(int(cfg["rbm"]["n_train_configs"]), n_avail)
    rng = np.random.default_rng(int(cfg["seed"]) + L * 997 + int(T * 1000))
    train_idx = rng.choice(n_avail, n_train, replace=False)
    train_configs = mc_configs[train_idx]
    print(f"  training set: {n_train} MC configs (of {n_avail} available)")

    # Build and train RBM
    n_visible = L * L
    n_hidden = int(cfg["rbm"]["n_hidden_factor"] * L * L)
    torch.manual_seed(int(cfg["seed"]) + L + int(T * 1000))
    rbm = IsingRBM(n_visible, n_hidden)
    print(f"  RBM: {n_visible} visible, {n_hidden} hidden, "
          f"{sum(p.numel() for p in rbm.parameters())} parameters")

    tcfg = RBMTrainConfig(
        n_epochs=int(cfg["rbm"]["n_epochs"]),
        batch_size=int(cfg["rbm"]["batch_size"]),
        lr=float(cfg["rbm"]["lr"]),
        cd_k=int(cfg["rbm"]["cd_k"]),
        log_every=max(1, int(cfg["rbm"]["n_epochs"]) // 10),
        device=device,
    )
    history = train_rbm(rbm, train_configs, tcfg)

    # Generate configurations from the trained RBM
    t0 = time.time()
    rbm_configs = generate_configs(
        rbm,
        L=L,
        n_configs=int(cfg["n_gen_configs"]),
        k_gibbs=int(cfg["gibbs_k_gen"]),
        n_chains=int(cfg["gibbs_n_chains"]),
        device=device,
    )
    walltime_gen = time.time() - t0
    print(f"  generation: {len(rbm_configs)} configs in {walltime_gen:.1f}s")

    # Save everything
    np.savez_compressed(
        out_path,
        rbm_configs=rbm_configs,
        rbm_W=rbm.W.detach().cpu().numpy(),
        rbm_b=rbm.b.detach().cpu().numpy(),
        rbm_c=rbm.c.detach().cpu().numpy(),
        loss_history=np.array(history.loss),
        theta_abs_history=np.array(history.theta_abs),
        L=L,
        T=T,
        n_train=n_train,
        n_gen=len(rbm_configs),
        n_hidden=n_hidden,
        cd_k=tcfg.cd_k,
        gibbs_k_gen=int(cfg["gibbs_k_gen"]),
        walltime_train=history.walltime,
        walltime_gen=walltime_gen,
    )
    print(f"  saved -> {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/rbm.yaml")
    ap.add_argument("--L", type=int, default=None,
                    help="If given, only train RBM for this lattice size")
    ap.add_argument("--T", type=float, default=None,
                    help="If given (with --L), train RBM only at this temperature")
    ap.add_argument("--device", type=str, default=None,
                    help="Override device (cpu / cuda / mps). Default: auto.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = ROOT / cfg["data_dir"]
    out_dir = ROOT / cfg["results_dir"]
    device = args.device or best_device()
    print(f"Device: {device}")
    print(f"MC data dir: {data_dir}")
    print(f"Output dir:  {out_dir}")

    Ls = [args.L] if args.L is not None else cfg["Ls"]
    for L in Ls:
        files = list_mc_files(data_dir, L)
        if not files:
            print(f"[warn] no MC files for L={L} in {data_dir}", flush=True)
            continue
        for fpath in files:
            T = parse_T_from_filename(fpath)
            if args.T is not None and abs(T - args.T) > 1e-3:
                continue
            train_one(L, T, fpath, cfg, out_dir, device)


if __name__ == "__main__":
    main()
