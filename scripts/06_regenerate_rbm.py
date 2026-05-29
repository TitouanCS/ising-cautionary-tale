#!/usr/bin/env python
"""
Regenerate RBM configurations using the trained weights stored in
data/results_rbm/L{L}_T{T}.npz, but with a different Gibbs initialization.

Useful for testing whether Gibbs chain initialization (random vs from MC data)
is the bottleneck for capturing the ordered phase at low T.

Usage:
    # Regenerate with MC-initialized chains:
    python scripts/06_regenerate_rbm.py --config configs/rbm.yaml --init mc

    # Or specific (L, T):
    python scripts/06_regenerate_rbm.py --config configs/rbm.yaml --init mc --L 12 --T 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rbm import IsingRBM, best_device
from src.rbm_train import generate_configs


def regenerate_one(L: int, T: float, cfg: dict, mc_dir: Path, rbm_dir: Path,
                    init: str, device: str) -> None:
    rbm_path = rbm_dir / f"L{L}_T{T:.4f}.npz"
    mc_path = mc_dir / f"L{L}_T{T:.4f}.npz"
    if not rbm_path.exists():
        print(f"[skip] {rbm_path.name} does not exist")
        return
    if not mc_path.exists():
        print(f"[skip] no MC data {mc_path.name}")
        return

    print(f"\n=== L={L}  T={T:.4f}  init={init} ===")

    # Reload trained RBM
    d = np.load(rbm_path)
    n_visible = int(d["L"]) ** 2
    n_hidden = int(d["n_hidden"])
    rbm = IsingRBM(n_visible, n_hidden)
    rbm.W.data = torch.from_numpy(d["rbm_W"])
    rbm.b.data = torch.from_numpy(d["rbm_b"])
    rbm.c.data = torch.from_numpy(d["rbm_c"])

    # Choose init source
    if init == "random":
        init_from = None
    elif init == "mc":
        mc_configs = np.load(mc_path)["configs"]
        init_from = mc_configs
        print(f"  Initializing Gibbs chains from {len(mc_configs)} MC configs")
    else:
        raise ValueError(f"Unknown init mode: {init}")

    rbm_configs = generate_configs(
        rbm, L=L,
        n_configs=int(cfg["n_gen_configs"]),
        k_gibbs=int(cfg["gibbs_k_gen"]),
        n_chains=int(cfg["gibbs_n_chains"]),
        device=device,
        init_from=init_from,
    )

    # Update the saved RBM file with new configs (backup the old key)
    new_data = dict(d)
    new_data["rbm_configs"] = rbm_configs
    new_data["gibbs_init"] = init
    np.savez_compressed(rbm_path, **new_data)
    print(f"  Re-saved {rbm_path.name} with {len(rbm_configs)} new configs")

    # Quick check
    from src.observables import energy_per_config, magnetization_per_config
    E = energy_per_config(rbm_configs).mean()
    M = np.abs(magnetization_per_config(rbm_configs)).mean()
    print(f"  Diagnostic: <eps>={E:+.3f}, <|M|>={M:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/rbm.yaml")
    ap.add_argument("--init", type=str, default="mc", choices=["random", "mc"],
                    help="How to initialize Gibbs chains. 'mc' uses training "
                         "data, 'random' uses uniform ±1.")
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc_dir = ROOT / cfg["data_dir"]
    rbm_dir = ROOT / cfg["results_dir"]
    device = args.device or best_device()
    print(f"Device: {device}")

    Ls = [args.L] if args.L is not None else cfg["Ls"]
    for L in Ls:
        rbm_files = sorted(rbm_dir.glob(f"L{L}_T*.npz"))
        for fpath in rbm_files:
            T = float(fpath.stem.split("_T")[1])
            if args.T is not None and abs(T - args.T) > 1e-3:
                continue
            regenerate_one(L, T, cfg, mc_dir, rbm_dir, args.init, device)


if __name__ == "__main__":
    main()