#!/usr/bin/env python
"""
Compare statistics of MC and RBM ensembles. Reproduces qualitatively
figures 5, 6 and 7 of Azizi-Pleimling 2021.

Outputs:
    data/results_rbm/fig_energy_mag.png   --- Fig 5: <eps>(T) and <|M|>(T)
    data/results_rbm/fig_PE.png           --- Fig 6: P(E) at low and high T
    data/results_rbm/fig_correlations.png --- Fig 7: C(r) at multiple T
    data/results_rbm/summary.json

Usage:
    python scripts/05_analyze_rbm.py --config configs/rbm.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.observables import (  # noqa: E402
    energy_per_config,
    magnetization_per_config,
    correlation_function,
    energy_histogram,
)


def load_pair(L: int, T: float, mc_dir: Path, rbm_dir: Path):
    """Return (mc_configs, rbm_configs) for a given (L, T), or (None, None) if missing."""
    mc_path = mc_dir / f"L{L}_T{T:.4f}.npz"
    rbm_path = rbm_dir / f"L{L}_T{T:.4f}.npz"
    if not mc_path.exists() or not rbm_path.exists():
        return None, None
    mc = np.load(mc_path)["configs"]
    rbm = np.load(rbm_path)["rbm_configs"]
    return mc, rbm


def list_temperatures(mc_dir: Path, L: int) -> list[float]:
    Ts = []
    for f in sorted(mc_dir.glob(f"L{L}_T*.npz")):
        try:
            Ts.append(float(f.stem.split("_T")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(Ts)


def plot_energy_and_mag(Ls: list[int], mc_dir: Path, rbm_dir: Path,
                         out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(Ls)))
    summary = {}

    for L, col in zip(Ls, colors):
        Ts = list_temperatures(mc_dir, L)
        eps_mc, eps_rbm, M_mc, M_rbm = [], [], [], []
        for T in Ts:
            mc, rbm = load_pair(L, T, mc_dir, rbm_dir)
            if mc is None:
                continue
            eps_mc.append(energy_per_config(mc).mean())
            eps_rbm.append(energy_per_config(rbm).mean())
            M_mc.append(np.abs(magnetization_per_config(mc)).mean())
            M_rbm.append(np.abs(magnetization_per_config(rbm)).mean())
        Ts = np.array(Ts[:len(eps_mc)])
        ax1.plot(Ts, eps_mc, "--", color=col, alpha=0.7, label=f"MC  L={L}")
        ax1.plot(Ts, eps_rbm, "o", color=col, markersize=6, label=f"RBM L={L}")
        ax2.plot(Ts, M_mc, "--", color=col, alpha=0.7, label=f"MC  L={L}")
        ax2.plot(Ts, M_rbm, "o", color=col, markersize=6, label=f"RBM L={L}")
        summary[f"L={L}"] = {
            "T": Ts.tolist(),
            "eps_mc": eps_mc, "eps_rbm": eps_rbm,
            "absM_mc": M_mc, "absM_rbm": M_rbm,
        }

    ax1.set_xlabel("Temperature $T$")
    ax1.set_ylabel(r"$\langle \varepsilon \rangle$")
    ax1.set_title("Energy density: MC vs RBM")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Temperature $T$")
    ax2.set_ylabel(r"$\langle |M| \rangle$")
    ax2.set_title("Magnetization density: MC vs RBM")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"  saved {out_path}")
    return summary


def plot_PE(L: int, T_targets: list[float], mc_dir: Path, rbm_dir: Path,
             out_path: Path):
    """P(E) at a few selected temperatures."""
    available_Ts = list_temperatures(mc_dir, L)
    Ts = []
    for Tt in T_targets:
        if available_Ts:
            Ts.append(min(available_Ts, key=lambda t: abs(t - Tt)))
    Ts = sorted(set(Ts))

    n = len(Ts)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, T in zip(axes, Ts):
        mc, rbm = load_pair(L, T, mc_dir, rbm_dir)
        if mc is None:
            ax.set_title(f"T = {T:.2f}  (no data)")
            continue
        # Use a common bin range that covers both
        E_mc = energy_per_config(mc)
        E_rbm = energy_per_config(rbm)
        E_range = (min(E_mc.min(), E_rbm.min()) - 0.05,
                   max(E_mc.max(), E_rbm.max()) + 0.05)
        c_mc, p_mc = energy_histogram(mc, n_bins=50, E_range=E_range)
        c_rbm, p_rbm = energy_histogram(rbm, n_bins=50, E_range=E_range)
        ax.plot(c_mc, p_mc, "o-", label="MC", color="#1f77b4", markersize=4)
        ax.plot(c_rbm, p_rbm, "s-", label="RBM", color="#d62728", markersize=4)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("P")
        ax.set_title(f"L = {L}, T = {T:.2f}")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"  saved {out_path}")


def plot_correlations(Ls: list[int], T_targets: list[float],
                       mc_dir: Path, rbm_dir: Path, out_path: Path):
    fig, axes = plt.subplots(1, len(Ls), figsize=(5.5 * len(Ls), 4.5))
    if len(Ls) == 1:
        axes = [axes]

    for ax, L in zip(axes, Ls):
        available_Ts = list_temperatures(mc_dir, L)
        Ts = sorted({min(available_Ts, key=lambda t: abs(t - Tt))
                     for Tt in T_targets if available_Ts})
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ts)))
        for T, col in zip(Ts, colors):
            mc, rbm = load_pair(L, T, mc_dir, rbm_dir)
            if mc is None:
                continue
            r_max = L // 2
            C_mc = correlation_function(mc, r_max)
            C_rbm = correlation_function(rbm, r_max)
            r = np.arange(r_max + 1)
            ax.plot(r, C_mc, "o--", color=col, alpha=0.6,
                    label=f"T = {T:.2f}  MC", markersize=4)
            ax.plot(r, C_rbm, "s-", color=col,
                    label=f"T = {T:.2f}  RBM", markersize=4)
        ax.set_xlabel("$r$")
        ax.set_ylabel("$C(r)$")
        ax.set_title(f"L = {L}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/rbm.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc_dir = ROOT / cfg["data_dir"]
    rbm_dir = ROOT / cfg["results_dir"]
    Ls = cfg["Ls"]

    print("\n--- Fig 5 equivalent: <eps>(T) and <|M|>(T) ---")
    summary = plot_energy_and_mag(Ls, mc_dir, rbm_dir, rbm_dir / "fig_energy_mag.png")

    print("\n--- Fig 6 equivalent: P(E) at low and high T ---")
    # The paper shows L=12 at T=1 and T=3; we plot the largest L we have
    L_for_PE = max(Ls)
    plot_PE(L_for_PE, [1.0, 3.0], mc_dir, rbm_dir, rbm_dir / "fig_PE.png")

    print("\n--- Fig 7 equivalent: C(r) at several T ---")
    plot_correlations(Ls, [1.0, 2.0, cfg["T_c"], 3.0],
                      mc_dir, rbm_dir, rbm_dir / "fig_correlations.png")

    # Save summary numerically too
    with open(rbm_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {rbm_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
