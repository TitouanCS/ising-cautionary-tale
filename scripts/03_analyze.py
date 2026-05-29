#!/usr/bin/env python
"""
Analyze CNN outputs: plot <p_ordered>(T) for all L, fit T_c and nu by
data collapse, and produce the rescaled finite-size scaling plot.

Outputs:
  data/results/fig_pT.png        --- raw <p>(T, L) curves with inset
  data/results/fig_collapse.png  --- data collapse (T - T_c) L^{1/nu}
  data/results/summary.json      --- fitted parameters

Usage:
    python scripts/03_analyze.py --config configs/default.yaml
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

from src.analysis import estimate_Tc_from_crossing, fit_collapse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = ROOT / cfg["results_dir"]
    T_c_exact = cfg["T_c"]

    # Load CNN curves
    data_by_L: dict[int, dict[str, np.ndarray]] = {}
    for L in cfg["Ls"]:
        path = results_dir / f"cnn_output_L{L}.npz"
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        d = np.load(path)
        data_by_L[L] = {
            "T": d["T"], "p_mean": d["p_mean"], "p_std": d["p_std"],
            "test_acc": float(d["test_acc"]),
        }

    if not data_by_L:
        raise SystemExit("No CNN outputs found.")

    # ---------------- Figure 1: <p>(T) curves + zoomed inset ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(data_by_L)))
    for (L, d), c in zip(sorted(data_by_L.items()), colors):
        ax.errorbar(d["T"], d["p_mean"], yerr=d["p_std"], fmt="o-",
                    label=f"L = {L}  (acc = {d['test_acc']:.3f})",
                    color=c, capsize=2, markersize=4)
    ax.axvline(T_c_exact, color="red", lw=1, ls="--",
               label=f"$T_c$ Onsager = {T_c_exact:.4f}")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel(r"$\langle p_{\rm ordered} \rangle$")
    ax.set_title("CNN output vs. temperature (test set)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Inset: zoom around T_c
    axin = ax.inset_axes([0.55, 0.55, 0.42, 0.4])
    for (L, d), c in zip(sorted(data_by_L.items()), colors):
        axin.plot(d["T"], d["p_mean"], "o-", color=c, markersize=3)
    axin.axvline(T_c_exact, color="red", lw=1, ls="--")
    axin.axhline(0.5, color="gray", lw=0.5, ls=":")
    axin.set_xlim(T_c_exact - 0.4, T_c_exact + 0.4)
    axin.set_ylim(0.0, 1.0)
    axin.set_title("zoom near $T_c$", fontsize=9)
    axin.tick_params(labelsize=8)

    fig.tight_layout()
    out_pT = results_dir / "fig_pT.png"
    fig.savefig(out_pT, dpi=160)
    print(f"Saved {out_pT}")

    # ---------------- T_c estimates from 0.5-crossing ----------------
    Tc_estimates = {}
    for L, d in sorted(data_by_L.items()):
        Tc_estimates[L] = estimate_Tc_from_crossing(d["T"], d["p_mean"])
        print(f"  T*(L={L})  where <p>=0.5  =>  {Tc_estimates[L]:.4f}")
    print(f"  Exact T_c (Onsager)          =>  {T_c_exact:.4f}")

    # ---------------- Fit T_c, nu by data collapse ----------------
    T_by_L = {L: d["T"] for L, d in data_by_L.items()}
    p_by_L = {L: d["p_mean"] for L, d in data_by_L.items()}
    T_c_fit, nu_fit, residual = fit_collapse(T_by_L, p_by_L,
                                              T_c0=T_c_exact, nu0=1.0)
    print(f"\nData collapse fit:")
    print(f"  T_c   = {T_c_fit:.4f}  (exact: {T_c_exact:.4f})")
    print(f"  nu    = {nu_fit:.4f}  (exact: 1.0)")
    print(f"  resid = {residual:.5f}")

    # ---------------- Figure 2: collapsed curves ----------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for (L, d), c in zip(sorted(data_by_L.items()), colors):
        x = (d["T"] - T_c_fit) * (L ** (1.0 / nu_fit))
        ax2.plot(x, d["p_mean"], "o-", label=f"L = {L}", color=c, markersize=4)
    ax2.set_xlabel(r"$(T - T_c)\, L^{1/\nu}$")
    ax2.set_ylabel(r"$\langle p_{\rm ordered} \rangle$")
    ax2.set_title(
        f"Finite-size scaling collapse  "
        f"($T_c$ = {T_c_fit:.4f}, $\\nu$ = {nu_fit:.3f})"
    )
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out_collapse = results_dir / "fig_collapse.png"
    fig2.savefig(out_collapse, dpi=160)
    print(f"Saved {out_collapse}")

    # Save summary
    summary = {
        "T_c_exact": T_c_exact,
        "T_c_fit": T_c_fit,
        "nu_fit": nu_fit,
        "collapse_residual": residual,
        "Tc_from_crossing": {str(L): float(v) for L, v in Tc_estimates.items()},
        "test_acc": {str(L): d["test_acc"] for L, d in data_by_L.items()},
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
