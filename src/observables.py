"""
Per-configuration observables used for comparing MC and RBM ensembles.

All functions take a (N, L, L) int8 array of spin configurations and return
either per-configuration arrays (so distributions and averages can both be
computed) or aggregated quantities.
"""
from __future__ import annotations

import numpy as np


def energy_per_config(configs: np.ndarray) -> np.ndarray:
    """
    Energy density per configuration, eps = E / L^2 with
        E = -sum_<ij> S_i S_j   (PBC, J = 1).

    Returns shape (N,) float64.
    """
    s = configs.astype(np.int32)
    right = np.roll(s, -1, axis=2)
    down = np.roll(s, -1, axis=1)
    E_total = -(s * right + s * down).sum(axis=(1, 2))
    L = configs.shape[1]
    return E_total.astype(np.float64) / (L * L)


def magnetization_per_config(configs: np.ndarray) -> np.ndarray:
    """Magnetization density per configuration, M = (1/L^2) sum_ij S_ij."""
    L = configs.shape[1]
    return configs.astype(np.int32).sum(axis=(1, 2)) / (L * L)


def correlation_function(configs: np.ndarray, r_max: int | None = None
                          ) -> np.ndarray:
    """
    Space-dependent correlation function as defined in Eq. (5) of Azizi-Pleimling:

        C(r) = < (1/(2L^2)) * sum_{i,j} S_{i,j} * (S_{i+r,j} + S_{i,j+r}) >

    where < > is the ensemble average over configurations.

    Returns shape (r_max + 1,) float64 with C[0] == 1 by construction.
    """
    L = configs.shape[1]
    if r_max is None:
        r_max = L // 2
    s = configs.astype(np.float32)
    C = np.zeros(r_max + 1, dtype=np.float64)
    for r in range(r_max + 1):
        right = np.roll(s, -r, axis=2)
        down = np.roll(s, -r, axis=1)
        prod = 0.5 * (s * right + s * down)
        # spatial mean per config, then ensemble mean
        C[r] = prod.mean()
    return C


def energy_histogram(configs: np.ndarray, n_bins: int = 60,
                     E_range: tuple[float, float] | None = None,
                     align_to_lattice: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Distribution of the energy density.

    If align_to_lattice is True (default), return one point per *visited*
    integer-E value with its empirical probability. This is exact and
    artifact-free, but the number of returned points depends on the data.

    If align_to_lattice is False, use a standard fixed-bin histogram.

    Returns (centers, probabilities) such that probabilities sum to 1
    (or empty arrays if no data in range).
    """
    L = configs.shape[1]
    L2 = L * L
    E = energy_per_config(configs)

    if align_to_lattice:
        E_int = np.rint(E * L2).astype(np.int64)
        if E_range is not None:
            mask = (E >= E_range[0]) & (E <= E_range[1])
            E_int = E_int[mask]
        if len(E_int) == 0:
            return np.array([]), np.array([])
        unique, counts = np.unique(E_int, return_counts=True)
        centers = unique.astype(np.float64) / L2
        probs = counts / counts.sum()
        return centers, probs

    if E_range is None:
        E_range = (E.min(), E.max())
    hist, edges = np.histogram(E, bins=n_bins, range=E_range, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs = hist / hist.sum() if hist.sum() > 0 else hist
    return centers, probs


__all__ = [
    "energy_per_config",
    "magnetization_per_config",
    "correlation_function",
    "energy_histogram",
]