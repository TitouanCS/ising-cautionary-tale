"""
Finite-size scaling analysis: data collapse of <p_ordered>(T, L) onto a
universal curve f((T - T_c) * L^(1/nu)).

We provide:
  - estimate_Tc_from_crossing: find T such that <p>(T) = 0.5 by linear interp
  - fit_collapse: optimize T_c and nu by minimizing the spread of the
    collapsed curves
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def estimate_Tc_from_crossing(T_arr: np.ndarray, p_arr: np.ndarray,
                              target: float = 0.5) -> float:
    """Linear interpolation to find T at which p crosses `target`."""
    # Sort by T
    order = np.argsort(T_arr)
    T_arr = T_arr[order]
    p_arr = p_arr[order]
    # Find first sign change of (p - target)
    diff = p_arr - target
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            # Linear interp between T_arr[i] and T_arr[i+1]
            denom = diff[i + 1] - diff[i]
            if abs(denom) < 1e-12:
                return 0.5 * (T_arr[i] + T_arr[i + 1])
            return T_arr[i] - diff[i] * (T_arr[i + 1] - T_arr[i]) / denom
    # No crossing: return NaN
    return float("nan")


def _collapse_spread(params: np.ndarray, T_by_L: dict[int, np.ndarray],
                     p_by_L: dict[int, np.ndarray]) -> float:
    """
    Symmetric pairwise collapse cost.

    For each pair of lattice sizes (L1, L2) and each data point (T, p) of L1
    whose rescaled coordinate x falls inside the x-range of L2, we add the
    squared difference between p_L1 and p_L2 interpolated at the same x.
    We do the same with L1 and L2 swapped.

    Using actual data points (rather than an artificial dense grid) avoids
    a bias where smoother interpolations would artificially lower the cost.
    """
    T_c, inv_nu = params
    if inv_nu <= 0 or T_c < 0:
        return 1e6

    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L, T in T_by_L.items():
        x = (T - T_c) * (L ** inv_nu)
        order = np.argsort(x)
        curves[L] = (x[order], p_by_L[L][order])

    Ls = list(curves.keys())
    cost = 0.0
    count = 0
    for i, L1 in enumerate(Ls):
        x1, p1 = curves[L1]
        for L2 in Ls[i + 1:]:
            x2, p2 = curves[L2]
            # Evaluate L2's curve at the x-positions of L1's data points,
            # restricted to L2's x-range.
            in_range = (x1 >= x2[0]) & (x1 <= x2[-1])
            if in_range.sum() >= 3:
                p2_interp = np.interp(x1[in_range], x2, p2)
                cost += np.mean((p1[in_range] - p2_interp) ** 2)
                count += 1
            # And vice versa.
            in_range = (x2 >= x1[0]) & (x2 <= x1[-1])
            if in_range.sum() >= 3:
                p1_interp = np.interp(x2[in_range], x1, p1)
                cost += np.mean((p2[in_range] - p1_interp) ** 2)
                count += 1
    if count == 0:
        return 1e6
    return cost / count


def fit_collapse(T_by_L: dict[int, np.ndarray], p_by_L: dict[int, np.ndarray],
                 T_c0: float = 2.27, nu0: float = 1.0,
                 multistart: bool = True
                 ) -> tuple[float, float, float]:
    """
    Fit T_c and nu by minimizing the collapse spread.

    Uses multistart Nelder-Mead to avoid local minima: tries several
    (T_c, nu) starting points around the user-provided initial guess.

    Returns
    -------
    T_c, nu, residual
    """
    best = (T_c0, nu0, float("inf"))

    if multistart:
        starts = [
            (T_c0, 1.0),
            (T_c0, 0.8),
            (T_c0, 1.2),
            (T_c0 - 0.05, 1.0),
            (T_c0 + 0.05, 1.0),
        ]
    else:
        starts = [(T_c0, nu0)]

    for Tc_init, nu_init in starts:
        x0 = np.array([Tc_init, 1.0 / nu_init])
        res = minimize(_collapse_spread, x0, args=(T_by_L, p_by_L),
                       method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 5000})
        if res.fun < best[2]:
            best = (float(res.x[0]), float(1.0 / res.x[1]), float(res.fun))
    return best


__all__ = ["estimate_Tc_from_crossing", "fit_collapse"]
