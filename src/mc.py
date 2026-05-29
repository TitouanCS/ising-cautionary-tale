"""
Monte Carlo simulation of the 2D Ising model with conserved magnetization
(Kawasaki spin-exchange dynamics). Numba-accelerated.

Conventions:
  - spins[i, j] in {-1, +1}, shape (L, L), periodic BCs
  - Hamiltonian: H = -sum_{<ij>} S_i S_j  (coupling J = 1, kB = 1)
  - One MC sweep = L*L attempted nearest-neighbor exchanges
  - Magnetization is conserved exactly (Kawasaki preserves sum of spins)
"""
from __future__ import annotations

import numpy as np
from numba import njit


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------

def init_config(L: int, M0: float, rng: np.random.Generator,
                mode: str = "random") -> np.ndarray:
    """
    Build an initial L x L configuration with magnetization density exactly M0.

    Modes
    -----
    "random" (default):
        Shuffle a flat array with the correct number of +1 and -1 spins.
        Used at high T or when no prior assumption on the ground state is wanted.

    "stripes":
        Initialize directly into the (approximate) low-T ground state for
        M0 = 0 on a square with PBC: two horizontal bands, top L//2 rows are
        +1 and bottom L//2 rows are -1. Useful at low T to avoid trapping in
        local minima of Kawasaki dynamics (e.g. 5+7 stripes instead of 6+6).
        Requires M0 = 0 and even L for exact magnetization. Falls back to
        "random" otherwise.

    M0 must be such that (1 + M0) * L^2 / 2 is an integer.
    """
    N = L * L
    n_plus = int(round((1.0 + M0) * N / 2.0))
    if n_plus < 0 or n_plus > N:
        raise ValueError(f"M0={M0} is out of range for L={L}")

    if mode == "stripes" and abs(M0) < 1e-9 and L % 2 == 0:
        spins = np.ones((L, L), dtype=np.int8)
        spins[L // 2:] = -1
        return spins

    if mode not in ("random", "stripes"):
        raise ValueError(f"Unknown init mode '{mode}', expected 'random' or 'stripes'")

    flat = np.empty(N, dtype=np.int8)
    flat[:n_plus] = 1
    flat[n_plus:] = -1
    rng.shuffle(flat)
    return flat.reshape(L, L).copy()


# ----------------------------------------------------------------------
# Energy and magnetization (host-side, vectorized)
# ----------------------------------------------------------------------

def total_energy(spins: np.ndarray) -> float:
    """
    Total energy H = -sum_<ij> S_i S_j with periodic BCs.
    """
    s = spins.astype(np.int32)
    right = np.roll(s, -1, axis=1)
    down = np.roll(s, -1, axis=0)
    return float(-(s * right).sum() - (s * down).sum())


def energy_density(spins: np.ndarray) -> float:
    L = spins.shape[0]
    return total_energy(spins) / (L * L)


def magnetization_density(spins: np.ndarray) -> float:
    return float(spins.sum()) / spins.size


# ----------------------------------------------------------------------
# Kawasaki step (Numba)
# ----------------------------------------------------------------------
# We pick a random site and one of 4 directions, then attempt to swap
# with that neighbor. If both spins are equal, the move is a no-op and
# we still count it as an attempt (this is the standard convention and
# is the simplest correct algorithm; it satisfies detailed balance).
#
# Delta E for an exchange of two opposite neighboring spins s1, s2 = -s1:
#   dE = 2 * s1 * (nb1 - nb2)
# where nb1 = sum of neighbors of site 1 EXCLUDING site 2,
#       nb2 = sum of neighbors of site 2 EXCLUDING site 1.
# Derivation: swapping (s1, s2) with s1 != s2 is equivalent to flipping
# both spins. The bond between them stays fixed. The "outer" neighbor
# contributions yield 2 s1 nb1 + 2 s2 nb2 = 2 s1 (nb1 - nb2).


@njit(cache=True, fastmath=True)
def _kawasaki_sweep(spins: np.ndarray, beta: float, L: int) -> None:
    """One MC sweep: L*L attempted exchanges. Modifies spins in place."""
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        d = np.random.randint(0, 4)
        if d == 0:
            i2, j2 = i, (j + 1) % L
        elif d == 1:
            i2, j2 = (i + L - 1) % L, j
        elif d == 2:
            i2, j2 = i, (j + L - 1) % L
        else:
            i2, j2 = (i + 1) % L, j

        s1 = spins[i, j]
        s2 = spins[i2, j2]
        if s1 == s2:
            continue  # no-op exchange

        # Sums of neighbors EXCLUDING the partner site
        nb1 = (
            spins[(i + 1) % L, j]
            + spins[(i + L - 1) % L, j]
            + spins[i, (j + 1) % L]
            + spins[i, (j + L - 1) % L]
        ) - s2  # subtract the partner contribution
        nb2 = (
            spins[(i2 + 1) % L, j2]
            + spins[(i2 + L - 1) % L, j2]
            + spins[i2, (j2 + 1) % L]
            + spins[i2, (j2 + L - 1) % L]
        ) - s1

        dE = 2 * s1 * (nb1 - nb2)
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] = s2
            spins[i2, j2] = s1


@njit(cache=True)
def _run_sweeps(spins: np.ndarray, beta: float, n_sweeps: int) -> None:
    """Run multiple MC sweeps on `spins` in place."""
    L = spins.shape[0]
    for _ in range(n_sweeps):
        _kawasaki_sweep(spins, beta, L)


# ----------------------------------------------------------------------
# Sampling driver
# ----------------------------------------------------------------------

def generate_configurations(
    L: int,
    T: float,
    M0: float,
    n_configs: int,
    n_thermalize: int,
    n_decorrelate: int,
    seed: int,
    return_energies: bool = True,
    init_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Generate `n_configs` configurations at temperature T on an L x L lattice
    with conserved magnetization density M0.

    Parameters
    ----------
    init_mode : "random" or "stripes"
        How to seed the MC chain. "stripes" places the system directly into
        the M0=0 ground state (two bands of width L/2) which avoids local
        minima at low T. See `init_config` for details.

    Returns
    -------
    configs : np.ndarray, shape (n_configs, L, L), dtype int8
    energies : np.ndarray, shape (n_configs,), dtype float32, or None
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    beta = 1.0 / T
    spins = init_config(L, M0, rng, mode=init_mode)

    _run_sweeps(spins, beta, n_thermalize)

    configs = np.empty((n_configs, L, L), dtype=np.int8)
    energies = np.empty(n_configs, dtype=np.float32) if return_energies else None

    for k in range(n_configs):
        _run_sweeps(spins, beta, n_decorrelate)
        configs[k] = spins
        if energies is not None:
            energies[k] = total_energy(spins) / (L * L)

    return configs, energies


# ----------------------------------------------------------------------
# Decorrelation heuristic
# ----------------------------------------------------------------------

def decorrelation_sweeps(L: int, T: float, T_c: float = 2.269,
                         base: int = 30, z: float = 2.17) -> int:
    """
    Heuristic for the number of MC sweeps between snapshots.

    Far from T_c, base sweeps are usually enough. Near T_c we scale up by
    a critical-slowing-down factor L^z (Kawasaki dynamic exponent ~2.17).
    The transition between the two regimes is smooth via a Gaussian bump.
    """
    width = 0.25  # temperature width of the critical region
    bump = np.exp(-((T - T_c) / width) ** 2)
    factor = 1.0 + bump * (L ** z) / (8.0 ** z)  # normalize so L=8 -> factor=2 at peak
    return max(int(base), int(base * factor))


__all__ = [
    "init_config",
    "total_energy",
    "energy_density",
    "magnetization_density",
    "generate_configurations",
    "decorrelation_sweeps",
]