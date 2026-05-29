#!/usr/bin/env python
"""
Generate Monte Carlo configurations for the 2D Ising model with conserved
magnetization, using Kawasaki dynamics.

For each (L, T) pair, writes a separate .npz file:
    {data_dir}/L{L}_T{T:.4f}.npz
containing keys 'configs' (N, L, L) int8 and 'energies' (N,) float32.

Three execution modes:

1. Single task (debugging or SLURM array):
    python scripts/01_generate_mc.py --config configs/default.yaml --task-id 17

2. Sequential, all tasks (slow):
    python scripts/01_generate_mc.py --config configs/default.yaml

3. Parallel within a single process (recommended on DCE):
    python scripts/01_generate_mc.py --config configs/default.yaml --parallel 16

To enumerate task indices:
    python scripts/01_generate_mc.py --config configs/default.yaml --list-tasks
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mc import generate_configurations, decorrelation_sweeps  # noqa: E402


def build_task_list(cfg: dict) -> list[tuple[int, float]]:
    Ls = cfg["Ls"]
    T_grid = np.linspace(cfg["T_min"], cfg["T_max"], cfg["n_T"])
    return [(L, float(T)) for L in Ls for T in T_grid]


def run_one(L: int, T: float, cfg: dict, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"L{L}_T{T:.4f}.npz"
    if out_path.exists():
        return f"[skip] {out_path.name} (exists)"

    decorr = decorrelation_sweeps(
        L=L, T=T, T_c=cfg["T_c"], base=cfg["decorrelation_base"]
    )
    seed = abs(hash((L, round(T, 4), cfg["seed"]))) % (2**31 - 1)
    init_mode = cfg.get("init_mode", "random")

    t0 = time.time()
    configs, energies = generate_configurations(
        L=L,
        T=T,
        M0=cfg["M0"],
        n_configs=cfg["n_configs_per_T"],
        n_thermalize=cfg["n_thermalize_sweeps"],
        n_decorrelate=decorr,
        seed=seed,
        return_energies=True,
        init_mode=init_mode,
    )
    dt = time.time() - t0

    np.savez_compressed(
        out_path,
        configs=configs,
        energies=energies,
        L=L,
        T=T,
        M0=cfg["M0"],
        decorrelate=decorr,
        seed=seed,
    )

    # Sanity checks
    M_actual = configs.sum(axis=(1, 2)) / (L * L)
    assert np.allclose(M_actual, cfg["M0"], atol=1e-6), \
        f"Magnetization not conserved! Got {M_actual[:5]}..."

    return (f"[done] L={L} T={T:.4f}  n={len(configs)}  decorr={decorr}sw  "
            f"<eps>={energies.mean():+.4f}  time={dt:.1f}s")


def _worker(args: tuple) -> str:
    """Worker function for ProcessPoolExecutor."""
    L, T, cfg, out_dir = args
    return run_one(L, T, cfg, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--task-id", type=int, default=None,
                    help="If given, run only this index from the task list")
    ap.add_argument("--list-tasks", action="store_true",
                    help="Print the task list and exit")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Number of parallel worker processes (default 1)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tasks = build_task_list(cfg)
    if args.list_tasks:
        for i, (L, T) in enumerate(tasks):
            print(f"{i:3d}  L={L}  T={T:.4f}")
        print(f"\nTotal: {len(tasks)} tasks")
        return

    out_dir = ROOT / cfg["data_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task_id is not None:
        if not (0 <= args.task_id < len(tasks)):
            raise SystemExit(f"task-id {args.task_id} out of range [0, {len(tasks)})")
        L, T = tasks[args.task_id]
        print(run_one(L, T, cfg, out_dir), flush=True)
        return

    if args.parallel <= 1:
        # Sequential
        t_start = time.time()
        for L, T in tasks:
            print(run_one(L, T, cfg, out_dir), flush=True)
        print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")
        return

    # Parallel
    n_workers = min(args.parallel, len(tasks))
    print(f"Running {len(tasks)} tasks on {n_workers} worker processes "
          f"(host = {os.uname().nodename})", flush=True)
    t_start = time.time()
    # We schedule the heaviest tasks (large L, T near T_c) first so the
    # slowest ones don't trail behind at the end.
    heavy_first = sorted(
        tasks,
        key=lambda LT: -(LT[0] ** 2) * decorrelation_sweeps(LT[0], LT[1], cfg["T_c"]),
    )
    work = [(L, T, cfg, out_dir) for (L, T) in heavy_first]
    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_worker, w) for w in work]
        for fut in as_completed(futures):
            msg = fut.result()
            n_done += 1
            print(f"  [{n_done:3d}/{len(work)}] {msg}", flush=True)
    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()