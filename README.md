# Ising-CNN: a reproduction of Azizi & Pleimling (2021)

A reimplementation of the supervised-learning part of
*A Cautionary Tale for Machine Learning Generated Configurations in the
Presence of a Conserved Quantity* (A. Azizi and M. Pleimling, *Scientific
Reports* **11**:6395, 2021).

A convolutional neural network is trained to classify configurations of the
2D Ising model with **conserved magnetization** ($M_0 = 0$, Kawasaki
dynamics) into ordered ($T < T_c$) and disordered ($T > T_c$) phases.
The average CNN output $\langle p_{\text{ordered}}\rangle(T)$ is then shown
to exhibit a finite-size scaling collapse governed by the 2D Ising
correlation-length exponent $\nu = 1$.

---

## Pipeline

1. **Monte Carlo generation** (Numba). For each $(L, T)$, $N$ configurations
   are sampled by a Kawasaki Markov chain: nearest-neighbour spin exchanges
   with Metropolis accept/reject. Magnetization is conserved exactly.
   Decorrelation between snapshots is increased near $T_c$ to account for
   critical slowing down.

2. **CNN training** (PyTorch). One model per system size $L$. Architecture:
   2 × (Conv 3×3 + MaxPool 2×2) + Dense → 2 (softmax). Circular padding for
   the periodic boundary conditions. Cross-entropy loss, Adam optimizer,
   early stopping on validation accuracy.

3. **Analysis** (SciPy + matplotlib). Plot $\langle p\rangle(T)$ for each
   $L$, estimate $T_c$ from the $0.5$-crossing, and jointly fit $T_c$ and
   $\nu$ by minimizing the collapse dispersion on the rescaled variable
   $(T - T_c)L^{1/\nu}$.

---

## Repository structure

```
ising_cnn/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml             # physical and training hyperparameters
├── src/
│   ├── mc.py                    # Kawasaki Monte Carlo (Numba)
│   ├── model.py                 # CNN
│   ├── dataset.py               # PyTorch Datasets
│   ├── train.py                 # training loop
│   └── analysis.py              # finite-size scaling
├── scripts/
│   ├── 00_smoke_test.py         # quick end-to-end test (~1 min on laptop)
│   ├── 01_generate_mc.py        # MC generation
│   ├── 02_train_cnn.py          # CNN training
│   └── 03_analyze.py            # plots + collapse fit
└── data/                        # created at runtime
    ├── raw/                     # MC configurations (.npz)
    └── results/                 # CNN weights, curves, figures
```

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/TitouanCS/ising-cautionary-tale.git
cd ising-cautionary-tale
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick start — smoke test (~1 min)

Before running the full pipeline, validate the chain end-to-end on a small
system ($L = 8$):

```bash
python scripts/00_smoke_test.py
```

This checks that (i) the Kawasaki Monte Carlo conserves magnetization,
(ii) the CNN learns a sigmoid in the expected range, and (iii) all I/O
paths are correctly set up.

---

## Full pipeline

The three stages are run sequentially. All hyperparameters (system sizes,
temperature grid, number of configurations, training schedule) live in
`configs/default.yaml`.

### 1. Generate Monte Carlo configurations

```bash
python scripts/01_generate_mc.py --config configs/default.yaml
```

Generates configurations in parallel across CPU cores using
`multiprocessing.Pool`. The heaviest tasks (large $L$, $T$ near $T_c$)
are scheduled first to minimize tail effects. Output written to
`data/raw/`.

### 2. Train the CNNs

```bash
python scripts/02_train_cnn.py --config configs/default.yaml
```

Trains one CNN per system size sequentially. PyTorch will automatically
detect and use a GPU if available; otherwise it runs on CPU. Trained
weights and per-epoch metrics written to `data/results/`.

### 3. Produce figures and fit the collapse

```bash
python scripts/03_analyze.py --config configs/default.yaml
```

Outputs in `data/results/`:
- `fig_pT.png` — $\langle p\rangle(T)$ for each $L$, with a zoom around
  $T_c$.
- `fig_collapse.png` — collapse $\langle p\rangle$ vs.\
  $(T - T_c)L^{1/\nu}$ with the fitted exponents.
- `summary.json` — fitted $T_c$ and $\nu$, plus test accuracies.

Expected values at convergence:
- $T_c^{\text{fit}} \approx 2.27$ (exact Onsager value: $\approx 2.2692$)
- $\nu \approx 1.0$ for the asymptotic regime; finite-size effects on
  small $L$ may yield slightly lower values ($\sim 0.82$ for
  $L \in \{20, 30, 40\}$).
- test accuracy $\geq 0.97$ for $L \geq 30$.

---

## Runtime estimates (local)

With the default configuration (`n_configs_per_T = 20000`, 30 temperatures,
$L \in \{20, 30, 40\}$):

| Stage                  | Sequential (1 core) | Parallel (8 cores) |
|------------------------|---------------------|--------------------|
| MC generation, $L=20$  | ~5 min              | <1 min             |
| MC generation, $L=30$  | ~25 min             | ~4 min             |
| MC generation, $L=40$  | ~2.5 h              | ~20 min            |
| CNN training, all $L$  | ~15 min (CPU)       | a few min on GPU   |
| Analysis               | seconds             | seconds            |

Timings measured on a typical modern laptop (Intel i7-class). For a quick
look, reduce `n_configs_per_T` and the temperature grid in
`configs/default.yaml`; reproducibility of the collapse holds qualitatively
down to $\sim 5000$ configurations per $(L, T)$.

---

## Implementation choices vs.\ the paper

| Choice                         | Paper          | This implementation                   |
|--------------------------------|----------------|---------------------------------------|
| Configurations per $(L, T)$    | $10^5$         | $2 \times 10^4$ (configurable)        |
| System sizes $L$               | 20, 30, 40, 50 | 20, 30, 40 (configurable)             |
| Temperature grid               | 23 points in $[1.0, 3.2]$ | 30 points in $[1.0, 3.5]$ (configurable) |
| Convolutional padding          | unspecified    | circular (periodic)                   |
| Snapshot decorrelation         | unspecified    | adaptive near $T_c$                   |
| Stopping criterion             | 3 epochs without val improvement | same                |

---

## Sanity checks built into the pipeline

- `scripts/01_generate_mc.py` verifies that every generated configuration
  has the target magnetization — fails fast if the Kawasaki dynamics is
  miscoded.
- `scripts/00_smoke_test.py` reproduces the full pipeline on $L = 8$,
  sufficient to validate the chain without significant cost.
- Mean energy is computed and stored in every `.npz` and can be compared
  to limiting cases (energy density $\to -2$ as $T \to 0$ for $M_0 = 0$
  with straight interfaces, $\to 0$ as $T \to \infty$).

---

## Notes on the RBM part

The original paper has a second half on Restricted Boltzmann Machines,
which this repository does not currently include in production form.
Reproductions of the RBM results (energy density tracking, magnetization
drift, $P(E)$ pathology) used in the accompanying report were performed
in standalone notebooks and are available on request.

---

## Reference

A. Azizi and M. Pleimling, *A cautionary tale for machine learning generated
configurations in presence of a conserved quantity*, **Scientific Reports
11**:6395 (2021). [doi:10.1038/s41598-021-85683-8](https://doi.org/10.1038/s41598-021-85683-8)

---

## License

MIT.

