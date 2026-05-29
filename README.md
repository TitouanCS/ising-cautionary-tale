# Ising-CNN — Réimplémentation de la partie CNN de Azizi & Pleimling (2021)

Reproduction de la partie « apprentissage supervisé » de l'article *A cautionary
tale for machine learning generated configurations in presence of a conserved
quantity* (Sci. Rep. 11:6395, 2021).

On entraîne un CNN à classifier des configurations du modèle d'Ising 2D avec
magnétisation conservée ($M_0 = 0$, dynamique de Kawasaki) en deux classes :
phase ordonnée ($T < T_c$) ou désordonnée ($T > T_c$). On montre ensuite que
la sortie moyenne du CNN, $\langle p_{\text{ordonné}} \rangle(T)$, exhibe un
*finite-size scaling* gouverné par l'exposant critique $\nu = 1$ de l'Ising 2D.

---

## Pipeline

1. **Génération MC** (Numba) : pour chaque $(L, T)$, on tire $N$ configurations
   avec une chaîne Kawasaki (échange de spins voisins, accept/reject Metropolis).
   La magnétisation est exactement conservée. Décorrélation entre snapshots
   ajustée près de $T_c$ pour tenir compte du *critical slowing down*.

2. **Entraînement CNN** (PyTorch) : un modèle par taille $L$. Architecture
   2 × (Conv 3×3 + MaxPool 2×2) + Dense → 2 (softmax). Padding circulaire pour
   respecter les conditions périodiques. Cross-entropy + Adam, early stopping
   sur la validation.

3. **Analyse** (SciPy + matplotlib) : tracé de $\langle p \rangle(T)$ par $L$,
   estimation de $T_c$ par le croisement à 0.5, fit conjoint de $T_c$ et $\nu$
   par minimisation de la dispersion *collapse* sur la variable rééchelonnée
   $(T - T_c) L^{1/\nu}$.

---

## Arborescence

```
ising_cnn/
├── README.md
├── requirements.txt
├── configs/default.yaml         # hyperparamètres physiques et d'entraînement
├── src/
│   ├── mc.py                    # Monte Carlo Kawasaki (Numba)
│   ├── model.py                 # CNN
│   ├── dataset.py               # PyTorch Datasets
│   ├── train.py                 # boucle d'entraînement
│   └── analysis.py              # finite-size scaling
├── scripts/
│   ├── 00_smoke_test.py         # test rapide bout-en-bout (1 min sur laptop)
│   ├── 01_generate_mc.py        # génération MC (parallélisable SLURM array)
│   ├── 02_train_cnn.py          # entraînement CNN
│   ├── 03_analyze.py            # plots + fit collapse
│   └── slurm/
│       ├── gen_mc.sbatch
│       └── train.sbatch
└── data/                        # créé à l'exécution
    ├── raw/                     # configurations MC (.npz)
    └── results/                 # poids CNN, courbes, figures
```

---

## Utilisation

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Smoke test (local, ~1 minute)

```bash
python scripts/00_smoke_test.py
```

À utiliser **avant** de soumettre quoi que ce soit sur le DCE. Vérifie que la
chaîne MC tourne, que la magnétisation est conservée, et que le CNN apprend
une sigmoïde dans la bonne plage.

### Sur le DCE — flux complet

Le DCE impose **max 1 job par utilisateur**, donc pas de SLURM array. On
utilise un seul job par étape, qui parallélise en interne.

Partitions utilisées :
- **`cpu_prod`** (nœuds `kyle`, SkyLake 2×8 cœurs, 64 GB, walltime 12h) → MC
- **`cpu_inter`** (walltime 2h, autorise `srun`/`salloc`) → smoke test interactif

1. **Lister les tâches MC** pour vérifier la configuration (3 tailles × 30 températures = 90 tâches) :
   ```bash
   python scripts/01_generate_mc.py --list-tasks
   ```

2. **Soumettre le job MC** (un seul, ~30 min - 2h walltime selon la machine) :
   ```bash
   sbatch scripts/slurm/gen_mc.sbatch
   ```
   Le script demande 16 cœurs sur un nœud `kyle` et lance 16 tâches MC en
   parallèle via `multiprocessing.Pool`. Les tâches les plus lourdes (grand
   $L$, $T$ proche de $T_c$) sont schedulées en premier pour minimiser le
   *tail effect*.

3. **Soumettre le training** (une fois la MC terminée) :
   ```bash
   sbatch scripts/slurm/train.sbatch
   ```
   Entraîne les 3 CNN séquentiellement sur CPU (~15 min total). Si tu veux
   utiliser un GPU, change `--partition` et ajoute `--gres=gpu:1` dans le
   sbatch — PyTorch détecte le GPU automatiquement.

4. **Analyse** (peut tourner sur le frontend ou en local après rapatriement) :
   ```bash
   python scripts/03_analyze.py --config configs/default.yaml
   ```

### Smoke test interactif sur le DCE

Si tu veux valider le pipeline avant le gros run :
```bash
salloc --partition=cpu_inter --cpus-per-task=4 --time=00:30:00
# une fois sur le nœud :
python scripts/00_smoke_test.py
```

### Sortie attendue

Dans `data/results/` :
- `fig_pT.png` — courbes $\langle p \rangle(T)$ pour les trois $L$, avec zoom
  autour de $T_c$.
- `fig_collapse.png` — collapse $\langle p \rangle$ vs $(T - T_c) L^{1/\nu}$.
  Les trois courbes doivent se superposer.
- `summary.json` — $T_c$ et $\nu$ fittés, accuracies de test.

Valeurs attendues à la convergence :
- $T_c^{\text{fit}} \approx 2.27$ (vs. Onsager exact $\approx 2.2692$)
- $\nu \approx 1.0$
- accuracy test $\geq 0.97$ pour $L \geq 30$

---

## Estimation du coût (DCE, partition cpu_prod sur kyle)

Avec `n_configs_per_T = 20000` (au lieu des $10^5$ du papier) :

| $L$  | Temps MC / température (1 cœur Numba, kyle) | × 30 températures (séquentiel) |
|------|----------------------------------------------|------------------------------------|
| 20   | ~10 s                                        | ~5 min                             |
| 30   | ~1 min                                       | ~25 min                            |
| 40   | ~5 min                                       | ~1h30                              |

Avec **16 cœurs en parallèle** sur un nœud `kyle` (multiprocessing) : walltime
total ~**15-30 min** suivant la charge réelle du critical slowing down.

Entraînement CNN : quelques minutes par $L$ sur CPU 8 cœurs, plus rapide si tu
réquisitionnes un GPU à la place.

---

## Choix d'implémentation par rapport au papier

| Choix                              | Papier | Cette implémentation |
|------------------------------------|--------|----------------------|
| Configs par $(L, T)$               | $10^5$ | $2 \times 10^4$ (config) |
| Tailles $L$                        | 20, 30, 40, 50 | 20, 30, 40 (config) |
| Grille de $T$                      | 23 points $[1.0, 3.2]$ | 30 points $[1.0, 3.5]$ (config) |
| Padding conv                       | non précisé | circulaire (périodique) |
| Décorrélation entre snapshots      | non précisé | adaptative près de $T_c$ |
| Critère d'arrêt                    | 3 epochs sans amélioration val | idem |

---

## Sanity checks intégrés

- `01_generate_mc.py` vérifie que toutes les configurations générées ont bien
  la magnétisation cible (échoue si Kawasaki est mal codé).
- `00_smoke_test.py` reproduit le pipeline complet sur $L = 8$, ce qui est
  suffisant pour valider la chaîne sans coûter cher.
- Énergie moyenne calculée et écrite dans chaque `.npz` — on peut comparer aux
  valeurs connues (par exemple energy density $\to -2$ à $T \to 0$ pour
  $M_0 = 0$ avec interfaces, et $\to 0$ à $T \to \infty$).
