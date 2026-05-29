#!/bin/bash
#SBATCH --job-name=ising_mc
#SBATCH --output=logs/mc_%j.out
#SBATCH --error=logs/mc_%j.err
#SBATCH --partition=cpu_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # all physical cores on a kyle node (SkyLake)
#SBATCH --mem=32G
#SBATCH --time=08:00:00           # safe upper bound; real run ~30 min - 2h

# Single-job design: max jobs/user = 1 on the DCE means no SLURM arrays.
# We grab a whole kyle node and parallelize the 90 (L, T) tasks across its
# 16 physical cores using Python multiprocessing.

set -euo pipefail
mkdir -p logs

# --- Environment ---
# Edit these lines to match your DCE setup. Two common patterns:
#
# (a) Conda / module-based environment:
# module load anaconda3
# source activate ising
#
# (b) Local venv:
# source ~/venvs/ising/bin/activate

cd "${SLURM_SUBMIT_DIR}"

# Disable BLAS / OpenMP threading inside workers — each worker is single-
# threaded and we want all parallelism to come from multiprocessing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Use the actual number of cores SLURM allocated
N_WORKERS="${SLURM_CPUS_PER_TASK:-16}"

echo "Host: $(hostname)"
echo "Workers: ${N_WORKERS}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "---"

python scripts/01_generate_mc.py \
    --config configs/default.yaml \
    --parallel "${N_WORKERS}"
