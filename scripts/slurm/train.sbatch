#!/bin/bash
#SBATCH --job-name=ising_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=cpu_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Trains one CNN per L sequentially. The model is tiny (~3-7k params),
# so CPU training takes a few minutes per L; total ~15 min walltime.
# If a GPU partition is available and you'd prefer that, change --partition
# to your GPU partition and add #SBATCH --gres=gpu:1; PyTorch will pick it
# up automatically (see TrainConfig.device).

set -euo pipefail
mkdir -p logs

# Activate your environment here, e.g.:
# source ~/venvs/ising/bin/activate

cd "${SLURM_SUBMIT_DIR}"

# Cap intra-op threading to avoid oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "---"

python scripts/02_train_cnn.py --config configs/default.yaml
