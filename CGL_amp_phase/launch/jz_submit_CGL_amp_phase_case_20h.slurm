#!/bin/bash
#SBATCH --job-name=CGLA_CASE
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3

module purge
module load pytorch-gpu/py3/2.1.1

PROJECT_DIR="$WORK/These_DeepOnet_CGL/CGL_amp_phase"
cd "$PROJECT_DIR" || { echo "❌ Impossible d'accéder à $PROJECT_DIR"; exit 1; }

export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p slurm/log

CONFIG_PATH="${CONFIG_PATH:-configs/cgl_case_alpha075_beta0_mu0_tchar.yaml}"
RESUME_DIR="${RESUME_DIR:-}"

echo "🚀 Job lancé sur le noeud : $SLURMD_NODENAME"
echo "📂 Dossier de travail : $PWD"
echo "🧾 Config : $CONFIG_PATH"
echo "🔄 Resume : ${RESUME_DIR:-<none>}"
nvidia-smi

if [ -n "$RESUME_DIR" ]; then
  srun python scripts/train_cgl_amp_phase.py --resume "$RESUME_DIR" --config "$CONFIG_PATH"
else
  srun python scripts/train_cgl_amp_phase.py --config "$CONFIG_PATH"
fi
