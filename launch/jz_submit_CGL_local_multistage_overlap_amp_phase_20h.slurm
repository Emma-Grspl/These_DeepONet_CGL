#!/bin/bash
#SBATCH --job-name=CGL_LMS_OV
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

PROJECT_DIR="$WORK/These_DeepOnet_CGL"
cd "$PROJECT_DIR" || { echo "❌ Impossible d'accéder à $PROJECT_DIR"; exit 1; }

export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p slurm/log

CONFIG_PATH="${CONFIG_PATH:-configs/cgl_single_case_local_multistage_overlap_amp_phase_alpha075_beta0_mu0_t1.yaml}"
RESUME_DIR="${RESUME_DIR:-}"

echo "🚀 Job lancé sur le noeud : $SLURMD_NODENAME"
echo "📂 Dossier de travail : $PWD"
echo "🧾 Config : $CONFIG_PATH"
echo "🔄 Resume : ${RESUME_DIR:-<none>}"
nvidia-smi

if [ -n "$RESUME_DIR" ]; then
  srun python scripts/train_cgl_local_multistage_amp_phase.py --config "$CONFIG_PATH" --resume "$RESUME_DIR"
else
  srun python scripts/train_cgl_local_multistage_amp_phase.py --config "$CONFIG_PATH"
fi
