#!/bin/bash
#SBATCH --job-name=CGL_LM_CS
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
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH doit etre fourni}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"

cd "$PROJECT_DIR" || { echo "❌ Impossible d'accéder à $PROJECT_DIR"; exit 1; }

export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p slurm/log
mkdir -p .mplconfig
export MPLCONFIGDIR=$PWD/.mplconfig

echo "🚀 Job lancé sur le noeud : $SLURMD_NODENAME"
echo "📂 Dossier de travail : $PWD"
echo "🧾 Config : $CONFIG_PATH"
echo "🔄 Resume : ${RESUME_RUN_DIR:-<none>}"
nvidia-smi

if [ -n "$RESUME_RUN_DIR" ]; then
  srun python scripts/train_cgl_local_physics_mononet_amp_phase.py \
    --config "$CONFIG_PATH" \
    --resume "$RESUME_RUN_DIR"
else
  srun python scripts/train_cgl_local_physics_mononet_amp_phase.py \
    --config "$CONFIG_PATH"
fi
