#!/bin/bash
#SBATCH --job-name=CGL_1HARD_R15
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

echo "🚀 Job lancé sur le noeud : $SLURMD_NODENAME"
echo "📂 Dossier de travail : $PWD"
nvidia-smi

srun python scripts/train_cgl_resume.py \
  --resume /lustre/fswork/projects/rech/fdb/usv13rn/These_DeepOnet_CGL/results/CGL_Navigator_Run_20260330-152012/checkpoints/ckpt_t0.6500.pth \
  --config configs/cgl_single_case_hard_resume_t15.yaml
