#!/bin/bash
#SBATCH --job-name=DeepONet_ADR      # Nom du job
#SBATCH --output=slurm/log/%x_%j.out # %x=nom du job, %j=ID du job
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100                  # <--- Ton compte projet
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # On reste sur 1 GPU pour l'instant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10           # Idéal pour charger les données
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00              # Max pour qos_gpu-t3
#SBATCH --qos=qos_gpu-t3

# Nettoyage et chargement des modules
module purge
module load pytorch-gpu/py3/2.1.1

# Création des dossiers de sortie sur Jean Zay
mkdir -p slurm/log
mkdir -p outputs/checkpoints

echo "🚀 Job lancé sur $SLURMD_NODENAME"
nvidia-smi

# --- Lancement avec le BON CHEMIN ---
# On appelle le script qui est dans le dossier scripts/
# On lui donne explicitement le chemin de la config
srun python scripts/train_resume_nlse.py --config configs/nlse.yaml
