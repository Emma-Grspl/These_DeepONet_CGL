#!/bin/bash
#SBATCH --job-name=CGL      # Nom du job
#SBATCH --output=slurm/log/%x_%j.out # Log standard
#SBATCH --error=slurm/log/%x_%j.err  # Log d'erreur
#SBATCH -A fdb@v100                  # <--- VÉRIFIE CE CODE PROJET !
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                 # 1 GPU V100 32Go
#SBATCH --cpus-per-task=10           # 10 CPUs pour préparer les batchs rapidement
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00              # 20h max (QOS t3)
#SBATCH --qos=qos_gpu-t3

# 1. Chargement des modules
module purge
module load pytorch-gpu/py3/2.1.1

# 2. (OPTIONNEL) Activation de ton environnement virtuel si tu en as un
# source $WORK/mon_env/bin/activate  <-- Décommente si besoin

# 3. Sécurité pour les imports Python
# Force Python à chercher les modules 'src' dans le dossier courant
export PYTHONPATH=$PWD:$PYTHONPATH

# 4. Création des dossiers de logs Slurm (Python crée les siens ailleurs)
mkdir -p slurm/log

echo "🚀 Job lancé sur le noeud : $SLURMD_NODENAME"
echo "📂 Dossier de travail : $PWD"
nvidia-smi

# 5. Lancement
# Note : Python va créer automatiquement results/CGL_run_DATE/
srun python scripts/train_cgl.py --resume latest --config configs/cgl_config.yaml
