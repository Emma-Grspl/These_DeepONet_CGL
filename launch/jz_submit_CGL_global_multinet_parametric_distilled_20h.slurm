#!/bin/bash
#SBATCH --job-name=CGL_GMN_PD
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

python - "$PROJECT_DIR" "$CONFIG_PATH" <<'PY'
import os
import sys
import yaml

project_dir = sys.argv[1]
config_path = sys.argv[2]
with open(config_path, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

anchors = cfg.get("teacher_distillation", {}).get("anchors", [])
missing = []
for anchor in anchors:
    label = str(anchor.get("label", "<sans_label>"))
    checkpoint = anchor.get("teacher_checkpoint")
    checkpoint_str = "" if checkpoint is None else str(checkpoint)
    if not checkpoint_str or checkpoint_str.startswith("FILL_ME_"):
        missing.append(f"{label}: checkpoint non renseigne ({checkpoint_str or '<vide>'})")
        continue
    resolved = checkpoint_str if os.path.isabs(checkpoint_str) else os.path.join(project_dir, checkpoint_str)
    if not os.path.exists(resolved):
        missing.append(f"{label}: fichier introuvable ({checkpoint_str})")

if missing:
    print("❌ Validation teacher_distillation echouee :")
    for row in missing:
        print(f"   - {row}")
    sys.exit(1)
PY

if [ -n "$RESUME_RUN_DIR" ]; then
  srun python scripts/train_cgl_global_multinet_parametric_distilled_amp_phase.py \
    --config "$CONFIG_PATH" \
    --resume "$RESUME_RUN_DIR"
else
  srun python scripts/train_cgl_global_multinet_parametric_distilled_amp_phase.py \
    --config "$CONFIG_PATH"
fi
