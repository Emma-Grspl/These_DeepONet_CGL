import sys
import os
import argparse
import yaml
import torch
from datetime import datetime

# --- GESTION DES CHEMINS ---
project_root = os.getcwd()
sys.path.append(project_root)

# --- IMPORTS ---
from src.models.cgl_deeponet import CGL_PI_DeepONet
# MODIFICATION ICI : On importe le Navigateur (la nouvelle logique)
from src.training.trainer_CGL import train_navigator 

# --- HELPER CONFIG ---
class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)
    def __getitem__(self, item): return self._dict[item]
    def get(self, key, default=None): return self._dict.get(key, default)

def main():
    parser = argparse.ArgumentParser()
    # On garde ton défaut (vérifie juste que le fichier existe bien)
    parser.add_argument("--config", type=str, default="configs/cgl_config.yaml") 
    args = parser.parse_args()

    # 1. SETUP RUN
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"CGL_Navigator_Run_{timestamp}" # Petit rename pour distinguer
    run_dir = os.path.join(project_root, "results", run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"🚀 NOUVEAU START (NAVIGATOR) : {run_dir}")

    # 2. CONFIG
    with open(args.config, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # On force la sauvegarde dans le dossier du run actuel
    yaml_data['training']['save_dir'] = ckpt_dir 
    cfg = ConfigObj(yaml_data)

    # 3. DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")

    # 4. MODÈLE
    model = CGL_PI_DeepONet(cfg).to(device)

    # 5. ENTRAÎNEMENT (FORCE ZÉRO REPRISE)
    try:
        # MODIFICATION ICI : Appel au Navigateur
        # Le None force le Warmup au début
        train_navigator(model, cfg, explicit_resume_path=None)

        # FINAL
        torch.save(model.state_dict(), os.path.join(run_dir, "model_final_cgl.pth"))
        print("\n✅ Terminé avec succès !")

    except Exception as e:
        print(f"\n❌ Erreur Critique : {e}")
        raise e

if __name__ == "__main__":
    main()