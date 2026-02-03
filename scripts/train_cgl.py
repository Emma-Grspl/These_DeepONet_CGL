import sys
import os
import argparse
import yaml
import torch
from datetime import datetime

# --- GESTION DES CHEMINS ---
# Indispensable pour que Python trouve 'src' quand on lance depuis la racine ou un dossier script
project_root = os.getcwd()
sys.path.append(project_root)

# --- IMPORTS ---
try:
    from src.models.CGL_PI_DeepOnet import CGL_PI_DeepONet
    from src.training.trainer_cgle import train_cgle_curriculum
    print("✅ Imports CGL réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("Vérifiez que vous êtes bien à la racine du projet (là où il y a le dossier src).")
    sys.exit(1)

# --- HELPER CONFIG ---
class ConfigObj:
    """Transforme un dictionnaire en objet (cfg.physics.alpha au lieu de cfg['physics']['alpha'])"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, value) # Pas récursif ici pour simplifier, mais suffisant pour cfg.physics['alpha']
            else:
                setattr(self, key, value)
        self.__dict__.update(dictionary)

def main():
    # 0. ARGUMENTS (Permet de changer de yaml via Slurm)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgle.yaml", help="Chemin vers le fichier de config YAML")
    args = parser.parse_args()

    # 1. SETUP DOSSIER DE SAUVEGARDE
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"CGL_run_{timestamp}"
    run_dir = os.path.join(project_root, "results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"🚀 Lancement Entraînement CGL (Jean Zay / Local)")
    print(f"📁 Dossier de sortie : {run_dir}")

    # 2. CHARGEMENT CONFIG
    print(f"📖 Chargement de la config : {args.config}")
    if not os.path.exists(args.config):
        print(f"❌ Fichier config introuvable : {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # On injecte le dossier de sauvegarde dans la config pour que le trainer l'utilise
    if 'training' not in yaml_data: yaml_data['training'] = {}
    yaml_data['training']['save_dir'] = run_dir
    
    # Création de l'objet Config
    cfg = ConfigObj(yaml_data)

    # 3. DEVICE
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"📱 Device : CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("📱 Device : MPS (Mac)")
    else:
        device = torch.device("cpu")
        print("📱 Device : CPU")

    # 4. INITIALISATION MODÈLE
    print("🏗️  Initialisation du modèle CGL_PI_DeepONet...")
    model = CGL_PI_DeepONet(cfg).to(device)

    # 5. ENTRAÎNEMENT
    try:
        # On passe la main au Curriculum Trainer
        # Note : Le trainer va utiliser cfg.training['save_dir'] pour les checkpoints
        train_cgle_curriculum(model, cfg)

        # 6. SAUVEGARDE FINALE
        final_path = os.path.join(run_dir, "model_final_cgl.pth")
        torch.save(model.state_dict(), final_path)
        print(f"\n✅ Modèle final sauvegardé : {final_path}")

    except KeyboardInterrupt:
        print("\n🛑 Interruption utilisateur (Ctrl+C).")
        save_path = os.path.join(run_dir, "model_INTERRUPTED.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Sauvegarde d'urgence : {save_path}")

    except Exception as e:
        print(f"\n❌ Erreur critique pendant l'entraînement : {e}")
        save_path = os.path.join(run_dir, "model_CRASHED.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Sauvegarde d'urgence : {save_path}")
        raise e

if __name__ == "__main__":
    main()