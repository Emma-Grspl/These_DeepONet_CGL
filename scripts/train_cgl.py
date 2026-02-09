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
try:
    # CORRECTION : On pointe vers le bon fichier (cgl_deeponet.py)
    from src.models.cgl_deeponet import CGL_PI_DeepONet
    from src.training.trainer_CGL import train_cgle_curriculum 
    print("✅ Imports CGL réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("Vérifiez que le fichier src/models/cgl_deeponet.py existe bien.")
    sys.exit(1)

# --- HELPER CONFIG ---
class ConfigObj:
    """
    Wrapper hybride : permet l'accès cfg.key ET cfg['key'].
    """
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, value) 
            else:
                setattr(self, key, value)
    
    def __getitem__(self, item):
        return self._dict[item]
    
    def get(self, key, default=None):
        return self._dict.get(key, default)

def main():
    # 0. ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_config.yaml", help="Chemin vers le fichier de config YAML")
    args = parser.parse_args()

    # 1. SETUP DOSSIER DE SAUVEGARDE
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"CGL_run_{timestamp}"
    run_dir = os.path.join(project_root, "results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Sous-dossier checkpoints
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"🚀 Lancement Entraînement CGL")
    print(f"📁 Dossier de sortie : {run_dir}")

    # 2. CHARGEMENT CONFIG
    print(f"📖 Chargement de la config : {args.config}")
    if not os.path.exists(args.config):
        print(f"❌ Fichier config introuvable : {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # INJECTION DU SAVE DIR DANS LA CONFIG
    # Crucial pour que le trainer sache où sauvegarder les checkpoints intermédiaires
    if 'training' not in yaml_data: yaml_data['training'] = {}
    yaml_data['training']['save_dir'] = ckpt_dir 
    
    # Sauvegarde de la config utilisée
    with open(os.path.join(run_dir, "config_used.yaml"), 'w') as f:
        yaml.dump(yaml_data, f)

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
    # ... initialisation du modèle ...

    # 👇 METS LE CHEMIN EXACT QUE TU AS TROUVÉ AVEC 'find' 👇
    OLD_CHECKPOINT = "/lustre/fswork/projects/rech/fdb/usv13rn/These_DeepOnet_CGL/results/CGL_run_20260207-175414/checkpoints/ckpt_t0.09.pth"
    
    # 5. ENTRAÎNEMENT
    try:
        # On passe la main au Curriculum Trainer (le chef d'orchestre)
        train_cgle_curriculum(model, cfg, explicit_resume_path=OLD_CHECKPOINT)

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