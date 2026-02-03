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
    # CORRECTION 1 : Attention à la casse (CGL vs cgle)
    from src.models.CGL_PI_DeepOnet import CGL_PI_DeepONet
    from src.training.trainer_CGL import train_cgle_curriculum 
    print("✅ Imports CGL réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("Vérifiez que vous êtes bien à la racine du projet et que les noms de fichiers (CGL/cgle) correspondent.")
    sys.exit(1)

# --- HELPER CONFIG ---
class ConfigObj:
    """
    Wrapper hybride : permet l'accès cfg.key ET cfg['key'].
    Utile car certains scripts utilisent l'un ou l'autre.
    """
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # On ne récursive pas pour garder l'accès dict sur les enfants (ex: cfg.physics['alpha'])
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
    # CORRECTION 2 : Nom du fichier yaml par défaut
    parser.add_argument("--config", type=str, default="configs/cgl_config.yaml", help="Chemin vers le fichier de config YAML")
    args = parser.parse_args()

    # 1. SETUP DOSSIER DE SAUVEGARDE
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"CGL_run_{timestamp}"
    run_dir = os.path.join(project_root, "results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # On crée aussi un sous-dossier pour les checkpoints intermédiaires
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
    # Pour que le trainer sache où enregistrer
    if 'training' not in yaml_data: yaml_data['training'] = {}
    yaml_data['training']['save_dir'] = ckpt_dir 
    
    # Sauvegarde de la config utilisée dans le dossier de résultat (Bonne pratique !)
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
    # Le modèle attend le dictionnaire brut ou l'objet ConfigObj (ça marche car ConfigObj a __getitem__)
    model = CGL_PI_DeepONet(cfg).to(device)

    # 5. ENTRAÎNEMENT
    try:
        # On passe la main au Curriculum Trainer
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