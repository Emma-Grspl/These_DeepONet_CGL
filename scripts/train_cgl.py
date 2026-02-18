import sys
import os
import argparse
import yaml
import torch
import glob
from datetime import datetime

# --- GESTION DES CHEMINS ---
project_root = os.getcwd()
sys.path.append(project_root)

# --- IMPORTS ---
from src.models.cgl_deeponet import CGL_PI_DeepONet
from src.training.trainer_CGL import train_navigator 

# --- HELPER CONFIG ---
class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)
    def __getitem__(self, item): return self._dict[item]
    def get(self, key, default=None): return self._dict.get(key, default)

def find_latest_run_dir(base_results_dir):
    """Trouve le dossier de run le plus récent."""
    # On cherche tous les dossiers commençant par CGL_
    search_path = os.path.join(base_results_dir, "CGL_*")
    all_runs = glob.glob(search_path)
    
    # On filtre pour ne garder que les dossiers
    all_runs = [d for d in all_runs if os.path.isdir(d)]
    
    if not all_runs:
        return None
        
    # On trie par date de modification (le plus récent en dernier)
    latest_run = max(all_runs, key=os.path.getmtime)
    return latest_run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_config.yaml", help="Chemin vers le fichier de config YAML")
    
    # --- NOUVEL ARGUMENT MAGIQUE ---
    # Si on met juste --resume, ça vaut "latest". Sinon on peut mettre un chemin.
    parser.add_argument("--resume", nargs='?', const="latest", default=None, 
                        help="Mode reprise : 'latest' pour le dernier run, ou chemin spécifique.")
    
    args = parser.parse_args()

    results_root = os.path.join(project_root, "results")
    os.makedirs(results_root, exist_ok=True)

    # 1. DÉCISION DU DOSSIER DE RUN
    if args.resume:
        # MODE REPRISE
        if args.resume == "latest":
            run_dir = find_latest_run_dir(results_root)
            if not run_dir:
                raise ValueError("❌ Aucun run précédent trouvé pour --resume latest !")
        else:
            # L'utilisateur a donné un chemin (relatif ou absolu)
            run_dir = args.resume
            if not os.path.exists(run_dir):
                 # Essai en relatif par rapport à results
                 run_dir = os.path.join(results_root, args.resume)
                 if not os.path.exists(run_dir):
                     raise ValueError(f"❌ Dossier introuvable : {args.resume}")
        
        print(f"🔄 MODE REPRISE ACTIVÉ : {run_dir}")
        print("   (Le script cherchera le dernier checkpoint valide dans ce dossier)")
        
    else:
        # MODE NOUVEAU RUN
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"CGL_Navigator_Run_{timestamp}"
        run_dir = os.path.join(results_root, run_name)
        print(f"🚀 NOUVEAU START : {run_dir}")

    # Création des sous-dossiers (si pas existants)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. CONFIG
    with open(args.config, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # On force la sauvegarde dans le dossier déterminé plus haut
    yaml_data['training']['save_dir'] = ckpt_dir 
    cfg = ConfigObj(yaml_data)

    # 3. DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")

    # 4. MODÈLE
    model = CGL_PI_DeepONet(cfg).to(device)

    # 5. ENTRAÎNEMENT
    try:
        # Le Navigator va scanner ckpt_dir. 
        # S'il trouve des fichiers (Mode Resume), il reprend.
        # S'il ne trouve rien (Mode Nouveau), il commence à t=0.
        train_navigator(model, cfg, explicit_resume_path=None)

        # FINAL
        torch.save(model.state_dict(), os.path.join(run_dir, "model_final_cgl.pth"))
        print("\n✅ Terminé avec succès !")

    except Exception as e:
        print(f"\n❌ Erreur Critique : {e}")
        raise e

if __name__ == "__main__":
    main()