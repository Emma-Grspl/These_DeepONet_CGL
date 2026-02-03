import sys
import os
import yaml
import torch
import numpy as np

# 1. Ajouter la racine du projet au path
sys.path.append(os.getcwd())

# 2. Imports
from src.plot.plot_snapshot import plot_temporal_snapshots

# --- HELPER CLASS (La solution à ton problème) ---
class ConfigObj:
    """Transforme un dictionnaire en objet pour permettre l'accès cfg.physics"""
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            # On définit les attributs (cfg.physics)
            setattr(self, key, value)
    
    # Permet aussi l'accès dictionnaire (cfg['physics']) au cas où
    def __getitem__(self, item):
        return self._dict[item]

# 3. Charger la configuration
config_path = "configs/cgl_config.yaml"
if not os.path.exists(config_path):
    print(f"❌ Erreur: Config introuvable à {config_path}")
    sys.exit(1)

with open(config_path, 'r') as f:
    yaml_dict = yaml.safe_load(f)
    # C'est ici que la magie opère : on convertit le dict en objet
    cfg = ConfigObj(yaml_dict)

# 4. Définir tes paramètres de test
# On teste un Soliton (Sech) qui bouge vers la droite (V=1.0)
test_params = {
    'alpha': 0.5,    
    'beta': 1.0,     
    'mu': 0.5,       
    'V': 1.0,        
    'A': 1.0,        
    'w0': 1.0,       
    'x0': 0.0,       
    'k': 0.0,        
    'type': 2        
}

# 5. Lancer le plot (Mode Solveur Uniquement -> model=None)
print("📸 Génération du snapshot...")
save_file = "outputs/solver_only/test_solver_only_tanh_1.png"

# Création du dossier results si inexistant
os.makedirs("results", exist_ok=True)

try:
    plot_temporal_snapshots(
        cfg=cfg, 
        params_dict=test_params, 
        model=None, 
        save_path=save_file,
        show=False
    )
    print(f"✅ Terminé ! Image sauvée ici : {save_file}")
except Exception as e:
    print(f"❌ Erreur pendant le plot : {e}")
    import traceback
    traceback.print_exc()