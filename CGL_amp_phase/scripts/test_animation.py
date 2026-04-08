import sys
import os
import yaml
import torch
import numpy as np

sys.path.append(os.getcwd())

# Import de la nouvelle fonction
from src.plot.plot_animation import animate_cgl_solution

# Helper Config
class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)
    def __getitem__(self, item): return self._dict[item]
    def get(self, key, default=None): return self._dict.get(key, default)

# Chargement Config
config_path = "configs/cgl_config.yaml"
with open(config_path, 'r') as f:
    cfg = ConfigObj(yaml.safe_load(f))

# Paramètres (Exemple : Soliton qui se déplace)
test_params = {
    'alpha': 0.5, 'beta': 1.0, 'mu': 0.5, 
    'V': 1.0,       # Vitesse visible
    'A': 1.0, 'w0': 1.0, 'x0': 0.0, 'k': 2.0,
    'type': 0       
}

print("🏃 Lancement du test animation...")

# On teste SANS modèle pour commencer (juste le solveur)
# Si tu as un modèle chargé, tu peux le passer dans 'model='
animate_cgl_solution(
    cfg=cfg, 
    params_dict=test_params, 
    model=None, 
    save_path="outputs/solver_only/test_anim_solver_gauss_k_2.gif",
    frames=100  # 100 frames pour que ce soit rapide à générer
)