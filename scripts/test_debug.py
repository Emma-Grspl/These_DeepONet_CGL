import sys
import os
import torch
import yaml

# Ajout du path pour trouver src
sys.path.append(os.getcwd())

from src.models.CGL_PI_DeepOnet import CGL_PI_DeepONet

# Helper Config (Le même que d'habitude)
class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            if isinstance(value, dict): setattr(self, key, value)
            else: setattr(self, key, value)
    def __getitem__(self, item): return self._dict[item]
    def get(self, key, default=None): return self._dict.get(key, default)

def test_run():
    print("🛠️  DÉBUT DU CRASH-TEST LOCAL")
    
    # 1. Chargement Config
    print("1. Chargement Config...")
    with open("configs/cgl_config.yaml", 'r') as f:
        cfg = ConfigObj(yaml.safe_load(f))
    
    # 2. Init Modèle (CPU pour aller vite)
    print("2. Init Modèle...")
    device = torch.device("cpu") # On teste en CPU local
    model = CGL_PI_DeepONet(cfg).to(device)
    
    # 3. Création d'un Batch Fictif (Batch size = 2)
    # Branch input : 9 paramètres
    # [alpha, beta, mu, V, A, w0, x0, k, type]
    print("3. Génération données bidon...")
    batch_size = 2
    branch_input = torch.randn(batch_size, 9).to(device)
    
    # Trunk input : 2 coordonnées (x, t)
    trunk_input = torch.randn(batch_size, 2).to(device)
    
    # 4. Forward Pass (C'est là que ça plantait !)
    print("4. Test Forward Pass...")
    try:
        u_re, u_im = model(branch_input, trunk_input)
        print(f"   ✅ Sortie OK : re={u_re.shape}, im={u_im.shape}")
    except Exception as e:
        print(f"   ❌ CRASH PENDANT LE FORWARD : \n{e}")
        return

    print("🎉 SUCCÈS : Le modèle tourne sans erreur de typage.")

if __name__ == "__main__":
    test_run()