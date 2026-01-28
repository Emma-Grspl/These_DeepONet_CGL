import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.Diffractive_PI_DeepOnet import PI_DeepONet_Robust
from src.utils.solver import CrankNicolsonSolver

# Helper Config
class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def main():
    # 1. Config & Modèle
    config_path = "configs/diffractive.yaml"
    with open(config_path, 'r') as f:
        cfg = Config(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI_DeepONet_Robust(cfg).to(device)

    # Chargement des poids
    ckpt_path = "outputs/diffractive_final.pth"
    if os.path.exists(ckpt_path):
        print(f"Chargement de {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("⚠️ Pas de checkpoint trouvé. On utilise un modèle non entraîné (juste pour tester le script).")

    model.eval()

    # 2. Paramètres du Cas Test
    A_val = 1.0
    w0_val = 1.0
    f_val = 1000.0
    print(f"🧪 Test Case : A={A_val}, w0={w0_val}, f={f_val}")

    # 3. Solveur Crank-Nicolson (Vérité Terrain)
    print("🧮 Exécution Crank-Nicolson...")
    solver = CrankNicolsonSolver(cfg, nr=500, nz=200) # Résolution moyenne pour aller vite
    z_cn, r_cn, E_cn = solver.solve(A_val, w0_val, f_val)
    E_cn_mod = np.abs(E_cn)

    # 4. Prédiction PINN (Sur la même grille)
    print("🧠 Exécution PINN...")
    # On doit créer une grille (r, z) correspondant à celle du CN
    R_grid, Z_grid = np.meshgrid(r_cn, z_cn)

    r_flat = torch.tensor(R_grid.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    z_flat = torch.tensor(Z_grid.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    coords = torch.cat([r_flat, z_flat], dim=1)

    # Branch input constant
    n_pts = len(r_flat)
    branch = torch.tensor([[A_val, w0_val, f_val]], device=device).repeat(n_pts, 1)

    with torch.no_grad():
        p_re, p_im = model(branch, coords)
        p_mod = torch.sqrt(p_re**2 + p_im**2).cpu().numpy().reshape(Z_grid.shape)

    # 5. Plot Comparatif
    print("🎨 Génération du Plot...")
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(12, 5))

    # Plot CN
    plt.subplot(1, 3, 1)
    plt.imshow(E_cn_mod, extent=[0, cfg.physics['r_max'], cfg.physics['z_max'], 0], aspect='auto', cmap='inferno')
    plt.title("Crank-Nicolson (Truth)")
    plt.xlabel("r (mm)"); plt.ylabel("z (mm)")
    plt.colorbar()

    # Plot PINN
    plt.subplot(1, 3, 2)
    plt.imshow(p_mod, extent=[0, cfg.physics['r_max'], cfg.physics['z_max'], 0], aspect='auto', cmap='inferno')
    plt.title("PINN Prediction")
    plt.xlabel("r (mm)")
    plt.colorbar()

    # Plot Erreur
    err = np.abs(E_cn_mod - p_mod)
    plt.subplot(1, 3, 3)
    plt.imshow(err, extent=[0, cfg.physics['r_max'], cfg.physics['z_max'], 0], aspect='auto', cmap='seismic')
    plt.title("Différence Absolue")
    plt.xlabel("r (mm)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("outputs/plots/comparison_cn_pinn.png")
    print("✅ Sauvegardé dans outputs/plots/comparison_cn_pinn.png")

if __name__ == "__main__":
    main()
