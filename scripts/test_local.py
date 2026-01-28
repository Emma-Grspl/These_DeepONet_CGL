import os
import sys

# --- FIX CRITIQUE WINDOWS/INTEL ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
import time
from types import SimpleNamespace

# 1. AJOUT DU PATH RACINE
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

print(f"[INFO] Racine du projet detectee : {root_dir}")

# 2. IMPORTS DU PROJET
print("[INFO] Test des imports...")
try:
    from src.models.Diffractive_PI_DeepOnet import PI_DeepONet_Robust
    from src.data.generators import get_ic_batch_sobolev, get_pde_batch_z_limited
    from src.physics.diffractive import pde_residual_corrected
    from src.utils.solver import CrankNicolsonSolver
    from src.utils.metrics import evaluate_robust_metrics_smart
    print("[OK] Imports reussis.")
except ImportError as e:
    print(f"[ERREUR] Import : {e}")
    sys.exit(1)

# Helper pour la config
class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
    def __getitem__(self, item):
        return getattr(self, item)

def main():
    print("\n[START] DEMARRAGE DU DRY RUN (TEST LOCAL)...")

    # A. CHARGEMENT CONFIG
    config_path = os.path.join(root_dir, "configs", "diffractive.yaml")
    if not os.path.exists(config_path):
        print(f"[ERREUR] Config introuvable : {config_path}")
        return

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Config(cfg_dict)
    print("[OK] Configuration chargee.")

    # B. GPU CHECK
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device utilise : {device}")

    # C. MODÈLE
    try:
        model = PI_DeepONet_Robust(cfg).to(device)
        print("[OK] Modele instancie avec succes.")
        # Petit test forward
        dummy_branch = torch.randn(2, 3).to(device)
        dummy_coords = torch.randn(2, 2).to(device)
        _ = model(dummy_branch, dummy_coords)
        print("   -> Forward pass OK.")
    except Exception as e:
        print(f"[ERREUR] Modele : {e}")
        return

    # D. DATA GENERATION
    try:
        print("[INFO] Test Generation Donnees...")
        br, co, tr, ti, dtr, dti = get_ic_batch_sobolev(4, cfg, device)
        br_pde, co_pde = get_pde_batch_z_limited(4, cfg, device, z_limit=10.0)
        print("[OK] Batches generes.")
    except Exception as e:
        print(f"[ERREUR] Data : {e}")
        return

    # E. PHYSIQUE
    try:
        print("[INFO] Test Calcul Physique...")
        r_re, r_im = pde_residual_corrected(model, br_pde, co_pde, cfg)
        loss = torch.mean(r_re**2) + torch.mean(r_im**2)
        loss.backward()
        print(f"[OK] Physique OK (Loss dummy: {loss.item():.2e})")
    except Exception as e:
        print(f"[ERREUR] Physique : {e}")
        return

    # F. SOLVEUR
    try:
        print("[INFO] Test Solveur Crank-Nicolson...")
        solver = CrankNicolsonSolver(cfg, nr=50, nz=10)
        _, _, _ = solver.solve(A_val=1.0, w0_val=1.0, f_val=1000.0)
        print("[OK] Solveur OK.")
    except Exception as e:
        print(f"[ERREUR] Solveur : {e}")
        return

    print("\n[SUCCES] TOUT EST OK ! LE CODE EST PRET.")

if __name__ == "__main__":
    main()
