import sys
import os
import torch
import numpy as np

# Ajout du chemin racine
sys.path.append(os.getcwd())

def test_all():
    print("🔍 DÉBUT DU DIAGNOSTIC COMPLET...\n")
    
    # --- 1. TEST CONFIG MOCK ---
    print("1️⃣  Création d'une Config Mock...", end=" ")
    try:
        class ConfigObj:
            def __init__(self, d): self._d = d; vars(self).update(d)
            def __getitem__(self, k): return self._d[k]
            def get(self, k, default=None): return self._d.get(k, default)
            
            # Gestion récursive pour physics.bounds etc
            @property
            def physics(self): return self._d['physics']
            @property
            def model(self): return self._d['model']
            @property
            def training(self): return self._d['training']

        # Config minimale pour faire tourner les tests
        mock_cfg_dict = {
            'physics': {
                'x_domain': [-20, 20], 't_max': 10.0,
                'bounds': {'A': [0.1, 3.0], 'w0': [0.5, 6.0], 'x0': [0.0, 0.0], 'k': [-2.0, 2.0]},
                'equation_params': {'alpha': [0,1], 'beta': [-1,1], 'mu': [-0.5,1], 'V': [-2,2]}
            },
            'model': {
                'latent_dim': 32, 'branch_layers': [32, 32], 'trunk_layers': [32, 32],
                'fourier_dim': 16, 'fourier_scales': [1.0, 10.0]
            },
            'training': {
                'batch_size_pde': 16, 'batch_size_ic': 16,
                'max_macro_loops': 1, 'weights': {'pde_loss': 1, 'ic_loss': 1}
            }
        }
        cfg = ConfigObj(mock_cfg_dict)
        print("✅ OK")
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return

    # --- 2. TEST DATA GENERATORS ---
    print("2️⃣  Test Data Generators (src.data.generators)...", end=" ")
    try:
        from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle
        
        # Test PDE Batch
        b, c, p = get_pde_batch_cgle(10, cfg, "cpu")
        assert b.shape == (10, 9) # 9 params
        assert c.shape == (10, 2) # x, t
        
        # Test IC Batch (Sobolev)
        b_ic, c_ic, u_re, u_im, ux_re, ux_im = get_ic_batch_cgle(10, cfg, "cpu")
        assert ux_re.shape == (10, 1) # Vérif présence dérivées
        
        print("✅ OK (Sobolev & PDE valides)")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback; traceback.print_exc()
        return

    # --- 3. TEST MODEL ARCHITECTURE ---
    print("3️⃣  Test Model (src.models.cgl_deeponet)...", end=" ")
    try:
        from src.models.cgl_deeponet import CGL_PI_DeepONet
        model = CGL_PI_DeepONet(cfg)
        
        # Forward pass dummy
        dummy_params = torch.randn(10, 9)
        dummy_coords = torch.randn(10, 2)
        out_re, out_im = model(dummy_params, dummy_coords)
        
        assert out_re.shape == (10, 1)
        print("✅ OK (ModifiedMLP & Forward pass)")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback; traceback.print_exc()
        return

    # --- 4. TEST SOLVER ---
    print("4️⃣  Test Solver (src.utils.solver_cgl)...", end=" ")
    try:
        from src.utils.solver_cgl import get_ground_truth_CGL
        # Test rapide sur une petite grille
        params = {'alpha': 0.1, 'beta': 0.1, 'mu': 0.1, 'V': 1.0, 'A': 1, 'w0': 1, 'x0': 0, 'k': 0, 'type': 0}
        X, T, U = get_ground_truth_CGL(params, -10, 10, 0.1, Nx=32, Nt=10)
        assert U.shape == (32, 10) or U.shape == (10, 32) # Peu importe l'ordre tant que ça sort
        print("✅ OK (Scipy & Matrices)")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback; traceback.print_exc()
        return

    # --- 5. TEST TRAINER IMPORTS ---
    print("5️⃣  Test Trainer Imports (src.training.trainer_CGL)...", end=" ")
    try:
        from src.training.trainer_CGL import train_cgle_curriculum, robust_optimize
        # On vérifie juste que les fonctions existent
        assert callable(train_cgle_curriculum)
        assert callable(robust_optimize)
        print("✅ OK (King of the Hill logic loaded)")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback; traceback.print_exc()
        return

    print("\n🎉 TOUT EST VERT ! Tu peux lancer le training.")

if __name__ == "__main__":
    test_all()
