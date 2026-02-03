import torch
import numpy as np
import sys
import os

# On importe le solveur numérique qu'on vient de créer
from src.utils.solver_nlse import NLSESolver

def evaluate_nlse_metrics(model, cfg, n_samples=10, z_eval=0.0):
    """
    Évalue le modèle NLSE en le comparant au Solveur Numérique (Split-Step).
    ATTENTION : C'est plus lent que la version analytique, donc on garde n_samples faible (ex: 10).
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Instantiation du solveur (CPU)
    # On prend une résolution suffisante pour être précis
    solver = NLSESolver(cfg, nr=1000, nz=1000)
    
    total_l2 = 0.0
    total_samples = 0
    
    # Limites pour le sampling aléatoire
    b = cfg.physics['bounds']
    
    with torch.no_grad():
        for i in range(n_samples):
            # 1. Tirage aléatoire d'un cas de test (CPU pour le solveur)
            A_val = np.random.uniform(b['A'][0], b['A'][1])
            
            log_w0 = np.random.uniform(np.log10(b['w0'][0]), np.log10(b['w0'][1]))
            w0_val = 10**log_w0
            
            log_f = np.random.uniform(np.log10(b['f'][0]), np.log10(b['f'][1]))
            f_val = 10**log_f
            
            # 2. Calcul de la Vérité Terrain (Ground Truth) avec le Solveur
            # Le solveur renvoie toute la grille (z, r, E)
            zs, rs, E_true_grid = solver.solve(A_val, w0_val, f_val)
            
            # 3. Extraction du profil au point z_eval
            # On cherche l'index z le plus proche de z_eval
            idx_z = (np.abs(zs - z_eval)).argmin()
            
            # Profil radial vrai (complexe) à z_eval
            E_true_profile = E_true[idx_z, :] # Shape (Nr,)
            mod_true = np.abs(E_true_profile)
            
            # 4. Prédiction du Modèle (Neural Network)
            # On doit interroger le modèle sur les mêmes points r que le solveur
            r_tensor = torch.tensor(rs, dtype=torch.float32, device=device).view(-1, 1)
            z_tensor = torch.full_like(r_tensor, z_eval)
            coords = torch.cat([r_tensor, z_tensor], dim=1)
            
            # Inputs Branch (A, w0, f) répétés pour tous les points r
            branch_input = torch.tensor([[A_val, w0_val, f_val]], dtype=torch.float32, device=device)
            branch_batch = branch_input.repeat(len(rs), 1)
            
            p_re, p_im = model(branch_batch, coords)
            p_mod = torch.sqrt(p_re**2 + p_im**2).cpu().numpy().flatten()
            
            # 5. Calcul de l'erreur L2 Relative sur ce profil
            # On normalise par l'énergie totale pour que l'erreur soit en %
            numerator = np.sum((p_mod - mod_true)**2)
            denominator = np.sum(mod_true**2)
            
            # Sécurité division par zero
            if denominator < 1e-12: denominator = 1e-12
                
            error_l2 = np.sqrt(numerator / denominator)
            
            total_l2 += error_l2
            total_samples += 1
            
    avg_l2 = total_l2 / total_samples
    return avg_l2, 0.0 # On renvoie 0 pour peak_err pour l'instant (moins critique)