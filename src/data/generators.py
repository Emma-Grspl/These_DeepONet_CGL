import torch
import numpy as np

def get_ic_batch_cgle(n_samples, cfg, device):
    """
    Générateur de Conditions Initiales "Mixed" pour CGL (1D Complexe).
    Génère u0(x) et sa dérivée spatiale exacte du0/dx pour la loss Sobolev.
    
    Types supportés (choisis aléatoirement) :
    0: Gaussian Pulse : A * exp(-(x-x0)^2 / w0^2) * exp(i*k*x)
    1: Sech Pulse     : A * sech((x-x0) / w0) * exp(i*k*x)
    2: Tanh Front     : A * tanh((x-x0) / w0) * exp(i*k*x)
    """
    b = cfg.physics['bounds']
    
    # --- 1. Sampling des Paramètres Physiques (Branch Input) ---
    # Pour CGL, le réseau prend en entrée [alpha, beta, mu, V] + Paramètres de l'IC
    # Mais ici, on se concentre sur les paramètres de l'IC pour le batch
    
    # Amplitude A
    A = torch.rand(n_samples, 1, device=device) * (b['A'][1] - b['A'][0]) + b['A'][0]
    
    # Largeur w0 (Log-scale)
    w0_min, w0_max = b['w0']
    w0 = 10 ** (torch.rand(n_samples, 1, device=device) * np.log10(w0_max/w0_min) + np.log10(w0_min))
    
    # Position x0
    x0 = torch.rand(n_samples, 1, device=device) * (b['x0'][1] - b['x0'][0]) + b['x0'][0]
    
    # Vecteur d'onde k (pour la phase complexe exp(ikx))
    k_wav = torch.rand(n_samples, 1, device=device) * (b['k'][1] - b['k'][0]) + b['k'][0]

    # Type d'IC (0, 1, 2)
    # On crée un masque pour vectoriser le calcul
    types = torch.randint(0, 3, (n_samples, 1), device=device).float()
    
    # Branch Input : On concatène tout ce qui définit l'IC
    # Note : Si alpha/beta sont variables, ils devraient être ajoutés ici. 
    # Pour l'instant on met les params géométriques.
    branch = torch.cat([A, w0, x0, k_wav, types], dim=1)

    # --- 2. Coordonnées Spatiales (Trunk Input) ---
    # x entre x_min et x_max
    x_min, x_max = cfg.physics['x_domain']
    x = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
    t = torch.zeros_like(x) # t=0 pour l'IC
    coords = torch.cat([x, t], dim=1).requires_grad_(True)

    # --- 3. Construction de u0(x) et du0/dx (Sobolev) ---
    # Variable centrée réduite X = (x - x0) / w0
    # Attention aux divisions par zéro ou w0 trop petit
    X = (x - x0) / (w0 + 1e-8)
    
    # Terme de phase : P = exp(i * k * x)
    phase_val = k_wav * x
    P_re = torch.cos(phase_val)
    P_im = torch.sin(phase_val)
    
    # Dérivée de la phase : dP/dx = i * k * P
    dP_re = -k_wav * P_im
    dP_im =  k_wav * P_re

    # Initialisation des enveloppes (E) et leurs dérivées (dE)
    E_val = torch.zeros_like(x)
    dE_val = torch.zeros_like(x)

    # --- TYPE 0 : GAUSSIAN ---
    mask_0 = (types == 0)
    if mask_0.any():
        # E = A * exp(-X^2)
        # dE = E * (-2X / w0)
        gauss_arg = -X**2
        E_gauss = A * torch.exp(gauss_arg)
        dE_gauss = E_gauss * (-2 * X / (w0 + 1e-8))
        
        E_val = torch.where(mask_0, E_gauss, E_val)
        dE_val = torch.where(mask_0, dE_gauss, dE_val)

    # --- TYPE 1 : SECH (Soliton) ---
    mask_1 = (types == 1)
    if mask_1.any():
        # E = A * sech(X) = A / cosh(X)
        # dE = -A * sech(X) * tanh(X) / w0
        cosh_X = torch.cosh(X)
        E_sech = A / (cosh_X + 1e-8)
        dE_sech = -E_sech * torch.tanh(X) / (w0 + 1e-8)
        
        E_val = torch.where(mask_1, E_sech, E_val)
        dE_val = torch.where(mask_1, dE_sech, dE_val)

    # --- TYPE 2 : TANH (Front) ---
    mask_2 = (types == 2)
    if mask_2.any():
        # E = A * tanh(X)
        # dE = A * (1 - tanh^2(X)) / w0
        tanh_X = torch.tanh(X)
        E_tanh = A * tanh_X
        dE_tanh = (A / (w0 + 1e-8)) * (1 - tanh_X**2)
        
        E_val = torch.where(mask_2, E_tanh, E_val)
        dE_val = torch.where(mask_2, dE_tanh, dE_val)

    # --- 4. Combinaison (Produit Enveloppe * Phase) ---
    # u = E * P
    # u_re = E * P_re
    # u_im = E * P_im
    t_re = E_val * P_re
    t_im = E_val * P_im

    # Dérivée (Règle du produit) : u' = E'P + E P'
    # du_re = dE * P_re + E * dP_re
    # du_im = dE * P_im + E * dP_im
    dt_re = dE_val * P_re + E_val * dP_re
    dt_im = dE_val * P_im + E_val * dP_im

    return branch, coords, t_re, t_im, dt_re, dt_im


def get_pde_batch_cgle(n_samples, cfg, device, t_limit=None):
    """
    Générateur de points de collocation pour la PDE.
    Similaire à get_ic mais avec t > 0.
    """
    b = cfg.physics['bounds']
    x_min, x_max = cfg.physics['x_domain']
    
    # 1. Branch Params (Identique à IC)
    A = torch.rand(n_samples, 1, device=device) * (b['A'][1] - b['A'][0]) + b['A'][0]
    
    w0_min, w0_max = b['w0']
    w0 = 10 ** (torch.rand(n_samples, 1, device=device) * np.log10(w0_max/w0_min) + np.log10(w0_min))
    
    x0 = torch.rand(n_samples, 1, device=device) * (b['x0'][1] - b['x0'][0]) + b['x0'][0]
    k_wav = torch.rand(n_samples, 1, device=device) * (b['k'][1] - b['k'][0]) + b['k'][0]
    
    types = torch.randint(0, 3, (n_samples, 1), device=device).float()
    
    branch = torch.cat([A, w0, x0, k_wav, types], dim=1)

    # 2. Coordonnées Spatiales et Temporelles
    # x : Uniforme dans le domaine
    x = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
    
    # t : Uniforme jusqu'à t_limit (Curriculum Learning)
    if t_limit is None:
        t_limit = cfg.physics['t_max']
    
    t = torch.rand(n_samples, 1, device=device) * t_limit
    
    coords = torch.cat([x, t], dim=1).requires_grad_(True)

    return branch, coords