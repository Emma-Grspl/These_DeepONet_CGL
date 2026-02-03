import torch
import numpy as np

def get_ic_batch_cgle(n_samples, cfg, device):
    """
    Générateur de Conditions Initiales + Paramètres d'équation pour CGL.
    """
    b = cfg.physics['bounds']
    eq_params = cfg.physics['equation_params'] # Assure-toi d'avoir ça dans ton YAML
    
    # --- 1. Sampling des Paramètres Physiques de l'équation (Branch Input) ---
    # Pour un solveur généraliste, ces paramètres varient à chaque sample !
    
    # Alpha (Diffusion complexe)
    alpha = torch.rand(n_samples, 1, device=device) * (eq_params['alpha'][1] - eq_params['alpha'][0]) + eq_params['alpha'][0]
    
    # Beta (Non-linéarité complexe)
    beta = torch.rand(n_samples, 1, device=device) * (eq_params['beta'][1] - eq_params['beta'][0]) + eq_params['beta'][0]
    
    # Mu (Gain/Perte linéaire)
    mu = torch.rand(n_samples, 1, device=device) * (eq_params['mu'][1] - eq_params['mu'][0]) + eq_params['mu'][0]
    
    # V (Vitesse d'advection)
    V = torch.rand(n_samples, 1, device=device) * (eq_params['V'][1] - eq_params['V'][0]) + eq_params['V'][0]

    # --- 2. Sampling des Paramètres de l'IC (Branch Input) ---
    
    # Amplitude A
    A = torch.rand(n_samples, 1, device=device) * (b['A'][1] - b['A'][0]) + b['A'][0]
    
    # Largeur w0 (Log-scale)
    w0_min, w0_max = b['w0']
    w0 = 10 ** (torch.rand(n_samples, 1, device=device) * np.log10(w0_max/w0_min) + np.log10(w0_min))
    
    # Position x0
    x0 = torch.rand(n_samples, 1, device=device) * (b['x0'][1] - b['x0'][0]) + b['x0'][0]
    
    # Vecteur d'onde k
    k_wav = torch.rand(n_samples, 1, device=device) * (b['k'][1] - b['k'][0]) + b['k'][0]

    # Type d'IC (0: Gauss, 1: Sech, 2: Tanh)
    types = torch.randint(0, 3, (n_samples, 1), device=device).float()
    
    # --- 3. Construction du Branch Input ---
    # Le réseau prend TOUT : paramètres de l'équation + paramètres de l'IC
    # Ordre : [alpha, beta, mu, V, A, w0, x0, k, type] (9 paramètres)
    branch = torch.cat([alpha, beta, mu, V, A, w0, x0, k_wav, types], dim=1)

    # --- 4. Coordonnées Spatiales (Trunk Input) ---
    x_min, x_max = cfg.physics['x_domain']
    x = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
    t = torch.zeros_like(x) # t=0 pour l'IC
    coords = torch.cat([x, t], dim=1).requires_grad_(True)

    # --- 5. Construction Exacte de u0(x) et du0/dx ---
    # (Calcul identique à la version précédente)
    X = (x - x0) / (w0 + 1e-8)
    phase_val = k_wav * x
    P_re = torch.cos(phase_val)
    P_im = torch.sin(phase_val)
    dP_re = -k_wav * P_im
    dP_im =  k_wav * P_re

    E_val = torch.zeros_like(x)
    dE_val = torch.zeros_like(x)

    # TYPE 0 : GAUSSIAN
    mask_0 = (types == 0)
    if mask_0.any():
        gauss_arg = -X**2
        E_gauss = A * torch.exp(gauss_arg)
        dE_gauss = E_gauss * (-2 * X / (w0 + 1e-8))
        E_val = torch.where(mask_0, E_gauss, E_val)
        dE_val = torch.where(mask_0, dE_gauss, dE_val)

    # TYPE 1 : SECH
    mask_1 = (types == 1)
    if mask_1.any():
        cosh_X = torch.cosh(X)
        E_sech = A / (cosh_X + 1e-8)
        dE_sech = -E_sech * torch.tanh(X) / (w0 + 1e-8)
        E_val = torch.where(mask_1, E_sech, E_val)
        dE_val = torch.where(mask_1, dE_sech, dE_val)

    # TYPE 2 : TANH
    mask_2 = (types == 2)
    if mask_2.any():
        tanh_X = torch.tanh(X)
        E_tanh = A * tanh_X
        dE_tanh = (A / (w0 + 1e-8)) * (1 - tanh_X**2)
        E_val = torch.where(mask_2, E_tanh, E_val)
        dE_val = torch.where(mask_2, dE_tanh, dE_val)

    # Recombinaison
    t_re = E_val * P_re
    t_im = E_val * P_im
    dt_re = dE_val * P_re + E_val * dP_re
    dt_im = dE_val * P_im + E_val * dP_im

    return branch, coords, t_re, t_im, dt_re, dt_im


def get_pde_batch_cgle(n_samples, cfg, device, t_limit=None):
    """
    Générateur pour la PDE Collocation.
    Retourne aussi les paramètres de l'équation sous forme de dictionnaire
    pour faciliter le calcul du résidu dans la loss function.
    """
    b = cfg.physics['bounds']
    eq_params = cfg.physics['equation_params']
    x_min, x_max = cfg.physics['x_domain']
    
    # 1. Equation Params
    alpha = torch.rand(n_samples, 1, device=device) * (eq_params['alpha'][1] - eq_params['alpha'][0]) + eq_params['alpha'][0]
    beta = torch.rand(n_samples, 1, device=device) * (eq_params['beta'][1] - eq_params['beta'][0]) + eq_params['beta'][0]
    mu = torch.rand(n_samples, 1, device=device) * (eq_params['mu'][1] - eq_params['mu'][0]) + eq_params['mu'][0]
    V = torch.rand(n_samples, 1, device=device) * (eq_params['V'][1] - eq_params['V'][0]) + eq_params['V'][0]

    # 2. IC Params
    A = torch.rand(n_samples, 1, device=device) * (b['A'][1] - b['A'][0]) + b['A'][0]
    w0_min, w0_max = b['w0']
    w0 = 10 ** (torch.rand(n_samples, 1, device=device) * np.log10(w0_max/w0_min) + np.log10(w0_min))
    x0 = torch.rand(n_samples, 1, device=device) * (b['x0'][1] - b['x0'][0]) + b['x0'][0]
    k_wav = torch.rand(n_samples, 1, device=device) * (b['k'][1] - b['k'][0]) + b['k'][0]
    types = torch.randint(0, 3, (n_samples, 1), device=device).float()
    
    # 3. Branch Input complet
    branch = torch.cat([alpha, beta, mu, V, A, w0, x0, k_wav, types], dim=1)

    # 4. Coordonnées (Trunk Input)
    x = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
    
    if t_limit is None:
        t_limit = cfg.physics['t_max']
    t = torch.rand(n_samples, 1, device=device) * t_limit
    
    coords = torch.cat([x, t], dim=1).requires_grad_(True)

    # 5. Dictionnaire de paramètres pour la Physics Loss
    # Cela permet à pde_cgl.py d'accéder directement à 'alpha' sans re-slicing complexe
    params_dict = {
        "alpha": alpha,
        "beta": beta,
        "mu": mu,
        "V": V
    }

    return branch, coords, params_dict