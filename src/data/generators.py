import numpy as np
import torch

def get_ic_batch_cgle(batch_size, cfg, device):
    """
    Génère un batch CI (t=0) : UNIQUEMENT SECH (1) et TANH (2).
    """
    if isinstance(cfg, dict):
        eq_p = cfg['physics']['equation_params']
        bounds = cfg['physics']['bounds']
        x_domain = cfg['physics']['x_domain']
    else:
        eq_p = cfg.physics['equation_params']
        bounds = cfg.physics['bounds']
        x_domain = cfg.physics['x_domain']

    # 1. Paramètres Physiques
    alpha = np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1], batch_size)
    beta  = np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1], batch_size)
    mu    = np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1], batch_size)
    V     = np.random.uniform(eq_p['V'][0],     eq_p['V'][1], batch_size)
    
    # 2. Paramètres de la CI
    A = np.random.uniform(bounds['A'][0], bounds['A'][1], batch_size)
    w0 = 10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1]), batch_size)
    x0 = np.random.uniform(bounds['x0'][0], bounds['x0'][1], batch_size)
    k  = np.random.uniform(bounds['k'][0], bounds['k'][1], batch_size)
    
    # --- MODIFICATION ICI ---
    # On tire entre [1, 3[ => soit 1, soit 2.
    type_id = np.random.randint(1, 3, batch_size) 
    # ------------------------
    
    # 3. Coordonnées Spatiales
    x = np.random.uniform(x_domain[0], x_domain[1], batch_size)
    t = np.zeros(batch_size)

    # 4. Calcul Analytique
    X = x - x0
    exp_ikx = np.exp(1j * k * x)
    
    u_val = np.zeros(batch_size, dtype=np.complex128)
    u_x   = np.zeros(batch_size, dtype=np.complex128)
    
    # --- Gaussian (Type 0) : SUPPRIMÉ ---
    # Le code est toujours là pour mémoire mais ne sera jamais exécuté
    # car type_id ne vaut jamais 0.

    # --- Sech (Type 1) ---
    mask_1 = (type_id == 1)
    if np.any(mask_1):
        arg = X[mask_1] / w0[mask_1]
        M = A[mask_1] / np.cosh(arg)
        M_prime = - (M / w0[mask_1]) * np.tanh(arg)
        
        u_val[mask_1] = M * exp_ikx[mask_1]
        u_x[mask_1]   = exp_ikx[mask_1] * (M_prime + 1j * k[mask_1] * M)

    # --- Tanh (Type 2) ---
    mask_2 = (type_id == 2)
    if np.any(mask_2):
        arg = X[mask_2] / w0[mask_2]
        M = A[mask_2] * np.tanh(arg)
        sech_sq = 1.0 / (np.cosh(arg)**2)
        M_prime = (A[mask_2] / w0[mask_2]) * sech_sq
        
        u_val[mask_2] = M * exp_ikx[mask_2]
        u_x[mask_2]   = exp_ikx[mask_2] * (M_prime + 1j * k[mask_2] * M)

    # 5. Conversion en Tenseurs
    params = np.stack([alpha, beta, mu, V, A, w0, x0, k, type_id.astype(float)], axis=1)
    coords = np.stack([x, t], axis=1)
    
    branch_tensor = torch.tensor(params, dtype=torch.float32).to(device)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    
    u_val_re = torch.tensor(u_val.real, dtype=torch.float32).unsqueeze(1).to(device)
    u_val_im = torch.tensor(u_val.imag, dtype=torch.float32).unsqueeze(1).to(device)
    u_x_re   = torch.tensor(u_x.real, dtype=torch.float32).unsqueeze(1).to(device)
    u_x_im   = torch.tensor(u_x.imag, dtype=torch.float32).unsqueeze(1).to(device)

    return branch_tensor, coords_tensor, u_val_re, u_val_im, u_x_re, u_x_im

def get_pde_batch_cgle(n_samples, cfg, device, t_limit=None):
    """
    Générateur PDE : UNIQUEMENT SECH (1) et TANH (2).
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
    
    # --- MODIFICATION ICI ---
    # torch.randint(low, high) exclut high. Donc [1, 3) => 1 ou 2.
    types = torch.randint(1, 3, (n_samples, 1), device=device).float()
    # ------------------------
    
    # 3. Branch Input
    branch = torch.cat([alpha, beta, mu, V, A, w0, x0, k_wav, types], dim=1)

    # 4. Trunk Input
    x = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
    if t_limit is None: t_limit = cfg.physics['t_max']
    t = torch.rand(n_samples, 1, device=device) * t_limit
    coords = torch.cat([x, t], dim=1).requires_grad_(True)

    # 5. Params Dict pour Physics Loss
    params_dict = {"alpha": alpha, "beta": beta, "mu": mu, "V": V}

    return branch, coords, params_dict
