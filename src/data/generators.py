import numpy as np   # <--- C'est cette ligne qui manque !
import torch
def get_ic_batch_cgle(batch_size, cfg, device):
    """
    Génère un batch pour la Condition Initiale (t=0) AVEC les dérivées spatiales (Sobolev).
    Retourne: branch, coords, u_val_re, u_val_im, u_x_re, u_x_im
    """
    if isinstance(cfg, dict):
        eq_p = cfg['physics']['equation_params']
        bounds = cfg['physics']['bounds']
        x_domain = cfg['physics']['x_domain']
    else:
        eq_p = cfg.physics['equation_params']
        bounds = cfg.physics['bounds']
        x_domain = cfg.physics['x_domain']

    # 1. Paramètres Physiques (Branch Input)
    alpha = np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1], batch_size)
    beta  = np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1], batch_size)
    mu    = np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1], batch_size)
    V     = np.random.uniform(eq_p['V'][0],     eq_p['V'][1], batch_size)
    
    # 2. Paramètres de la CI
    A = np.random.uniform(bounds['A'][0], bounds['A'][1], batch_size)
    w0 = 10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1]), batch_size)
    x0 = np.random.uniform(bounds['x0'][0], bounds['x0'][1], batch_size)
    k  = np.random.uniform(bounds['k'][0], bounds['k'][1], batch_size)
    
    # Choix du type de CI (0: Gaussian, 1: Sech, 2: Tanh)
    type_id = np.random.randint(0, 3, batch_size)
    
    # 3. Coordonnées Spatiales (Trunk Input)
    x = np.random.uniform(x_domain[0], x_domain[1], batch_size)
    t = np.zeros(batch_size) # t=0

    # 4. Calcul Analytique des Valeurs (u) et Dérivées (u_x)
    # u(x) = M(x) * exp(i*k*x)
    # u'(x) = M'(x)*exp(i*k*x) + i*k*M(x)*exp(i*k*x) = exp(i*k*x) * [M'(x) + i*k*M(x)]
    
    X = x - x0
    exp_ikx = np.exp(1j * k * x)
    
    u_val = np.zeros(batch_size, dtype=np.complex128)
    u_x   = np.zeros(batch_size, dtype=np.complex128)
    
    # --- Gaussian: M(x) = A * exp(-X^2 / w0^2) ---
    mask_0 = (type_id == 0)
    if np.any(mask_0):
        # M = A * exp(-X^2/w0^2)
        M = A[mask_0] * np.exp(-(X[mask_0]**2) / (w0[mask_0]**2))
        # M' = M * (-2*X/w0^2)
        M_prime = M * (-2 * X[mask_0] / (w0[mask_0]**2))
        
        u_val[mask_0] = M * exp_ikx[mask_0]
        u_x[mask_0]   = exp_ikx[mask_0] * (M_prime + 1j * k[mask_0] * M)

    # --- Sech: M(x) = A * sech(X/w0) ---
    mask_1 = (type_id == 1)
    if np.any(mask_1):
        arg = X[mask_1] / w0[mask_1]
        # M = A / cosh(arg)
        M = A[mask_1] / np.cosh(arg)
        # M' = -A/w0 * sech(arg) * tanh(arg) = -M/w0 * tanh(arg)
        M_prime = - (M / w0[mask_1]) * np.tanh(arg)
        
        u_val[mask_1] = M * exp_ikx[mask_1]
        u_x[mask_1]   = exp_ikx[mask_1] * (M_prime + 1j * k[mask_1] * M)

    # --- Tanh (Hole/Shock): M(x) = A * tanh(X/w0) ---
    mask_2 = (type_id == 2)
    if np.any(mask_2):
        arg = X[mask_2] / w0[mask_2]
        # M = A * tanh(arg)
        M = A[mask_2] * np.tanh(arg)
        # M' = A/w0 * sech^2(arg)
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