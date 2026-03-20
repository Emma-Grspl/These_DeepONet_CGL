import numpy as np
import torch
from src.physics.pde_cgl import pde_residual_cgle


def _sample_cgl_branch(n_samples, cfg, device):
    """Tire un batch de paramètres CGL cohérent avec le domaine d'entraînement."""
    b = cfg['physics']['bounds'] if isinstance(cfg, dict) else cfg.physics['bounds']
    eq_p = cfg['physics']['equation_params'] if isinstance(cfg, dict) else cfg.physics['equation_params']

    alpha = torch.rand(n_samples, 1, device=device) * (eq_p['alpha'][1] - eq_p['alpha'][0]) + eq_p['alpha'][0]
    beta = torch.rand(n_samples, 1, device=device) * (eq_p['beta'][1] - eq_p['beta'][0]) + eq_p['beta'][0]
    mu = torch.rand(n_samples, 1, device=device) * (eq_p['mu'][1] - eq_p['mu'][0]) + eq_p['mu'][0]
    V = torch.rand(n_samples, 1, device=device) * (eq_p['V'][1] - eq_p['V'][0]) + eq_p['V'][0]

    A = torch.rand(n_samples, 1, device=device) * (b['A'][1] - b['A'][0]) + b['A'][0]
    w0 = 10 ** (torch.rand(n_samples, 1, device=device) * np.log10(b['w0'][1] / b['w0'][0]) + np.log10(b['w0'][0]))
    x0 = torch.rand(n_samples, 1, device=device) * (b['x0'][1] - b['x0'][0]) + b['x0'][0]
    k_wav = torch.rand(n_samples, 1, device=device) * (b['k'][1] - b['k'][0]) + b['k'][0]
    types = torch.zeros((n_samples, 1), device=device).float()

    return torch.cat([alpha, beta, mu, V, A, w0, x0, k_wav, types], dim=1)


def _params_dict_from_branch(branch):
    return {"alpha": branch[:, 0:1], "beta": branch[:, 1:2], "mu": branch[:, 2:3], "V": branch[:, 3:4]}


def _sample_causal_times(n_samples, device, t_prev, t_curr):
    if t_prev <= 1e-5:
        return torch.rand(n_samples, 1, device=device) * t_curr

    n_past = int(0.3 * n_samples)
    t_past = torch.rand(n_past, 1, device=device) * t_prev
    t_front = torch.rand(n_samples - n_past, 1, device=device) * (t_curr - t_prev) + t_prev
    return torch.cat([t_past, t_front], dim=0)


def _sample_coords_from_branch(branch, cfg, device, t_prev, t_curr):
    x_min, x_max = cfg['physics']['x_domain'] if isinstance(cfg, dict) else cfg.physics['x_domain']
    n_samples = branch.shape[0]
    w0 = branch[:, 5:6]
    x0 = branch[:, 6:7]
    t = _sample_causal_times(n_samples, device, t_prev, t_curr)

    n_center = int(0.8 * n_samples)
    W_t = w0[:n_center] * torch.sqrt(1.0 + (2.0 * t[:n_center])**2)
    x_center = x0[:n_center] + torch.randn(n_center, 1, device=device) * W_t * 1.5
    x_uniform = torch.rand(n_samples - n_center, 1, device=device) * (x_max - x_min) + x_min
    x = torch.cat([x_center, x_uniform], dim=0)
    x = torch.clamp(x, x_min, x_max)

    idx = torch.randperm(n_samples, device=device)
    branch, x, t = branch[idx], x[idx], t[idx]
    coords = torch.cat([x, t], dim=1).requires_grad_(True)
    return branch, coords, _params_dict_from_branch(branch)


def _cases_to_branch(case_rows, cfg, device, n_samples):
    if not case_rows or n_samples <= 0:
        return None

    idx = np.random.randint(0, len(case_rows), size=n_samples)
    selected = np.array([[case_rows[i][k] for k in ['alpha', 'beta', 'mu', 'V', 'A', 'w0', 'x0', 'k', 'type']] for i in idx], dtype=np.float32)
    return torch.tensor(selected, dtype=torch.float32, device=device)

def get_ic_batch_cgle(batch_size, cfg, device):
    """
    Génère un batch de CI (t=0) : UNIQUEMENT GAUSSIENNE (0).
    Sert uniquement à fournir des coordonnées pour le calcul des BC (Bords).
    """
    eq_p = cfg['physics']['equation_params'] if isinstance(cfg, dict) else cfg.physics['equation_params']
    bounds = cfg['physics']['bounds'] if isinstance(cfg, dict) else cfg.physics['bounds']
    x_domain = cfg['physics']['x_domain'] if isinstance(cfg, dict) else cfg.physics['x_domain']

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
    
    type_id = np.zeros(batch_size, dtype=np.int32) # GAUSSIENNE SEULEMENT
    
    # 3. Coordonnées Spatiales (Focus 80% au centre)
    x = np.zeros(batch_size)
    n_center = int(0.8 * batch_size)
    x[:n_center] = np.random.normal(loc=x0[:n_center], scale=w0[:n_center]*1.5, size=n_center)
    x[n_center:] = np.random.uniform(x_domain[0], x_domain[1], batch_size - n_center)
    x = np.clip(x, x_domain[0], x_domain[1])
    
    t = np.zeros(batch_size)

    # 4. Conversion en Tenseurs
    params = np.stack([alpha, beta, mu, V, A, w0, x0, k, type_id.astype(float)], axis=1)
    coords = np.stack([x, t], axis=1)
    
    branch_tensor = torch.tensor(params, dtype=torch.float32).to(device)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)

    return branch_tensor, coords_tensor


def get_pde_batch_cgle_causal(n_samples, cfg, device, t_prev, t_curr):
    """
    Générateur PDE avec Time Marching Causal (30% mémoire passé, 70% front actif)
    et Focus Spatial Dynamique (Suit l'étalement W(t) de la Gaussienne).
    """
    branch = _sample_cgl_branch(n_samples, cfg, device)
    return _sample_coords_from_branch(branch, cfg, device, t_prev, t_curr)


def get_pde_batch_cgle_causal_mixed(n_samples, cfg, device, t_prev, t_curr, case_groups=None, group_weights=None):
    """Batch PDE causal avec mélange de cas ciblés et tirage global."""
    if not case_groups:
        return get_pde_batch_cgle_causal(n_samples, cfg, device, t_prev, t_curr)

    weights = {'hard': 0.5, 'medium': 0.2, 'global': 0.3}
    if group_weights:
        weights.update(group_weights)

    n_hard = int(n_samples * weights['hard']) if case_groups.get('hard') else 0
    n_medium = int(n_samples * weights['medium']) if case_groups.get('medium') else 0
    n_global = max(0, n_samples - n_hard - n_medium)

    chunks = []
    hard_branch = _cases_to_branch(case_groups.get('hard', []), cfg, device, n_hard)
    if hard_branch is not None:
        chunks.append(hard_branch)
    medium_branch = _cases_to_branch(case_groups.get('medium', []), cfg, device, n_medium)
    if medium_branch is not None:
        chunks.append(medium_branch)
    if n_global > 0:
        chunks.append(_sample_cgl_branch(n_global, cfg, device))

    if not chunks:
        return get_pde_batch_cgle_causal(n_samples, cfg, device, t_prev, t_curr)

    branch = torch.cat(chunks, dim=0)
    if branch.shape[0] < n_samples:
        extra = _sample_cgl_branch(n_samples - branch.shape[0], cfg, device)
        branch = torch.cat([branch, extra], dim=0)
    elif branch.shape[0] > n_samples:
        idx = torch.randperm(branch.shape[0], device=device)[:n_samples]
        branch = branch[idx]

    return _sample_coords_from_branch(branch, cfg, device, t_prev, t_curr)


def get_pde_batch_cgle_global(n_samples, cfg, device, t_max_local):
    """
    Générateur PDE Global (Uniforme sur tout [0, t_max_local]).
    Avec Focus Spatial Dynamique adapté au temps.
    """
    x_min, x_max = cfg['physics']['x_domain'] if isinstance(cfg, dict) else cfg.physics['x_domain']
    branch = _sample_cgl_branch(n_samples, cfg, device)
    w0 = branch[:, 5:6]
    x0 = branch[:, 6:7]

    # 3. Échantillonnage Temporel Uniforme D'ABORD
    t = torch.rand(n_samples, 1, device=device) * t_max_local

    # 4. Échantillonnage Spatial Focus DYNAMIQUE (80% centre)
    n_center = int(0.8 * n_samples)
    
    W_t = w0[:n_center] * torch.sqrt(1.0 + (2.0 * t[:n_center])**2)
    
    x_center = x0[:n_center] + torch.randn(n_center, 1, device=device) * W_t * 1.5
    x_uniform = torch.rand(n_samples - n_center, 1, device=device) * (x_max - x_min) + x_min
    
    x = torch.cat([x_center, x_uniform], dim=0)
    x = torch.clamp(x, x_min, x_max)

    # 5. Shuffle final
    idx = torch.randperm(n_samples)
    branch, x, t = branch[idx], x[idx], t[idx]

    coords = torch.cat([x, t], dim=1).requires_grad_(True)
    return branch, coords, _params_dict_from_branch(branch)


def get_interface_batch_cgle(n_samples, cfg, device, t_value):
    """Batch pour les pertes intégrales et la continuité causale à temps fixé."""
    x_min, x_max = cfg['physics']['x_domain'] if isinstance(cfg, dict) else cfg.physics['x_domain']
    t_tensor = torch.tensor(float(t_value), device=device)

    branch = _sample_cgl_branch(n_samples, cfg, device)
    w0 = branch[:, 5:6]
    x0 = branch[:, 6:7]

    W_t = w0 * torch.sqrt(1.0 + (2.0 * t_tensor) ** 2)
    n_center = int(0.8 * n_samples)
    x_center = x0[:n_center] + torch.randn(n_center, 1, device=device) * W_t[:n_center] * 1.5
    x_uniform = torch.rand(n_samples - n_center, 1, device=device) * (x_max - x_min) + x_min
    x = torch.cat([x_center, x_uniform], dim=0)
    x = torch.clamp(x, x_min, x_max)

    t = torch.full((n_samples, 1), float(t_value), device=device)
    idx = torch.randperm(n_samples, device=device)
    branch = branch[idx]
    coords = torch.cat([x[idx], t[idx]], dim=1).requires_grad_(True)
    return branch, coords, _params_dict_from_branch(branch)


def get_mass_balance_batch_cgle(n_cases, n_x, cfg, device, t_low, t_high):
    """Grille régulière en x pour approximer les intégrales de balance de masse."""
    x_min, x_max = cfg['physics']['x_domain'] if isinstance(cfg, dict) else cfg.physics['x_domain']
    branch_cases = _sample_cgl_branch(n_cases, cfg, device)

    if t_high <= t_low:
        t_cases = torch.full((n_cases, 1), float(t_high), device=device)
    else:
        t_cases = torch.rand(n_cases, 1, device=device) * (t_high - t_low) + t_low

    x_grid = torch.linspace(x_min, x_max, n_x, device=device).view(1, n_x, 1).repeat(n_cases, 1, 1)
    t_grid = t_cases.view(n_cases, 1, 1).repeat(1, n_x, 1)
    coords = torch.cat([x_grid, t_grid], dim=2).view(-1, 2).requires_grad_(True)
    branch = branch_cases.unsqueeze(1).repeat(1, n_x, 1).view(-1, branch_cases.shape[1])
    return branch, coords, _params_dict_from_branch(branch), n_cases, n_x


def get_rar_batch(model, cfg, device, t_prev, t_curr, n_candidates=10000, n_selected=2000):
    """Génère des points PDE là où le résidu est fort sur la zone temporelle actuelle."""
    b_p, c_p, p_p = get_pde_batch_cgle_causal(n_candidates, cfg, device, t_prev, t_curr)
    
    with torch.set_grad_enabled(True):
        rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
        ur, ui = model(b_p, c_p)
        amp_sq = ur**2 + ui**2
        amp_grads = torch.autograd.grad(amp_sq.sum(), c_p, create_graph=False)[0]
        amp_grad_norm = torch.sqrt(amp_grads[:, 0:1] ** 2 + amp_grads[:, 1:2] ** 2 + 1e-12)

        residual = torch.sqrt(rr**2 + ri**2).detach()
        energy_boost = amp_sq.detach() / (amp_sq.detach().mean() + 1e-9)
        grad_boost = amp_grad_norm.detach() / (amp_grad_norm.detach().mean() + 1e-9)
        residual = residual * (1.0 + 0.5 * energy_boost + 0.25 * grad_boost)
    
    _, indices = torch.topk(residual.view(-1), n_selected)
    return b_p[indices], c_p[indices].detach().requires_grad_(True), {k: v[indices].detach() for k, v in p_p.items()}
