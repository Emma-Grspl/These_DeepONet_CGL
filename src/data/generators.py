import torch
import numpy as np

def get_ic_batch_sobolev(n_samples, cfg, device):
    # cfg est un objet, on accède aux dicts via attributs
    b = cfg.physics['bounds']

    A = torch.rand(n_samples, 1, device=device) * (b['A'][1] - b['A'][0]) + b['A'][0]
    w0_min, w0_max = b['w0']
    w0 = 10 ** (torch.rand(n_samples, 1, device=device) * np.log10(w0_max/w0_min) + np.log10(w0_min))
    f = torch.rand(n_samples, 1, device=device) * (b['f'][1] - b['f'][0]) + b['f'][0]
    branch = torch.cat([A, w0, f], dim=1)

    # Sniper sampling (80% in beam, 20% outside)
    n_in = int(0.8 * n_samples)
    n_out = n_samples - n_in
    r_in = torch.rand(n_in, 1, device=device) * 2.5 * w0[:n_in]
    r_out = torch.rand(n_out, 1, device=device) * cfg.physics['r_max']
    r = torch.cat([r_in, r_out], dim=0)
    z = torch.zeros_like(r)
    coords = torch.cat([r, z], dim=1).requires_grad_(True)

    # Target Values & Gradients
    k = cfg.physics['k']
    arg_gauss = -(r**2)/(w0**2)
    amp = torch.sqrt(A) * torch.exp(arg_gauss)
    phase = -(k * r**2)/(2*f)

    cos_p = torch.cos(phase); sin_p = torch.sin(phase)
    t_re = amp * cos_p; t_im = amp * sin_p

    # Sobolev Targets
    dA_dr = amp * (-2*r / w0**2)
    dP_dr = -(k * r) / f
    dt_re = dA_dr * cos_p - amp * sin_p * dP_dr
    dt_im = dA_dr * sin_p + amp * cos_p * dP_dr

    return branch, coords, t_re, t_im, dt_re, dt_im

def get_pde_batch_z_limited(n_samples, cfg, device, z_limit):
    b = cfg.physics['bounds']

    A = torch.rand(n_samples, 1, device=device) * (b['A'][1] - b['A'][0]) + b['A'][0]
    w0 = torch.rand(n_samples, 1, device=device) * (b['w0'][1] - b['w0'][0]) + b['w0'][0]
    f = torch.rand(n_samples, 1, device=device) * (b['f'][1] - b['f'][0]) + b['f'][0]
    branch = torch.cat([A, w0, f], dim=1)

    r = torch.abs(torch.randn(n_samples, 1, device=device)) * (cfg.physics['r_max'] / 2.0)
    r = torch.clamp(r, 0, cfg.physics['r_max'])
    z = torch.rand(n_samples, 1, device=device) * z_limit
    coords = torch.cat([r, z], dim=1)

    return branch, coords
