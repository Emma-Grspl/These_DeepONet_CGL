import torch

def pde_residual_corrected(model, branch, coords, cfg):
    coords.requires_grad_(True)
    E_re, E_im = model(branch, coords)

    # 1. Gradients Premiers
    grads_re = torch.autograd.grad(E_re, coords, torch.ones_like(E_re), create_graph=True)[0]
    grads_im = torch.autograd.grad(E_im, coords, torch.ones_like(E_im), create_graph=True)[0]

    u_z = grads_re[:, 1:2]
    v_z = grads_im[:, 1:2]
    u_r = grads_re[:, 0:1]
    v_r = grads_im[:, 0:1]

    # 2. Gradients Seconds (Laplacien Radial)
    u_rr = torch.autograd.grad(u_r, coords, torch.ones_like(u_r), create_graph=True)[0][:, 0:1]
    v_rr = torch.autograd.grad(v_r, coords, torch.ones_like(v_r), create_graph=True)[0][:, 0:1]

    r = coords[:, 0:1]
    # Ajout epsilon pour éviter div/0 à r=0
    lap_u = u_rr + (1.0/(r+1e-6)) * u_r
    lap_v = v_rr + (1.0/(r+1e-6)) * v_r

    # 3. Schrödinger Paraxiale
    k_scale = 2.0 * cfg.physics['k']

    res_re = -k_scale * v_z + lap_u
    res_im =  k_scale * u_z + lap_v

    return res_re, res_im
