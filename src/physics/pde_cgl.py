import torch

def pde_residual_cgle(model, branch, coords, equation_params, cfg):
    """
    Calcule le résidu de l'équation CGL généraliste.
    
    Args:
        model: Le DeepONet
        branch: Les inputs du branch net
        coords: [x, t]
        equation_params: Dictionnaire contenant les tenseurs {alpha, beta, mu, V}
                         (C'est le params_dict retourné par generators.py)
        cfg: La config (utilisée uniquement pour debug ou constantes globales)
    """
    
    # 1. Récupération des paramètres VARIABLES (Tenseurs [Batch, 1])
    # On utilise les vraies valeurs générées pour ce batch spécifique
    alpha = equation_params['alpha']
    beta  = equation_params['beta']
    mu    = equation_params['mu']
    V     = equation_params['V']

    # 2. Prédiction du modèle
    u_re, u_im = model(branch, coords)
    
    # 3. Calcul des Dérivées Premières (Gradients)
    grads_re = torch.autograd.grad(u_re, coords, torch.ones_like(u_re), create_graph=True)[0]
    grads_im = torch.autograd.grad(u_im, coords, torch.ones_like(u_im), create_graph=True)[0]
    
    du_dx_re = grads_re[:, 0:1]
    du_dt_re = grads_re[:, 1:2]
    
    du_dx_im = grads_im[:, 0:1]
    du_dt_im = grads_im[:, 1:2]

    # 4. Calcul des Dérivées Secondes
    grads_2_re = torch.autograd.grad(du_dx_re, coords, torch.ones_like(du_dx_re), create_graph=True)[0]
    grads_2_im = torch.autograd.grad(du_dx_im, coords, torch.ones_like(du_dx_im), create_graph=True)[0]
    
    d2u_dx2_re = grads_2_re[:, 0:1]
    d2u_dx2_im = grads_2_im[:, 0:1]

    # 5. Assemblage des Termes
    
    # --- A. Terme de Diffusion : (1 + i*alpha) * (Re'' + i*Im'') ---
    diff_re = d2u_dx2_re - alpha * d2u_dx2_im
    diff_im = d2u_dx2_im + alpha * d2u_dx2_re
    
    # --- B. Terme Linéaire : mu * u ---
    lin_re = mu * u_re
    lin_im = mu * u_im
    
    # --- C. Terme Non-Linéaire : -(1 + i*beta) * |u|^2 * u ---
    I = u_re**2 + u_im**2
    nl_re = -I * (u_re - beta * u_im)
    nl_im = -I * (u_im + beta * u_re)

    # --- D. Terme Advection : - V * du/dx ---
    # Le signe est correct ici car on l'a passé à droite de l'égalité
    adv_re = -V * du_dx_re
    adv_im = -V * du_dx_im

    # 6. Résidu Final
    res_re = du_dt_re - (diff_re + lin_re + nl_re + adv_re)
    res_im = du_dt_im - (diff_im + lin_im + nl_im + adv_im)

    return res_re, res_im