import torch

def pde_residual_cgle(model, branch, coords, cfg):
    """
    Calcule le résidu de l'équation de Ginzburg-Landau Complexe (1D) :
    
    du/dt + V*du/dx = (1 + i*alpha)*d2u/dx2 + mu*u - (1 + i*beta)*|u|^2*u
    
    Variables :
    u(x,t) est complexe : u_re + i*u_im
    coords : [x, t]
    """
    # 1. Récupération des paramètres physiques
    # Ils sont dans cfg.physics
    alpha = cfg.physics['alpha']
    beta  = cfg.physics['beta']
    mu    = cfg.physics['mu']
    V     = cfg.physics['V'] # Vitesse d'advection (souvent 0 ou petite)

    # 2. Prédiction du modèle
    u_re, u_im = model(branch, coords)
    
    # 3. Calcul des Dérivées Premières (Gradients)
    # coords[:, 0] = x
    # coords[:, 1] = t
    
    grads_re = torch.autograd.grad(u_re, coords, torch.ones_like(u_re), create_graph=True)[0]
    grads_im = torch.autograd.grad(u_im, coords, torch.ones_like(u_im), create_graph=True)[0]
    
    du_dx_re = grads_re[:, 0:1]
    du_dt_re = grads_re[:, 1:2]
    
    du_dx_im = grads_im[:, 0:1]
    du_dt_im = grads_im[:, 1:2]

    # 4. Calcul des Dérivées Secondes (Laplacien 1D spatial uniquement)
    # On dérive du_dx par rapport à coords pour avoir d2u/dx2
    
    grads_2_re = torch.autograd.grad(du_dx_re, coords, torch.ones_like(du_dx_re), create_graph=True)[0]
    grads_2_im = torch.autograd.grad(du_dx_im, coords, torch.ones_like(du_dx_im), create_graph=True)[0]
    
    d2u_dx2_re = grads_2_re[:, 0:1]
    d2u_dx2_im = grads_2_im[:, 0:1]

    # 5. Assemblage des Termes (Arithmétique Complexe)

    # --- A. Terme de Diffusion : (1 + i*alpha) * (Re'' + i*Im'') ---
    # Partie Réelle : 1*Re'' - alpha*Im''
    diff_re = d2u_dx2_re - alpha * d2u_dx2_im
    # Partie Imaginaire : 1*Im'' + alpha*Re''
    diff_im = d2u_dx2_im + alpha * d2u_dx2_re
    
    # --- B. Terme Linéaire : mu * u ---
    lin_re = mu * u_re
    lin_im = mu * u_im
    
    # --- C. Terme Non-Linéaire (Cubique) : -(1 + i*beta) * |u|^2 * u ---
    # Intensité I = Re^2 + Im^2
    I = u_re**2 + u_im**2
    
    # Produit : (1+ib) * (u_re + i*u_im) = (u_re - beta*u_im) + i*(u_im + beta*u_re)
    # Donc avec le signe moins et l'intensité :
    nl_re = -I * (u_re - beta * u_im)
    nl_im = -I * (u_im + beta * u_re)

    # --- D. Terme Advection (passé à droite) : - V * du/dx ---
    adv_re = -V * du_dx_re
    adv_im = -V * du_dx_im

    # 6. Résidu Final
    # Equation : du/dt - (Diff + Lin + NL + Adv) = 0
    # Note : J'ai passé le terme V*du/dx à droite (donc signe moins) pour isoler du/dt
    
    res_re = du_dt_re - (diff_re + lin_re + nl_re + adv_re)
    res_im = du_dt_im - (diff_im + lin_im + nl_im + adv_im)

    return res_re, res_im