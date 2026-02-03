import torch

def pde_residual_nlse(model, branch, coords, cfg):
    """
    Calcule le résidu de la NLSE Complète :
    dE/dz = i*Laplacien(E) + i*Kerr*|E|^2*E - i*Dr*|E|^(2K)*E - Kl*|E|^(2(K-1))*E
    """
    # 1. Récupération des constantes depuis la config
    # On normalise ou on prend les valeurs brutes selon ton choix (ici valeurs brutes normalisées)
    K = 8
    C_kerr = 1.0186
    C_plasma = 1.0552
    C_abs = 3.4925e-06
    
    # 2. Prédiction du modèle (E = u + iv)
    u, v = model(branch, coords)
    
    # 3. Calcul des dérivées premières (Gradients)
    grads_u = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    grads_v = torch.autograd.grad(v, coords, torch.ones_like(v), create_graph=True)[0]
    
    du_dr = grads_u[:, 0:1]
    du_dz = grads_u[:, 1:2]
    dv_dr = grads_v[:, 0:1]
    dv_dz = grads_v[:, 1:2]

    # 4. Calcul des dérivées secondes (Laplacien)
    d2u_dr2 = torch.autograd.grad(du_dr, coords, torch.ones_like(du_dr), create_graph=True)[0][:, 0:1]
    d2v_dr2 = torch.autograd.grad(dv_dr, coords, torch.ones_like(dv_dr), create_graph=True)[0][:, 0:1]

    r = coords[:, 0:1]
    # Laplacien cylindrique : d²f/dr² + (1/r)*df/dr
    # Astuce numérique : 1/(r+eps) pour éviter la division par zéro
    lap_u = d2u_dr2 + (1.0 / (r + 1e-9)) * du_dr
    lap_v = d2v_dr2 + (1.0 / (r + 1e-9)) * dv_dr

    # 5. Termes Non-Linéaires (NL)
    # Intensité |E|^2 = u^2 + v^2
    I = u**2 + v**2
    
    # Terme Kerr : i * C_kerr * I * E
    # Re(i * I * (u + iv)) = -I*v
    # Im(i * I * (u + iv)) =  I*u
    kerr_re = -C_kerr * I * v
    kerr_im =  C_kerr * I * u
    
    # Terme Plasma (Défocalisation) : -i * C_plasma * I^K * E
    # Re(-i * I^K * (u + iv)) =  I^K * v
    # Im(-i * I^K * (u + iv)) = -I^K * u
    # Note: K=8 est grand, attention aux explosions numériques si I > 1.0 !
    I_K = torch.pow(I, K)
    plasma_re =  C_plasma * I_K * v
    plasma_im = -C_plasma * I_K * u
    
    # Terme Absorption (Multiphoton) : - C_abs * I^(K-1) * E
    # Re(-C_abs * I^(K-1) * (u + iv)) = -C_abs * I^(K-1) * u
    # Im(-C_abs * I^(K-1) * (u + iv)) = -C_abs * I^(K-1) * v
    I_Km1 = torch.pow(I, K - 1)
    abs_re = -C_abs * I_Km1 * u
    abs_im = -C_abs * I_Km1 * v

    # 6. Assemblage de l'équation (Residuals)
    # Partie Réelle de dE/dz - Partie Réelle du RHS = 0
    # RHS_Re = -lap_v + kerr_re + plasma_re + abs_re
    res_re = du_dz - (-lap_v + kerr_re + plasma_re + abs_re)
    
    # Partie Imaginaire de dE/dz - Partie Imaginaire du RHS = 0
    # RHS_Im = lap_u + kerr_im + plasma_im + abs_im
    res_im = dv_dz - (lap_u + kerr_im + plasma_im + abs_im)

    return res_re, res_im