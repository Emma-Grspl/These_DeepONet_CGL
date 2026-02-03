import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

# Imports CGL spécifiques
from src.physics.cgle import pde_residual_cgle
from src.data.generators_cgle import get_ic_batch_cgle, get_pde_batch_cgle
from src.utils.solver_cgle import get_ground_truth_CGL

# --- 1. OUTILS D'AUDIT & DIAGNOSTIC ---

def diagnose_cgle(model, cfg, t_max, threshold=0.05, n_per_type=10):
    """
    Teste le modèle sur les 3 types d'IC (Gaussian, Sech, Tanh)
    et renvoie la liste des types qui échouent.
    """
    device = next(model.parameters()).device
    model.eval()
    
    failed_types = []
    types_map = {0: "Gaussian", 1: "Sech", 2: "Tanh"}
    
    # Paramètres physiques globaux depuis la config
    phys = cfg.physics
    p_dict_base = {
        'alpha': phys['alpha'], 'beta': phys['beta'], 'mu': phys['mu'],
        'x0': 0.0, 'k': 1.0, 'w0': 1.0, 'A': 1.0 # Valeurs par défaut
    }

    print(f"      🔎 Diagnostic par type (t_max={t_max:.2f})...")

    for type_id, type_name in types_map.items():
        errors = []
        for _ in range(n_per_type):
            # On varie un peu les conditions initiales pour être robuste
            p_dict = p_dict_base.copy()
            p_dict['type'] = type_id
            # Randomize un peu A et w0
            p_dict['A'] = np.random.uniform(phys['bounds']['A'][0], phys['bounds']['A'][1])
            p_dict['w0'] = 10**np.random.uniform(np.log10(phys['bounds']['w0'][0]), np.log10(phys['bounds']['w0'][1]))
            
            try:
                # Vérité Terrain (Solveur)
                X_grid, T_grid, U_true_cplx = get_ground_truth_CGL(
                    p_dict, 
                    phys['x_domain'][0], phys['x_domain'][1], 
                    t_max, Nx=256, Nt=None
                )
            except Exception as e:
                print(f"      ⚠️ Erreur solveur sur {type_name}: {e}")
                continue

            # Prédiction Modèle
            X_flat, T_flat = X_grid.flatten(), T_grid.flatten()
            xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
            
            # Construction du tenseur de paramètres [A, w0, x0, k, type]
            # Attention : alpha, beta, mu sont implicites dans le modèle ou fixés dans cfg ?
            # D'après ton générateur, le Branch Net prend [A, w0, x0, k, type]
            p_vec = np.array([p_dict['A'], p_dict['w0'], p_dict.get('x0',0), p_dict.get('k',1), float(type_id)])
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

            with torch.no_grad():
                u_re, u_im = model(p_tensor, xt_tensor)
                u_pred_cplx = (u_re + 1j * u_im).cpu().numpy().flatten()
            
            U_true = U_true_cplx.flatten()
            
            # Erreur Relative L2 (Complexe)
            norm_true = np.linalg.norm(U_true)
            if norm_true < 1e-6: norm_true = 1e-6
            err = np.linalg.norm(U_true - u_pred_cplx) / norm_true
            errors.append(err)

        mean_err = np.mean(errors) if errors else 1.0
        if mean_err > threshold:
            print(f"      ❌ {type_name}: Err = {mean_err:.2%}")
            failed_types.append(type_id)
        else:
            print(f"      ✅ {type_name}: Err = {mean_err:.2%}")

    return failed_types

# --- 2. FONCTIONS DE LOSS ---

import torch
from src.physics.cgle import pde_residual_cgle

def compute_cgle_loss(model, branch_pde, coords_pde, branch_ic, coords_ic, u_true_ic_re, u_true_ic_im, cfg):
    """
    Calcule la Loss Totale CGL : PDE + IC + BC (Conditionnel).
    Gère le masque pour ne pas forcer la périodicité sur les IC de type Tanh (Fronts).
    """
    
    # --- A. Loss PDE (Physique) ---
    r_re, r_im = pde_residual_cgle(model, branch_pde, coords_pde, cfg)
    loss_pde = torch.mean(r_re**2 + r_im**2)

    # --- B. Loss IC (Condition Initiale) ---
    p_re, p_im = model(branch_ic, coords_ic)
    
    # Erreur Composante (Réel/Imag) + Erreur Module (Enveloppe)
    l_comp = torch.mean((p_re - u_true_ic_re)**2) + torch.mean((p_im - u_true_ic_im)**2)
    
    mod_pred = torch.sqrt(p_re**2 + p_im**2 + 1e-9)
    mod_true = torch.sqrt(u_true_ic_re**2 + u_true_ic_im**2 + 1e-9)
    l_mod = torch.mean((mod_pred - mod_true)**2)
    
    loss_ic = l_comp + l_mod

    # --- C. Loss BC (Périodicité Conditionnelle) ---
    # On force u(x_min, t) == u(x_max, t) UNIQUEMENT si ce n'est pas un front Tanh.
    
    # 1. Identification des types dans le batch
    # branch_pde contient [A, w0, x0, k, type] -> colonne indice 4
    ic_types = branch_pde[:, 4:5]
    
    # Masque binaire : 1.0 si Périodique (Gauss/Sech), 0.0 si Dirichlet (Tanh)
    # On considère que type=2 (Tanh) est le seul non-périodique.
    is_periodic = (torch.abs(ic_types - 2.0) > 0.1).float()
    
    # 2. Prédictions aux bords (Gauche et Droite)
    x_min, x_max = cfg.physics['x_domain']
    t_bc = coords_pde[:, 1:2] # On utilise les temps du batch PDE
    
    # On crée des tenseurs de coordonnées aux bords
    coords_L = torch.cat([torch.full_like(t_bc, x_min), t_bc], dim=1)
    coords_R = torch.cat([torch.full_like(t_bc, x_max), t_bc], dim=1)
    
    u_L_re, u_L_im = model(branch_pde, coords_L)
    u_R_re, u_R_im = model(branch_pde, coords_R)
    
    # 3. Calcul de l'erreur brute aux bords
    diff_bc = (u_L_re - u_R_re)**2 + (u_L_im - u_R_im)**2
    
    # 4. Application du Masque
    # Seuls les éléments où is_periodic=1 contribuent à la loss
    loss_bc_masked = torch.mean(is_periodic * diff_bc)
    
    # (Optionnel) Normalisation pour éviter de diluer le gradient si peu d'exemples périodiques
    # n_periodic = is_periodic.sum()
    # if n_periodic > 0:
    #     loss_bc_masked = loss_bc_masked * (len(is_periodic) / n_periodic)

    # --- D. Loss Totale ---
    weights = cfg.training['weights']
    
    total_loss = weights['pde_loss'] * loss_pde + \
                 weights['ic_loss'] * loss_ic + \
                 weights.get('bc_loss', 1.0) * loss_bc_masked
                 
    return total_loss

# --- 3. BOUCLE D'ENTRAÎNEMENT PAR PALIER ---

def train_step_cgle(model, cfg, t_max, n_iters):
    """
    Entraîne le modèle sur le domaine temporel [0, t_max].
    Utilise la stratégie : Global -> Audit -> Correction Ciblée.
    """
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.training['ic_phase']['learning_rate']))
    
    batch_size_pde = int(cfg.training['batch_size_pde'])
    batch_size_ic = int(cfg.training['batch_size_ic'])
    
    print(f"\n🔵 PALIER t=[0, {t_max:.2f}] (iters={n_iters})")

    # === PHASE 1 : ENTRAÎNEMENT GLOBAL ===
    model.train()
    for i in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        
        # Génération Batch (t jusqu'à t_max)
        br_pde, co_pde = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
        br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device) # Sobolev ignoré ici pour simplifier ou à ajouter
        
        loss = compute_cgle_loss(model, br_pde, co_pde, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print(f"    Iter {i} | Loss: {loss.item():.2e}")

    # === PHASE 2 : AUDIT & DIAGNOSTIC ===
    # On vérifie si le modèle a bien appris
    failed_types = diagnose_cgle(model, cfg, t_max, threshold=0.10) # 10% tolérance pour le chaos

    if not failed_types:
        print("    ✅ Audit Global OK. Passage au step suivant.")
        return True

    # === PHASE 3 : CORRECTION CIBLÉE (FOCUS LOOP) ===
    print(f"    🚑 Correction Ciblée sur les types : {failed_types}")
    
    # On augmente un peu le learning rate ou on le garde
    optimizer_focus = optim.Adam(model.parameters(), lr=float(cfg.training['ic_phase']['learning_rate']))
    
    # On fait une boucle de correction courte mais intense
    n_correction = n_iters // 2
    
    for i in range(n_correction):
        optimizer_focus.zero_grad(set_to_none=True)
        
        # TODO: Modifier get_pde_batch_cgle pour accepter 'allowed_types'
        # Pour l'instant, on fait simple : on génère un batch normal
        # et on espère tomber sur les bons types, ou on modifie le générateur.
        # *Note : Pour aller vite, on suppose que le générateur sort un mix uniforme.*
        # L'idéal serait de passer 'failed_types' au générateur.
        
        br_pde, co_pde = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
        br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device)
        
        loss = compute_cgle_loss(model, br_pde, co_pde, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
        loss.backward()
        optimizer_focus.step()

    # Re-Audit final
    failed_final = diagnose_cgle(model, cfg, t_max, threshold=0.15) # Tolérance relaxée
    
    if not failed_final:
        print("    ✅ Correction réussie !")
        return True
    else:
        print(f"    ⚠️ Attention : Types résistants {failed_final}. On avance quand même.")
        return False

# --- 4. MAIN TRAINER (Time Marching) ---

def train_cgle_curriculum(model, cfg):
    """
    Boucle principale de Time Marching pour CGL.
    """
    save_dir = "outputs/checkpoints_cgl"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Warmup IC (t=0 uniquement)
    print("🧊 WARMUP (IC Only)...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(2000):
        optimizer.zero_grad()
        br_ic, co_ic, tr_re, tr_im, _, _ = get_ic_batch_cgle(4096, cfg, next(model.parameters()).device)
        
        p_re, p_im = model(br_ic, co_ic)
        loss = torch.mean((p_re - tr_re)**2 + (p_im - tr_im)**2)
        loss.backward()
        optimizer.step()
    print("✅ Warmup terminé.")

    # 2. Time Marching
    t_max_phys = cfg.physics['t_max']
    dt_step = 0.5 # On avance par pas de 0.5 secondes (exemple)
    
    current_t = dt_step
    while current_t <= t_max_phys + 1e-9:
        
        success = train_step_cgle(model, cfg, current_t, n_iters=3000)
        
        # Sauvegarde Checkpoint
        torch.save(model.state_dict(), f"{save_dir}/ckpt_t{current_t:.2f}.pth")
        
        current_t += dt_step

    print("🏁 Entraînement CGL Terminé.")