import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

# Imports CGL spécifiques
from src.physics.pde_cgl import pde_residual_cgle
from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle
from src.utils.solver_cgl import get_ground_truth_CGL 

# --- 1. OUTILS D'AUDIT & DIAGNOSTIC ---

def diagnose_cgle(model, cfg, t_max, threshold=0.05, n_per_type=10):
    """
    Teste le modèle sur les 3 types d'IC avec des paramètres physiques aléatoires.
    """
    device = next(model.parameters()).device
    model.eval()
    
    failed_types = []
    types_map = {0: "Gaussian", 1: "Sech", 2: "Tanh"}
    
    # Gestion robustesse accès config (Dict ou Objet)
    if isinstance(cfg, dict):
        eq_p = cfg['physics']['equation_params']
        bounds = cfg['physics']['bounds']
        x_domain = cfg['physics']['x_domain']
    else:
        eq_p = cfg.physics['equation_params']
        bounds = cfg.physics['bounds']
        x_domain = cfg.physics['x_domain']

    print(f"      🔎 Diagnostic par type (t_max={t_max:.2f})...")

    for type_id, type_name in types_map.items():
        errors = []
        for _ in range(n_per_type):
            # 1. Tirage aléatoire des paramètres
            alpha = np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1])
            beta  = np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1])
            mu    = np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1])
            V     = np.random.uniform(eq_p['V'][0],     eq_p['V'][1])
            
            # 2. Paramètres de l'IC
            A = np.random.uniform(bounds['A'][0], bounds['A'][1])
            w0 = 10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1]))
            x0 = 0.0 
            k  = 1.0 

            p_dict = {
                'alpha': alpha, 'beta': beta, 'mu': mu, 'V': V,
                'A': A, 'w0': w0, 'x0': x0, 'k': k, 'type': type_id
            }
            
            try:
                # Vérité Terrain
                X_grid, T_grid, U_true_cplx = get_ground_truth_CGL(
                    p_dict, 
                    x_domain[0], x_domain[1], 
                    t_max, Nx=256, Nt=None
                )
            except Exception as e:
                continue

            # Prédiction Modèle
            X_flat, T_flat = X_grid.flatten(), T_grid.flatten()
            xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
            
            p_vec = np.array([alpha, beta, mu, V, A, w0, x0, k, float(type_id)])
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

            with torch.no_grad():
                u_re, u_im = model(p_tensor, xt_tensor)
                u_pred_cplx = (u_re + 1j * u_im).cpu().numpy().flatten()
            
            U_true = U_true_cplx.flatten()
            
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

def compute_cgle_loss(model, branch_pde, coords_pde, pde_params, branch_ic, coords_ic, u_true_ic_re, u_true_ic_im, cfg):
    """
    Calcule la Loss Totale : PDE + IC + BC.
    """
    
    # --- A. Loss PDE ---
    r_re, r_im = pde_residual_cgle(model, branch_pde, coords_pde, pde_params, cfg)
    loss_pde = torch.mean(r_re**2 + r_im**2)

    # --- B. Loss IC ---
    p_re, p_im = model(branch_ic, coords_ic)
    l_comp = torch.mean((p_re - u_true_ic_re)**2) + torch.mean((p_im - u_true_ic_im)**2)
    mod_pred = torch.sqrt(p_re**2 + p_im**2 + 1e-9)
    mod_true = torch.sqrt(u_true_ic_re**2 + u_true_ic_im**2 + 1e-9)
    l_mod = torch.mean((mod_pred - mod_true)**2)
    loss_ic = l_comp + l_mod

    # --- C. Loss BC ---
    ic_types = branch_pde[:, 8:9]
    is_periodic = (torch.abs(ic_types - 2.0) > 0.1).float()
    
    # Accès robuste x_domain
    if isinstance(cfg, dict): x_domain = cfg['physics']['x_domain']
    else: x_domain = cfg.physics['x_domain']

    x_min, x_max = x_domain
    t_bc = coords_pde[:, 1:2]
    coords_L = torch.cat([torch.full_like(t_bc, x_min), t_bc], dim=1)
    coords_R = torch.cat([torch.full_like(t_bc, x_max), t_bc], dim=1)
    u_L_re, u_L_im = model(branch_pde, coords_L)
    u_R_re, u_R_im = model(branch_pde, coords_R)
    diff_bc = (u_L_re - u_R_re)**2 + (u_L_im - u_R_im)**2
    loss_bc_masked = torch.mean(is_periodic * diff_bc)
    
    # --- D. Total ---
    # Accès robuste weights
    if isinstance(cfg, dict): weights = cfg['training']['weights']
    else: weights = cfg.training['weights']

    total_loss = weights['pde_loss'] * loss_pde + \
                 weights['ic_loss'] * loss_ic + \
                 weights.get('bc_loss', 1.0) * loss_bc_masked
                 
    return total_loss

# --- 3. BOUCLE D'ENTRAÎNEMENT PAR PALIER ---

def train_step_cgle(model, cfg, t_max, n_iters):
    """
    Entraîne le modèle sur [0, t_max] avec stratégie Adam -> Audit -> Adam Correction -> L-BFGS.
    """
    device = next(model.parameters()).device
    
    # Accès robuste learning rate & batch size
    if isinstance(cfg, dict):
        base_lr = float(cfg['training']['ic_phase']['learning_rate'])
        batch_size_pde = int(cfg['training']['batch_size_pde'])
        batch_size_ic = int(cfg['training']['batch_size_ic'])
    else:
        base_lr = float(cfg.training['ic_phase']['learning_rate'])
        batch_size_pde = int(cfg.training['batch_size_pde'])
        batch_size_ic = int(cfg.training['batch_size_ic'])

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    print(f"\n🔵 PALIER t=[0, {t_max:.2f}] (iters={n_iters})")

    # === PHASE 1 : ENTRAÎNEMENT GLOBAL (Adam) ===
    model.train()
    for i in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        br_pde, co_pde, pde_params = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
        br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device)
        
        loss = compute_cgle_loss(model, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print(f"    Iter {i} | Loss: {loss.item():.2e}")

    # === PHASE 2 : AUDIT ===
    # Accès robuste audit threshold
    threshold = 0.05
    if not isinstance(cfg, dict):
        threshold = cfg.time_marching.get('audit_threshold', 0.05)
    
    failed_types = diagnose_cgle(model, cfg, t_max, threshold=threshold)

    if not failed_types:
        print("    ✅ Audit Global OK. Passage au step suivant.")
        return True

    # === PHASE 3 : CORRECTION CIBLÉE (Adam) ===
    print(f"    🚑 Correction Ciblée Adam sur les types : {failed_types}")
    
    optimizer_focus = optim.Adam(model.parameters(), lr=base_lr * 0.5)
    n_correction = n_iters + 1000
    
    for i in range(n_correction):
        optimizer_focus.zero_grad(set_to_none=True)
        br_pde, co_pde, pde_params = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
        br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device)
        
        loss = compute_cgle_loss(model, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
        loss.backward()
        optimizer_focus.step()

    # Re-Audit après Adam correction
    failed_after_adam = diagnose_cgle(model, cfg, t_max, threshold=0.08)
    
    if not failed_after_adam:
        print("    ✅ Correction Adam réussie !")
        return True

    # === PHASE 4 : ULTIMATE RESCUE (L-BFGS) ===
    print(f"    💀 Adam a échoué. Lancement L-BFGS (Optimisation Second Ordre)...")
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), 
                                  lr=1.0,               
                                  max_iter=100,          
                                  max_eval=25,
                                  history_size=50,
                                  tolerance_grad=1e-5,
                                  tolerance_change=1.0 * np.finfo(float).eps,
                                  line_search_fn="strong_wolfe") 

    n_lbfgs_steps = 35 
    
    for l_step in range(n_lbfgs_steps):
        # Batch fixe pour la closure
        br_pde, co_pde, pde_params = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
        br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device)

        def closure():
            optimizer_lbfgs.zero_grad()
            loss_val = compute_cgle_loss(model, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
            loss_val.backward()
            return loss_val

        loss_lbfgs = optimizer_lbfgs.step(closure)
        
        if l_step % 2 == 0:
            print(f"      L-BFGS Step {l_step}/{n_lbfgs_steps} | Loss: {loss_lbfgs.item():.2e}")

    # Re-Audit Final
    failed_final = diagnose_cgle(model, cfg, t_max, threshold=0.15)
    
    if not failed_final:
        print("    ✅ Correction L-BFGS réussie !")
        return True
    else:
        print(f"    ⚠️ ECHEC FINAL : Types résistants {failed_final}. On avance quand même.")
        return False

# --- 4. MAIN TRAINER ---

# --- 4. MAIN TRAINER ---

def train_cgle_curriculum(model, cfg):
    # Gestion robustesse dossier save
    if isinstance(cfg, dict):
        save_dir = cfg['training'].get('save_dir', "outputs/checkpoints_cgl")
        ic_iter = int(cfg['training']['ic_phase']['iterations'])
        
        t_max_phys = cfg['physics']['t_max']
        dt_step = cfg['time_marching'].get('dt_step', 0.5)
        iters_per_step = cfg['time_marching'].get('iters_per_step', 3000)
    else:
        # Si ConfigObj
        # 1. On récupère le dictionnaire 'training' via l'attribut
        training_dict = cfg.training 
        
        save_dir = training_dict.get('save_dir', "outputs/checkpoints_cgl")
        
        # 2. CORRECTION ICI : ic_phase est DANS training_dict
        ic_iter = int(training_dict['ic_phase']['iterations']) 
        
        t_max_phys = cfg.physics['t_max']
        
        # time_marching est à la racine (suite à notre modif yaml)
        # Mais attention si tu utilises ConfigObj, time_marching est un attribut qui contient un dict
        dt_step = cfg.time_marching.get('dt_step', 0.5)
        iters_per_step = cfg.time_marching.get('iters_per_step', 3000)

    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Warmup IC
    print("🧊 WARMUP (IC Only)...")
    print(f"   Iterations: {ic_iter}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(ic_iter):
        optimizer.zero_grad()
        br_ic, co_ic, tr_re, tr_im, _, _ = get_ic_batch_cgle(4096, cfg, next(model.parameters()).device)
        
        p_re, p_im = model(br_ic, co_ic)
        loss = torch.mean((p_re - tr_re)**2 + (p_im - tr_im)**2)
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print(f"   Warmup Iter {i}/{ic_iter} | Loss: {loss.item():.2e}")
            
    print("✅ Warmup terminé.")

    # 2. Time Marching
    current_t = dt_step
    while current_t <= t_max_phys + 1e-9:
        
        success = train_step_cgle(model, cfg, current_t, n_iters=iters_per_step)
        
        ckpt_path = os.path.join(save_dir, f"ckpt_t{current_t:.2f}.pth")
        torch.save(model.state_dict(), ckpt_path)
        current_t += dt_step

    print("🏁 Entraînement CGL Terminé.")