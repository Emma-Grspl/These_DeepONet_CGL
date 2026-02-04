import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

# Imports CGL spécifiques
from src.physics.pde_cgl import pde_residual_cgle
from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle
from src.utils.solver_cgl import get_ground_truth_CGL 

# ==============================================================================
# 1. OUTILS D'AUDIT (GLOBAL & SPÉCIFIQUE)
# ==============================================================================

def audit_global_fast(model, cfg, t_max, threshold=0.05, n_samples=20):
    device = next(model.parameters()).device
    model.eval()
    errors = []
    
    if isinstance(cfg, dict):
        eq_p = cfg['physics']['equation_params']
        bounds = cfg['physics']['bounds']
        x_domain = cfg['physics']['x_domain']
    else:
        eq_p = cfg.physics['equation_params']
        bounds = cfg.physics['bounds']
        x_domain = cfg.physics['x_domain']

    for _ in range(n_samples):
        try:
            alpha = np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1])
            beta  = np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1])
            mu    = np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1])
            V     = np.random.uniform(eq_p['V'][0],     eq_p['V'][1])
            A = np.random.uniform(bounds['A'][0], bounds['A'][1])
            w0 = 10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1]))
            x0, k = 0.0, 1.0
            type_id = np.random.choice([0, 1, 2]) 

            p_dict = {'alpha': alpha, 'beta': beta, 'mu': mu, 'V': V, 'A': A, 'w0': w0, 'x0': x0, 'k': k, 'type': type_id}
            
            X_grid, T_grid, U_true_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], t_max, Nx=256, Nt=None)
            
            X_flat, T_flat = X_grid.flatten(), T_grid.flatten()
            xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
            p_vec = np.array([alpha, beta, mu, V, A, w0, x0, k, float(type_id)])
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

            with torch.no_grad():
                u_re, u_im = model(p_tensor, xt_tensor)
                u_pred_cplx = (u_re + 1j * u_im).cpu().numpy().flatten()
            
            U_true = U_true_cplx.flatten()
            norm_true = np.linalg.norm(U_true) + 1e-7
            err = np.linalg.norm(U_true - u_pred_cplx) / norm_true
            errors.append(err)
        except Exception: continue

    if not errors: return False, 1.0
    mean_err = np.mean(errors)
    return mean_err < threshold, mean_err

def diagnose_cgle(model, cfg, t_max, threshold=0.05, n_per_type=10):
    device = next(model.parameters()).device
    model.eval()
    failed_types = []
    types_map = {0: "Gaussian", 1: "Sech", 2: "Tanh"}
    
    if isinstance(cfg, dict):
        eq_p = cfg['physics']['equation_params']
        bounds = cfg['physics']['bounds']
        x_domain = cfg['physics']['x_domain']
    else:
        eq_p = cfg.physics['equation_params']
        bounds = cfg.physics['bounds']
        x_domain = cfg.physics['x_domain']

    print(f"      🔎 Diagnostic Spécifique (t_max={t_max:.2f})...")

    for type_id, type_name in types_map.items():
        errors = []
        for _ in range(n_per_type):
            alpha = np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1])
            beta  = np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1])
            mu    = np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1])
            V     = np.random.uniform(eq_p['V'][0],     eq_p['V'][1])
            A = np.random.uniform(bounds['A'][0], bounds['A'][1])
            w0 = 10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1]))
            x0, k = 0.0, 1.0

            p_dict = {'alpha': alpha, 'beta': beta, 'mu': mu, 'V': V, 'A': A, 'w0': w0, 'x0': x0, 'k': k, 'type': type_id}
            
            try:
                X_grid, T_grid, U_true_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], t_max, Nx=256, Nt=None)
            except: continue

            X_flat, T_flat = X_grid.flatten(), T_grid.flatten()
            xt_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
            p_vec = np.array([alpha, beta, mu, V, A, w0, x0, k, float(type_id)])
            p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

            with torch.no_grad():
                u_re, u_im = model(p_tensor, xt_tensor)
                u_pred = (u_re + 1j * u_im).cpu().numpy().flatten()
            
            U_true = U_true_cplx.flatten()
            err = np.linalg.norm(U_true - u_pred) / (np.linalg.norm(U_true) + 1e-7)
            errors.append(err)

        mean_err = np.mean(errors) if errors else 1.0
        if mean_err > threshold:
            print(f"      ❌ {type_name}: Err = {mean_err:.2%}")
            failed_types.append(type_id)
        else:
            print(f"      ✅ {type_name}: Err = {mean_err:.2%}")

    return failed_types

# ==============================================================================
# 2. FONCTION LOSS
# ==============================================================================

def compute_cgle_loss(model, branch_pde, coords_pde, pde_params, branch_ic, coords_ic, u_true_ic_re, u_true_ic_im, cfg):
    # Loss PDE
    r_re, r_im = pde_residual_cgle(model, branch_pde, coords_pde, pde_params, cfg)
    loss_pde = torch.mean(r_re**2 + r_im**2)

    # Loss IC
    p_re, p_im = model(branch_ic, coords_ic)
    l_comp = torch.mean((p_re - u_true_ic_re)**2) + torch.mean((p_im - u_true_ic_im)**2)
    l_mod = torch.mean((torch.sqrt(p_re**2 + p_im**2 + 1e-9) - torch.sqrt(u_true_ic_re**2 + u_true_ic_im**2 + 1e-9))**2)
    loss_ic = l_comp + l_mod

    # Loss BC
    ic_types = branch_pde[:, 8:9]
    is_periodic = (torch.abs(ic_types - 2.0) > 0.1).float()
    
    if isinstance(cfg, dict): x_domain = cfg['physics']['x_domain']
    else: x_domain = cfg.physics['x_domain']
    
    t_bc = coords_pde[:, 1:2]
    coords_L = torch.cat([torch.full_like(t_bc, x_domain[0]), t_bc], dim=1)
    coords_R = torch.cat([torch.full_like(t_bc, x_domain[1]), t_bc], dim=1)
    u_L_re, u_L_im = model(branch_pde, coords_L)
    u_R_re, u_R_im = model(branch_pde, coords_R)
    loss_bc_masked = torch.mean(is_periodic * ((u_L_re - u_R_re)**2 + (u_L_im - u_R_im)**2))

    # Total
    if isinstance(cfg, dict): weights = cfg['training']['weights']
    else: weights = cfg.training['weights']

    return weights['pde_loss'] * loss_pde + weights['ic_loss'] * loss_ic + weights.get('bc_loss', 1.0) * loss_bc_masked

# ==============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT (RETRY LOGIC)
# ==============================================================================

def train_step_cgle(model, cfg, t_max, n_iters, threshold=0.05):
    device = next(model.parameters()).device
    
    # Params
    if isinstance(cfg, dict):
        base_lr = float(cfg['training']['ic_phase']['learning_rate'])
        batch_size_pde = int(cfg['training']['batch_size_pde'])
        batch_size_ic = int(cfg['training']['batch_size_ic'])
        max_retry = cfg['training'].get('max_retry', 3) 
    else:
        base_lr = float(cfg.training['ic_phase']['learning_rate'])
        batch_size_pde = int(cfg.training['batch_size_pde'])
        batch_size_ic = int(cfg.training['batch_size_ic'])
        max_retry = cfg.training.get('max_retry', 3)

    print(f"\n🔵 PALIER t=[0, {t_max:.2f}] (iters base={n_iters}, retry={max_retry})")

    global_success = False
    current_lr = base_lr

    for attempt in range(max_retry):
        
        # --- MODE LBFGS ---
        if attempt == max_retry - 1:
            print(f"  👉 Tentative Globale {attempt+1}/{max_retry} : LBFGS Finisher")
            optimizer = optim.LBFGS(model.parameters(), lr=0.2, max_iter=100, tolerance_grad=1e-5, line_search_fn="strong_wolfe")
            for _ in range(20): 
                br_pde, co_pde, pde_params = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
                br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device)
                def closure():
                    optimizer.zero_grad()
                    loss = compute_cgle_loss(model, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    return loss
                try: optimizer.step(closure)
                except: pass

        # --- MODE ADAM ---
        else:
            print(f"  👉 Tentative Globale {attempt+1}/{max_retry} : Adam (LR={current_lr:.2e})")
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            model.train()
            
            # Augmentation de la durée à chaque échec
            current_iters = int(n_iters + (3000 * attempt))
            print(f"     ⏳ Durée étendue à {current_iters} itérations.")
            
            for i in range(current_iters):
                optimizer.zero_grad(set_to_none=True)
                br_pde, co_pde, pde_params = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
                br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device)
                
                loss = compute_cgle_loss(model, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
                if torch.isnan(loss): 
                    print("    💀 Loss NaN. Break.")
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if (i+1)%1000==0: print(f"    Iter {i+1}/{current_iters} | Loss: {loss.item():.2e}")

        # --- AUDIT GLOBAL RAPIDE ---
        success, err = audit_global_fast(model, cfg, t_max, threshold=threshold)
        
        if success:
            print(f"    📊 Audit Global OK ({err:.2%}). Passage au check spécifique...")
            global_success = True
            break
        else:
            print(f"    📊 Audit Global KO ({err:.2%}).")
            if attempt < max_retry - 2: 
                current_lr *= 0.5
                print(f"    ↘️  Decay LR : {current_lr:.2e}")

    if not global_success: 
        print("🛑 Echec Global. Abandon du step.")
        return False

    # --------------------------------------------------------------------------
    # PHASE 2 : DIAGNOSTIC SPÉCIFIQUE
    # --------------------------------------------------------------------------
    failed_types = diagnose_cgle(model, cfg, t_max, threshold=threshold)
    if not failed_types:
        print("  ✅ Validation Totale : Tous types OK.")
        return True

    # --------------------------------------------------------------------------
    # PHASE 3 : CORRECTION CIBLÉE
    # --------------------------------------------------------------------------
    print(f"\n🚑 Correction Ciblée nécessaire sur {failed_types}...")
    current_lr = base_lr 
    
    for attempt in range(max_retry):
        
        n_iters_focus = int((n_iters + 3000) + (3000 * attempt))
        
        if attempt == max_retry - 1:
            print(f"  ☢️ [Focus Loop] Tentative {attempt+1}/{max_retry} : LBFGS Ultimate")
            optimizer = optim.LBFGS(model.parameters(), lr=0.2, max_iter=200, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
            for _ in range(30):
                br_pde, co_pde, pde_params = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
                br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device)
                def closure():
                    optimizer.zero_grad()
                    loss = compute_cgle_loss(model, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    return loss
                try: optimizer.step(closure)
                except: pass
        else:
            print(f"  🚑 [Focus Loop] Tentative {attempt+1}/{max_retry} : Adam (LR={current_lr:.2e}, Iters={n_iters_focus})")
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            for i in range(n_iters_focus):
                optimizer.zero_grad(set_to_none=True)
                br_pde, co_pde, pde_params = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_max)
                br_ic, co_ic, tr_ic_re, tr_ic_im, _, _ = get_ic_batch_cgle(batch_size_ic, cfg, device)
                
                loss = compute_cgle_loss(model, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im, cfg)
                if torch.isnan(loss): break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        failed_now = diagnose_cgle(model, cfg, t_max, threshold=threshold)
        if not failed_now:
            print(f"  ✅ Correction réussie à la tentative {attempt+1} !")
            return True
        else:
            print(f"  ❌ Toujours des erreurs: {failed_now}")
            if attempt < max_retry - 2: current_lr *= 0.5

    # Check Final Relaxé
    failed_final = diagnose_cgle(model, cfg, t_max, threshold=0.08)
    if not failed_final:
        print("✅ Validé in-extremis (Tolérance relaxée).")
        return True
    
    print("🛑 ECHEC FINAL.")
    return False

# ==============================================================================
# 4. MAIN TRAINER (GESTION DES ZONES)
# ==============================================================================

def train_cgle_curriculum(model, cfg):
    if isinstance(cfg, dict):
        save_dir = cfg['training'].get('save_dir', "outputs/checkpoints_cgl")
        ic_iter = int(cfg['training']['ic_phase']['iterations'])
        t_max_phys = cfg['physics']['t_max']
        # On récupère les zones du yaml
        zones = cfg['time_marching'].get('zones', [])
    else:
        training_dict = cfg.training 
        save_dir = training_dict.get('save_dir', "outputs/checkpoints_cgl")
        ic_iter = int(training_dict['ic_phase']['iterations']) 
        t_max_phys = cfg.physics['t_max']
        zones = cfg.time_marching.get('zones', [])

    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    # --------------------------------------------------------------------------
    # 1. WARMUP (IC Only)
    # --------------------------------------------------------------------------
    print("🧊 WARMUP (IC Only)...")
    print(f"   👉 Phase A : Adam ({ic_iter} iters)")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, verbose=True)

    for i in range(ic_iter):
        optimizer.zero_grad()
        br_ic, co_ic, tr_re, tr_im, _, _ = get_ic_batch_cgle(4096, cfg, device)
        p_re, p_im = model(br_ic, co_ic)
        loss = torch.mean((p_re - tr_re)**2 + (p_im - tr_im)**2)
        
        if torch.isnan(loss):
             print("💀 NaN. Reset.")
             for pg in optimizer.param_groups: pg['lr'] *= 0.1
             continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        if i % 2000 == 0: print(f"   Warmup Adam {i}/{ic_iter} | Loss: {loss.item():.2e}")

    print(f"   👉 Phase B : L-BFGS (Finition)")
    batch_size_lbfgs = 4096 * 4 
    br_ic_fix, co_ic_fix, tr_re_fix, tr_im_fix, _, _ = get_ic_batch_cgle(batch_size_lbfgs, cfg, device)
    
    lbfgs_ic = optim.LBFGS(model.parameters(), lr=0.5, max_iter=2000, tolerance_grad=1e-9, line_search_fn="strong_wolfe")
    def closure_ic():
        lbfgs_ic.zero_grad()
        p_re, p_im = model(br_ic_fix, co_ic_fix)
        loss = torch.mean((p_re - tr_re_fix)**2 + (p_im - tr_im_fix)**2)
        loss.backward()
        return loss
    
    try: lbfgs_ic.step(closure_ic)
    except: pass
    print("✅ Warmup terminé.")

    # --------------------------------------------------------------------------
    # 2. TIME MARCHING PAR ZONES
    # --------------------------------------------------------------------------
    
    # Si pas de zones définies, on crée une zone par défaut (fallback)
    if not zones:
        print("⚠️ Aucune zone détectée, utilisation de la config par défaut.")
        zones = [{'t_end': t_max_phys, 'dt': 0.05, 'iters': 3000, 'audit_threshold': 0.05}]

    current_t = 0.0
    
    for zone in zones:
        z_end = zone['t_end']
        z_dt = zone['dt']
        z_iters = zone['iters']
        z_thresh = zone.get('audit_threshold', 0.05)
        
        print(f"\n🚀 ENTRÉE DANS LA ZONE : t_end={z_end}, dt={z_dt}, iters={z_iters}")

        # On avance dans le temps tant qu'on est dans la zone
        # On utilise une petite tolérance pour l'arrondi float
        while current_t < z_end - 1e-9:
            
            # Prochain pas
            next_t = current_t + z_dt
            
            # Si le pas dépasse la fin de zone, on cape (optionnel, mais propre)
            if next_t > z_end + 1e-9:
                next_t = z_end
            
            # Lancement de l'entraînement
            success = train_step_cgle(model, cfg, next_t, n_iters=z_iters, threshold=z_thresh)
            
            if not success:
                print("🛑 Echec critique du Time Marching. Arrêt.")
                return # On quitte tout

            # Sauvegarde et incrément
            current_t = next_t
            ckpt_path = os.path.join(save_dir, f"ckpt_t{current_t:.2f}.pth")
            torch.save(model.state_dict(), ckpt_path)

    print("🏁 Fin de toutes les zones. Entraînement terminé.")