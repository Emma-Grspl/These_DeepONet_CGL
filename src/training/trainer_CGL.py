import torch
import torch.optim as optim
import numpy as np
import os
import copy
from tqdm import tqdm

# Imports CGL spécifiques
from src.physics.pde_cgl import pde_residual_cgle
from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle
from src.utils.solver_cgl import get_ground_truth_CGL 

# ==============================================================================
# 1. OUTILS DE PONDÉRATION (RAMPE + NTK)
# ==============================================================================

def get_dynamic_pde_weight(model, t_current, cfg, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im):
    ramp_end = cfg['training'].get('ramp_end_t', 0.5)
    w_start = cfg['training'].get('pde_weight_start', 0.1)
    w_target = cfg['training'].get('pde_weight_target', 1.0)
    
    if t_current <= ramp_end:
        ratio = t_current / ramp_end
        current_w = w_start + ratio * (w_target - w_start)
        return current_w

    model.zero_grad()
    pred_ic_re, pred_ic_im = model(br_ic, co_ic)
    loss_ic = torch.mean((pred_ic_re - tr_ic_re)**2 + (pred_ic_im - tr_ic_im)**2)
    grad_ic = torch.autograd.grad(loss_ic, model.parameters(), retain_graph=True, allow_unused=True)
    
    r_re, r_im = pde_residual_cgle(model, br_pde, co_pde, pde_params, cfg)
    loss_pde = torch.mean(r_re**2 + r_im**2)
    grad_pde = torch.autograd.grad(loss_pde, model.parameters(), allow_unused=True)
    
    norm_ic = 0.0
    for g in grad_ic:
        if g is not None: norm_ic += torch.max(torch.abs(g)).item()
        
    norm_pde = 0.0
    for g in grad_pde:
        if g is not None: norm_pde += torch.max(torch.abs(g)).item()
        
    if norm_pde < 1e-8: return w_target
    ntk_ratio = norm_ic / norm_pde
    return 0.9 * w_target + 0.1 * ntk_ratio

# ==============================================================================
# 2. AUDIT & DIAGNOSTIC (STABILISÉ)
# ==============================================================================

def audit_global_fast(model, cfg, t_max, threshold=0.05, n_samples=100):
    """
    Audit global rapide.
    NOTE : On fixe la seed locale pour que l'audit soit comparable d'un cycle à l'autre.
    """
    device = next(model.parameters()).device
    model.eval()
    errors = []
    
    # Sauvegarde de l'état aléatoire actuel pour ne pas perturber l'entraînement
    rng_state = np.random.get_state()
    # On force une seed fixe pour l'Audit -> Comparaison "Pommes vs Pommes"
    np.random.seed(42) 
    
    if isinstance(cfg, dict):
        eq_p, bounds, x_domain = cfg['physics']['equation_params'], cfg['physics']['bounds'], cfg['physics']['x_domain']
    else:
        eq_p, bounds, x_domain = cfg.physics['equation_params'], cfg.physics['bounds'], cfg.physics['x_domain']

    for _ in range(n_samples):
        try:
            p_dict = {
                'alpha': np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1]),
                'beta':  np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1]),
                'mu':    np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1]),
                'V':     np.random.uniform(eq_p['V'][0],     eq_p['V'][1]),
                'A':     np.random.uniform(bounds['A'][0], bounds['A'][1]),
                'w0':    10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1])),
                'x0':    0.0, 'k': 1.0, 'type': np.random.choice([0, 1, 2])
            }
            
            audit_t = t_max if t_max > 1e-5 else 0.0
            
            if audit_t < 1e-5: # Warmup T=0
                X_grid, T_grid, U_true_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], 0.01, Nx=128, Nt=None)
                # ✅ CORRECTION : On prend tout l'espace à t=0
                U_true = U_true_cplx[:, 0]
                X_flat = X_grid[:, 0]
                T_flat = np.zeros_like(X_flat)
            else:
                X_grid, T_grid, U_true_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], audit_t, Nx=128, Nt=None)
                X_flat, T_flat, U_true = X_grid.flatten(), T_grid.flatten(), U_true_cplx.flatten()

            xt_t = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
            p_vec = np.array([p_dict[k] for k in ['alpha','beta','mu','V','A','w0','x0','k','type']])
            p_t = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)

            with torch.no_grad():
                ur, ui = model(p_t, xt_t)
                up = (ur + 1j*ui).cpu().numpy().flatten()
            
            norm_true = np.linalg.norm(U_true)
            if norm_true < 1e-9: norm_true = 1e-9
            
            err = np.linalg.norm(U_true - up) / norm_true
            errors.append(err)
        except: continue

    # Restauration de l'aléatoire pour l'entraînement suivant
    np.random.set_state(rng_state)

    if not errors: return False, 1.0
    mean_err = np.mean(errors)
    return mean_err < threshold, mean_err

# ==============================================================================
# 3. ROUTINE D'OPTIMISATION ROBUSTE (AVEC PRINTS)
# ==============================================================================

def robust_optimize(model, cfg, t_max, n_iters_base, context_str="Global", specific_type=None):
    device = next(model.parameters()).device
    max_macro = cfg['training'].get('max_macro_loops', 3)
    adam_retries = cfg['training'].get('nb_adam_retries', 3)
    
    # Learning Rate Initial
    start_lr = float(cfg['training']['ic_phase']['learning_rate'])
    
    check_interval = cfg['training'].get('check_interval', 2000)
    stagnation_thresh = cfg['training'].get('stagnation_threshold', 0.01)
    weights = cfg['training']['weights'].copy()
    
    champion_state = copy.deepcopy(model.state_dict())
    champion_loss = float('inf')
    
    current_lr = start_lr
    
    print(f"\n🏰 [Robust Optimize : {context_str}] t={t_max:.2f}")

    for macro in range(max_macro):
        print(f"  🔄 Macro Cycle {macro+1}/{max_macro} (LR Start={current_lr:.1e})")
        
        # --- ADAM SEQUENCE ---
        for attempt in range(adam_retries):
            model.load_state_dict(champion_state)
            
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            iters = n_iters_base + (2000 * attempt)
            pbar = tqdm(range(iters), desc=f"    Adam {attempt+1}/{adam_retries}", leave=False)
            
            losses_window = []
            local_best_loss = float('inf')
            local_best_state = copy.deepcopy(model.state_dict())
            
            current_pde_w = weights['pde_loss'] 
            
            for i in pbar:
                if t_max > 0.0 and i % 500 == 0:
                    b_p, c_p, p_p = get_pde_batch_cgle(1024, cfg, device, t_limit=t_max)
                    b_i, c_i, u_r, u_i, _, _ = get_ic_batch_cgle(1024, cfg, device)
                    current_pde_w = get_dynamic_pde_weight(model, t_max, cfg, b_p, c_p, p_p, b_i, c_i, u_r, u_i)
                
                optimizer.zero_grad(set_to_none=True)
                
                bs_pde = int(cfg['training']['batch_size_pde'])
                bs_ic = int(cfg['training']['batch_size_ic'])
                
                br_pde, co_pde, pde_params = get_pde_batch_cgle(bs_pde, cfg, device, t_limit=t_max)
                br_ic, co_ic, tr_ic_re, tr_ic_im, ux_re, ux_im = get_ic_batch_cgle(bs_ic, cfg, device)
                
                if t_max < 1e-5: # Warmup
                    co_ic.requires_grad_(True)
                    pr, pi = model(br_ic, co_ic)
                    l_val = torch.mean((pr-tr_ic_re)**2 + (pi-tr_ic_im)**2)
                    gr = torch.autograd.grad(pr.sum(), co_ic, create_graph=True)[0]
                    gi = torch.autograd.grad(pi.sum(), co_ic, create_graph=True)[0]
                    l_sob = torch.mean((gr[:,0:1]-ux_re)**2 + (gi[:,0:1]-ux_im)**2)
                    loss = l_val + 0.1 * l_sob
                else: 
                    r_re, r_im = pde_residual_cgle(model, br_pde, co_pde, pde_params, cfg)
                    l_pde = torch.mean(r_re**2 + r_im**2)
                    p_re, p_im = model(br_ic, co_ic)
                    l_ic = torch.mean((p_re - tr_ic_re)**2 + (p_im - tr_ic_im)**2)
                    loss = current_pde_w * l_pde + weights['ic_loss'] * l_ic
                
                if torch.isnan(loss): break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                curr_loss = loss.item()
                losses_window.append(curr_loss)
                
                # --- NOUVEAU : PRINT TOUTES LES 1000 ITERATIONS ---
                if i % 1000 == 0:
                    print(f"      [Iter {i}] Loss: {curr_loss:.2e} | PDE_W: {current_pde_w:.2f}")

                if curr_loss < local_best_loss:
                    local_best_loss = curr_loss
                    local_best_state = copy.deepcopy(model.state_dict())
                
                if i > 0 and i % check_interval == 0:
                    curr_avg = np.mean(losses_window[-check_interval:])
                    if len(losses_window) > check_interval:
                        prev_avg = np.mean(losses_window[-2*check_interval:-check_interval])
                        improvement = (prev_avg - curr_avg) / (prev_avg + 1e-9)
                        if improvement < stagnation_thresh:
                            print(f"      💤 Stagnation détectée à l'iter {i}. Arrêt anticipé.")
                            break 
                
                if i%100==0: pbar.set_postfix({"L": f"{curr_loss:.1e}"})

            if local_best_loss < champion_loss:
                champion_loss = local_best_loss
                champion_state = local_best_state
                print(f"    🚀 Nouveau Champion Adam ! (L={champion_loss:.2e})")
            
            current_lr *= 0.5

        # --- L-BFGS FINISHER ---
        print(f"    🔧 L-BFGS Finisher (from Champion)...")
        model.load_state_dict(champion_state)
        
        lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=800, line_search_fn="strong_wolfe")
        
        b_p_fix, c_p_fix, p_p_fix = get_pde_batch_cgle(4096, cfg, device, t_limit=t_max)
        b_i_fix, c_i_fix, tr_re_fix, tr_im_fix, ux_re, ux_im = get_ic_batch_cgle(4096, cfg, device)
        
        def closure():
            lbfgs.zero_grad()
            if t_max < 1e-5:
                c_i_fix.requires_grad_(True)
                pr, pi = model(b_i_fix, c_i_fix)
                l_val = torch.mean((pr-tr_re_fix)**2 + (pi-tr_im_fix)**2)
                gr = torch.autograd.grad(pr.sum(), c_i_fix, create_graph=True)[0]
                gi = torch.autograd.grad(pi.sum(), c_i_fix, create_graph=True)[0]
                loss = l_val + 0.1 * torch.mean((gr[:,0:1]-ux_re)**2 + (gi[:,0:1]-ux_im)**2)
            else:
                rr, ri = pde_residual_cgle(model, b_p_fix, c_p_fix, p_p_fix, cfg)
                pr, pi = model(b_i_fix, c_i_fix)
                loss = current_pde_w * torch.mean(rr**2 + ri**2) + weights['ic_loss'] * torch.mean((pr-tr_re_fix)**2 + (pi-tr_im_fix)**2)
            loss.backward()
            return loss
        
        try: lbfgs.step(closure)
        except: pass
        
        final_lbfgs_loss = closure().item()
        if final_lbfgs_loss < champion_loss:
            champion_loss = final_lbfgs_loss
            champion_state = copy.deepcopy(model.state_dict())
            print(f"    🚀 L-BFGS a amélioré ! (L={champion_loss:.2e})")
        else:
            print(f"    ⚠️ L-BFGS n'a pas amélioré. Restauration Champion (L={champion_loss:.2e}).")
            model.load_state_dict(champion_state) # ⚠️ TRES IMPORTANT : ON FORCE LE RETOUR AU CHAMPION

        # --- AUDIT FINAL DU CYCLE ---
        thresh = cfg['training'].get('target_error_ic', 0.03) if t_max < 1e-5 else cfg['training'].get('target_error_global', 0.05)
        success, err = audit_global_fast(model, cfg, t_max, threshold=thresh)
        
        if success:
            print(f"    🏆 VICTOIRE ! Audit OK ({err:.2%}).")
            return True
        else:
            print(f"    ❌ Audit KO ({err:.2%}). On relance un Macro Cycle.")
            
    print("🛑 ECHEC FINAL : Max Macro Loops atteint.")
    return False

def train_cgle_curriculum(model, cfg):
    # (Identique à avant, juste pour compléter le fichier)
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints_cgl")
    os.makedirs(save_dir, exist_ok=True)
    
    print("🧊 WARMUP (IC + Sobolev)...")
    ok = robust_optimize(model, cfg, 0.0, 10000, context_str="Warmup") # Augmenté iters pour warmup
    if not ok: return
    
    zones = cfg['time_marching']['zones']
    current_t = 0.0
    for zone in zones:
        z_end, z_dt, z_iters = zone['t_end'], zone['dt'], zone['iters']
        print(f"\n🚀 ZONE : t_end={z_end}, dt={z_dt}")
        while current_t < z_end - 1e-9:
            next_t = min(current_t + z_dt, z_end)
            ok = robust_optimize(model, cfg, next_t, z_iters, context_str="Global")
            if not ok: return
            current_t = next_t
            torch.save({'model': model.state_dict(), 't': current_t}, os.path.join(save_dir, f"ckpt_t{current_t:.2f}.pth"))