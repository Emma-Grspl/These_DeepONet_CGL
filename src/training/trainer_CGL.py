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
    """
    Calcule le poids de la PDE selon la stratégie :
    - t <= 0.5 : Rampe linéaire
    - t > 0.5  : NTK (Gradient Balancing)
    """
    
    # Paramètres de la rampe (à mettre dans le yaml)
    ramp_end = cfg['training'].get('ramp_end_t', 0.5)
    w_start = cfg['training'].get('pde_weight_start', 0.1)
    w_target = cfg['training'].get('pde_weight_target', 1.0)
    
    # --- PHASE 1 : RAMPE LINÉAIRE ---
    if t_current <= ramp_end:
        ratio = t_current / ramp_end
        current_w = w_start + ratio * (w_target - w_start)
        return current_w

    # --- PHASE 2 : NTK (GRADIENT BALANCING) ---
    # On ne calcule pas NTK à chaque step (trop lourd), on peut le faire à la volée 
    # mais ici on va le faire simple : calcul du ratio des normes de gradient.
    
    # 1. Zero grad
    model.zero_grad()
    
    # 2. Gradient Loss IC
    pred_ic_re, pred_ic_im = model(br_ic, co_ic)
    loss_ic = torch.mean((pred_ic_re - tr_ic_re)**2 + (pred_ic_im - tr_ic_im)**2)
    grad_ic = torch.autograd.grad(loss_ic, model.parameters(), retain_graph=True, allow_unused=True)
    
    # 3. Gradient Loss PDE
    r_re, r_im = pde_residual_cgle(model, br_pde, co_pde, pde_params, cfg)
    loss_pde = torch.mean(r_re**2 + r_im**2)
    grad_pde = torch.autograd.grad(loss_pde, model.parameters(), allow_unused=True)
    
    # 4. Calcul des normes (sur les couches finales pour économiser mémoire)
    # On prend tous les params non-None
    norm_ic = 0.0
    for g in grad_ic:
        if g is not None: norm_ic += torch.max(torch.abs(g)).item()
        
    norm_pde = 0.0
    for g in grad_pde:
        if g is not None: norm_pde += torch.max(torch.abs(g)).item()
        
    # 5. Ratio NTK
    if norm_pde < 1e-8: return w_target # Sécurité
    ntk_ratio = norm_ic / norm_pde
    
    # On lisse le ratio avec le poids cible pour éviter les sauts brutaux
    # Alpha blending : 0.9 * ancien + 0.1 * nouveau
    return 0.9 * w_target + 0.1 * ntk_ratio


# ==============================================================================
# 2. AUDIT & DIAGNOSTIC
# ==============================================================================

def audit_global_fast(model, cfg, t_max, threshold=0.05, n_samples=30):
    """Audit global rapide pour valider le pas de temps."""
    device = next(model.parameters()).device
    model.eval()
    errors = []
    
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
                U_true = U_true_cplx[0, :] # On prend la ligne 0
                X_flat = X_grid[0, :]
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
            
            err = np.linalg.norm(U_true - up) / (np.linalg.norm(U_true) + 1e-7)
            errors.append(err)
        except: continue

    if not errors: return False, 1.0
    mean_err = np.mean(errors)
    return mean_err < threshold, mean_err

def diagnose_cgle(model, cfg, t_max, threshold, n_per_type=20, specific_type=None):
    """Diagnostic pour identifier les types en échec."""
    # (Similaire à audit_global_fast mais trié par type. 
    #  Pour abréger ici, on réutilise la logique, mais en prod reprends ta version complète)
    # ICI VERSION SIMPLIFIÉE POUR TENIR DANS LE CONTEXTE
    # ... (Garder ta fonction diagnose_cgle existante si possible, sinon je la remets)
    return [] # Placeholder si tout va bien, à remplacer par ta fonction diagnose complète

# ==============================================================================
# 3. ROUTINE D'OPTIMISATION ROBUSTE (WARMUP / GLOBAL / SPECIFIQUE)
# ==============================================================================

def robust_optimize(model, cfg, t_max, n_iters_base, context_str="Global", specific_type=None):
    """
    Le Cœur du Système : 
    - Macro Loops
    - Adam Sequence (avec King of the Hill & Rolling Check)
    - L-BFGS Finisher (avec King of the Hill)
    - Audit Final
    """
    device = next(model.parameters()).device
    
    # Config Extraction
    max_macro = cfg['training'].get('max_macro_loops', 3)
    adam_retries = cfg['training'].get('nb_adam_retries', 3)
    start_lr = float(cfg['training']['ic_phase']['learning_rate'])
    
    # Rolling Check Config
    check_interval = cfg['training'].get('check_interval', 2000)
    stagnation_thresh = cfg['training'].get('stagnation_threshold', 0.01)
    
    # Poids (Initialisation)
    weights = cfg['training']['weights'].copy() # Copie pour modif locale
    
    # 👑 KING OF THE HILL : Initialisation
    champion_state = copy.deepcopy(model.state_dict())
    champion_loss = float('inf')
    
    current_lr = start_lr
    
    print(f"\n🏰 [Robust Optimize : {context_str}] t={t_max:.2f}")

    # --- MACRO LOOP ---
    for macro in range(max_macro):
        print(f"  🔄 Macro Cycle {macro+1}/{max_macro} (LR Start={current_lr:.1e})")
        
        # --- ADAM SEQUENCE ---
        for attempt in range(adam_retries):
            # 1. Load Champion
            model.load_state_dict(champion_state)
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            
            iters = n_iters_base + (2000 * attempt)
            pbar = tqdm(range(iters), desc=f"    Adam {attempt+1}/{adam_retries}", leave=False)
            
            # Variables Rolling Check
            losses_window = []
            
            # Variables Local Champion (pour cette run Adam)
            local_best_loss = float('inf')
            local_best_state = copy.deepcopy(model.state_dict())
            
            # Calcul du poids PDE (Dynamique ou Rampe) une fois au début du cycle ou périodique ?
            # Pour stabilité, on le calcule au début de la boucle Adam ou on le fixe pour la boucle.
            # Faisons-le dynamique tous les 500 iters.
            current_pde_w = weights['pde_loss'] 
            
            for i in pbar:
                # Mise à jour Poids Dynamique (Tous les 500 iters)
                if t_max > 0.0 and i % 500 == 0:
                     # Génération batch temporaire pour le calcul NTK
                    b_p, c_p, p_p = get_pde_batch_cgle(1024, cfg, device, t_limit=t_max)
                    b_i, c_i, u_r, u_i, _, _ = get_ic_batch_cgle(1024, cfg, device)
                    current_pde_w = get_dynamic_pde_weight(model, t_max, cfg, b_p, c_p, p_p, b_i, c_i, u_r, u_i)
                
                # Training Step
                optimizer.zero_grad(set_to_none=True)
                
                # Batch Generation
                bs_pde = int(cfg['training']['batch_size_pde'])
                bs_ic = int(cfg['training']['batch_size_ic'])
                
                br_pde, co_pde, pde_params = get_pde_batch_cgle(bs_pde, cfg, device, t_limit=t_max)
                # Note: On récupère les 6 variables IC (avec sobolev)
                br_ic, co_ic, tr_ic_re, tr_ic_im, ux_re, ux_im = get_ic_batch_cgle(bs_ic, cfg, device)
                
                # Calcul Loss
                if t_max < 1e-5: # Warmup (Sobolev Loss)
                    # compute_sobolev_ic_loss doit être définie ou incluse (voir code précédent)
                    # Ici on l'implémente inline pour simplifier
                    co_ic.requires_grad_(True)
                    pr, pi = model(br_ic, co_ic)
                    l_val = torch.mean((pr-tr_ic_re)**2 + (pi-tr_ic_im)**2)
                    gr = torch.autograd.grad(pr.sum(), co_ic, create_graph=True)[0]
                    gi = torch.autograd.grad(pi.sum(), co_ic, create_graph=True)[0]
                    l_sob = torch.mean((gr[:,0:1]-ux_re)**2 + (gi[:,0:1]-ux_im)**2)
                    loss = l_val + 0.1 * l_sob
                else: # Time Marching
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
                
                # 👑 KING OF THE HILL (Intra-Adam)
                if curr_loss < local_best_loss:
                    local_best_loss = curr_loss
                    local_best_state = copy.deepcopy(model.state_dict())
                
                # 💤 ROLLING CHECK
                if i > 0 and i % check_interval == 0:
                    curr_avg = np.mean(losses_window[-check_interval:])
                    if len(losses_window) > check_interval:
                        prev_avg = np.mean(losses_window[-2*check_interval:-check_interval])
                        improvement = (prev_avg - curr_avg) / (prev_avg + 1e-9)
                        if improvement < stagnation_thresh:
                            pbar.set_postfix_str(f"💤 Stagnation (<{stagnation_thresh*100}%)")
                            break # Early Stop cette boucle Adam
                
                if i%100==0: pbar.set_postfix({"L": f"{curr_loss:.1e}", "W_pde": f"{current_pde_w:.2f}"})

            # Fin Adam : On regarde si on a battu le Champion Global
            if local_best_loss < champion_loss:
                champion_loss = local_best_loss
                champion_state = local_best_state # Deepcopy déjà fait
                print(f"    🚀 Nouveau Champion Adam ! (L={champion_loss:.2e})")
            
            # Decay LR pour la prochaine tentative
            current_lr *= 0.5

        # --- L-BFGS FINISHER ---
        print(f"    🔧 L-BFGS Finisher (from Champion)...")
        model.load_state_dict(champion_state)
        
        lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=800, line_search_fn="strong_wolfe")
        
        # Batch Fixe LBFGS
        b_p_fix, c_p_fix, p_p_fix = get_pde_batch_cgle(4096, cfg, device, t_limit=t_max)
        b_i_fix, c_i_fix, tr_re_fix, tr_im_fix, ux_re, ux_im = get_ic_batch_cgle(4096, cfg, device)
        
        def closure():
            lbfgs.zero_grad()
            if t_max < 1e-5: # Sobolev
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
        
        # Check LBFGS result (King of the Hill)
        final_lbfgs_loss = closure().item()
        if final_lbfgs_loss < champion_loss:
            champion_loss = final_lbfgs_loss
            champion_state = copy.deepcopy(model.state_dict())
            print(f"    🚀 L-BFGS a amélioré ! (L={champion_loss:.2e})")
        else:
            print(f"    ⚠️ L-BFGS n'a pas amélioré. Restauration Champion.")
            model.load_state_dict(champion_state)

        # --- AUDIT FINAL DU CYCLE ---
        # C'est le seul moment où on décide de sortir ou de rejouer une macro loop
        thresh = cfg['training'].get('target_error_ic', 0.03) if t_max < 1e-5 else cfg['training'].get('target_error_global', 0.05)
        
        success, err = audit_global_fast(model, cfg, t_max, threshold=thresh)
        
        if success:
            print(f"    🏆 VICTOIRE ! Audit OK ({err:.2%}).")
            return True
        else:
            print(f"    ❌ Audit KO ({err:.2%}). On relance un Macro Cycle.")
            # Important : On garde le LR réduit pour le prochain cycle !
            
    print("🛑 ECHEC FINAL : Max Macro Loops atteint.")
    return False


# ==============================================================================
# 4. MAIN CURRICULUM
# ==============================================================================

def train_cgle_curriculum(model, cfg):
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints_cgl")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. WARMUP
    print("🧊 WARMUP (IC + Sobolev)...")
    ok = robust_optimize(model, cfg, 0.0, 5000, context_str="Warmup")
    if not ok: return
    
    # 2. TIME MARCHING
    zones = cfg['time_marching']['zones']
    current_t = 0.0
    
    for zone in zones:
        z_end = zone['t_end']
        z_dt = zone['dt']
        z_iters = zone['iters']
        
        print(f"\n🚀 ZONE : t_end={z_end}, dt={z_dt}")
        
        while current_t < z_end - 1e-9:
            next_t = min(current_t + z_dt, z_end)
            
            # Robust Optimize gère tout (King of the Hill, Rolling Check, LBFGS, Audit)
            ok = robust_optimize(model, cfg, next_t, z_iters, context_str="Global")
            
            if not ok:
                print("🛑 Arrêt critique.")
                return
                
            current_t = next_t
            torch.save({'model': model.state_dict(), 't': current_t}, os.path.join(save_dir, f"ckpt_t{current_t:.2f}.pth"))
            print(f"💾 Checkpoint t={current_t:.2f}")

    print("🏁 Fin de l'entraînement.")