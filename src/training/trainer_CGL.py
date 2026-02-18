import torch
import torch.optim as optim
import numpy as np
import copy
import os
import csv
from tqdm import tqdm
import glob
import re

# Imports CGL
from src.physics.pde_cgl import pde_residual_cgle
from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle
from src.utils.solver_cgl import get_ground_truth_CGL 

# ==============================================================================
# 0. OUTILS & LOGGING
# ==============================================================================

def init_csv_logs(save_dir):
    log_path = os.path.join(save_dir, "training_diagnostics.csv")
    if not os.path.exists(log_path):
        os.makedirs(save_dir, exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["t_max", "step", "loss_global", "loss_ic", "loss_pde", "r_grad", "cos_phi", "tr_ratio", "weight_pde"])
    return log_path

def get_zone_config(t_target, cfg):
    """Récupère le nombre d'itérations prévu dans le YAML pour ce temps t."""
    zones = cfg['time_marching']['zones']
    selected_iters = zones[-1]['iters'] # Défaut = dernière zone
    
    for zone in zones:
        if t_target <= zone['t_end']:
            selected_iters = zone['iters']
            break
            
    # Sécurité si le YAML a 0 ou une valeur bizarre
    if selected_iters < 100: selected_iters = 5000
    return selected_iters

def get_dynamic_weights(t_current, cfg):
    """Calcule les poids initiaux (avant adaptation) selon le temps."""
    t_ramp_end = float(cfg['training'].get('ramp_end_t', 0.1))
    
    # Rampe PDE (Montante)
    w_pde_start = float(cfg['training'].get('pde_weight_start', 1e-4))
    w_pde_target = float(cfg['training'].get('pde_weight_target', 1.0))
    
    # Rampe IC (Descendante)
    w_ic_start = float(cfg['training']['weights'].get('ic_loss_start', 10.0))
    w_ic_target = float(cfg['training']['weights'].get('ic_loss_target', 1.0))

    if t_current <= t_ramp_end:
        ratio = t_current / t_ramp_end
        pde_w = w_pde_start + ratio * (w_pde_target - w_pde_start)
        ic_w = w_ic_start - ratio * (w_ic_start - w_ic_target)
    else:
        pde_w = w_pde_target
        ic_w = w_ic_target
        
    return pde_w, ic_w

# ==============================================================================
# 1. GÉNÉRATEURS DE BATCH
# ==============================================================================

def get_biased_batch_generator(cfg, device, target_types, t_limit):
    """Générateur 80/20 pour cibler les faiblesses."""
    def generator(batch_size_pde, batch_size_ic):
        n_tgt_ic = int(0.8 * batch_size_ic)
        n_gen_ic = batch_size_ic - n_tgt_ic
        
        # 1. IC General (20%)
        b_gen, c_gen, tr_re_gen, tr_im_gen, ux_re_gen, ux_im_gen = get_ic_batch_cgle(n_gen_ic, cfg, device)
        
        # 2. IC Ciblée (80%)
        list_b, list_c, list_tr_re, list_tr_im, list_ux_re, list_ux_im = [], [], [], [], [], []
        curr, safety = 0, 0
        while curr < n_tgt_ic and safety < 50:
            tb, tc, t_re, t_im, tu_re, tu_im = get_ic_batch_cgle(n_tgt_ic * 2, cfg, device)
            mask = torch.zeros(tb.shape[0], dtype=torch.bool, device=device)
            for t_id in target_types: mask |= (tb[:, 8].long() == t_id)
            if mask.sum() > 0:
                list_b.append(tb[mask]); list_c.append(tc[mask])
                list_tr_re.append(t_re[mask]); list_tr_im.append(t_im[mask])
                list_ux_re.append(tu_re[mask]); list_ux_im.append(tu_im[mask])
                curr += mask.sum().item()
            safety += 1
        
        if list_b:
            b_tgt = torch.cat(list_b)[:n_tgt_ic]; c_tgt = torch.cat(list_c)[:n_tgt_ic]
            tr_re_tgt = torch.cat(list_tr_re)[:n_tgt_ic]; tr_im_tgt = torch.cat(list_tr_im)[:n_tgt_ic]
            ux_re_tgt = torch.cat(list_ux_re)[:n_tgt_ic]; ux_im_tgt = torch.cat(list_ux_im)[:n_tgt_ic]
            b_ic = torch.cat([b_gen, b_tgt]); c_ic = torch.cat([c_gen, c_tgt])
            tr_ic_re = torch.cat([tr_re_gen, tr_re_tgt]); tr_ic_im = torch.cat([tr_im_gen, tr_im_tgt])
            ux_re = torch.cat([ux_re_gen, ux_re_tgt]); ux_im = torch.cat([ux_im_gen, ux_im_tgt])
        else:
            b_ic, c_ic, tr_ic_re, tr_ic_im, ux_re, ux_im = get_ic_batch_cgle(batch_size_ic, cfg, device)

        perm_ic = torch.randperm(b_ic.size(0))
        b_ic, c_ic = b_ic[perm_ic], c_ic[perm_ic]
        tr_ic_re, tr_ic_im = tr_ic_re[perm_ic], tr_ic_im[perm_ic]
        ux_re, ux_im = ux_re[perm_ic], ux_im[perm_ic]

        # 3. PDE Batch (Biaisé aussi)
        if t_limit > 1e-5:
            n_tgt_pde = int(0.8 * batch_size_pde)
            n_gen_pde = batch_size_pde - n_tgt_pde
            bg, cg, _ = get_pde_batch_cgle(n_gen_pde, cfg, device, t_limit=t_limit)
            bt, ct, _ = get_pde_batch_cgle(n_tgt_pde, cfg, device, t_limit=t_limit)
            forced_types = np.random.choice(target_types, size=(n_tgt_pde, 1))
            bt[:, 8] = torch.tensor(forced_types, dtype=torch.float32, device=device).squeeze()
            b_pde = torch.cat([bg, bt]); c_pde = torch.cat([cg, ct])
            b_pde, c_pde = b_pde[torch.randperm(b_pde.size(0))], c_pde[torch.randperm(c_pde.size(0))]
            p_params = {"alpha": b_pde[:,0:1], "beta": b_pde[:,1:2], "mu": b_pde[:,2:3], "V": b_pde[:,3:4]}
        else:
            b_pde, c_pde, p_params = None, None, None

        return b_pde, c_pde, p_params, b_ic, c_ic, tr_ic_re, tr_ic_im, ux_re, ux_im
    return generator

def get_standard_batch_generator(cfg, device, t_limit):
    """Générateur 'Harmonisé' : IC 50/50 (Vide/Plein), PDE Uniforme."""
    def generator(batch_size_pde, batch_size_ic):
        # A. Moitié standard
        n_std = batch_size_ic // 2
        b1, c1, t1_re, t1_im, u1_re, u1_im = get_ic_batch_cgle(n_std, cfg, device)
        
        # B. Moitié "Focus" ([-6, 6])
        n_focus = batch_size_ic - n_std
        cfg_focus = copy.deepcopy(cfg)
        cfg_focus['physics']['x_domain'] = [-6.0, 6.0]
        b2, c2, t2_re, t2_im, u2_re, u2_im = get_ic_batch_cgle(n_focus, cfg_focus, device)
        
        # C. Fusion
        b_ic = torch.cat([b1, b2]); c_ic = torch.cat([c1, c2])
        tr_re = torch.cat([t1_re, t2_re]); tr_im = torch.cat([t1_im, t2_im])
        ux_re = torch.cat([u1_re, u2_re]); ux_im = torch.cat([u1_im, u2_im])
        
        perm = torch.randperm(batch_size_ic)
        b_ic, c_ic = b_ic[perm], c_ic[perm]
        tr_re, tr_im = tr_re[perm], tr_im[perm]
        ux_re, ux_im = ux_re[perm], ux_im[perm]

        # 2. PDE (Uniforme)
        if t_limit > 1e-5:
            b_p, c_p, p_p = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_limit)
        else:
            b_p, c_p, p_p = None, None, None
            
        return b_p, c_p, p_p, b_ic, c_ic, tr_re, tr_im, ux_re, ux_im

    return generator

# ==============================================================================
# 2. AUDIT
# ==============================================================================

def run_audit(model, cfg, t_max, threshold=0.03, n_global=60, n_specific=30, verbose=True):
    device = next(model.parameters()).device
    model.eval()
    
    phys = cfg['physics'] if isinstance(cfg, dict) else cfg.physics
    allowed_types = phys.get('initial_conditions', [1, 2])
    type_names = {0: "Gaussian", 1: "Sech", 2: "Tanh"}
    
    rng_state = np.random.get_state()
    np.random.seed(42) 

    eq_p, bounds, x_domain = phys['equation_params'], phys['bounds'], phys['x_domain']

    def evaluate_point(p_dict, t_eval):
        # FIX : Sécurité t très petit -> Comparaison IC analytique
        t_for_solver = 0.01 if t_eval < 0.01 else t_eval
        
        X, T, U_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], t_for_solver, Nx=128, Nt=None)
        
        if t_eval < 0.01:
            U_true = U_cplx[:, 0]
            X_flat = X[:, 0]
            T_flat = np.zeros_like(X_flat) + t_eval 
        else:
            U_true = U_cplx.flatten()
            X_flat = X.flatten()
            T_flat = T.flatten()
            
        xt_t = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        p_vec = np.array([p_dict[k] for k in ['alpha','beta','mu','V','A','w0','x0','k','type']])
        p_t = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)
        
        with torch.no_grad():
            ur, ui = model(p_t, xt_t)
            up = (ur + 1j*ui).cpu().numpy().flatten()
            
        norm = np.linalg.norm(U_true)
        return np.linalg.norm(U_true - up) / (norm if norm > 1e-9 else 1e-9)

    # 1. GLOBAL
    g_errs = []
    for _ in range(n_global):
        try:
            p = {'alpha': np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1]),
                 'beta':  np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1]),
                 'mu':    np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1]),
                 'V':     np.random.uniform(eq_p['V'][0],     eq_p['V'][1]),
                 'A':     np.random.uniform(bounds['A'][0], bounds['A'][1]),
                 'w0':    10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1])),
                 'x0': 0.0, 'k': 1.0, 
                 'type': np.random.choice(allowed_types)}
            g_errs.append(evaluate_point(p, t_max if t_max > 1e-5 else 0.0))
        except: continue
    
    global_score = np.mean(g_errs) if g_errs else 1.0
    passed_global = global_score < threshold
    
    if verbose:
        status_icon = "✅" if passed_global else "❌"
        print(f"    🌍 Audit Global  : {global_score:.2%} [{status_icon}]")

    # 2. SPÉCIFIQUE
    failed_types = []
    if verbose: print(f"    🔎 Audit Spécifique :")
    
    for t_id in allowed_types:
        t_errs = []
        for _ in range(n_specific):
            try:
                p = {'alpha': np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1]),
                     'beta':  np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1]),
                     'mu':    np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1]),
                     'V':     np.random.uniform(eq_p['V'][0],     eq_p['V'][1]),
                     'A':     np.random.uniform(bounds['A'][0], bounds['A'][1]),
                     'w0':    10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1])),
                     'x0': 0.0, 'k': 1.0, 'type': t_id}
                t_errs.append(evaluate_point(p, t_max if t_max > 1e-5 else 0.0))
            except: continue
        
        score = np.mean(t_errs) if t_errs else 1.0
        status = "✅" if score < threshold else "❌"
        
        if verbose: 
            print(f"      - {type_names[t_id]:<10} : {score:.2%} {status}")
            
        if score > threshold: 
            failed_types.append(t_id)

    np.random.set_state(rng_state)
    return passed_global, failed_types, global_score

# ==============================================================================
# 3. L'OUVRIER (TRAIN WORKER)
# ==============================================================================

def train_worker(model, cfg, t_max, start_lr, n_iters, batch_gen_func, context_name, 
                 global_best_state, global_best_score, use_lbfgs=True, stop_on_explosion=False):
    """
    L'OUVRIER : Exécute N itérations d'Adam + NTK-Light + L-BFGS (Optionnel).
    Gère les poids adaptatifs et les sauvegardes.
    """
    weights = cfg['training']['weights'].copy()
    save_dir = cfg['training'].get('save_dir', "outputs")
    log_path = init_csv_logs(save_dir)

    current_champion_state = copy.deepcopy(global_best_state)
    current_champion_score = global_best_score
    current_lr = start_lr
    
    # Init Poids Adaptatifs
    current_pde_w, current_ic_w = get_dynamic_weights(t_max, cfg)
    
    # --- ADAM LOOP ---
    model.load_state_dict(current_champion_state)
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=0.0)
    
    pbar = tqdm(range(n_iters), desc=f"  👷 [{context_name}]", leave=False)
    
    for i in pbar:
        # 1. Batch
        b_p, c_p, p_p, b_i, c_i, tr_re, tr_im, ux_re, ux_im = batch_gen_func(
            cfg['training']['batch_size_pde'], cfg['training']['batch_size_ic']
        )

        optimizer.zero_grad(set_to_none=True)
        
        # 2. Forward & Loss
        if t_max < 1e-5: # Warmup Mode
             c_i.requires_grad_(True)
             pr, pi = model(b_i, c_i)
             l_ic_raw = torch.mean((pr-tr_re)**2 + (pi-tr_im)**2)
             gr = torch.autograd.grad(pr.sum(), c_i, create_graph=True)[0]
             gi = torch.autograd.grad(pi.sum(), c_i, create_graph=True)[0]
             l_pde_raw = torch.tensor(0.0).to(b_i.device) # Dummy
             loss = l_ic_raw + 1.0 * torch.mean((gr[:,0:1]-ux_re)**2 + (gi[:,0:1]-ux_im)**2)
        else:
             rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
             pr, pi = model(b_i, c_i)
             l_pde_raw = torch.mean(rr**2 + ri**2)
             l_ic_raw = torch.mean((pr-tr_re)**2 + (pi-tr_im)**2)

             # 3. Poids Adaptatifs & Diagnostics (tous les 1000 pas)
             if i % 1000 == 0:
                  params = [p for p in model.parameters() if p.requires_grad]
                  g_ic = torch.autograd.grad(l_ic_raw, params, retain_graph=True, allow_unused=True)
                  g_pde = torch.autograd.grad(l_pde_raw, params, retain_graph=True, allow_unused=True)
                  v_ic = torch.cat([g.flatten() for g in g_ic if g is not None])
                  v_pde = torch.cat([g.flatten() for g in g_pde if g is not None])
                  
                  n_ic, n_pde = torch.norm(v_ic) + 1e-9, torch.norm(v_pde) + 1e-9
                  r_grad = (n_pde / n_ic).item()
                  cos_phi = torch.nn.functional.cosine_similarity(v_ic, v_pde, dim=0).item()
                  tr_ratio = (n_ic.item()**2) / (n_pde.item()**2 + 1e-9)
                  
                  # Update Poids (Clamped)
                  ideal_w = max(1.0 / (r_grad + 1e-9), 1e-4)
                  current_pde_w = 0.9 * current_pde_w + 0.1 * ideal_w

                  # Log
                  with open(log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([f"{t_max:.4f}", i, f"{l_ic_raw.item():.2e}", f"{l_pde_raw.item():.2e}", 
                                         f"{r_grad:.2f}", f"{cos_phi:.3f}", f"{tr_ratio:.2e}", f"{current_pde_w:.2e}"])
                  tqdm.write(f"\n📊 [t={t_max:.4f} | {i:04d}] R_grad: {r_grad:.1f} | Cos(φ): {cos_phi:.3f} | W_pde: {current_pde_w:.1e}")

             # BC Neumann
             idx_bc = torch.randperm(b_p.size(0))[:int(b_p.size(0)*0.25)]
             b_bc = b_p[idx_bc]; c_bc = c_p[idx_bc].clone()
             x_min, x_max = cfg['physics']['x_domain']
             c_left = c_bc.clone(); c_left[:, 0] = x_min; c_right = c_bc.clone(); c_right[:, 0] = x_max
             b_all_bc = torch.cat([b_bc, b_bc], dim=0); c_all_bc = torch.cat([c_left, c_right], dim=0)
             c_all_bc.requires_grad_(True)
             ur_bc, ui_bc = model(b_all_bc, c_all_bc)
             grads_r = torch.autograd.grad(ur_bc.sum(), c_all_bc, create_graph=True)[0]
             grads_i = torch.autograd.grad(ui_bc.sum(), c_all_bc, create_graph=True)[0]
             loss_bc = torch.mean(grads_r[:, 0:1]**2 + grads_i[:, 0:1]**2)

             # LOSS TOTALE CORRIGÉE (Pas de Key Error)
             loss = current_pde_w * l_pde_raw + current_ic_w * l_ic_raw + weights.get('bc_loss', 1.0) * loss_bc

        # Sécurité Diag
        if stop_on_explosion and loss.item() > 100.0:
            return False, current_champion_state, 1e9

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Tracking Best Local
        if loss.item() < 100.0: # Pas d'explosion
             # Logic simplifiée : on garde le dernier état stable par défaut
             pass

    # --- L-BFGS ---
    if use_lbfgs and not stop_on_explosion and t_max > 1e-5:
        tqdm.write("    🔧 L-BFGS Finisher...")
        lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=800, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
            pr, pi = model(b_i, c_i)
            # Loss corrigée L-BFGS
            loss_bfgs = current_pde_w * torch.mean(rr**2 + ri**2) + current_ic_w * torch.mean((pr-tr_re)**2 + (pi-tr_im)**2)
            loss_bfgs.backward()
            return loss_bfgs
        try: lbfgs.step(closure)
        except: pass

    # Audit Final avec Affichage Forcé
    print(f"    📊 Audit de Fin de Cycle [{context_name}] :")
    passed, _, score = run_audit(model, cfg, t_max, threshold=cfg['training'].get('target_error_global', 0.05), verbose=True)
    
    if score < current_champion_score:
        current_champion_state = copy.deepcopy(model.state_dict())
        current_champion_score = score

    return passed, current_champion_state, current_champion_score

# ==============================================================================
# 4. LE CONTRÔLEUR
# ==============================================================================

def run_controller(model, cfg, t_target, dt, global_iters_yaml):
    """
    1. Diag (4000 it) -> 2. Global (YAML) -> 3. Audit -> 4. Spécifique (YAML)
    """
    device = next(model.parameters()).device
    base_lr = float(cfg['time_marching'].get('learning_rate', 1e-4))
    target_err = cfg['training'].get('target_error_global', 0.05)
    
    # ÉTAPE 1 : DIAGNOSTIC (4000 it)
    print(f"    🛡️ [Controller] Diagnostic (4000 it) pour valider dt={dt:.4f}...")
    diag_state = copy.deepcopy(model.state_dict())
    
    # Le diag est silencieux par défaut, mais affichera l'audit final grâce au fix
    diag_success, _, diag_score = train_worker(
        model, cfg, t_target, start_lr=base_lr, n_iters=4000, 
        batch_gen_func=get_standard_batch_generator(cfg, device, t_target),
        context_name="DIAG",
        global_best_state=diag_state, global_best_score=1e9,
        use_lbfgs=False, stop_on_explosion=True
    )
    
    if diag_score > 10.0:
        print(f"    💥 Diagnostic Échoué (Score: {diag_score:.2f}). DT/LR trop élevés.")
        model.load_state_dict(diag_state)
        return False, diag_state

    # ÉTAPE 2 : GLOBAL (iters du YAML)
    print(f"    ✅ Diag OK. Lancement Global ({global_iters_yaml} it).")
    glob_success, best_state, best_score = train_worker(
        model, cfg, t_target, start_lr=base_lr, n_iters=global_iters_yaml,
        batch_gen_func=get_standard_batch_generator(cfg, device, t_target),
        context_name="GLOBAL",
        global_best_state=model.state_dict(), global_best_score=1.0,
        use_lbfgs=True
    )
    model.load_state_dict(best_state)

    # ÉTAPE 3 : AUDIT DE DÉCISION
    print(f"    🧐 Audit de Contrôle (Entraînement Spécifique requis ?) :")
    pass_global, failed_types, score_final = run_audit(model, cfg, t_target, threshold=target_err, verbose=True)
    
    if pass_global or not failed_types:
        return True, best_state

    # ÉTAPE 4 : SPÉCIFIQUE (Si échec partiel)
    print(f"    🎯 Échec Global ({score_final:.2%}). Entraînement Spécifique sur {failed_types}.")
    current_best_state = best_state 
    
    for t_id in failed_types:
        spec_success, spec_state, spec_score = train_worker(
            model, cfg, t_target, start_lr=base_lr, n_iters=global_iters_yaml,
            batch_gen_func=get_biased_batch_generator(cfg, device, [t_id], t_target),
            context_name=f"SPEC_{t_id}",
            global_best_state=current_best_state, global_best_score=best_score,
            use_lbfgs=True
        )
        if spec_success:
            current_best_state = spec_state
            model.load_state_dict(spec_state)
        else:
            print(f"    ❌ Échec Spécifique Type {t_id} ({spec_score:.2%}).")

    print(f"    📊 Audit Final Ultime :")
    pass_final, _, score_final_ult = run_audit(model, cfg, t_target, threshold=target_err, verbose=True)
    
    if pass_final:
        print(f"    🏆 Sauvetage Réussi (Score: {score_final_ult:.2%}) !")
        return True, current_best_state
    else:
        print(f"    🛑 Échec Final (Score: {score_final_ult:.2%}).")
        return False, current_best_state

# ==============================================================================
# 5. LE NAVIGATEUR
# ==============================================================================

def train_navigator(model, cfg, explicit_resume_path=None):
    """Pilote automatique du temps."""
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    t_curr = 0.0
    dt = 0.002
    t_max = cfg['physics']['t_max']
    
    # WARMUP
    print("🧊 [Navigator] Warmup t=0.00...")
    # On force 15000 iters pour le Warmup
    ok_warmup, state_warmup, _ = train_worker(
        model, cfg, 0.0, start_lr=3e-4, n_iters=15000,
        batch_gen_func=get_standard_batch_generator(cfg, next(model.parameters()).device, 0.0),
        context_name="WARMUP",
        global_best_state=model.state_dict(), global_best_score=1.0,
        use_lbfgs=False
    )
    
    if not ok_warmup:
        print("❌ Échec Warmup. Arrêt.")
        return
        
    print("\n📊 [PREUVE WARMUP] Audit Détaillé à t=0 :")
    model.load_state_dict(state_warmup)
    run_audit(model, cfg, 0.0, threshold=0.035, verbose=True)
    print("✅ Warmup Validé. Démarrage de la propagation.\n")
    
    # TIME LOOP
    print("🧭 [Navigator] Démarrage Séquence.")
    
    while t_curr < t_max:
        t_next = t_curr + dt
        if t_next > t_max: t_next = t_max
        iters_yaml = get_zone_config(t_next, cfg)
        
        print(f"\n🚀 [Navigator] Cap t={t_next:.4f} (+{dt:.4f}) | YAML Iters: {iters_yaml}")
        
        # 1. Easy Win
        print(f"    🔎 Vérification Easy Win (Le modèle sait-il déjà prédire ?) :")
        pass_easy, _, score = run_audit(model, cfg, t_next, threshold=cfg['training'].get('target_error_global', 0.05), verbose=True)
        
        if pass_easy:
            print(f"    🎉 EASY WIN VALIDÉ (Score: {score:.2%}). On saute l'entraînement.")
            t_curr = t_next
            dt = min(dt * 1.2, 0.1)
            torch.save({'model': model.state_dict(), 't': t_curr}, os.path.join(save_dir, f"ckpt_t{t_curr:.4f}.pth"))
            continue

        # 2. Controller
        success, new_state = run_controller(model, cfg, t_next, dt, global_iters_yaml=iters_yaml)
        
        if success:
            model.load_state_dict(new_state)
            t_curr = t_next
            torch.save({'model': model.state_dict(), 't': t_curr}, os.path.join(save_dir, f"ckpt_t{t_curr:.4f}.pth"))
        else:
            print(f"    🛑 ÉCHEC CRITIQUE. Réduction dt.")
            dt *= 0.5
            if dt < 1e-5:
                print("💀 DT trop petit. Arrêt.")
                break
            print(f"    🔄 Nouvel essai dt={dt:.5f}")