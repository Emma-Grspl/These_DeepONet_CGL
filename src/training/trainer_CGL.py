import torch
import torch.optim as optim
import numpy as np
import os
import copy
from tqdm import tqdm
import glob
import re

# Imports CGL
from src.physics.pde_cgl import pde_residual_cgle
from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle
from src.utils.solver_cgl import get_ground_truth_CGL 

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

        # 3. PDE Batch
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
    """
    Générateur 'Harmonisé' : 
    - Pour la PDE : 100% Uniforme (Respecte la physique globale)
    - Pour l'IC : 50% Uniforme (Apprend le vide) + 50% Focus (Apprend le pic)
    """
    def generator(batch_size_pde, batch_size_ic):
        # --- 1. GÉNÉRATION IC HARMONISÉE (Mix 50/50) ---
        
        # A. Moitié standard (Exploration globale [-20, 20])
        n_std = batch_size_ic // 2
        b1, c1, t1_re, t1_im, u1_re, u1_im = get_ic_batch_cgle(n_std, cfg, device)
        
        # B. Moitié "Focus" (Concentration sur [-6, 6])
        # On crée une config temporaire pour forcer le générateur à tirer au centre
        n_focus = batch_size_ic - n_std
        cfg_focus = copy.deepcopy(cfg)
        cfg_focus['physics']['x_domain'] = [-6.0, 6.0] # Zone d'activité de Sech/Tanh
        
        b2, c2, t2_re, t2_im, u2_re, u2_im = get_ic_batch_cgle(n_focus, cfg_focus, device)
        
        # C. Fusion et Mélange
        # On concatène
        b_ic = torch.cat([b1, b2])
        c_ic = torch.cat([c1, c2])
        tr_re = torch.cat([t1_re, t2_re]); tr_im = torch.cat([t1_im, t2_im])
        ux_re = torch.cat([u1_re, u2_re]); ux_im = torch.cat([u1_im, u2_im])
        
        # On mélange aléatoirement (Shuffle) pour que le batch ne soit pas trié
        perm = torch.randperm(batch_size_ic)
        b_ic, c_ic = b_ic[perm], c_ic[perm]
        tr_re, tr_im = tr_re[perm], tr_im[perm]
        ux_re, ux_im = ux_re[perm], ux_im[perm]

        # --- 2. GÉNÉRATION PDE (Standard - On ne touche à rien !) ---
        if t_limit > 1e-5:
            # On utilise 'cfg' (le vrai), donc domaine complet [-20, 20]
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
    
    # Lecture des types autorisés depuis le YAML
    phys = cfg['physics'] if isinstance(cfg, dict) else cfg.physics
    allowed_types = phys.get('initial_conditions', [1, 2])
    type_names = {0: "Gaussian", 1: "Sech", 2: "Tanh"}
    
    rng_state = np.random.get_state()
    np.random.seed(42) 

    eq_p, bounds, x_domain = phys['equation_params'], phys['bounds'], phys['x_domain']

    def evaluate_point(p_dict, t_eval):
        if t_eval < 1e-5:
            X, T, U_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], 0.01, Nx=128, Nt=None)
            U_true, X_flat, T_flat = U_cplx[:, 0], X[:, 0], np.zeros_like(X[:, 0])
        else:
            X, T, U_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], t_eval, Nx=128, Nt=None)
            U_true, X_flat, T_flat = U_cplx.flatten(), X.flatten(), T.flatten()
            
        xt_t = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        p_vec = np.array([p_dict[k] for k in ['alpha','beta','mu','V','A','w0','x0','k','type']])
        p_t = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)
        
        with torch.no_grad():
            ur, ui = model(p_t, xt_t)
            up = (ur + 1j*ui).cpu().numpy().flatten()
        norm = np.linalg.norm(U_true)
        return np.linalg.norm(U_true - up) / (norm if norm > 1e-9 else 1e-9)

    # --- 1. AUDIT GLOBAL ---
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
    
    # Affichage systématique du Global
    if verbose:
        status_icon = "✅" if passed_global else "❌"
        print(f"    🌍 Audit Global  : {global_score:.2%} [{status_icon}]")

    # --- 2. AUDIT SPÉCIFIQUE ---
    # On l'exécute MAINTENANT quoi qu'il arrive au Global pour avoir l'info
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
# 3. MOTEUR D'ENTRAÎNEMENT UNIFIÉ (CORE ENGINE)
# ==============================================================================

def get_dynamic_weights(t_current, cfg):
    t_ramp_end = cfg['training'].get('ramp_end_t', 0.1) # Défaut 0.1 pour rampe rapide
    
    # --- Rampe PDE (Montante) ---
    w_pde_start = cfg['training'].get('pde_weight_start', 0.01)
    w_pde_target = cfg['training'].get('pde_weight_target', 1.0)
    
    # --- Rampe IC (Descendante) - Double Rampe ---
    w_ic_start = cfg['training']['weights'].get('ic_loss_start', 10.0)
    w_ic_target = cfg['training']['weights'].get('ic_loss_target', 1.0) # Ou 2.0 selon config

    if t_current <= t_ramp_end:
        # Calcul du ratio de progression (0 à 1)
        ratio = t_current / t_ramp_end
        
        pde_w = w_pde_start + ratio * (w_pde_target - w_pde_start)
        ic_w = w_ic_start - ratio * (w_ic_start - w_ic_target) # Elle descend
    else:
        pde_w = w_pde_target
        ic_w = w_ic_target
        
    return pde_w, ic_w

def core_optimization_loop(model, cfg, t_max, start_lr, batch_gen_func, context_name, 
                           global_best_state, global_best_score, use_lbfgs=True):
    device = next(model.parameters()).device
    adam_retries = cfg['training'].get('nb_adam_retries', 2)
    
    # Seuils
    if t_max < 1e-5:
        target_err = cfg['training'].get('target_error_ic', 0.05)
    else:
        target_err = cfg['training'].get('target_error_global', 0.05)
        
    weights = cfg['training']['weights'].copy()
    check_interval = cfg['training'].get('check_interval', 2000)
    stagnation_thresh = cfg['training'].get('stagnation_threshold', 1e-4)
    
    champion_state = copy.deepcopy(global_best_state)
    champion_loss = float('inf') 
    champion_audit_score = global_best_score 

    current_lr = start_lr
    early_exit_success = False 
    
    # NOUVEAU : On mémorise si Adam a détecté une erreur spécifique
    adam_detected_failures = [] 
    
    print(f"  ⚔️  Start {context_name} Training (LR={current_lr:.1e})...")
    
    for attempt in range(adam_retries):
        model.load_state_dict(champion_state)
        optimizer = optim.Adam(model.parameters(), lr=current_lr)
        
        n_iter = 5000 + (2000 * attempt)
        pbar = tqdm(range(n_iter), desc=f"    [{context_name}] Adam {attempt+1}/{adam_retries}", leave=False)
        
        losses_window = []
        current_pde_w = 0.5
        local_run_best_loss = float('inf')
        local_run_best_state = None

        for i in pbar:
            # 1. Fetch
            b_p, c_p, p_p, b_i, c_i, tr_re, tr_im, ux_re, ux_im = batch_gen_func(
                cfg['training']['batch_size_pde'], cfg['training']['batch_size_ic']
            )
            
            # 2. Weight Update
            if t_max > 0 and i % 500 == 0:
                current_pde_w, current_ic_w = get_dynamic_weights(t_max, cfg)
                weights['ic_loss'] = current_ic_w 

            # 3. Step
            optimizer.zero_grad(set_to_none=True)
            
            if t_max < 1e-5: # WARMUP
                c_i.requires_grad_(True)
                pr, pi = model(b_i, c_i)
                l_val = torch.mean((pr-tr_re)**2 + (pi-tr_im)**2)
                gr = torch.autograd.grad(pr.sum(), c_i, create_graph=True)[0]
                gi = torch.autograd.grad(pi.sum(), c_i, create_graph=True)[0]
                loss = l_val + 0.1 * torch.mean((gr[:,0:1]-ux_re)**2 + (gi[:,0:1]-ux_im)**2)
            else: # TIME MARCHING
                rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
                pr, pi = model(b_i, c_i)
                
                # BC Neumann
                idx_bc = torch.randperm(b_p.size(0))[:int(b_p.size(0)*0.25)]
                b_bc = b_p[idx_bc]; c_bc = c_p[idx_bc].clone()
                x_min, x_max = cfg['physics']['x_domain']
                c_left = c_bc.clone(); c_left[:, 0] = x_min
                c_right = c_bc.clone(); c_right[:, 0] = x_max
                b_all_bc = torch.cat([b_bc, b_bc], dim=0); c_all_bc = torch.cat([c_left, c_right], dim=0)
                c_all_bc.requires_grad_(True)
                ur_bc, ui_bc = model(b_all_bc, c_all_bc)
                grad_outputs = torch.ones_like(ur_bc)
                grads_r = torch.autograd.grad(ur_bc, c_all_bc, grad_outputs=grad_outputs, create_graph=True)[0]
                grads_i = torch.autograd.grad(ui_bc, c_all_bc, grad_outputs=grad_outputs, create_graph=True)[0]
                loss_bc = torch.mean(grads_r[:, 0:1]**2 + grads_i[:, 0:1]**2)

                loss = current_pde_w * torch.mean(rr**2 + ri**2) \
                     + weights['ic_loss'] * torch.mean((pr-tr_re)**2 + (pi-tr_im)**2) \
                     + weights.get('bc_loss', 0.25) * loss_bc

            loss.backward()
            optimizer.step()
            
            curr_loss = loss.item()
            losses_window.append(curr_loss)
            
            if curr_loss < local_run_best_loss:
                local_run_best_loss = curr_loss
                local_run_best_state = copy.deepcopy(model.state_dict())
            
            if i > 0 and i % check_interval == 0:
                curr_avg = np.mean(losses_window[-check_interval:])
                if len(losses_window) > check_interval:
                    prev_avg = np.mean(losses_window[-2*check_interval:-check_interval])
                    if (prev_avg - curr_avg)/(prev_avg+1e-9) < stagnation_thresh:
                        print(f"      💤 Stagnation. Stop Adam {attempt+1}.")
                        break
        
        if local_run_best_loss < champion_loss:
            champion_loss = local_run_best_loss
            champion_state = local_run_best_state
            print(f"    🚀 Nouveau Champion Local (L={champion_loss:.2e})")

        # 🛡️ AUDIT INTERMÉDIAIRE (Dans la boucle Adam)
        model.load_state_dict(champion_state)
        # verbose=False ici pour ne pas spammer, on affiche juste le résumé
        passed_g, failed_t, current_score = run_audit(model, cfg, t_max, threshold=target_err, n_global=40, n_specific=20, verbose=False)
        
        status_icon = "✅" if passed_g else "❌"
        print(f"    📊 Fin Adam {attempt+1}: Audit Global={current_score:.2%} {status_icon} | Failed={failed_t}")

        # LOGIQUE DE SORTIE
        if passed_g:
            if len(failed_t) == 0:
                print(f"    ✅ Audit Intermédiaire PARFAIT ! Sortie anticipée.")
                early_exit_success = True 
                break 
            else:
                print(f"    ⚠️ Audit Global OK mais Spécifique KO {failed_t}. Sortie pour Correction.")
                # ON SAUVEGARDE L'ECHEC SPÉCIFIQUE
                adam_detected_failures = failed_t 
                early_exit_success = True 
                break 
        
        current_lr *= 0.5
    
    # --- FINISHER L-BFGS ---
    # On ne lance L-BFGS que si on n'a PAS détecté de problème spécifique
    # Si Adam a vu un problème spécifique, L-BFGS ne va pas aider, il faut re-entraîner le spécifique.
    if use_lbfgs and not early_exit_success and len(adam_detected_failures) == 0:
        print(f"    🔧 L-BFGS Finisher ({context_name})...")
        model.load_state_dict(champion_state)
        state_before_lbfgs = copy.deepcopy(model.state_dict())
        _, _, score_before = run_audit(model, cfg, t_max, threshold=target_err, n_global=40, n_specific=0, verbose=False)
        print(f"        -> Audit avant L-BFGS : {score_before:.2%}")

        lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=800, line_search_fn="strong_wolfe")
        bp, cp, pp, bi, ci, tr, ti, ux, ui = batch_gen_func(cfg['training']['batch_size_pde']*2, cfg['training']['batch_size_ic']*2)

        def closure():
            lbfgs.zero_grad()
            if t_max < 1e-5:
                ci.requires_grad_(True)
                pr, pi = model(bi, ci)
                l_v = torch.mean((pr-tr)**2 + (pi-ti)**2)
                gr = torch.autograd.grad(pr.sum(), ci, create_graph=True)[0]
                gi = torch.autograd.grad(pi.sum(), ci, create_graph=True)[0]
                loss = l_v + 0.1 * torch.mean((gr[:,0:1]-ux)**2 + (gi[:,0:1]-ui)**2)
            else:
                rr, ri = pde_residual_cgle(model, bp, cp, pp, cfg)
                pr, pi = model(bi, ci)
                
                idx_bc = torch.randperm(bp.size(0))[:int(bp.size(0)*0.25)]
                b_bc = bp[idx_bc]; c_bc = cp[idx_bc].clone()
                x_min, x_max = cfg['physics']['x_domain']
                c_left = c_bc.clone(); c_left[:, 0] = x_min
                c_right = c_bc.clone(); c_right[:, 0] = x_max
                b_all_bc = torch.cat([b_bc, b_bc], dim=0); c_all_bc = torch.cat([c_left, c_right], dim=0)
                c_all_bc.requires_grad_(True)
                ur_bc, ui_bc = model(b_all_bc, c_all_bc)
                grad_outputs = torch.ones_like(ur_bc)
                grads_r = torch.autograd.grad(ur_bc, c_all_bc, grad_outputs=grad_outputs, create_graph=True)[0]
                grads_i = torch.autograd.grad(ui_bc, c_all_bc, grad_outputs=grad_outputs, create_graph=True)[0]
                loss_bc = torch.mean(grads_r[:, 0:1]**2 + grads_i[:, 0:1]**2)

                loss = current_pde_w * torch.mean(rr**2 + ri**2) \
                     + weights['ic_loss'] * torch.mean((pr-tr)**2 + (pi-ti)**2) \
                     + weights.get('bc_loss', 0.25) * loss_bc
            loss.backward()
            return loss
        
        try: lbfgs.step(closure)
        except: pass
        
        _, _, score_after = run_audit(model, cfg, t_max, threshold=target_err, n_global=40, n_specific=0, verbose=False)
        print(f"        -> Audit après L-BFGS : {score_after:.2%}")
        
        if score_after > score_before:
            print(f"    ⚠️ L-BFGS REJETÉ (Dégradation Audit). ROLLBACK.")
            model.load_state_dict(state_before_lbfgs)
        else:
            print(f"    🚀 L-BFGS Validé ! (Gain Audit: {score_before - score_after:.2%})")
            champion_state = copy.deepcopy(model.state_dict())
            champion_audit_score = score_after
            
    elif early_exit_success:
        print(f"    ⏩ L-BFGS Skipped (Early Exit Triggered).")
    else:
        print(f"    ⏩ L-BFGS Skipped (Config).")
        model.load_state_dict(champion_state)

    # --- AUDIT FINAL & DÉCISIF ---
    print(f"    🏁 Audit Final de validation ({context_name}) :")
    # verbose=True pour FORCER l'affichage complet ici
    passed_g, failed_t, final_score = run_audit(model, cfg, t_max, threshold=target_err, n_global=100, n_specific=50, verbose=True)
    
    # 🚨 CORRECTION CRITIQUE : RÉ-INJECTION DE LA MÉMOIRE D'ADAM
    # Si Adam a détecté un problème spécifique, on le remet dans la liste
    # même si l'audit final l'a raté.
    if len(adam_detected_failures) > 0:
        # On ajoute les échecs d'Adam s'ils ne sont pas déjà là
        for f in adam_detected_failures:
            if f not in failed_t:
                failed_t.append(f)
        print(f"    🚨 MÉMOIRE : Ré-injection des échecs détectés par Adam {adam_detected_failures}")
    
    if final_score < global_best_score:
        return passed_g, failed_t, current_lr, champion_state, final_score
    else:
        return passed_g, failed_t, current_lr, global_best_state, global_best_score
# ==============================================================================
# 4. ORCHESTRATEUR
# ==============================================================================

def robust_optimize(model, cfg, t_max, n_iters_base, context_str="Global", start_lr_override=None):
    max_macro = cfg['training'].get('max_macro_loops', 3)
    target_err = cfg['training'].get('target_error_global', 0.05)
    device = next(model.parameters()).device
    
    # Choix du LR de base : Override (Transition) > IC Phase (Warmup) > Time Marching (Normal)
    if start_lr_override:
        start_lr = start_lr_override
    elif t_max < 1e-5:
        start_lr = float(cfg['training']['ic_phase']['learning_rate'])
    else:
        # On va chercher le LR spécifique Time Marching, par défaut 1e-4 si pas présent
        start_lr = float(cfg['time_marching'].get('learning_rate', 1e-4))
    
    current_lr = start_lr
    
    # --- 1. AUDIT INITIAL (AVANT DE FAIRE QUOI QUE CE SOIT) ---
    global_best_state = copy.deepcopy(model.state_dict())
    
    # On fait un audit COMPLET (Global + Spécifique) tout de suite
    passed_g, failed_types, init_score = run_audit(model, cfg, t_max, threshold=target_err, n_global=40, n_specific=20, verbose=False)
    global_best_score = init_score
    
    status_icon = "✅" if passed_g else "❌"
    print(f"\n🏰 [Robust Optimize : {context_str}] t={t_max:.2f} | Initial Score: {init_score:.2%} {status_icon} | Failed Specific: {failed_types}")

    # --- LOGIQUE DE DÉCISION INTELLIGENTE ---
    run_global_phase = True

    # CAS 1 : Tout est parfait -> On ne fait RIEN, on passe au t suivant
    if passed_g and len(failed_types) == 0:
        print(f"  ⏩ SKIP TOTAL : Le modèle est déjà parfait ({init_score:.2%}). Passage au t suivant !")
        return True

    # CAS 2 : Global OK, mais Spécifique KO -> On saute le Global, on va direct au Spécifique
    if passed_g and len(failed_types) > 0:
        print(f"  ↪️  SKIP GLOBAL : Global OK ({init_score:.2%}), focus immédiat sur Spécifique {failed_types}.")
        run_global_phase = False # On désactive la phase globale pour ce cycle
        
    # CAS 3 : Global KO -> On laisse run_global_phase = True
    # ------------------------------------

    for macro in range(max_macro):
        print(f"  🔄 Macro Cycle {macro+1}/{max_macro} (Best: {global_best_score:.2%})")

        # 1️⃣ PHASE GLOBALE (Seulement si nécessaire)
        if run_global_phase:
            gen_std = get_standard_batch_generator(cfg, device, t_max)
            
            ok, f_types, next_lr, best_state, best_score = core_optimization_loop(
                model, cfg, t_max, current_lr, gen_std, "GLOBAL", 
                global_best_state, global_best_score, use_lbfgs=True
            )
            
            # Mise à jour
            current_lr = next_lr 
            if best_score < global_best_score:
                global_best_score = best_score
                global_best_state = best_state
            
            # Mise à jour des échecs pour la phase spécifique
            failed_types = f_types
            
            # Si tout est réglé après le global, on sort !
            if ok and not failed_types:
                print("    🏆 VICTOIRE TOTALE (après Global) ! Passage à la suite.")
                model.load_state_dict(global_best_state)
                return True
        else:
            # Si on a sauté le global au premier tour, on le réactive pour le tour suivant
            # (Au cas où l'entrainement spécifique aurait dégradé le global)
            run_global_phase = True 
            
        # 2️⃣ PHASE SPÉCIFIQUE (Si besoin)
        if failed_types:
            print(f"    ⚠️ CIBLAGE : Correction requise pour {failed_types}.")
            gen_biased = get_biased_batch_generator(cfg, device, failed_types, t_max)
            
            ok_spec, failed_spec, next_lr_spec, best_state_spec, best_score_spec = core_optimization_loop(
                model, cfg, t_max, current_lr, gen_biased, "SPECIFIC", 
                global_best_state, global_best_score, use_lbfgs=False # Souvent mieux sans LBFGS sur le spécifique, ou True selon tes goûts
            )
            
            current_lr = next_lr_spec
            if best_score_spec < global_best_score:
                global_best_score = best_score_spec
                global_best_state = best_state_spec

            if ok_spec and not failed_spec:
                 print("    ✅ CORRECTION SPÉCIFIQUE RÉUSSIE !")
                 model.load_state_dict(global_best_state)
                 return True
            else:
                 print(f"    ❌ CORRECTION PARTIELLE (Reste: {failed_spec}).")
                 failed_types = failed_spec # On met à jour pour la prochaine boucle
        
    print("🛑 ECHEC FINAL (Max Macro loops atteint).")
    model.load_state_dict(global_best_state)
    return False

import glob
import re
import os
import torch

def train_cgle_curriculum(model, cfg, explicit_resume_path=None):
    """
    Args:
        explicit_resume_path (str): Chemin complet vers un checkpoint précis (ex: .../ckpt_t0.09.pth)
                                    Si fourni, force la reprise depuis ce fichier.
    """
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    current_t = 0.0
    loaded = False

    # --- 1. REPRISE FORCÉE (Si chemin fourni) ---
    if explicit_resume_path and os.path.exists(explicit_resume_path):
        print(f"🔄 REPRISE FORCÉE : Chargement de {explicit_resume_path}")
        checkpoint = torch.load(explicit_resume_path, map_location=next(model.parameters()).device)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            current_t = checkpoint.get('t', 0.0)
        else:
            model.load_state_dict(checkpoint)
            # Si le t n'est pas dans le fichier, on essaie de le deviner du nom
            try:
                match = re.search(r"t(\d+\.\d+)", os.path.basename(explicit_resume_path))
                if match: current_t = float(match.group(1))
            except: pass
            
        print(f"✅ Modèle chargé depuis source externe. Reprise à t = {current_t:.4f}")
        loaded = True

    # --- 2. REPRISE AUTOMATIQUE (Dans le dossier courant) ---
    if not loaded:
        print(f"📂 Recherche de reprise locale dans : {save_dir}")
        search_pattern = os.path.join(save_dir, "ckpt_t*.pth")
        checkpoints = glob.glob(search_pattern)
        
        if checkpoints:
            last_ckpt = None
            max_t = -1.0
            for ckpt_path in checkpoints:
                try:
                    match = re.search(r"ckpt_t(\d+\.\d+)\.pth", os.path.basename(ckpt_path))
                    if match:
                        t_val = float(match.group(1))
                        if t_val > max_t:
                            max_t = t_val
                            last_ckpt = ckpt_path
                except: continue
            
            if last_ckpt:
                print(f"🔄 AUTO-RESUME : Chargement de {last_ckpt}")
                checkpoint = torch.load(last_ckpt)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
                current_t = max_t
                print(f"✅ Reprise locale à t = {current_t:.4f}")

    # --- SUITE (WARMUP & ZONES) ---
    
    # 1. WARMUP
    if current_t < 1e-5:
        print("🧊 WARMUP (IC + Sobolev)...")
        ok = robust_optimize(model, cfg, 0.0, 5000, context_str="Warmup")
        if not ok: return
    else:
        print(f"⏩ SKIP WARMUP (Déjà traité, t={current_t:.2f})")

    # 2. TIME MARCHING
    zones = cfg['time_marching']['zones']
    
    for zone in zones:
        z_end, z_dt, z_iters = zone['t_end'], zone['dt'], zone['iters']
        
        if current_t >= z_end - 1e-9:
            continue
            
        print(f"\n🚀 ZONE : t_end={z_end}, dt={z_dt} (Start t={current_t:.2f})")
        
        # Flag pour détecter le tout premier pas de cette zone
        is_new_zone = True 
        
        while current_t < z_end - 1e-9:
            next_t = current_t + z_dt
            if next_t > z_end: next_t = z_end
            
            # --- LOGIQUE DE LR ADAPTATIF ET SOFT START ---
            # 1. On récupère le LR de croisière (ex: 1e-4)
            base_lr = float(cfg['time_marching'].get('learning_rate', 1e-4))
            
            if is_new_zone:
                # 2. Transition de zone : Soft Start (Divisé par 2)
                step_lr = base_lr * 0.5
                print(f"  🛡️ Soft Start (Transition Zone) : LR={step_lr:.1e}")
                is_new_zone = False # Désactivé pour les prochains pas
            else:
                step_lr = base_lr
            
            # 3. Appel avec le LR calculé
            ok = robust_optimize(model, cfg, next_t, z_iters, 
                                 context_str="Global", start_lr_override=step_lr)
            if not ok: 
                print("🛑 Arrêt demandé ou échec critique.")
                return

            current_t = next_t
            
            # Sauvegarde dans le NOUVEAU dossier
            save_name = f"ckpt_t{current_t:.2f}.pth"
            save_path = os.path.join(save_dir, save_name)
            torch.save({'model': model.state_dict(), 't': current_t}, save_path)
            print(f"    💾 Checkpoint sauvegardé : {save_name}")

    print("🏁 Entraînement terminé avec succès !")