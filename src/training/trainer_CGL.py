iimport torch
import torch.optim as optim
import numpy as np
import os
import copy
from tqdm import tqdm

# Imports CGL
from src.physics.pde_cgl import pde_residual_cgle
from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle
from src.utils.solver_cgl import get_ground_truth_CGL 

# ==============================================================================
# 1. GÉNÉRATEUR DE BATCH BIAISÉ (80% / 20%)
# ==============================================================================

def get_biased_batch_generator(cfg, device, target_types, t_limit):
    """
    Retourne une FONCTION qui génère des batchs biaisés :
    - 80% target (les types échoués)
    - 20% global (pour maintenir la mémoire du reste)
    """
    def generator(batch_size_pde, batch_size_ic):
        # --- 1. IC BATCH ---
        # Proportions 80 / 20
        n_tgt_ic = int(0.8 * batch_size_ic)
        n_gen_ic = batch_size_ic - n_tgt_ic
        
        # Partie Générale (20%)
        b_gen, c_gen, tr_re_gen, tr_im_gen, ux_re_gen, ux_im_gen = get_ic_batch_cgle(n_gen_ic, cfg, device)
        
        # Partie Ciblée (80%) - Via boucle de rejet
        list_b, list_c, list_tr_re, list_tr_im, list_ux_re, list_ux_im = [], [], [], [], [], []
        curr = 0
        safety = 0
        while curr < n_tgt_ic and safety < 50:
            tb, tc, t_re, t_im, tu_re, tu_im = get_ic_batch_cgle(n_tgt_ic * 2, cfg, device)
            mask = torch.zeros(tb.shape[0], dtype=torch.bool, device=device)
            for t_id in target_types:
                mask |= (tb[:, 8].long() == t_id)
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
            
            # Fusion IC
            b_ic = torch.cat([b_gen, b_tgt]); c_ic = torch.cat([c_gen, c_tgt])
            tr_ic_re = torch.cat([tr_re_gen, tr_re_tgt]); tr_ic_im = torch.cat([tr_im_gen, tr_im_tgt])
            ux_re = torch.cat([ux_re_gen, ux_re_tgt]); ux_im = torch.cat([ux_im_gen, ux_im_tgt])
        else:
            # Fallback si échec génération ciblée
            b_ic, c_ic, tr_ic_re, tr_ic_im, ux_re, ux_im = get_ic_batch_cgle(batch_size_ic, cfg, device)

        # Shuffle IC
        perm_ic = torch.randperm(b_ic.size(0))
        b_ic, c_ic = b_ic[perm_ic], c_ic[perm_ic]
        tr_ic_re, tr_ic_im = tr_ic_re[perm_ic], tr_ic_im[perm_ic]
        ux_re, ux_im = ux_re[perm_ic], ux_im[perm_ic]

        # --- 2. PDE BATCH (Seulement si t_max > 0) ---
        if t_limit > 1e-5:
            n_tgt_pde = int(0.8 * batch_size_pde) # 80%
            n_gen_pde = batch_size_pde - n_tgt_pde
            
            # Général
            bg, cg, _ = get_pde_batch_cgle(n_gen_pde, cfg, device, t_limit=t_limit)
            # Ciblé (Force Type)
            bt, ct, _ = get_pde_batch_cgle(n_tgt_pde, cfg, device, t_limit=t_limit)
            forced_types = np.random.choice(target_types, size=(n_tgt_pde, 1))
            bt[:, 8] = torch.tensor(forced_types, dtype=torch.float32, device=device).squeeze()
            
            b_pde = torch.cat([bg, bt]); c_pde = torch.cat([cg, ct])
            perm_pde = torch.randperm(b_pde.size(0))
            b_pde, c_pde = b_pde[perm_pde], c_pde[perm_pde]
            
            p_params = {
                "alpha": b_pde[:,0:1], "beta": b_pde[:,1:2], "mu": b_pde[:,2:3], "V": b_pde[:,3:4]
            }
        else:
            b_pde, c_pde, p_params = None, None, None

        return b_pde, c_pde, p_params, b_ic, c_ic, tr_ic_re, tr_ic_im, ux_re, ux_im

    return generator

def get_standard_batch_generator(cfg, device, t_limit):
    """Générateur standard (Uniforme)"""
    def generator(batch_size_pde, batch_size_ic):
        b_ic, c_ic, tr_re, tr_im, ux_re, ux_im = get_ic_batch_cgle(batch_size_ic, cfg, device)
        if t_limit > 1e-5:
            b_p, c_p, p_p = get_pde_batch_cgle(batch_size_pde, cfg, device, t_limit=t_limit)
        else:
            b_p, c_p, p_p = None, None, None
        return b_p, c_p, p_p, b_ic, c_ic, tr_re, tr_im, ux_re, ux_im
    return generator

# ==============================================================================
# 2. AUDIT
# ==============================================================================

def run_audit(model, cfg, t_max, threshold=0.03, n_global=60, n_specific=30):
    device = next(model.parameters()).device
    model.eval()
    
    type_names = {0: "Gaussian", 1: "Sech", 2: "Tanh"}
    rng_state = np.random.get_state()
    np.random.seed(42) 

    if isinstance(cfg, dict): phys = cfg['physics']
    else: phys = cfg.physics
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
                 'x0': 0.0, 'k': 1.0, 'type': np.random.choice([0, 1, 2])}
            g_errs.append(evaluate_point(p, t_max if t_max > 1e-5 else 0.0))
        except: continue
    
    global_score = np.mean(g_errs) if g_errs else 1.0
    passed_global = global_score < threshold
    print(f"    🌍 Audit Global  : {global_score:.2%} [{'✅' if passed_global else '❌'}]")

    # 🛑 ARRÊT IMMÉDIAT SI GLOBAL KO
    if not passed_global:
        np.random.set_state(rng_state)
        return False, [] 

    # --- 2. AUDIT SPÉCIFIQUE (Seulement si Global OK) ---
    failed_types = []
    print(f"    🔎 Audit Spécifique (Détail) :")
    for t_id in [0, 1, 2]:
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
        print(f"      - {type_names[t_id]:<10} : {score:.2%} {status}")
        if score > threshold: failed_types.append(t_id)

    np.random.set_state(rng_state)
    return passed_global, failed_types

# ==============================================================================
# 3. MOTEUR D'ENTRAÎNEMENT UNIFIÉ (CORE ENGINE)
# ==============================================================================

def get_dynamic_pde_weight(model, t_current, cfg, br_pde, co_pde, pde_params, br_ic, co_ic, tr_ic_re, tr_ic_im):
    ramp_end = cfg['training'].get('ramp_end_t', 0.5)
    w_start = cfg['training'].get('pde_weight_start', 0.1)
    w_target = cfg['training'].get('pde_weight_target', 1.0)
    if t_current <= ramp_end: return w_start + (t_current/ramp_end)*(w_target-w_start)
    return w_target

def core_optimization_loop(model, cfg, t_max, start_lr, batch_gen_func, context_name, use_lbfgs=True):
    """
    Le Cœur du système : Exécute la boucle Adam (avec retries) + L-BFGS (optionnel) + Audit.
    Args:
        use_lbfgs (bool): Si True, lance L-BFGS à la fin. Si False, le saute (pour Specific Training).
    """
    device = next(model.parameters()).device
    adam_retries = cfg['training'].get('nb_adam_retries', 3)
    target_err = cfg['training'].get('target_error_ic', 0.03) if t_max < 1e-5 else cfg['training'].get('target_error_global', 0.05)
    weights = cfg['training']['weights'].copy()
    check_interval = cfg['training'].get('check_interval', 2000)
    stagnation_thresh = cfg['training'].get('stagnation_threshold', 0.01)
    
    champion_state = copy.deepcopy(model.state_dict())
    champion_loss = float('inf')
    current_lr = start_lr
    
    print(f"  ⚔️  Start {context_name} Training (LR={current_lr:.1e})...")
    
    for attempt in range(adam_retries):
        model.load_state_dict(champion_state)
        optimizer = optim.Adam(model.parameters(), lr=current_lr)
        
        n_iter = 5000 + (2000 * attempt)
        pbar = tqdm(range(n_iter), desc=f"    [{context_name}] Adam {attempt+1}/{adam_retries}", leave=False)
        
        losses_window = []
        current_pde_w = 0.5
        local_best_loss = float('inf')
        local_best_state = copy.deepcopy(model.state_dict())

        for i in pbar:
            # 1. Fetch Data
            b_p, c_p, p_p, b_i, c_i, tr_re, tr_im, ux_re, ux_im = batch_gen_func(
                cfg['training']['batch_size_pde'], cfg['training']['batch_size_ic']
            )
            
            # 2. Dynamic Weight Update
            if t_max > 0 and i % 500 == 0:
                current_pde_w = get_dynamic_pde_weight(model, t_max, cfg, b_p, c_p, p_p, b_i, c_i, tr_re, tr_im)

            optimizer.zero_grad(set_to_none=True)
            
            # 3. Loss Calculation
            if t_max < 1e-5:
                c_i.requires_grad_(True)
                pr, pi = model(b_i, c_i)
                l_val = torch.mean((pr-tr_re)**2 + (pi-tr_im)**2)
                gr = torch.autograd.grad(pr.sum(), c_i, create_graph=True)[0]
                gi = torch.autograd.grad(pi.sum(), c_i, create_graph=True)[0]
                l_sob = torch.mean((gr[:,0:1]-ux_re)**2 + (gi[:,0:1]-ux_im)**2)
                loss = l_val + 0.1 * l_sob
            else:
                rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
                pr, pi = model(b_i, c_i)
                loss = current_pde_w * torch.mean(rr**2 + ri**2) + weights['ic_loss'] * torch.mean((pr-tr_re)**2 + (pi-tr_im)**2)

            loss.backward()
            optimizer.step()
            
            curr_loss = loss.item()
            losses_window.append(curr_loss)
            
            if i % 1000 == 0:
                print(f"      [{context_name} Iter {i}] Loss: {curr_loss:.2e}")
            
            # King of the Hill Local
            if curr_loss < local_best_loss:
                local_best_loss = curr_loss
                local_best_state = copy.deepcopy(model.state_dict())
            
            # Stagnation Check
            if i > 0 and i % check_interval == 0:
                curr_avg = np.mean(losses_window[-check_interval:])
                if len(losses_window) > check_interval:
                    prev_avg = np.mean(losses_window[-2*check_interval:-check_interval])
                    if (prev_avg - curr_avg)/(prev_avg+1e-9) < stagnation_thresh:
                        print(f"      💤 Stagnation. Stop Adam {attempt+1}.")
                        break
        
        # End Adam Loop: Update Global Champion
        if local_best_loss < champion_loss:
            champion_loss = local_best_loss
            champion_state = local_best_state
            print(f"    🚀 Nouveau Champion (L={champion_loss:.2e})")

        # 🛡️ EARLY EXIT AUDIT (After each Adam loop)
        model.load_state_dict(champion_state)
        passed_g, failed_t = run_audit(model, cfg, t_max, threshold=target_err, n_global=40, n_specific=20)
        
        if passed_g and len(failed_t) == 0:
            print(f"    ✅ Audit Intermédiaire Validé ! Sortie anticipée.")
            break
        
        current_lr *= 0.5
    
    # --- FINISHER L-BFGS ---
    if use_lbfgs:
        print(f"    🔧 L-BFGS Finisher ({context_name})...")
        model.load_state_dict(champion_state)
        lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=800, line_search_fn="strong_wolfe")
        
        bp_fix, cp_fix, pp_fix, bi_fix, ci_fix, tr_re_fix, tr_im_fix, ux_re_fix, ux_im_fix = batch_gen_func(
            cfg['training']['batch_size_pde']*2, cfg['training']['batch_size_ic']*2
        )

        def closure():
            lbfgs.zero_grad()
            if t_max < 1e-5:
                ci_fix.requires_grad_(True)
                pr, pi = model(bi_fix, ci_fix)
                l_val = torch.mean((pr-tr_re_fix)**2 + (pi-tr_im_fix)**2)
                gr = torch.autograd.grad(pr.sum(), ci_fix, create_graph=True)[0]
                gi = torch.autograd.grad(pi.sum(), ci_fix, create_graph=True)[0]
                loss = l_val + 0.1 * torch.mean((gr[:,0:1]-ux_re_fix)**2 + (gi[:,0:1]-ux_im_fix)**2)
            else:
                rr, ri = pde_residual_cgle(model, bp_fix, cp_fix, pp_fix, cfg)
                pr, pi = model(bi_fix, ci_fix)
                loss = current_pde_w * torch.mean(rr**2 + ri**2) + weights['ic_loss'] * torch.mean((pr-tr_re_fix)**2 + (pi-tr_im_fix)**2)
            loss.backward()
            return loss
        
        try: lbfgs.step(closure)
        except: pass
        
        final_loss = closure().item()
        if final_loss < champion_loss:
            champion_loss = final_loss
            champion_state = copy.deepcopy(model.state_dict())
            print(f"    🚀 L-BFGS a amélioré ! (L={champion_loss:.2e})")
        else:
            model.load_state_dict(champion_state)
            print("    ⚠️ Restauration Champion pré-LBFGS.")
    else:
        print(f"    ⏩ L-BFGS Skipped for {context_name} (Safety Preservation).")
        model.load_state_dict(champion_state)

    # --- AUDIT FINAL DU CYCLE ---
    passed_g, failed_t = run_audit(model, cfg, t_max, threshold=target_err, n_global=100, n_specific=50)
    
    return passed_g, failed_t, current_lr, champion_state

# ==============================================================================
# 4. ORCHESTRATEUR
# ==============================================================================

def robust_optimize(model, cfg, t_max, n_iters_base, context_str="Global"):
    max_macro = cfg['training'].get('max_macro_loops', 3)
    start_lr = float(cfg['training']['ic_phase']['learning_rate'])
    device = next(model.parameters()).device
    
    current_lr = start_lr
    
    print(f"\n🏰 [Robust Optimize : {context_str}] t={t_max:.2f}")

    for macro in range(max_macro):
        print(f"  🔄 Macro Cycle {macro+1}/{max_macro}")

        # 1️⃣ PHASE GLOBALE (Standard + L-BFGS)
        gen_std = get_standard_batch_generator(cfg, device, t_max)
        
        ok, failed_types, next_lr, best_state = core_optimization_loop(
            model, cfg, t_max, current_lr, gen_std, context_name="GLOBAL", use_lbfgs=True
        )
        current_lr = next_lr 
        
        if ok and not failed_types:
            print("    🏆 VICTOIRE TOTALE ! Passage à la suite.")
            return True
            
        # 2️⃣ PHASE SPÉCIFIQUE (Si nécessaire) -> Pas de L-BFGS
        if failed_types:
            print(f"    ⚠️ ECHEC SPÉCIFIQUE sur {failed_types}. Lancement Mode CIBLÉ avec LR={current_lr:.1e}")
            
            # Générateur Biaisé (80% sur les failed_types)
            gen_biased = get_biased_batch_generator(cfg, device, failed_types, t_max)
            
            ok_spec, failed_spec, next_lr_spec, best_state_spec = core_optimization_loop(
                model, cfg, t_max, current_lr, gen_biased, context_name="SPECIFIC", use_lbfgs=False
            )
            
            current_lr = next_lr_spec
            
            if ok_spec and not failed_spec:
                 print("    ✅ CORRECTION SPÉCIFIQUE RÉUSSIE !")
                 return True
            else:
                 print(f"    ❌ CORRECTION PARTIELLE (Reste: {failed_spec}). On retente Macro suivante.")
        
    print("🛑 ECHEC FINAL.")
    return False

def train_cgle_curriculum(model, cfg):
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints_cgl")
    os.makedirs(save_dir, exist_ok=True)
    
    print("🧊 WARMUP (IC + Sobolev)...")
    ok = robust_optimize(model, cfg, 0.0, 5000, context_str="Warmup")
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