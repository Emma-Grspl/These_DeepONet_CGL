import torch
import torch.optim as optim
import numpy as np
import copy
import os
import csv
from tqdm import tqdm
import glob
import re

from src.physics.pde_cgl import pde_residual_cgle
from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle_causal, get_rar_batch, get_pde_batch_cgle_global
from src.utils.solver_cgl import get_ground_truth_CGL

def save_checkpoint_cgl(model, optimizer, t, ckpt_dir, name=None):
    """ Sauvegarde complète de la physique, des poids et de la dynamique d'entraînement """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Capture de l'état global (gère le cas où l'optimiseur n'est pas passé)
    state = {
        'model_state': model.state_dict(),
        't_curr': t
    }
    if optimizer is not None:
        state['optimizer_state'] = optimizer.state_dict()
    
    # 1. Historique : Sauvegarde du fichier spécifique à ce temps t
    if name is None:
        file_name = f"model_t_{t:.3f}.pth"
        save_path = os.path.join(ckpt_dir, file_name)
        torch.save(state, save_path)
    
    # 2. Reprise : Mise à jour constante du point de reprise automatique
    latest_path = os.path.join(ckpt_dir, "model_latest.pth")
    torch.save(state, latest_path)

# ==============================================================================
# 0. OUTILS
# ==============================================================================
class KingOfTheHill:
    def __init__(self, model):
        self.best_score = float('inf')
        self.best_state = copy.deepcopy(model.state_dict())
    def update(self, model, score):
        if score < self.best_score:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            return True
        return False
    def restore(self, model):
        model.load_state_dict(self.best_state)
        return self.best_score

def get_zone_config(t_target, cfg):
    zones = cfg['time_marching']['zones']
    selected_iters = zones[-1]['iters']
    for zone in zones:
        if t_target <= zone['t_end']:
            selected_iters = zone['iters']
            break
    if selected_iters < 100: selected_iters = 5000
    return selected_iters

def find_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir): return None, 0.0
    files = glob.glob(os.path.join(ckpt_dir, "ckpt_t*.pth"))
    if not files: return None, 0.0
    max_t = -1.0; best_file = None
    for f in files:
        match = re.search(r"ckpt_t([\d\.]+)\.pth", f)
        if match and float(match.group(1)) > max_t:
            max_t = float(match.group(1))
            best_file = f
    return best_file, max_t

# ==============================================================================
# 1. AUDIT DE VALIDATION (Local & Historique)
# ==============================================================================
def run_audit(model, cfg, t_max, threshold=0.05, n_global=60, verbose=False, historical=False):
    device = next(model.parameters()).device
    model.eval()
    rng_state = np.random.get_state()
    np.random.seed(42) 

    eq_p = cfg['physics']['equation_params']
    bounds = cfg['physics']['bounds']
    x_domain = cfg['physics']['x_domain']

    def evaluate_point(p_dict, t_eval):
        t_for_solver = 0.01 if t_eval < 0.01 else t_eval
        X, T, U_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], t_for_solver, Nx=128, Nt=None)
        
        U_true = U_cplx[:, 0] if t_eval < 0.01 else U_cplx.flatten()
        X_flat = X[:, 0] if t_eval < 0.01 else X.flatten()
        T_flat = np.zeros_like(X_flat) + t_eval if t_eval < 0.01 else T.flatten()
            
        xt_t = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        p_vec = np.array([p_dict[k] for k in ['alpha','beta','mu','V','A','w0','x0','k','type']])
        p_t = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(X_flat), 1).to(device)
        
        with torch.no_grad():
            ur, ui = model(p_t, xt_t)
            up = (ur + 1j*ui).cpu().numpy().flatten()
        norm = np.linalg.norm(U_true)
        return np.linalg.norm(U_true - up) / (norm if norm > 1e-9 else 1e-9)

    g_errs = []
    for _ in range(n_global):
        try:
            p = {'alpha': np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1]),
                 'beta':  np.random.uniform(eq_p['beta'][0],  eq_p['beta'][1]),
                 'mu':    np.random.uniform(eq_p['mu'][0],    eq_p['mu'][1]),
                 'V':     np.random.uniform(eq_p['V'][0],     eq_p['V'][1]),
                 'A':     np.random.uniform(bounds['A'][0], bounds['A'][1]),
                 'w0':    10**np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1])),
                 'x0': 0.0, 'k': 1.0, 'type': 0}
            
            # Tirage temporel : Point fixe (t_max) ou aléatoire dans le passé (historical)
            if historical and t_max > 1e-5:
                t_eval = np.random.uniform(0.0, t_max)
            else:
                t_eval = t_max if t_max > 1e-5 else 0.0
                
            g_errs.append(evaluate_point(p, t_eval))
        except: continue
    
    score = np.mean(g_errs) if g_errs else 1.0
    if verbose:
        tag = "Histo" if historical else "Local"
        print(f"    🌍 Audit L2 {tag} : {score:.2%} [{'✅' if score < threshold else '❌'}]")
    np.random.set_state(rng_state)
    return (score < threshold), score

# ==============================================================================
# 2. LE WORKER ADAM (Optimiseur Persistant)
# ==============================================================================
def train_worker(model, optimizer, cfg, t_prev, t_curr, current_lr, n_iters, disable_rar=False, is_global=False):
    king = KingOfTheHill(model)
    king.update(model, 1.0)
    
    bs_pde = cfg['training']['batch_size_pde']
    weights = cfg['training']['weights'].copy()
    
    # Mise à jour manuelle du LR sur l'optimiseur persistant
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    rar_active = False
    rar_b, rar_c, rar_p = None, None, None
    
    mode_tag = "[Adam Global]" if is_global else f"[Adam] dt={t_curr-t_prev:.4f}"
    pbar = tqdm(range(n_iters), desc=f"  👷 {mode_tag}", leave=False)
    
    for i in pbar:
        device = next(model.parameters()).device
        
        # Choix du générateur (Local/Causal vs Global/Rescue)
        if is_global:
            b_p, c_p, p_p = get_pde_batch_cgle_global(bs_pde, cfg, device, t_curr)
        else:
            b_p, c_p, p_p = get_pde_batch_cgle_causal(bs_pde, cfg, device, t_prev, t_curr)
        
        # RAR Injection
        if rar_active and rar_b is not None and b_p is not None:
            b_p = torch.cat([b_p, rar_b], dim=0)
            c_p = torch.cat([c_p, rar_c], dim=0)
            for k in p_p: p_p[k] = torch.cat([p_p[k], rar_p[k]], dim=0)

        optimizer.zero_grad(set_to_none=True)

        # Résidu PDE
        rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
        l_pde = torch.mean(rr**2 + ri**2)
        
        # BC Neumann
        idx_bc = torch.randperm(b_p.size(0))[:int(b_p.size(0)*0.25)]
        b_bc = b_p[idx_bc]
        c_bc_base = c_p[idx_bc].detach().clone()
        x_min, x_max = cfg['physics']['x_domain']
        
        c_left = c_bc_base.clone(); c_left[:, 0] = x_min
        c_right = c_bc_base.clone(); c_right[:, 0] = x_max
        b_all_bc = torch.cat([b_bc, b_bc], dim=0); c_all_bc = torch.cat([c_left, c_right], dim=0)
        c_all_bc.requires_grad_(True) 
        
        ur_bc, ui_bc = model(b_all_bc, c_all_bc)
        grads_r = torch.autograd.grad(ur_bc.sum(), c_all_bc, create_graph=True)[0]
        grads_i = torch.autograd.grad(ui_bc.sum(), c_all_bc, create_graph=True)[0]
        loss_bc = torch.mean(grads_r[:, 0:1]**2 + grads_i[:, 0:1]**2)

        # Loss totale
        loss = l_pde + weights.get('bc_loss', 1.0) * loss_bc

        if loss.item() > 10000:
            raise ValueError(f"Loss gigantesque ({loss.item():.2e} > 10^4).")
            
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss est devenue NaN ou Inf.")
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if not disable_rar and i > 0 and i % 2000 == 0:
            if i >= n_iters // 4:
                rar_active = True
                rar_b, rar_c, rar_p = get_rar_batch(model, cfg, device, t_prev, t_curr)
                
        if i % 1000 == 0:
            target = cfg['training'].get('target_error_global', 0.04)
            _, score = run_audit(model, cfg, t_curr, threshold=target, verbose=False, historical=is_global)
            king.update(model, score)
            
            if i > 0:
                tqdm.write(f"📊 [It {i}] Loss Totale: {loss.item():.2e} (PDE: {l_pde.item():.2e}) | L2 Audit: {score:.2%} | LR: {scheduler.get_last_lr()[0]:.1e}")

    king.restore(model) 
    return king.best_score

# ==============================================================================
# 3. LE DIAGNOSTIC (Fail-Fast)
# ==============================================================================
def run_diagnostic(model, optimizer, cfg, t_prev, t_curr, base_lr):
    print(f"    🛡️ Diagnostic (4000 it) de {t_prev:.3f} à {t_curr:.3f}...")
    diag_state = copy.deepcopy(model.state_dict())
    diag_opt_state = copy.deepcopy(optimizer.state_dict()) # Sauvegarde d'Adam
    
    target = cfg['training'].get('target_error_global', 0.04)
    _, score_in = run_audit(model, cfg, t_curr, threshold=target, verbose=False)
    
    try:
        score_out = train_worker(model, optimizer, cfg, t_prev, t_curr, base_lr, 4000, disable_rar=True)
    except Exception as e:
        print(f"      💥 Erreur ou vraie explosion pendant le Diag : {str(e)}")
        model.load_state_dict(diag_state)
        optimizer.load_state_dict(diag_opt_state)
        return False, "reduce_dt"

    model.load_state_dict(diag_state)
    optimizer.load_state_dict(diag_opt_state) # Rollback intégral pour préserver les momentums
    
    if score_out > score_in * 2.0 and score_in < 0.10:
        print(f"      ⚠️ Destruction (In: {score_in:.1%} -> Out: {score_out:.1%}). LR et dt trop grands.")
        return False, "reduce_both"
    elif score_out > 0.50:
        print("      ⚠️ Stagnation extrême.")
        return False, "reduce_dt"
        
    print(f"      ✅ Diag OK (Score projeté: {score_out:.1%}).")
    return True, "ok"

# ==============================================================================
# 4. LA MACRO LOOP (Sans L-BFGS)
# ==============================================================================
def run_macro_loop(model, optimizer, cfg, t_prev, t_curr, base_lr, n_iters, is_global=False):
    target = cfg['training'].get('target_error_global', 0.04)
    max_loops = cfg['training'].get('max_macro_loops', 3)
    
    base_state = copy.deepcopy(model.state_dict())
    base_opt_state = copy.deepcopy(optimizer.state_dict())
    
    current_lr = base_lr
    
    for loop in range(max_loops):
        print(f"\n    🔄 Macro-Loop {loop+1}/{max_loops} | LR: {current_lr:.1e}")
        model.load_state_dict(base_state) 
        optimizer.load_state_dict(base_opt_state)
        
        adam_score = train_worker(model, optimizer, cfg, t_prev, t_curr, current_lr, n_iters, is_global=is_global)
        print(f"      👉 Fin Adam : L2 = {adam_score:.2%}")
        
        if adam_score < target:
            return True, adam_score
            
        print("      ⚠️ Adam insuffisant. Nouvelle boucle avec LR*0.75.")
        current_lr *= 0.75

    model.load_state_dict(base_state)
    optimizer.load_state_dict(base_opt_state)
    return False, float('inf')

# ==============================================================================
# 5. POLISSAGE FINAL (Adam Global + L-BFGS)
# ==============================================================================
def run_polishing_loop(model, optimizer, cfg, t_max):
    target = 0.02 # Cible exigeante : < 2%
    device = next(model.parameters()).device
    
    print("\n    🧹 Dégrossissage Adam Global...")
    train_worker(model, optimizer, cfg, 0.0, t_max, 5e-5, 8000, is_global=True)
    
    print("    ⚙️ Finition au scalpel L-BFGS...")
    lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=50)
    
    def closure():
        lbfgs.zero_grad()
        b_p, c_p, p_p = get_pde_batch_cgle_global(cfg['training']['batch_size_pde'], cfg, device, t_max)
        rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
        l = torch.mean(rr**2 + ri**2)
        l.backward()
        return l
        
    try: lbfgs.step(closure)
    except: pass
    
    _, final_score = run_audit(model, cfg, t_max, threshold=target, verbose=True, historical=True)
    return final_score

# ==============================================================================
# 6. LE NAVIGATEUR
# ==============================================================================
def train_navigator(model, cfg, explicit_resume_path=None):
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    t_prev = 0.0
    
    # --- CORRECTION ICI ---
    dt = float(cfg['time_marching']['zones'][0]['dt']) 
    # ----------------------
    
    t_max = cfg['physics']['t_max']
    base_lr = float(cfg['time_marching'].get('learning_rate', 2e-4))
    
    # Création de l'Optimiseur Persistant
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    latest_ckpt, resume_t = find_latest_checkpoint(save_dir)
    if latest_ckpt:
        print(f"🔄 REPRISE : {os.path.basename(latest_ckpt)} (t={resume_t:.4f})")
        ckpt = torch.load(latest_ckpt)
        model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        t_prev = resume_t
        
    easy_win_streak = 0
    target = cfg['training'].get('target_error_global', 0.04)

    print("\n🧭 [Navigator] Démarrage de la séquence (Hard Constraint).")

    while t_prev < t_max:
        t_curr = min(t_prev + dt, t_max)
        print(f"\n🚀 Cap t={t_curr:.4f} (+{dt:.4f}) | Streak: {easy_win_streak}")
        
        # --- 1. Easy Win ---
        is_easy_win, score = run_audit(model, cfg, t_curr, threshold=target, verbose=True, historical=False)
        step_validated = False
        
        if is_easy_win:
            print(f"    🎉 EASY WIN ({score:.2%}).")
            step_validated = True
            easy_win_streak += 1
            if easy_win_streak >= 3:
                dt = min(dt * 1.2, 0.5)
                easy_win_streak = 0
                print(f"    📈 Bonus vitesse : dt passe à {dt:.4f}")
        else:
            easy_win_streak = 0
            
            # --- 2. Diagnostic (Fail-Fast) ---
            diag_ok, action = run_diagnostic(model, optimizer, cfg, t_prev, t_curr, base_lr)
            if not diag_ok:
                if action == "reduce_both": base_lr *= 0.75; dt *= 0.75
                elif action == "reduce_dt": dt *= 0.75
                print(f"    🔄 Repli tactique : dt={dt:.4f}, LR={base_lr:.1e}")
                continue
                
            # --- 3. Macro Loop (Local) ---
            iters = get_zone_config(t_curr, cfg)
            success, final_score = run_macro_loop(model, optimizer, cfg, t_prev, t_curr, base_lr, iters, is_global=False)
            
            if success:
                print(f"    ✅ Pas validé avec {final_score:.2%}")
                step_validated = True
            else:
                print("    🛑 Échec de la Macro-Loop locale. Réduction de dt.")
                dt *= 0.75
                
        # --- 4. Validation Historique & Rescue Loop ---
        if step_validated:
            # On vérifie si l'on n'a pas sacrifié le passé pour apprendre t_curr
            hist_ok, hist_score = run_audit(model, cfg, t_curr, threshold=target, verbose=True, historical=True)
            
            if not hist_ok:
                print(f"    ⚠️ Oubli catastrophique détecté (Audit Histo: {hist_score:.2%}). Lancement Rescue Loop.")
                # Entraînement global d'urgence sur [0, t_curr]
                success_rescue, _ = run_macro_loop(model, optimizer, cfg, 0.0, t_curr, base_lr, 5000, is_global=True)
                if not success_rescue:
                    print("    🛑 La Rescue Loop a peiné, mais on sauvegarde et on avance prudemment.")
                    dt *= 0.75
                    
            # Le pas est validé physiquement, on l'acte
            t_prev = t_curr
            save_checkpoint_cgl(model, optimizer, t_curr, save_dir, name=f"ckpt_t{t_curr:.4f}.pth")

    print("\n✨ Objectif temporel atteint. Lancement de la boucle de polissage final...")
    final_score = run_polishing_loop(model, optimizer, cfg, t_max)
    print(f"🏁 Entraînement terminé. Score global final : {final_score:.2%}")
    save_checkpoint_cgl(model, optimizer, t_max, save_dir, name="ckpt_FINAL.pth")