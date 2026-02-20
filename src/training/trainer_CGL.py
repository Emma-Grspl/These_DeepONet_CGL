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
from src.data.generators import get_ic_batch_cgle, get_pde_batch_cgle_causal, get_rar_batch
from src.utils.solver_cgl import get_ground_truth_CGL

import os
import torch

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
# 1. AUDIT DE VALIDATION
# ==============================================================================
def run_audit(model, cfg, t_max, threshold=0.05, n_global=60, verbose=False):
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
            g_errs.append(evaluate_point(p, t_max if t_max > 1e-5 else 0.0))
        except: continue
    
    score = np.mean(g_errs) if g_errs else 1.0
    if verbose:
        print(f"    🌍 Audit L2  : {score:.2%} [{'✅' if score < threshold else '❌'}]")
    np.random.set_state(rng_state)
    return (score < threshold), score

# ==============================================================================
# 2. LE WORKER ADAM (Hard Constraint)
# ==============================================================================
def train_worker(model, cfg, t_prev, t_curr, current_lr, n_iters):
    king = KingOfTheHill(model)
    king.update(model, 1.0)
    
    bs_pde = cfg['training']['batch_size_pde']
    weights = cfg['training']['weights'].copy()
    
    # Création de l'optimiseur local pour ce pas de temps
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    # StepLR: On divise par 2 le LR toutes les 2000 itérations
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    rar_active = False
    rar_b, rar_c, rar_p = None, None, None
    
    pbar = tqdm(range(n_iters), desc=f"  👷 [Adam] dt={t_curr-t_prev:.4f}", leave=False)
    
    for i in pbar:
        device = next(model.parameters()).device
        
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

        # Loss totale (Hard Constraint = pas de l_ic !)
        loss = l_pde + weights.get('bc_loss', 1.0) * loss_bc

        # Sécurité anti-explosion
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # RAR & Audit
        if i % 1000 == 0:
            if i >= n_iters // 4:
                rar_active = True
                rar_b, rar_c, rar_p = get_rar_batch(model, cfg, device, t_prev, t_curr)
            
            _, score = run_audit(model, cfg, t_curr, verbose=False)
            king.update(model, score)
            
            if i > 0:
                tqdm.write(f"📊 [It {i}] L2: {score:.2%} | LR: {scheduler.get_last_lr()[0]:.1e}")

    # Restaure le meilleur passage d'Adam avant de rendre la main à la macro-loop
    king.restore(model) 
    return king.best_score

# ==============================================================================
# 3. LE DIAGNOSTIC (Fail-Fast)
# ==============================================================================
def run_diagnostic(model, cfg, t_prev, t_curr, base_lr):
    """Crash-Test rapide (4000 it) pour valider dt et LR."""
    print(f"    🛡️ Diagnostic (4000 it) de {t_prev:.3f} à {t_curr:.3f}...")
    diag_state = copy.deepcopy(model.state_dict())
    
    _, score_in = run_audit(model, cfg, t_curr, verbose=False)
    
    try:
        score_out = train_worker(model, cfg, t_prev, t_curr, base_lr, 4000)
    except Exception as e:
        print("      💥 Explosion numérique détectée.")
        model.load_state_dict(diag_state)
        return False, "reduce_dt"

    model.load_state_dict(diag_state) # On rollback toujours après un Diag
    
    if score_out > score_in * 2.0 and score_in < 0.10:
        print(f"      ⚠️ Destruction (In: {score_in:.1%} -> Out: {score_out:.1%}). LR et dt trop grands.")
        return False, "reduce_both"
    elif score_out > 0.50:
        print("      ⚠️ Stagnation extrême.")
        return False, "reduce_dt"
        
    print(f"      ✅ Diag OK (Score projeté: {score_out:.1%}).")
    return True, "ok"

# ==============================================================================
# 4. LA MACRO LOOP & LE NAVIGATEUR
# ==============================================================================
def run_macro_loop(model, cfg, t_prev, t_curr, base_lr, n_iters):
    target = cfg['training'].get('target_error_global', 0.06)
    max_loops = cfg['training'].get('max_macro_loops', 3)
    base_state = copy.deepcopy(model.state_dict())
    
    current_lr = base_lr
    
    for loop in range(max_loops):
        print(f"\n    🔄 Macro-Loop {loop+1}/{max_loops} | LR: {current_lr:.1e}")
        model.load_state_dict(base_state) # Rollback de la boucle
        
        # 1. Adam
        adam_score = train_worker(model, cfg, t_prev, t_curr, current_lr, n_iters)
        print(f"      👉 Fin Adam : L2 = {adam_score:.2%}")
        
        if adam_score < target:
            return True, adam_score
            
        # 2. L-BFGS (Finition)
        print("      ⚙️ Scalpel L-BFGS...")
        adam_state = copy.deepcopy(model.state_dict()) # Point de sauvegarde Adam
        lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=40)
        
        def closure():
            lbfgs.zero_grad()
            b_p, c_p, p_p = get_pde_batch_cgle_causal(cfg['training']['batch_size_pde'], cfg, next(model.parameters()).device, t_prev, t_curr)
            rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
            l = torch.mean(rr**2 + ri**2)
            l.backward()
            return l
            
        try: lbfgs.step(closure)
        except: pass
        
        _, lbfgs_score = run_audit(model, cfg, t_curr, verbose=False)
        print(f"      👉 Fin L-BFGS : L2 = {lbfgs_score:.2%}")
        
        # Audit Post-LBFGS (Rollback conditionnel)
        if lbfgs_score < target:
            return True, lbfgs_score
        elif lbfgs_score > adam_score:
            print("      ⚠️ L-BFGS a abîmé les poids. Rollback à l'état Adam.")
            model.load_state_dict(adam_state)
            current_lr /= 2.0
        else:
            print("      ⚠️ L-BFGS a aidé mais insuffisant. Nouvelle boucle avec LR/2.")
            current_lr /= 2.0

    model.load_state_dict(base_state)
    return False, float('inf')

def train_navigator(model, cfg, explicit_resume_path=None):
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    t_prev = 0.0
    dt = 0.005 
    t_max = cfg['physics']['t_max']
    base_lr = float(cfg['time_marching'].get('learning_rate', 1e-4))
    
    latest_ckpt, resume_t = find_latest_checkpoint(save_dir)
    if latest_ckpt:
        print(f"🔄 REPRISE : {os.path.basename(latest_ckpt)} (t={resume_t:.4f})")
        ckpt = torch.load(latest_ckpt)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        t_prev = resume_t
        
    easy_win_streak = 0
    target = cfg['training'].get('target_error_global', 0.06)

    print("\n🧭 [Navigator] Démarrage de la séquence (Hard Constraint).")

    while t_prev < t_max:
        t_curr = min(t_prev + dt, t_max)
        print(f"\n🚀 Cap t={t_curr:.4f} (+{dt:.4f}) | Streak: {easy_win_streak}")
        
        # 1. Easy Win
        is_easy_win, score = run_audit(model, cfg, t_curr, threshold=target, verbose=True)
        if is_easy_win:
            print(f"    🎉 EASY WIN ({score:.2%}).")
            t_prev = t_curr
            easy_win_streak += 1
            if easy_win_streak >= 3:
                dt = min(dt * 1.2, 0.2)
                easy_win_streak = 0
                print(f"    📈 Bonus vitesse : dt passe à {dt:.4f}")
            torch.save({'model': model.state_dict(), 't': t_curr}, os.path.join(save_dir, f"ckpt_t{t_curr:.4f}.pth"))
            continue
            
        easy_win_streak = 0
        
        # 2. Diagnostic (Fail-Fast)
        diag_ok, action = run_diagnostic(model, cfg, t_prev, t_curr, base_lr)
        if not diag_ok:
            if action == "reduce_both": base_lr /= 2.0; dt *= 0.5
            elif action == "reduce_dt": dt *= 0.5
            print(f"    🔄 Repli tactique : dt={dt:.4f}, LR={base_lr:.1e}")
            continue
            
        # 3. Macro Loop (Trust-Region Training)
        iters = get_zone_config(t_curr, cfg)
        success, final_score = run_macro_loop(model, cfg, t_prev, t_curr, base_lr, iters)
        
        if success:
            print(f"    ✅ Pas validé avec {final_score:.2%}")
            t_prev = t_curr
            torch.save({'model': model.state_dict(), 't': t_curr}, os.path.join(save_dir, f"ckpt_t{t_curr:.4f}.pth"))
        else:
            print("    🛑 Échec de la Macro-Loop. Réduction de dt.")
            dt *= 0.5