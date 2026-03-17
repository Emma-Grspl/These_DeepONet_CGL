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

def save_checkpoint_cgl(model, optimizer, t, dt, ckpt_dir, name=None):
    """ Sauvegarde complète de la physique, des poids et de la dynamique d'entraînement """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Capture de l'état global
    state = {
        'model_state': model.state_dict(),
        't_curr': t,
        'dt': dt
    }
    if optimizer is not None:
        state['optimizer_state'] = optimizer.state_dict()
    
    # 1. Historique : Sauvegarde du fichier spécifique à ce temps t
    # CORRECTION : On utilise 'name' s'il existe, sinon on prend la valeur par défaut
    file_name = name if name is not None else f"model_t_{t:.4f}.pth"
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

def find_latest_checkpoint(ckpt_dir_or_file):
    """
    Si on donne un dossier, cherche le plus grand ckpt_tXXX.pth.
    Si on donne un fichier .pth (ex: model_latest.pth), le charge directement.
    """
    if not os.path.exists(ckpt_dir_or_file):
        return None, 0.0
        
    # Si c'est directement un fichier
    if os.path.isfile(ckpt_dir_or_file) and ckpt_dir_or_file.endswith('.pth'):
        try:
            ckpt = torch.load(ckpt_dir_or_file, map_location='cpu')
            t_curr = ckpt.get('t_curr', 0.0) # Récupère le temps s'il existe
            return ckpt_dir_or_file, t_curr
        except:
            return None, 0.0

    # Si c'est un dossier (comportement classique)
    files = glob.glob(os.path.join(ckpt_dir_or_file, "ckpt_t*.pth"))
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
# 2. LE WORKER ADAM UNIQUE & ADAPTATIF
# ==============================================================================
def train_step_adaptive(model, optimizer, cfg, t_prev, t_curr, base_lr, n_iters, is_global=False, disable_rar=False, target_error=0.03, allow_relaxation=True):
    king = KingOfTheHill(model)
    king.update(model, 1.0)
    
    bs_pde = cfg['training']['batch_size_pde']
    weights = cfg['training']['weights'].copy()
    
    # Réinitialisation du LR de départ pour ce pas
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr
        
    # Baisse douce : * 0.8 toutes les 2000 itérations
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
    
    rar_active = False
    rar_b, rar_c, rar_p = None, None, None
    
    # Cibles dynamiques
    target_strict = target_error
    target_relaxed = target_strict + 0.005  # Relaxation à +0.5%
    current_target = target_strict
    relax_threshold_iter = n_iters // 2  # On relaxe à mi-parcours
    
    mode_tag = "[Adam Global]" if is_global else f"[Adam] dt={t_curr-t_prev:.4f}"
    pbar = tqdm(range(n_iters), desc=f"  👷 {mode_tag}", leave=False)
    
    # --- Score initial pour le Fail-Fast interne ---
    _, score_in = run_audit(model, cfg, t_curr, threshold=target_strict, verbose=False, historical=is_global)
    tqdm.write(f"    🔍 Score initial : {score_in:.2%}")
    
    # --- RELOBRALO (EMA) ---
    ema_alpha = 0.999
    w_pde = 1.0
    w_bc = 1.0
    loss_pde_ema = None
    loss_bc_ema = None
    
    for i in pbar:
        device = next(model.parameters()).device
        
        # --- RELAXATION DE LA CIBLE ---
        if allow_relaxation and i == relax_threshold_iter:
            current_target = target_relaxed
            tqdm.write(f"    ⚠️ Mi-parcours atteint. Cible relaxée à {current_target:.2%}")

        if is_global:
            b_p, c_p, p_p = get_pde_batch_cgle_global(bs_pde, cfg, device, t_curr)
        else:
            b_p, c_p, p_p = get_pde_batch_cgle_causal(bs_pde, cfg, device, t_prev, t_curr)
        
        if rar_active and rar_b is not None and b_p is not None:
            b_p = torch.cat([b_p, rar_b], dim=0)
            c_p = torch.cat([c_p, rar_c], dim=0)
            for k in p_p: p_p[k] = torch.cat([p_p[k], rar_p[k]], dim=0)

        optimizer.zero_grad(set_to_none=True)

        rr, ri = pde_residual_cgle(model, b_p, c_p, p_p, cfg)
        l_pde = torch.mean(rr**2 + ri**2)
        
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

        # --- RELOBRALO : Mise à jour des EMA et Poids ---
        with torch.no_grad():
            if loss_pde_ema is None:
                loss_pde_ema = l_pde.item()
                loss_bc_ema = loss_bc.item()
            else:
                loss_pde_ema = ema_alpha * loss_pde_ema + (1 - ema_alpha) * l_pde.item()
                loss_bc_ema = ema_alpha * loss_bc_ema + (1 - ema_alpha) * loss_bc.item()
            
            tot_ema = loss_pde_ema + loss_bc_ema + 1e-9
            target_w_pde = min(tot_ema / (2 * loss_pde_ema + 1e-9), 5.0)
            target_w_bc = min(tot_ema / (2 * loss_bc_ema + 1e-9), 5.0)
            
            w_pde = ema_alpha * w_pde + (1 - ema_alpha) * target_w_pde
            w_bc = ema_alpha * w_bc + (1 - ema_alpha) * target_w_bc

        # Application de la pondération dynamique
        loss = w_pde * l_pde + w_bc * loss_bc

        if loss.item() > 10000:
            raise ValueError(f"Loss gigantesque ({loss.item():.2e} > 10^4).")
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss est devenue NaN ou Inf.")
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Garde-fou LR : Ne pas descendre sous 5e-6
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 5e-6:
                param_group['lr'] = 5e-6
        
        # RAR toutes les 2000 itérations
        if not disable_rar and i > 0 and i % 2000 == 0:
            if i >= n_iters // 4:
                rar_active = True
                rar_b, rar_c, rar_p = get_rar_batch(model, cfg, device, t_prev, t_curr)
                
        # Audit toutes les 1000 itérations
        if i % 1000 == 0:
            _, score = run_audit(model, cfg, t_curr, threshold=current_target, verbose=False, historical=is_global)
            king.update(model, score)

            if i > 0:
                tqdm.write(f"📊 [It {i}] Loss: {loss.item():.2e} | L2: {score:.2%} (Cible: {current_target:.2%}) | LR: {scheduler.get_last_lr()[0]:.1e}")

            # --- FAIL-FAST INTERNE à it=4000 ---
            if i == 4000:
                explosion = (score > score_in * 2.0 and score_in < 0.10)
                stagnation = (score > 0.50)
                if explosion or stagnation:
                    reason = "Explosion" if explosion else "Stagnation extrême"
                    tqdm.write(f"    💥 Fail-Fast interne [{reason}] à it=4000 (In: {score_in:.2%} → Now: {score:.2%}). Abandon.")
                    king.restore(model)
                    return False, score
                else:
                    tqdm.write(f"    ✅ Diag interne OK à it=4000 ({score:.2%}), poursuite de l'entraînement...")
                
            # --- ARRÊT PRÉMATURÉ ---
            if i > 0 and score < current_target:
                tqdm.write(f"    🎯 Cible atteinte ({score:.2%} < {current_target:.2%}) ! Arrêt anticipé.")
                king.restore(model)
                return True, score

    king.restore(model)
    return False, king.best_score

# (run_diagnostic supprimé : logique fusionnée dans train_step_adaptive à it=4000)

# ==============================================================================
# 4. POLISSAGE FINAL (Adam Global + L-BFGS)
# ==============================================================================
def run_polishing_loop(model, optimizer, cfg, t_max):
    target = 0.02 # Cible exigeante : < 2%
    device = next(model.parameters()).device
    
    print("\n    🧹 Dégrossissage Adam Global...")
    train_step_adaptive(model, optimizer, cfg, 0.0, t_max, 5e-5, 8000, is_global=True, target_error=target, allow_relaxation=False)
    
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
# 5. LE NAVIGATEUR
# ==============================================================================
def train_navigator(model, cfg, explicit_resume_path=None):
    # --- MODIFICATION ICI ---
    # Au lieu d'ajouter "checkpoints", on utilise directement le dossier fourni
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    # -------------------------
    
    t_prev = 0.0
    dt = float(cfg['time_marching']['zones'][0]['dt']) 
    dt_min = 0.1 # Plancher de tolérance strict (Garde-fou)
    t_max = cfg['physics']['t_max']
    base_lr = float(cfg['time_marching'].get('learning_rate', 2e-4))
    
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    # --- LOGIQUE DE REPRISE CORRIGÉE ---
    # Si tu as fourni un chemin explicite (ex: model_latest.pth), on l'utilise.
    # Sinon, on fouille dans le dossier de sauvegarde actuel.
    target_path = explicit_resume_path if explicit_resume_path else save_dir
    latest_ckpt, resume_t = find_latest_checkpoint(target_path)
    
    if latest_ckpt:
        print(f"🔄 REPRISE DEPUIS : {latest_ckpt} (Temps détecté: t={resume_t:.4f})")
        ckpt = torch.load(latest_ckpt, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Gestion des différentes structures de dictionnaires possibles
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt) # Cas où seuls les poids sont sauvés
            
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            
        t_prev = resume_t
        
        if 'dt' in ckpt:
            dt = max(ckpt['dt'], dt_min)
            print(f"   (dt restauré à {dt:.4f} [Plancher {dt_min} appliqué])")
    # -----------------------------------
        
    easy_win_streak = 0
    target = cfg['training'].get('target_error_global', 0.03)

    print("\n🧭 [Navigator] Démarrage de la séquence (Hard Constraint).")
    

    while t_prev < t_max:
        soft_accept_mode = False
        if dt < dt_min:
            print(f"\n    ⚠️ Attention: dt ({dt:.5f}) < dt_min ({dt_min}). Activation du mode Soft Accept.")
            dt = dt_min
            soft_accept_mode = True
            
        t_curr = min(t_prev + dt, t_max)
        print(f"\n🚀 Cap t={t_curr:.4f} (+{dt:.4f}) | Streak: {easy_win_streak}{' [SOFT ACCEPT]' if soft_accept_mode else ''}")
        
        # --- 1. Easy Win ---
        is_easy_win, score = run_audit(model, cfg, t_curr, threshold=target if not soft_accept_mode else target * 2.0, verbose=True, historical=False)
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

            # --- 2. La GRANDE Boucle Adaptative (Fail-Fast interne à it=4000) ---
            current_target = target if not soft_accept_mode else target * 2.0
            success, final_score = train_step_adaptive(
                model, optimizer, cfg, t_prev, t_curr, base_lr,
                n_iters=40000, is_global=False, target_error=current_target
            )

            if success or soft_accept_mode:
                if soft_accept_mode and not success:
                    print(f"    🛡️ Soft Accept forcé avec {final_score:.2%} (Déléguera le rattrapage au L-BFGS).")
                else:
                    print(f"    ✅ Pas validé avec {final_score:.2%}")
                step_validated = True
            else:
                print("    🛑 Échec de la boucle adaptative. Réduction de dt.")
                dt *= 0.75
                dt = max(dt, dt_min)  # Garde-fou : plancher strict
                
        # --- 4. Validation Historique & Rescue Loop ---
        if step_validated:
            hist_ok, hist_score = run_audit(model, cfg, t_curr, threshold=target if not soft_accept_mode else target * 2.0, verbose=True, historical=True)
            
            if not hist_ok and not soft_accept_mode:
                print(f"    ⚠️ Oubli catastrophique détecté (Audit Histo: {hist_score:.2%}). Lancement Rescue Loop.")
                success_rescue, _ = train_step_adaptive(model, optimizer, cfg, 0.0, t_curr, base_lr, n_iters=10000, is_global=True, target_error=target, allow_relaxation=False)
                if not success_rescue:
                    print("    🛑 La Rescue Loop a peiné, mais on sauvegarde et on avance prudemment.")
                    dt *= 0.75
                    dt = max(dt, dt_min) # Garde-fou : plancher strict
                    
            t_prev = t_curr
            save_checkpoint_cgl(model, optimizer, t_curr, dt, save_dir, name=f"ckpt_t{t_curr:.4f}.pth")

    print("\n✨ Objectif temporel atteint. Lancement de la boucle de polissage final...")
    final_score = run_polishing_loop(model, optimizer, cfg, t_max)
    print(f"🏁 Entraînement terminé. Score global final : {final_score:.2%}")
    save_checkpoint_cgl(model, optimizer, t_max, dt, save_dir, name="ckpt_FINAL.pth")