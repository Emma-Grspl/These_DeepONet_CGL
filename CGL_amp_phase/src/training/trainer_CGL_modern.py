import copy
import csv
import glob
import os
import re
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.physics.pde_cgl import pde_residual_cgle
from src.data.generators import (
    get_ic_batch_cgle,
    get_interface_batch_cgle,
    get_mass_balance_batch_cgle,
    get_pde_batch_cgle_causal,
    get_pde_batch_cgle_causal_mixed,
    get_pde_batch_cgle_global,
    get_rar_batch,
)
from src.utils.solver_cgl import get_ground_truth_CGL

def _atomic_torch_save(state, save_path, retries=3, retry_delay=1.0):
    directory = os.path.dirname(save_path)
    base_name = os.path.basename(save_path)
    last_error = None

    for attempt in range(retries):
        tmp_path = os.path.join(directory, f".{base_name}.tmp-{os.getpid()}-{attempt}")
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, save_path)
            return
        except Exception as exc:
            last_error = exc
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt < retries - 1:
                time.sleep(retry_delay)

    raise last_error


def save_checkpoint_cgl(model, optimizer, t, dt, ckpt_dir, name=None, extra_state=None):
    """Sauvegarde robuste de l'état d'entraînement avec reprise prioritaire sur model_latest."""
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        'model_state': model.state_dict(),
        't_curr': t,
        'dt': dt
    }
    if optimizer is not None:
        state['optimizer_state'] = optimizer.state_dict()
    if extra_state:
        state.update(extra_state)

    file_name = name if name is not None else f"model_t_{t:.4f}.pth"
    save_path = os.path.join(ckpt_dir, file_name)
    latest_path = os.path.join(ckpt_dir, "model_latest.pth")

    history_error = None
    try:
        _atomic_torch_save(state, save_path)
    except Exception as exc:
        history_error = exc
        print(f"    ⚠️ Sauvegarde historique impossible ({save_path}): {exc}")

    _atomic_torch_save(state, latest_path)

    if history_error is not None:
        print("    ↪️ model_latest.pth a bien été mis à jour malgré l'échec du checkpoint historique.")

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


def get_zone_dt(t_target, cfg):
    zones = cfg['time_marching']['zones']
    selected_dt = float(zones[-1]['dt'])
    for zone in zones:
        if t_target <= zone['t_end']:
            selected_dt = float(zone['dt'])
            break
    return selected_dt

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
    latest_path = os.path.join(ckpt_dir_or_file, "model_latest.pth")
    if not files:
        if os.path.isfile(latest_path):
            try:
                ckpt = torch.load(latest_path, map_location='cpu')
                return latest_path, ckpt.get('t_curr', 0.0)
            except Exception:
                return None, 0.0
        return None, 0.0
    max_t = -1.0; best_file = None
    for f in files:
        match = re.search(r"ckpt_t([\d\.]+)\.pth", f)
        if match and float(match.group(1)) > max_t:
            max_t = float(match.group(1))
            best_file = f
    return best_file, max_t


def _is_invalid_score(score):
    try:
        return not np.isfinite(float(score))
    except Exception:
        return True


def _get_physics_loss_cfg(cfg):
    defaults = {
        'pde_relative_weight': 0.5,
        'weak_weight': 0.05,
        'mass_weight': 0.05,
        'continuity_weight': 0.1,
        'mass_n_cases': 16,
        'mass_n_x': 128,
        'continuity_batch_size': 2048,
    }
    training_cfg = cfg['training'] if isinstance(cfg, dict) else cfg.training
    user_cfg = training_cfg.get('physics_losses', {})
    return {k: user_cfg.get(k, v) for k, v in defaults.items()}


def _get_early_stop_cfg(cfg):
    defaults = {
        'enabled': True,
        'min_iters': 12000,
        'patience_audits': 8,
        'min_score_improvement': 0.001,
        'min_loss_improvement_rel': 0.03,
    }
    training_cfg = cfg['training'] if isinstance(cfg, dict) else cfg.training
    user_cfg = training_cfg.get('early_stop', {})
    return {k: user_cfg.get(k, v) for k, v in defaults.items()}


def _get_hard_audit_cfg(cfg):
    defaults = {
        'enabled': True,
        'skip_first_step': True,
        'n_cases': 60,
        'medium_factor': 1.5,
        'mix_hard': 0.5,
        'mix_medium': 0.2,
        'mix_global': 0.3,
        'persistent_top_k': 20,
        'hard_top_fraction': 0.2,
        'medium_top_fraction': 0.2,
    }
    training_cfg = cfg['training'] if isinstance(cfg, dict) else cfg.training
    user_cfg = training_cfg.get('hard_audit', {})
    return {k: user_cfg.get(k, v) for k, v in defaults.items()}


def _get_target_cfg(cfg):
    defaults = {
        'base_target_error': None,
        'early_time_relaxation':
            [
                {'t_max': 0.1, 'target_error': 0.09},
                {'t_max': 0.2, 'target_error': 0.07},
                {'t_max': 0.5, 'target_error': 0.055},
            ],
    }
    training_cfg = cfg['training'] if isinstance(cfg, dict) else cfg.training
    merged = defaults.copy()
    user_cfg = training_cfg.get('target_schedule', {})
    merged.update(user_cfg)
    return merged


def _get_training_mode(cfg):
    training_cfg = cfg['training'] if isinstance(cfg, dict) else cfg.training
    return training_cfg.get('training_mode', 'navigator')


def _get_global_direct_cfg(cfg):
    defaults = {
        'total_iters': None,
        'chunk_iters': 10000,
        'target_error': None,
        'allow_relaxation': False,
        'disable_rar': False,
        'run_polishing': True,
    }
    training_cfg = cfg['training'] if isinstance(cfg, dict) else cfg.training
    user_cfg = training_cfg.get('global_direct', {})
    merged = defaults.copy()
    merged.update(user_cfg)
    return merged


def _get_step_target(cfg, t_curr):
    schedule_cfg = _get_target_cfg(cfg)
    base_target = schedule_cfg['base_target_error']
    if base_target is None:
        base_target = (cfg['training'] if isinstance(cfg, dict) else cfg.training).get('target_error_global', 0.03)

    for item in schedule_cfg.get('early_time_relaxation', []):
        if t_curr <= item['t_max'] + 1e-12:
            return float(max(base_target, item['target_error']))
    return float(base_target)


def _case_signature(case_row):
    return "|".join([
        f"{case_row['alpha']:.6f}",
        f"{case_row['beta']:.6f}",
        f"{case_row['mu']:.6f}",
        f"{case_row['V']:.6f}",
        f"{case_row['A']:.6f}",
        f"{case_row['w0']:.6f}",
        f"{case_row['x0']:.6f}",
        f"{case_row['k']:.6f}",
        f"{case_row['type']:.0f}",
    ])


def _sample_audit_case(cfg, t_eval):
    eq_p = cfg['physics']['equation_params']
    bounds = cfg['physics']['bounds']
    return {
        'alpha': np.random.uniform(eq_p['alpha'][0], eq_p['alpha'][1]),
        'beta': np.random.uniform(eq_p['beta'][0], eq_p['beta'][1]),
        'mu': np.random.uniform(eq_p['mu'][0], eq_p['mu'][1]),
        'V': np.random.uniform(eq_p['V'][0], eq_p['V'][1]),
        'A': np.random.uniform(bounds['A'][0], bounds['A'][1]),
        'w0': 10 ** np.random.uniform(np.log10(bounds['w0'][0]), np.log10(bounds['w0'][1])),
        'x0': 0.0,
        'k': np.random.uniform(bounds['k'][0], bounds['k'][1]),
        'type': 0,
        't_eval': t_eval,
    }


def _evaluate_audit_case(model, case_row, cfg):
    device = next(model.parameters()).device
    x_domain = cfg['physics']['x_domain']
    t_eval = case_row['t_eval']
    t_for_solver = 0.01 if t_eval < 0.01 else t_eval
    X, T, U_cplx = get_ground_truth_CGL(case_row, x_domain[0], x_domain[1], t_for_solver, Nx=512, Nt=None)

    U_true = U_cplx[:, 0] if t_eval < 0.01 else U_cplx.flatten()
    X_flat = X[:, 0] if t_eval < 0.01 else X.flatten()
    T_flat = np.zeros_like(X_flat) + t_eval if t_eval < 0.01 else T.flatten()
    xt_t = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32, device=device)
    p_vec = np.array([case_row[k] for k in ['alpha', 'beta', 'mu', 'V', 'A', 'w0', 'x0', 'k', 'type']], dtype=np.float32)
    p_t = torch.tensor(p_vec, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(X_flat), 1)

    with torch.no_grad():
        ur, ui = model(p_t, xt_t)
        up = (ur + 1j * ui).cpu().numpy().flatten()

    norm = np.linalg.norm(U_true)
    return np.linalg.norm(U_true - up) / (norm if norm > 1e-9 else 1e-9)


def run_hard_audit(model, cfg, t_curr, threshold, save_dir, n_cases=60, medium_factor=1.5):
    os.makedirs(save_dir, exist_ok=True)
    rng_state = np.random.get_state()
    np.random.seed(123 + int(round(t_curr * 1000)))
    records = []

    hard_cfg = _get_hard_audit_cfg(cfg)

    for _ in range(n_cases):
        try:
            case_row = _sample_audit_case(cfg, t_curr)
            score = _evaluate_audit_case(model, case_row, cfg)
            case_row['score'] = score
            case_row['signature'] = _case_signature(case_row)
            records.append(case_row)
        except Exception:
            continue

    np.random.set_state(rng_state)
    if not records:
        return {'hard': [], 'medium': [], 'easy': []}

    records.sort(key=lambda row: row['score'], reverse=True)
    n_records = len(records)
    n_hard = max(1, int(round(n_records * float(hard_cfg['hard_top_fraction']))))
    n_medium = max(1, int(round(n_records * float(hard_cfg['medium_top_fraction']))))

    for idx, row in enumerate(records):
        if idx < n_hard:
            row['bucket'] = 'hard'
        elif idx < n_hard + n_medium:
            row['bucket'] = 'medium'
        else:
            row['bucket'] = 'easy'

        if row['score'] <= threshold:
            row['bucket'] = 'easy'
        elif row['score'] <= medium_factor * threshold and row['bucket'] == 'hard':
            row['bucket'] = 'medium'

    audit_path = os.path.join(save_dir, "hard_audit_cases.csv")
    write_header = not os.path.exists(audit_path)
    with open(audit_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t_eval', 'alpha', 'beta', 'mu', 'V', 'A', 'w0', 'x0', 'k', 'type', 'score', 'bucket', 'signature'])
        if write_header:
            writer.writeheader()
        for row in records:
            writer.writerow(row)

    persistent_counts = {}
    for row in records:
        if row['bucket'] != 'easy':
            persistent_counts[row['signature']] = persistent_counts.get(row['signature'], 0) + 1

    groups = {'hard': [], 'medium': [], 'easy': []}
    for row in records:
        groups[row['bucket']].append(row)

    summary_path = os.path.join(save_dir, "hard_audit_summary.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t_eval', 'n_cases', 'n_hard', 'n_medium', 'n_easy', 'mean_score', 'max_score'])
        if write_header:
            writer.writeheader()
        writer.writerow({
            't_eval': t_curr,
            'n_cases': len(records),
            'n_hard': len(groups['hard']),
            'n_medium': len(groups['medium']),
            'n_easy': len(groups['easy']),
            'mean_score': float(np.mean([r['score'] for r in records])),
            'max_score': float(np.max([r['score'] for r in records])),
        })

    persistence = {}
    if os.path.exists(audit_path):
        with open(audit_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['bucket'] in {'hard', 'medium'}:
                    sig = row['signature']
                    item = persistence.setdefault(sig, {'count': 0, 'worst_score': 0.0, 'last_t_eval': 0.0, 'alpha': row['alpha'], 'beta': row['beta'], 'mu': row['mu'], 'V': row['V'], 'A': row['A'], 'w0': row['w0'], 'x0': row['x0'], 'k': row['k'], 'type': row['type']})
                    item['count'] += 1
                    item['worst_score'] = max(item['worst_score'], float(row['score']))
                    item['last_t_eval'] = max(item['last_t_eval'], float(row['t_eval']))

    persistent_path = os.path.join(save_dir, "hard_audit_persistent.csv")
    with open(persistent_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['signature', 'count', 'worst_score', 'last_t_eval', 'alpha', 'beta', 'mu', 'V', 'A', 'w0', 'x0', 'k', 'type'])
        writer.writeheader()
        for sig, item in sorted(persistence.items(), key=lambda kv: (-kv[1]['count'], -kv[1]['worst_score'])):
            writer.writerow({'signature': sig, **item})

    persistent_hard = []
    persistent_medium = []
    for sig, item in sorted(persistence.items(), key=lambda kv: (-kv[1]['count'], -kv[1]['worst_score'])):
        row = {
            'alpha': float(item['alpha']),
            'beta': float(item['beta']),
            'mu': float(item['mu']),
            'V': float(item['V']),
            'A': float(item['A']),
            'w0': float(item['w0']),
            'x0': 0.0,
            'k': float(item['k']),
            'type': float(item['type']),
            'score': float(item['worst_score']),
            'signature': sig,
        }
        if item['count'] >= 2:
            persistent_hard.append(row)
        else:
            persistent_medium.append(row)

    groups['persistent_hard'] = persistent_hard
    groups['persistent_medium'] = persistent_medium
    return groups


def _compute_relative_pde_loss(components):
    res_norm = torch.sqrt(components['res_re'] ** 2 + components['res_im'] ** 2 + 1e-12)
    term_norm = torch.sqrt(
        components['du_dt_re'] ** 2 + components['du_dt_im'] ** 2 +
        components['diff_re'] ** 2 + components['diff_im'] ** 2 +
        components['lin_re'] ** 2 + components['lin_im'] ** 2 +
        components['nl_re'] ** 2 + components['nl_im'] ** 2 +
        components['adv_re'] ** 2 + components['adv_im'] ** 2 +
        1e-12
    )
    return torch.mean((res_norm / (term_norm + 1e-6)) ** 2)


def _compute_weak_pde_loss(components, coords, cfg):
    x_min, x_max = cfg['physics']['x_domain']
    t_max = cfg['physics']['t_max']
    x_norm = 2.0 * (coords[:, 0:1] - x_min) / (x_max - x_min + 1e-9) - 1.0
    t_norm = 2.0 * coords[:, 1:2] / (t_max + 1e-9) - 1.0

    basis = [
        torch.ones_like(x_norm),
        x_norm,
        t_norm,
        torch.sin(np.pi * x_norm),
        torch.cos(np.pi * x_norm),
        x_norm * t_norm,
    ]
    weak_terms = []
    for phi in basis:
        weak_terms.append(torch.mean(components['res_re'] * phi) ** 2)
        weak_terms.append(torch.mean(components['res_im'] * phi) ** 2)
    return torch.stack(weak_terms).mean()


def _compute_mass_balance_loss(model, cfg, device, t_low, t_high, loss_cfg):
    branch, coords, params_dict, n_cases, n_x = get_mass_balance_batch_cgle(
        int(loss_cfg['mass_n_cases']),
        int(loss_cfg['mass_n_x']),
        cfg,
        device,
        t_low,
        t_high,
    )
    components = pde_residual_cgle(model, branch, coords, params_dict, cfg, return_components=True)

    u_sq = (components['u_re'] ** 2 + components['u_im'] ** 2).view(n_cases, n_x)
    u_quart = (u_sq ** 2)
    ux_sq = (components['du_dx_re'] ** 2 + components['du_dx_im'] ** 2).view(n_cases, n_x)

    density = u_sq.view(-1, 1)
    density_dt = torch.autograd.grad(density.sum(), coords, create_graph=True)[0][:, 1:2].view(n_cases, n_x)

    x_min, x_max = cfg['physics']['x_domain']
    domain_length = x_max - x_min

    dM_dt = domain_length * torch.mean(density_dt, dim=1, keepdim=True)
    M = domain_length * torch.mean(u_sq, dim=1, keepdim=True)
    grad_term = domain_length * torch.mean(ux_sq, dim=1, keepdim=True)
    quartic_term = domain_length * torch.mean(u_quart, dim=1, keepdim=True)
    mu_term = domain_length * torch.mean(params_dict['mu'].view(n_cases, n_x) * u_sq, dim=1, keepdim=True)
    rhs = -2.0 * grad_term + 2.0 * mu_term - 2.0 * quartic_term

    scale = torch.abs(rhs) + torch.abs(M) + 1e-4
    return torch.mean(((dM_dt - rhs) / scale) ** 2)


def _compute_continuity_loss(model, teacher_model, cfg, device, t_prev, loss_cfg):
    if teacher_model is None or t_prev <= 1e-8:
        return torch.tensor(0.0, device=device)

    branch, coords, _ = get_interface_batch_cgle(int(loss_cfg['continuity_batch_size']), cfg, device, t_prev)
    with torch.no_grad():
        teacher_re, teacher_im = teacher_model(branch, coords.detach())

    student_re, student_im = model(branch, coords)
    diff_sq = (student_re - teacher_re) ** 2 + (student_im - teacher_im) ** 2
    ref_sq = teacher_re ** 2 + teacher_im ** 2
    return torch.mean(diff_sq / (ref_sq + 1e-6))

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
        X, T, U_cplx = get_ground_truth_CGL(p_dict, x_domain[0], x_domain[1], t_for_solver, Nx=512, Nt=None)

        if historical:
            U_true = U_cplx[:, 0] if t_eval < 0.01 else U_cplx.flatten()
            X_flat = X[:, 0] if t_eval < 0.01 else X.flatten()
            T_flat = np.zeros_like(X_flat) + t_eval if t_eval < 0.01 else T.flatten()
        else:
            U_true = U_cplx[:, 0] if t_eval < 0.01 else U_cplx[:, -1]
            X_flat = X[:, 0]
            T_flat = np.zeros_like(X_flat) + t_eval
            
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
                 'x0': 0.0,
                 'k': np.random.uniform(bounds['k'][0], bounds['k'][1]),
                 'type': 0}
            
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
def train_step_adaptive(model, optimizer, cfg, t_prev, t_curr, base_lr, n_iters, is_global=False, disable_rar=False, target_error=0.03, allow_relaxation=True, teacher_model=None, case_groups=None, group_weights=None):
    king = KingOfTheHill(model)
    king.update(model, 1.0)
    
    bs_pde = cfg['training']['batch_size_pde']
    loss_cfg = _get_physics_loss_cfg(cfg)
    early_stop_cfg = _get_early_stop_cfg(cfg)
    audit_every = int(early_stop_cfg.get('audit_every', 1000))
    lr_decay_step = int(cfg['time_marching'].get('lr_decay_step', 5000))
    lr_decay_gamma = float(cfg['time_marching'].get('lr_decay_gamma', 0.85))
    
    # Réinitialisation du LR de départ pour ce pas
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    
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
    king.update(model, score_in)
    
    # --- RELOBRALO (EMA) ---
    ema_alpha = 0.999
    w_pde = 1.0
    w_bc = 1.0
    loss_pde_ema = None
    loss_bc_ema = None
    best_plateau_score = None
    best_plateau_loss = None
    plateau_audits = 0
    
    for i in pbar:
        device = next(model.parameters()).device
        
        # --- RELAXATION DE LA CIBLE ---
        if allow_relaxation and i == relax_threshold_iter:
            current_target = target_relaxed
            tqdm.write(f"    ⚠️ Mi-parcours atteint. Cible relaxée à {current_target:.2%}")

        if is_global:
            b_p, c_p, p_p = get_pde_batch_cgle_global(bs_pde, cfg, device, t_curr)
        else:
            if case_groups:
                b_p, c_p, p_p = get_pde_batch_cgle_causal_mixed(bs_pde, cfg, device, t_prev, t_curr, case_groups=case_groups, group_weights=group_weights)
            else:
                b_p, c_p, p_p = get_pde_batch_cgle_causal(bs_pde, cfg, device, t_prev, t_curr)
        
        if rar_active and rar_b is not None and b_p is not None:
            b_p = torch.cat([b_p, rar_b], dim=0)
            c_p = torch.cat([c_p, rar_c], dim=0)
            for k in p_p: p_p[k] = torch.cat([p_p[k], rar_p[k]], dim=0)

        optimizer.zero_grad(set_to_none=True)

        components = pde_residual_cgle(model, b_p, c_p, p_p, cfg, return_components=True)
        l_pde_abs = torch.mean(components['res_re'] ** 2 + components['res_im'] ** 2)
        l_pde_rel = _compute_relative_pde_loss(components)
        l_pde_weak = _compute_weak_pde_loss(components, c_p, cfg)
        l_pde = l_pde_abs + float(loss_cfg['pde_relative_weight']) * l_pde_rel + float(loss_cfg['weak_weight']) * l_pde_weak
        
        idx_bc = torch.randperm(b_p.size(0), device=device)[:int(b_p.size(0)*0.25)]
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
        n_bc = b_bc.size(0)
        ur_left, ur_right = ur_bc[:n_bc], ur_bc[n_bc:]
        ui_left, ui_right = ui_bc[:n_bc], ui_bc[n_bc:]
        du_dx_r_left, du_dx_r_right = grads_r[:n_bc, 0:1], grads_r[n_bc:, 0:1]
        du_dx_i_left, du_dx_i_right = grads_i[:n_bc, 0:1], grads_i[n_bc:, 0:1]
        loss_bc = torch.mean(
            (ur_left - ur_right) ** 2 +
            (ui_left - ui_right) ** 2 +
            (du_dx_r_left - du_dx_r_right) ** 2 +
            (du_dx_i_left - du_dx_i_right) ** 2
        )
        loss_mass = _compute_mass_balance_loss(model, cfg, device, 0.0 if is_global else t_prev, t_curr, loss_cfg)
        loss_continuity = _compute_continuity_loss(model, teacher_model, cfg, device, t_prev, loss_cfg)

        # --- RELOBRALO : Mise à jour des EMA et Poids ---
        with torch.no_grad():
            if loss_pde_ema is None:
                loss_pde_ema = l_pde.item()
                loss_bc_ema = (
                    loss_bc
                    + float(loss_cfg['mass_weight']) * loss_mass
                    + float(loss_cfg['continuity_weight']) * loss_continuity
                ).item()
            else:
                loss_pde_ema = ema_alpha * loss_pde_ema + (1 - ema_alpha) * l_pde.item()
                aux_loss = loss_bc + float(loss_cfg['mass_weight']) * loss_mass + float(loss_cfg['continuity_weight']) * loss_continuity
                loss_bc_ema = ema_alpha * loss_bc_ema + (1 - ema_alpha) * aux_loss.item()
            
            tot_ema = loss_pde_ema + loss_bc_ema + 1e-9
            target_w_pde = min(tot_ema / (2 * loss_pde_ema + 1e-9), 5.0)
            target_w_bc = min(tot_ema / (2 * loss_bc_ema + 1e-9), 5.0)
            
            w_pde = ema_alpha * w_pde + (1 - ema_alpha) * target_w_pde
            w_bc = ema_alpha * w_bc + (1 - ema_alpha) * target_w_bc

        # Application de la pondération dynamique
        aux_loss = loss_bc + float(loss_cfg['mass_weight']) * loss_mass + float(loss_cfg['continuity_weight']) * loss_continuity
        loss = w_pde * l_pde + w_bc * aux_loss

        if loss.item() > 10000:
            tqdm.write(f"    💥 Loss gigantesque détectée ({loss.item():.2e} > 10^4). Retour au meilleur état local.")
            king.restore(model)
            return False, king.best_score
        if torch.isnan(loss) or torch.isinf(loss):
            tqdm.write("    💥 Loss NaN/Inf détectée. Retour au meilleur état local.")
            king.restore(model)
            return False, king.best_score
            
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
                
        # Audit périodique solveur
        if i % audit_every == 0:
            _, score = run_audit(model, cfg, t_curr, threshold=current_target, verbose=False, historical=is_global)
            king.update(model, score)

            current_loss_ema = loss_pde_ema + loss_bc_ema if (loss_pde_ema is not None and loss_bc_ema is not None) else loss.item()
            score_improved = False
            loss_improved = False

            if best_plateau_score is None or score < best_plateau_score - float(early_stop_cfg['min_score_improvement']):
                best_plateau_score = score
                score_improved = True
            if best_plateau_loss is None or current_loss_ema < best_plateau_loss * (1.0 - float(early_stop_cfg['min_loss_improvement_rel'])):
                best_plateau_loss = current_loss_ema
                loss_improved = True

            if score_improved or loss_improved:
                plateau_audits = 0
            else:
                plateau_audits += 1

            if i > 0:
                tqdm.write(
                    f"📊 [It {i}] Loss: {loss.item():.2e} | L2: {score:.2%} "
                    f"| PDE(abs/rel/weak)=({l_pde_abs.item():.2e}/{l_pde_rel.item():.2e}/{l_pde_weak.item():.2e}) "
                    f"| Mass: {loss_mass.item():.2e} | Cont: {loss_continuity.item():.2e} "
                    f"| LR: {scheduler.get_last_lr()[0]:.1e}"
                )

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

            # --- EARLY STOP SUR PLATEAU ---
            if (
                early_stop_cfg['enabled']
                and i >= int(early_stop_cfg['min_iters'])
                and plateau_audits >= int(early_stop_cfg['patience_audits'])
            ):
                tqdm.write(
                    f"    ⏹️ Plateau détecté à it={i} "
                    f"(best L2: {king.best_score:.2%}, audits sans progrès: {plateau_audits}). Arrêt anticipé Adam."
                )
                break

    # --- L-BFGS FINISHER : Si on est proche de la cible à la fin d'Adam ---
    best_attained = king.best_score < current_target
    if not best_attained and king.best_score < current_target + 0.03:
        tqdm.write(f"    🎯 Proche de la cible ({king.best_score:.2%}). Lancement du Finisher L-BFGS...")
        king.restore(model)
        
        # Initialisation L-BFGS avec paramètres spécifiques
        optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=100)
        
        # Closure : Utilise les DERNIERS batches générés par Adam
        # On fige w_pde et w_bc (RELOBRALO) pour cette étape de finition.
        def closure():
            optimizer_lbfgs.zero_grad()
            
            # 1. Résidu PDE
            components = pde_residual_cgle(model, b_p, c_p, p_p, cfg, return_components=True)
            l_pde_abs = torch.mean(components['res_re'] ** 2 + components['res_im'] ** 2)
            l_pde_rel = _compute_relative_pde_loss(components)
            l_pde_weak = _compute_weak_pde_loss(components, c_p, cfg)
            l_pde = l_pde_abs + float(loss_cfg['pde_relative_weight']) * l_pde_rel + float(loss_cfg['weak_weight']) * l_pde_weak
            
            # 2. Conditions aux Bords
            ur_bc, ui_bc = model(b_all_bc, c_all_bc)
            grads_r = torch.autograd.grad(ur_bc.sum(), c_all_bc, create_graph=True)[0]
            grads_i = torch.autograd.grad(ui_bc.sum(), c_all_bc, create_graph=True)[0]
            loss_bc = torch.mean(grads_r[:, 0:1]**2 + grads_i[:, 0:1]**2)
            loss_mass = _compute_mass_balance_loss(model, cfg, device, 0.0 if is_global else t_prev, t_curr, loss_cfg)
            loss_continuity = _compute_continuity_loss(model, teacher_model, cfg, device, t_prev, loss_cfg)
            
            # 3. Loss Totale pondérée (poids RELOBRALO figés)
            aux_loss = loss_bc + float(loss_cfg['mass_weight']) * loss_mass + float(loss_cfg['continuity_weight']) * loss_continuity
            total_loss = w_pde * l_pde + w_bc * aux_loss
            total_loss.backward()
            return total_loss
            
        try:
            optimizer_lbfgs.step(closure)
            
            # Ultime run_audit pour voir si le finisher a réussi
            _, score = run_audit(model, cfg, t_curr, threshold=current_target, verbose=False, historical=is_global)
            if score < current_target:
                tqdm.write(f"    ✨ L-BFGS a réussi ! Score final: {score:.2%}")
                return True, score
            else:
                tqdm.write(f"    📉 L-BFGS n'a pas suffi (Score: {score:.2%}).")
                king.update(model, score) # On garde quand même le bénéfice s'il y en a un
        except Exception as e:
            tqdm.write(f"    ⚠️ Échec L-BFGS Finisher: {e}")

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
        components = pde_residual_cgle(model, b_p, c_p, p_p, cfg, return_components=True)
        l = torch.mean(components['res_re']**2 + components['res_im']**2)
        l.backward()
        return l
        
    try: lbfgs.step(closure)
    except: pass
    
    _, final_score = run_audit(model, cfg, t_max, threshold=target, verbose=True, historical=True)
    return final_score


def train_global_direct(model, cfg, explicit_resume_path=None):
    save_dir = cfg['training'].get('save_dir', "outputs/checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    t_max = float(cfg['physics']['t_max'])
    base_lr = float(cfg['time_marching'].get('learning_rate', 2e-4))
    direct_cfg = _get_global_direct_cfg(cfg)
    total_iters = direct_cfg['total_iters']
    if total_iters is None:
        total_iters = get_zone_config(t_max, cfg)
    total_iters = int(total_iters)
    chunk_iters = max(1, int(direct_cfg['chunk_iters']))
    target_error = direct_cfg['target_error']
    if target_error is None:
        target_error = (cfg['training'] if isinstance(cfg, dict) else cfg.training).get('target_error_global', 0.03)

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    target_path = explicit_resume_path if explicit_resume_path else save_dir
    latest_ckpt, _ = find_latest_checkpoint(target_path)

    iters_done = 0
    stage_idx = 0
    if latest_ckpt:
        print(f"🔄 REPRISE GLOBAL DIRECT DEPUIS : {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        iters_done = int(ckpt.get('global_direct_iters_done', 0))
        stage_idx = int(ckpt.get('global_direct_stage', 0))
        print(f"   (progression restaurée : {iters_done}/{total_iters} itérations, stage={stage_idx})")

    print("\n🌐 [Global Direct] Démarrage de l'entraînement direct sur tout l'horizon temporel.")

    while iters_done < total_iters:
        chunk = min(chunk_iters, total_iters - iters_done)
        print(
            f"\n🚀 Stage global direct {stage_idx + 1} | "
            f"iters {iters_done + 1}-{iters_done + chunk}/{total_iters} | horizon=[0, {t_max:.4f}]"
        )
        success, final_score = train_step_adaptive(
            model,
            optimizer,
            cfg,
            0.0,
            t_max,
            base_lr,
            n_iters=chunk,
            is_global=True,
            disable_rar=bool(direct_cfg['disable_rar']),
            target_error=float(target_error),
            allow_relaxation=bool(direct_cfg['allow_relaxation']),
        )
        iters_done += chunk
        stage_idx += 1
        save_checkpoint_cgl(
            model,
            optimizer,
            t_max,
            t_max,
            save_dir,
            name=f"ckpt_global_direct_stage{stage_idx:03d}.pth",
            extra_state={
                'global_direct_iters_done': iters_done,
                'global_direct_stage': stage_idx,
                'training_mode': 'global_direct',
            },
        )
        if _is_invalid_score(final_score):
            print("    💥 Score NaN/Inf détecté en global direct. Arrêt immédiat.")
            return
        if success:
            print(f"    ✅ Cible globale atteinte à {final_score:.2%}.")
            break

    if bool(direct_cfg['run_polishing']):
        print("\n✨ Global direct terminé. Lancement du polissage final...")
        final_score = run_polishing_loop(model, optimizer, cfg, t_max)
        print(f"🏁 Entraînement global direct terminé. Score final : {final_score:.2%}")
    else:
        _, final_score = run_audit(model, cfg, t_max, threshold=float(target_error), verbose=True, historical=True)
        print(f"🏁 Entraînement global direct terminé sans polissage. Score final : {final_score:.2%}")

    save_checkpoint_cgl(
        model,
        optimizer,
        t_max,
        t_max,
        save_dir,
        name="ckpt_FINAL.pth",
        extra_state={
            'global_direct_iters_done': iters_done,
            'global_direct_stage': stage_idx,
            'training_mode': 'global_direct',
        },
    )

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
    dt_min = float(cfg['time_marching'].get('dt_min', dt))
    t_max = cfg['physics']['t_max']
    base_lr = float(cfg['time_marching'].get('learning_rate', 2e-4))
    hard_audit_cfg = _get_hard_audit_cfg(cfg)
    
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
            zone_dt = get_zone_dt(max(t_prev + 1e-12, t_prev), cfg)
            dt = max(min(float(ckpt['dt']), zone_dt), dt_min)
            print(
                f"   (dt restauré à {dt:.4f} "
                f"[checkpoint={float(ckpt['dt']):.4f}, config={zone_dt:.4f}, plancher={dt_min:.4f}])"
            )
    # -----------------------------------
        
    easy_win_streak = 0
    target = cfg['training'].get('target_error_global', 0.03)
    first_attempt_done = False

    print("\n🧭 [Navigator] Démarrage de la séquence (Hard Constraint).")
    

    while t_prev < t_max:
        soft_accept_mode = False
        if dt < dt_min:
            print(f"\n    ⚠️ Attention: dt ({dt:.5f}) < dt_min ({dt_min}). Activation du mode Soft Accept.")
            dt = dt_min
            soft_accept_mode = True
            
        t_curr = min(t_prev + dt, t_max)
        step_target = _get_step_target(cfg, t_curr)
        print(f"\n🚀 Cap t={t_curr:.4f} (+{dt:.4f}) | Streak: {easy_win_streak}{' [SOFT ACCEPT]' if soft_accept_mode else ''}")
        
        # --- 1. Easy Win ---
        is_easy_win, score = run_audit(model, cfg, t_curr, threshold=step_target if not soft_accept_mode else step_target * 2.0, verbose=True, historical=False)
        if _is_invalid_score(score):
            print("    💥 Audit NaN/Inf détecté. Arrêt immédiat de l'entraînement pour éviter de gaspiller du GPU.")
            return
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
            current_target = step_target if not soft_accept_mode else step_target * 2.0
            zone_iters = get_zone_config(t_curr, cfg)
            case_groups = None
            group_weights = None
            skip_hard_audit = hard_audit_cfg['skip_first_step'] and not first_attempt_done
            if hard_audit_cfg['enabled'] and not is_easy_win and not skip_hard_audit:
                case_groups = run_hard_audit(
                    model,
                    cfg,
                    t_curr,
                    current_target,
                    save_dir,
                    n_cases=int(hard_audit_cfg['n_cases']),
                    medium_factor=float(hard_audit_cfg['medium_factor']),
                )
                group_weights = {
                    'hard': float(hard_audit_cfg['mix_hard']),
                    'medium': float(hard_audit_cfg['mix_medium']),
                    'global': float(hard_audit_cfg['mix_global']),
                }
                print(
                    f"    🧪 Hard audit dt: hard={len(case_groups['hard'])}, "
                    f"medium={len(case_groups['medium'])}, easy={len(case_groups['easy'])}"
                )
            teacher_model = copy.deepcopy(model).to(next(model.parameters()).device).eval()
            for param in teacher_model.parameters():
                param.requires_grad_(False)
            success, final_score = train_step_adaptive(
                model, optimizer, cfg, t_prev, t_curr, base_lr,
                n_iters=zone_iters, is_global=False, target_error=current_target, teacher_model=teacher_model,
                case_groups=case_groups, group_weights=group_weights
            )
            first_attempt_done = True
            if _is_invalid_score(final_score):
                print("    💥 Score NaN/Inf détecté après boucle adaptative. Arrêt immédiat de l'entraînement.")
                return

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
            hist_ok, hist_score = run_audit(model, cfg, t_curr, threshold=step_target if not soft_accept_mode else step_target * 2.0, verbose=True, historical=True)
            if _is_invalid_score(hist_score):
                print("    💥 Audit historique NaN/Inf détecté. Arrêt immédiat de l'entraînement.")
                return
            
            if not hist_ok and not soft_accept_mode:
                print(f"    ⚠️ Oubli catastrophique détecté (Audit Histo: {hist_score:.2%}). Lancement Rescue Loop.")
                rescue_iters = max(4000, min(10000, get_zone_config(t_curr, cfg) // 3))
                success_rescue, _ = train_step_adaptive(model, optimizer, cfg, 0.0, t_curr, base_lr, n_iters=rescue_iters, is_global=True, target_error=step_target, allow_relaxation=False)
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
