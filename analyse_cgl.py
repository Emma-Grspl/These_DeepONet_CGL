import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import yaml
import os
import sys
from tqdm import tqdm

# --- IMPORTS ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import conditionnel pour gérer l'environnement
try:
    from src.models.cgl_deeponet import CGL_PI_DeepONet
    from src.utils.solver_cgl import CGLSolver
except ImportError:
    try:
        from src.model.cgl_deeponet import CGL_PI_DeepONet
        from src.utils.solver_cgl import CGLSolver
    except ImportError:
        print("❌ Erreur critique : Impossible d'importer le modèle ou le solveur CGL.")
        sys.exit(1)

# =============================================================================
# 1. CHARGEMENT
# =============================================================================

def load_config(path="configs/cgl_config.yaml"):
    if os.path.exists(path):
        with open(path, 'r') as f: return yaml.safe_load(f)
    return {'physics': {'x_domain': [-20, 20], 't_max': 2.0}}

def load_model(model_path, cfg, device):
    print(f"📥 Chargement du modèle depuis : {model_path}")
    model = CGL_PI_DeepONet(cfg).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def predict_case(model, p_dict, cfg, t_max, device):
    # --- A. VÉRITÉ TERRAIN (SOLVEUR) ---
    class SolverConfig:
        physics = {
            'x_domain': cfg['physics']['x_domain'],
            't_max': t_max  # <--- CORRECTION : Plus de marge (+0.05 retiré)
        }
    
    # Détection BC & Setup
    bc_type = 'dirichlet' if int(p_dict['type']) == 2 else 'periodic'
    Nx = 256
    Nt_solver = int(SolverConfig.physics['t_max'] / 0.001) 
    
    solver = CGLSolver(SolverConfig(), params_dict=p_dict, nx=Nx, nt=Nt_solver, bc_type=bc_type)
    
    # Init u0
    x = solver.x
    X_s = (x - p_dict['x0']) / (p_dict['w0'] + 1e-9)
    Phase = np.exp(1j * p_dict['k'] * x)
    typ = int(p_dict['type'])
    
    if typ == 0: u0 = p_dict['A'] * np.exp(-X_s**2) * Phase
    elif typ == 1: u0 = p_dict['A'] / np.cosh(X_s) * Phase
    elif typ == 2: u0 = p_dict['A'] * np.tanh(X_s) * Phase
    else: u0 = np.zeros_like(x, dtype=np.complex64)
    
    # Résolution
    _, U_full, t_full = solver.solve(u0)
    
    # --- FILTRE DE SÉCURITÉ ---
    # On ne garde que les temps <= t_max (pour éviter tout dépassement d'arrondi)
    mask_t = t_full <= (t_max + 1e-6)
    t_full = t_full[mask_t]
    U_full = U_full[mask_t, :]
    
    # Sous-échantillonnage pour l'affichage (100 frames max)
    n_frames = min(100, len(t_full))
    idx_t = np.linspace(0, len(t_full)-1, n_frames, dtype=int)
    t_sub = t_full[idx_t]
    U_true_sub = U_full[idx_t, :]
    
    # --- B. PRÉDICTION DEEPONET ---
    x_flat = np.tile(x, len(t_sub))
    t_flat = np.repeat(t_sub, len(x))
    
    p_vec = np.array([p_dict['alpha'], p_dict['beta'], p_dict['mu'], p_dict['V'],
                      p_dict['A'], p_dict['w0'], p_dict['x0'], p_dict['k'], float(p_dict['type'])])
    
    p_tensor = torch.tensor(p_vec, dtype=torch.float32).to(device).repeat(len(x_flat), 1)
    xt_tensor = torch.tensor(np.stack([x_flat, t_flat], axis=1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        ur, ui = model(p_tensor, xt_tensor)
        ur = ur.cpu().numpy().reshape(len(t_sub), len(x))
        ui = ui.cpu().numpy().reshape(len(t_sub), len(x))
        
    U_pred_sub = ur + 1j * ui
    
    return x, t_sub, U_true_sub, U_pred_sub

# =============================================================================
# 2. ANALYSES VISUELLES (FIXES)
# =============================================================================

def run_analysis(model, cfg, device):
    # Paramètres fixes pour l'analyse visuelle
    test_cases = [
        {'name': 'Type_0_Gaussian', 'p': {'alpha': 0.5, 'beta': 0.5, 'mu': 0.5, 'V': 1.0, 'A': 1.0, 'w0': 1.0, 'x0': 0.0, 'k': 0.5, 'type': 0.0}},
        {'name': 'Type_1_Sech',     'p': {'alpha': 0.5, 'beta': 0.5, 'mu': 0.5, 'V': 1.0, 'A': 1.0, 'w0': 1.0, 'x0': 0.0, 'k': 0.5, 'type': 1.0}},
        {'name': 'Type_2_Tanh',     'p': {'alpha': 0.5, 'beta': 0.5, 'mu': 0.5, 'V': 1.0, 'A': 1.0, 'w0': 1.0, 'x0': 0.0, 'k': 0.5, 'type': 2.0}}
    ]
    
    target_times = [0.0, 0.03, 0.06, 0.09]
    base_output = "outputs/Analysis_t0.09"
    os.makedirs(base_output, exist_ok=True)
    
    print(f"🚀 Lancement de l'analyse visuelle...")

    for case in test_cases:
        name = case['name']
        p = case['p']
        print(f"   👉 {name}")
        
        path_root = os.path.join(base_output, name)
        os.makedirs(path_root, exist_ok=True)

        # 1. Prédiction
        x, t, u_true, u_pred = predict_case(model, p, cfg, 0.09, device)
        
        # 2. Metrics
        metrics = {}
        comps_data = {
            'Reelle': (u_true.real, u_pred.real),
            'Imaginaire': (u_true.imag, u_pred.imag),
            'Module': (np.abs(u_true), np.abs(u_pred))
        }
        
        for k_comp, (ut, up) in comps_data.items():
            metrics[k_comp] = np.linalg.norm(ut - up) / (np.linalg.norm(ut) + 1e-9)
            
        with open(os.path.join(path_root, "metrics.txt"), "w") as f:
            f.write(f"--- Erreurs L2 Relatives Moyennes ({name}) ---\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4%}\n")

        # --- A. SNAPSHOTS ---
        colors = cm.viridis(np.linspace(0.1, 0.9, len(target_times)))
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Liste pour itérer proprement sur les axes et les données
        plot_list = [('Partie Réelle', comps_data['Reelle']), 
                     ('Partie Imaginaire', comps_data['Imaginaire']), 
                     ('Module', comps_data['Module'])]

        for i_ax, (title, (ut, up)) in enumerate(plot_list):
            ax = axes[i_ax]
            for j, t_tgt in enumerate(target_times):
                idx = np.argmin(np.abs(t - t_tgt))
                lbl = f"t={t_tgt}" if i_ax == 0 else ""
                ax.plot(x, ut[idx], color=colors[j], linestyle='-', alpha=0.5)
                ax.plot(x, up[idx], color=colors[j], linestyle=':', linewidth=2, label=lbl)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if i_ax == 0: ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig(os.path.join(path_root, "Snapshots_Evolution.png"), dpi=200)
        plt.close()

        # --- B. ANIMATION ---
        # On recrée les données pour l'animation
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        lines = [] # Va stocker (line_true, line_pred, data_true, data_pred)
        
        for i_ax, (title, (ut, up)) in enumerate(plot_list):
            ax = axes[i_ax]
            ax.set_title(title)
            ax.set_xlim(x[0], x[-1])
            
            all_v = np.concatenate([ut.flatten(), up.flatten()])
            margin = (all_v.max() - all_v.min()) * 0.1
            if margin == 0: margin = 0.1
            ax.set_ylim(all_v.min() - margin, all_v.max() + margin)
            
            l1, = ax.plot([], [], 'k-', label='Ref')
            l2, = ax.plot([], [], 'r--', linewidth=2, label='Pred')
            lines.append((l1, l2, ut, up)) # On stocke tout ici
            ax.grid(True)
            if i_ax == 0: ax.legend(loc='upper right')

        # 👇 LA CORRECTION EST ICI 👇
        def update(frame):
            artists = [] # On ne retourne que les objets graphiques
            for (l1, l2, ut, up) in lines:
                # Mise à jour des données
                l1.set_data(x, ut[frame])
                l2.set_data(x, up[frame])
                # Ajout à la liste des objets à redessiner
                artists.append(l1)
                artists.append(l2)
            return artists 
        # 👆 FIN DE LA CORRECTION 👆

        # Sous-échantillonnage pour le GIF (évite d'avoir 1000 frames)
        step_frame = max(1, len(t)//50)
        frames = range(0, len(t), step_frame)
        
        # Note : Sur Mac, si blit=True plante encore, passe-le à False
        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
        ani.save(os.path.join(path_root, "Comparison_Animation.gif"), writer='pillow', fps=15)
        plt.close()
# =============================================================================
# 3. STATISTIQUES MASSIVES
# =============================================================================

def run_massive_stats(model, cfg, device, n_samples=1000):
    print(f"\n📊 LANCEMENT DES STATISTIQUES FINALES (N={n_samples})...")
    
    # 1. Plages de paramètres
    # NOTE : On fixe x0=0 car le modèle a été entraîné uniquement sur des pulses centrés.
    ranges = {
        'alpha': [0.0, 1.0], 
        'beta': [-1.5, 1.5], 
        'mu': [-0.5, 1.0], 
        'V': [-2.0, 2.0],
        'A': [0.5, 2.0], 
        'w0': [0.5, 5.0], 
        'x0': [0.0, 0.0],  # <--- CRITIQUE : On reste dans la distribution apprise
        'k': [-2.0, 2.0]
    }
    
    types = [0, 1, 2]
    type_names = ['Gaussian', 'Sech', 'Tanh']
    
    # Stockage des résultats par composante
    stats = {t: {'Mod': [], 'Real': [], 'Imag': []} for t in types}
    t_eval = 0.09 # Temps final à évaluer
    
    # 2. Boucle de Calcul
    for typ in types:
        print(f"   👉 Traitement : {type_names[typ]}")
        
        # Barre de progression
        for _ in tqdm(range(n_samples), desc=f"Simulations {type_names[typ]}"):
            # Tirage aléatoire
            p = {k: np.random.uniform(v[0], v[1]) for k, v in ranges.items()}
            p['type'] = float(typ)
            
            try:
                # Prédiction
                _, _, u_true, u_pred = predict_case(model, p, cfg, t_eval, device)
                
                # Vérification
                if len(u_true) == 0: continue

                # Extraction dernier temps
                ut, up = u_true[-1], u_pred[-1]
                
                # Normes pour l'erreur relative
                norm_mod = np.linalg.norm(np.abs(ut)) + 1e-9
                norm_re  = np.linalg.norm(ut.real) + 1e-9
                norm_im  = np.linalg.norm(ut.imag) + 1e-9
                
                # Calcul des erreurs
                err_mod = np.linalg.norm(np.abs(ut) - np.abs(up)) / norm_mod
                err_re  = np.linalg.norm(ut.real - up.real) / norm_re
                err_im  = np.linalg.norm(ut.imag - up.imag) / norm_im
                
                # Ajout seulement si numérique valide
                if not np.isnan(err_mod) and not np.isinf(err_mod):
                    stats[typ]['Mod'].append(err_mod)
                    stats[typ]['Real'].append(err_re)
                    stats[typ]['Imag'].append(err_im)
                    
            except Exception:
                # On ignore les erreurs de solveur pour ne pas bloquer le script
                continue

    # 3. Génération du Graphique (Barres Groupées)
    print("\n📈 Génération du graphique final...")
    components = ['Real', 'Imag', 'Mod']
    
    # Configuration des barres
    x_pos = np.arange(len(types))
    width = 0.25  # Largeur des barres
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Bleu, Orange, Vert
    
    plt.figure(figsize=(12, 7))
    
    for i, comp in enumerate(components):
        means = []
        stds = []
        
        # Calcul des moyennes/std pour chaque type
        for t in types:
            data = stats[t][comp]
            if data:
                means.append(np.mean(data))
                stds.append(np.std(data))
            else:
                means.append(0)
                stds.append(0)
        
        # Position décalée pour grouper les barres
        pos = x_pos + (i - 1) * width
        
        # Dessin des barres
        bars = plt.bar(pos, means, width, yerr=stds, capsize=4, 
                       label=comp, color=colors[i], alpha=0.85, edgecolor='black')
        
        # Ajout des étiquettes de valeur au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                         f"{height:.1%}", ha='center', va='bottom', 
                         fontsize=9, fontweight='bold')

    # Mise en forme du graphique
    plt.xlabel("Type de Condition Initiale", fontweight='bold')
    plt.ylabel(f"Erreur Relative L2 (t={t_eval})", fontweight='bold')
    plt.title(f"Performance Moyenne du Modèle (sur {n_samples} simulations aléatoires)", fontsize=14)
    plt.xticks(x_pos, type_names, fontsize=11)
    plt.legend(title="Composante")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(bottom=0) # Commence à 0
    
    # Sauvegarde
    output_path = "outputs/Analysis_t0.09/Statistiques_Globales.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Analyse terminée ! Graphique sauvegardé ici : {output_path}")
if __name__ == "__main__":
    CKPT_PATH = "ckpt_t0.09.pth"
    if not os.path.exists(CKPT_PATH):
        print(f"❌ {CKPT_PATH} introuvable.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device : {device}")

    cfg = load_config()
    model = load_model(CKPT_PATH, cfg, device)
    
    #run_analysis(model, cfg, device)
    run_massive_stats(model, cfg, device, n_samples=500) # Augmente à 1000 pour la vraie science