import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Imports locaux nécessaires
from src.utils.solver_cgl import get_ground_truth_CGL

def plot_temporal_snapshots(cfg, params_dict, model=None, save_path=None, show=False):
    """
    Trace des snapshots de la solution à 5 instants différents (0, t/6, t/4, t/2, t_max).
    Colonnes : Module |u|, Partie Réelle, Partie Imaginaire.
    
    Args:
        cfg: Configuration (pour les bounds, x_domain, etc.)
        params_dict: Dictionnaire des paramètres (alpha, beta, A, etc.)
        model: (Optionnel) Le modèle DeepONet entraîné.
        save_path: (Optionnel) Chemin de sauvegarde.
        show: (Bool) Afficher l'image.
    """
    
    # --- 1. Génération de la Vérité Terrain (Solveur) ---
    print(f"📊 Génération de la solution exacte pour params: {params_dict}")
    
    # Gestion souple de l'accès à la config (Dict ou Objet)
    if isinstance(cfg, dict):
        x_min, x_max = cfg['physics']['x_domain']
        t_max = cfg['physics']['t_max']
    else:
        x_min, x_max = cfg.physics['x_domain']
        t_max = cfg.physics['t_max']
    
    # On récupère la grille complète
    X_grid, T_grid, U_true = get_ground_truth_CGL(
        params_dict, x_min, x_max, t_max, Nx=512, Nt=1000
    )
    
    # Axes 1D
    x = X_grid[:, 0]
    t = T_grid[0, :]
    
    # --- 2. Prédiction du Modèle (si fourni) ---
    U_pred = None
    if model is not None:
        device = next(model.parameters()).device
        model.eval()
        
        X_flat = X_grid.flatten()
        T_flat = T_grid.flatten()
        coords = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        
        # Branch Input (9 paramètres)
        p_vec = np.array([
            params_dict['alpha'], params_dict['beta'], params_dict['mu'], params_dict.get('V', 0.0),
            params_dict['A'], params_dict['w0'], params_dict['x0'], params_dict['k'], float(params_dict['type'])
        ])
        branch = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(coords), 1).to(device)
        
        with torch.no_grad():
            u_re, u_im = model(branch, coords)
            u_cplx = (u_re + 1j * u_im).cpu().numpy().flatten()
            
        U_pred = u_cplx.reshape(X_grid.shape)

    # --- 3. Sélection des Instants ---
    ratios = [0.0, 1/6, 1/4, 1/2, 1.0]
    time_indices = []
    for r in ratios:
        target_t = t_max * r
        idx = (np.abs(t - target_t)).argmin()
        time_indices.append(idx)

    # --- 4. Plotting (3 Colonnes maintenant) ---
    # Taille augmentée en largeur pour accommoder la 3ème colonne
    fig, axes = plt.subplots(5, 3, figsize=(18, 12), sharex=True)
    
    # Ajustement des espaces
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    
    title_str = (f"CGL Snapshots | $\\alpha$={params_dict['alpha']:.2f}, $\\beta$={params_dict['beta']:.2f}, "
                 f"$\\mu$={params_dict['mu']:.2f}, V={params_dict.get('V',0):.2f}")
    fig.suptitle(title_str, fontsize=16, y=0.95)

    for i, t_idx in enumerate(time_indices):
        current_t = t[t_idx]
        u_true_t = U_true[:, t_idx]
        
        if U_pred is not None:
            u_pred_t = U_pred[:, t_idx]
        
        # Labels communs
        ylabel_str = f"t = {current_t:.2f}s"

        # --- Colonne 1 : Module |u| ---
        ax_mod = axes[i, 0]
        ax_mod.plot(x, np.abs(u_true_t), 'k-', label='Exact' if i==0 else "", linewidth=1.5)
        if U_pred is not None:
            ax_mod.plot(x, np.abs(u_pred_t), 'r--', label='DeepONet' if i==0 else "", linewidth=1.5)
            
        ax_mod.set_ylabel(f"{ylabel_str}\n|u|", fontsize=11, fontweight='bold')
        if i == 0: ax_mod.set_title("Module |u|", fontsize=13)
        if i == 4: ax_mod.set_xlabel("x")
        ax_mod.grid(True, alpha=0.3)

        # --- Colonne 2 : Partie Réelle Re(u) ---
        ax_re = axes[i, 1]
        ax_re.plot(x, np.real(u_true_t), 'k-', linewidth=1.5)
        if U_pred is not None:
            ax_re.plot(x, np.real(u_pred_t), 'r--', linewidth=1.5)
            
        if i == 0: ax_re.set_title("Réel Re(u)", fontsize=13)
        if i == 4: ax_re.set_xlabel("x")
        ax_re.grid(True, alpha=0.3)

        # --- Colonne 3 : Partie Imaginaire Im(u) (NOUVEAU) ---
        ax_im = axes[i, 2]
        ax_im.plot(x, np.imag(u_true_t), 'k-', linewidth=1.5)
        if U_pred is not None:
            ax_im.plot(x, np.imag(u_pred_t), 'r--', linewidth=1.5)
            
        if i == 0: ax_im.set_title("Imaginaire Im(u)", fontsize=13)
        if i == 4: ax_im.set_xlabel("x")
        ax_im.grid(True, alpha=0.3)

    # Légende unique
    if U_pred is not None:
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        unique_labels = dict(zip(labels, lines))
        fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=12)

    # --- 5. Sauvegarde ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot sauvegardé (3 colonnes) : {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)