import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Imports locaux nécessaires
from src.utils.solver_cgl import get_ground_truth_CGL


def plot_temporal_snapshots(
    cfg,
    params_dict,
    model=None,
    save_path=None,
    show=False,
    time_ratios=None,
    x_view=None,
    classical_color="black",
    time_cmap="RdPu",
):
    """
    Trace des snapshots de la solution sur une seule figure compacte.
    Par défaut : 3 instants (0, t/2, t_max).
    Lignes : Module |u|, Partie Réelle, Partie Imaginaire.
    Chaque ligne superpose les différents temps.
    
    Args:
        cfg: Configuration (pour les bounds, x_domain, etc.)
        params_dict: Dictionnaire des paramètres (alpha, beta, A, etc.)
        model: (Optionnel) Le modèle DeepONet entraîné.
        save_path: (Optionnel) Chemin de sauvegarde.
        show: (Bool) Afficher l'image.
        time_ratios: (Optionnel) Liste des fractions de t_max à tracer.
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
    ratios = time_ratios if time_ratios is not None else [0.0, 0.5, 1.0]
    time_indices = []
    for r in ratios:
        target_t = t_max * r
        idx = (np.abs(t - target_t)).argmin()
        time_indices.append(idx)

    n_snapshots = len(time_indices)

    # --- 4. Plotting compact sur une seule figure ---
    cmap = plt.get_cmap(time_cmap)
    colors = cmap(np.linspace(0.45, 0.9, n_snapshots))
    view_x_min, view_x_max = x_view if x_view is not None else (x_min, x_max)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    plt.subplots_adjust(hspace=0.28)
    
    title_str = (f"CGL Snapshots | $\\alpha$={params_dict['alpha']:.2f}, $\\beta$={params_dict['beta']:.2f}, "
                 f"$\\mu$={params_dict['mu']:.2f}, V={params_dict.get('V',0):.2f}")
    fig.suptitle(title_str, fontsize=16, y=0.95)

    for i, t_idx in enumerate(time_indices):
        current_t = t[t_idx]
        u_true_t = U_true[:, t_idx]
        
        if U_pred is not None:
            u_pred_t = U_pred[:, t_idx]
        
        label = f"t = {current_t:.2f}"

        # --- Ligne 1 : Module |u| ---
        ax_mod = axes[0]
        ax_mod.plot(x, np.abs(u_true_t), linestyle=":", color=classical_color, alpha=0.7, linewidth=1.2)
        if U_pred is not None:
            ax_mod.plot(x, np.abs(u_pred_t), color=colors[i], label=label, linewidth=2.0)
        ax_mod.set_ylabel("|u|", fontsize=11, fontweight="bold")
        ax_mod.set_title("Snapshots superposés : module", fontsize=13, fontweight="bold")
        ax_mod.grid(True, alpha=0.3)

        # --- Ligne 2 : Partie Réelle Re(u) ---
        ax_re = axes[1]
        ax_re.plot(x, np.real(u_true_t), linestyle=":", color=classical_color, alpha=0.7, linewidth=1.2)
        if U_pred is not None:
            ax_re.plot(x, np.real(u_pred_t), color=colors[i], label=label, linewidth=2.0)
        ax_re.set_ylabel("Re(u)", fontsize=11, fontweight="bold")
        ax_re.set_title("Snapshots superposés : partie réelle", fontsize=13, fontweight="bold")
        ax_re.grid(True, alpha=0.3)

        # --- Ligne 3 : Partie Imaginaire Im(u) ---
        ax_im = axes[2]
        ax_im.plot(x, np.imag(u_true_t), linestyle=":", color=classical_color, alpha=0.7, linewidth=1.2)
        if U_pred is not None:
            ax_im.plot(x, np.imag(u_pred_t), color=colors[i], label=label, linewidth=2.0)
        ax_im.set_ylabel("Im(u)", fontsize=11, fontweight="bold")
        ax_im.set_title("Snapshots superposés : partie imaginaire", fontsize=13, fontweight="bold")
        ax_im.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")
    for ax in axes:
        ax.set_xlim(view_x_min, view_x_max)

    if U_pred is not None:
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=classical_color,
                linestyle=":",
                linewidth=1.5,
                label="Solveur classique",
            )
        ]
        handles += [
            plt.Line2D([0], [0], color=colors[i], linewidth=2.0, label=f"DeepONet, t = {t[time_indices[i]]:.2f}")
            for i in range(n_snapshots)
        ]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.96), fontsize=11)

    # --- 5. Sauvegarde ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot sauvegardé (3 colonnes) : {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
