import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Imports locaux nécessaires
from src.utils.solver_cgl import get_ground_truth_CGL

def plot_temporal_snapshots(cfg, params_dict, model=None, save_path=None, show=False):
    """
    Trace des snapshots de la solution à 5 instants différents (0, t/6, t/4, t/2, t_max).
    
    Args:
        cfg: Configuration (nécessaire pour les bounds, etc.)
        params_dict: Dictionnaire des paramètres pour ce cas précis (alpha, beta, A, etc.)
        model: (Optionnel) Le modèle DeepONet entraîné. Si None, on trace juste le solveur.
        save_path: (Optionnel) Chemin pour sauvegarder l'image.
        show: (Bool) Afficher l'image ou non.
    """
    
    # --- 1. Génération de la Vérité Terrain (Solveur) ---
    print(f"📊 Génération de la solution exacte pour params: {params_dict}")
    x_min, x_max = cfg.physics['x_domain']
    t_max = cfg.physics['t_max']
    
    # On récupère la grille complète (Nx, Nt)
    X_grid, T_grid, U_true = get_ground_truth_CGL(
        params_dict, x_min, x_max, t_max, Nx=512, Nt=1000
    )
    
    # Axes 1D uniques
    x = X_grid[:, 0]
    t = T_grid[0, :]
    
    # --- 2. Prédiction du Modèle (si fourni) ---
    U_pred = None
    if model is not None:
        device = next(model.parameters()).device
        model.eval()
        
        # Préparation des inputs
        # X_grid, T_grid aplatis
        X_flat = X_grid.flatten()
        T_flat = T_grid.flatten()
        
        # Coords [x, t]
        coords = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        
        # Branch Input (9 paramètres répétés)
        # Ordre : [alpha, beta, mu, V, A, w0, x0, k, type]
        p_vec = np.array([
            params_dict['alpha'], params_dict['beta'], params_dict['mu'], params_dict.get('V', 0.0),
            params_dict['A'], params_dict['w0'], params_dict['x0'], params_dict['k'], float(params_dict['type'])
        ])
        branch = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(coords), 1).to(device)
        
        with torch.no_grad():
            u_re, u_im = model(branch, coords)
            # Reconstitution complexe
            u_cplx = (u_re + 1j * u_im).cpu().numpy().flatten()
            
        # Reshape pour matcher la grille (Nx, Nt)
        U_pred = u_cplx.reshape(X_grid.shape)

    # --- 3. Sélection des Instants ---
    # Ratios demandés : 0, 1/6, 1/4, 1/2, 1
    ratios = [0.0, 1/6, 1/4, 1/2, 1.0]
    time_indices = []
    
    for r in ratios:
        target_t = t_max * r
        # Trouve l'index le plus proche
        idx = (np.abs(t - target_t)).argmin()
        time_indices.append(idx)

    # --- 4. Plotting ---
    # 5 lignes (temps), 2 colonnes (Module, Partie Réelle)
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    
    # Titre global avec les params
    title_str = (f"CGL Snapshots | $\\alpha$={params_dict['alpha']:.2f}, $\\beta$={params_dict['beta']:.2f}, "
                 f"$\\mu$={params_dict['mu']:.2f}, V={params_dict.get('V',0):.2f}")
    fig.suptitle(title_str, fontsize=14)

    for i, t_idx in enumerate(time_indices):
        current_t = t[t_idx]
        
        # Données Exactes
        u_true_t = U_true[:, t_idx]
        
        # -- Colonne 1 : Module |u| --
        ax_mod = axes[i, 0]
        ax_mod.plot(x, np.abs(u_true_t), 'k-', label='Exact' if i==0 else "", linewidth=1.5)
        
        if U_pred is not None:
            u_pred_t = U_pred[:, t_idx]
            ax_mod.plot(x, np.abs(u_pred_t), 'r--', label='DeepONet' if i==0 else "", linewidth=1.5)
            
        ax_mod.set_ylabel(f"t = {current_t:.2f}s\n|u|")
        if i == 0: ax_mod.set_title("Module |u|")
        if i == 4: ax_mod.set_xlabel("Position x")
        ax_mod.grid(True, alpha=0.3)

        # -- Colonne 2 : Partie Réelle Re(u) --
        ax_re = axes[i, 1]
        ax_re.plot(x, np.real(u_true_t), 'k-', linewidth=1.5)
        
        if U_pred is not None:
            ax_re.plot(x, np.real(u_pred_t), 'r--', linewidth=1.5)
            
        ax_re.set_ylabel("Re(u)")
        if i == 0: ax_re.set_title("Partie Réelle Re(u)")
        if i == 4: ax_re.set_xlabel("Position x")
        ax_re.grid(True, alpha=0.3)

    # Légende unique en haut
    if U_pred is not None:
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # On ne garde que les uniques pour éviter les doublons
        unique_labels = dict(zip(labels, lines))
        fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(0.95, 0.95))

    # --- 5. Sauvegarde ---
    if save_path:
        # Assure que le dossier parent existe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot sauvegardé : {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig) # Libère la mémoire