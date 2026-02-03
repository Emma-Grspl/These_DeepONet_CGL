import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import os

from src.utils.solver_cgl import get_ground_truth_CGL

def animate_cgl_solution(cfg, params_dict, model=None, save_path="results/animation.gif", frames=200):
    """
    Génère une animation comparant la Vérité Terrain et le Modèle (optionnel).
    3 Panneaux : Module, Réel, Imaginaire.
    
    Args:
        cfg: Configuration.
        params_dict: Paramètres physiques.
        model: (Optionnel) Modèle DeepONet.
        save_path: Chemin de sortie (.gif ou .mp4).
        frames: Nombre de frames total pour l'animation (sous-échantillonnage temporel).
    """
    
    # --- 1. Préparation des Données ---
    print(f"🎬 Préparation des données pour l'animation...")
    
    if isinstance(cfg, dict):
        x_min, x_max = cfg['physics']['x_domain']
        t_max = cfg['physics']['t_max']
    else:
        x_min, x_max = cfg.physics['x_domain']
        t_max = cfg.physics['t_max']

    # Vérité Terrain
    # On prend Nt assez grand pour avoir une belle résolution, puis on sous-échantillonnera pour la vidéo
    X_grid, T_grid, U_true = get_ground_truth_CGL(
        params_dict, x_min, x_max, t_max, Nx=512, Nt=1000
    )
    
    x = X_grid[:, 0]
    t_full = T_grid[0, :]
    
    # Prédiction Modèle (si présent)
    U_pred = None
    if model is not None:
        print("🤖 Calcul de la prédiction du modèle...")
        device = next(model.parameters()).device
        model.eval()
        
        X_flat = X_grid.flatten()
        T_flat = T_grid.flatten()
        coords = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)
        
        p_vec = np.array([
            params_dict['alpha'], params_dict['beta'], params_dict['mu'], params_dict.get('V', 0.0),
            params_dict['A'], params_dict['w0'], params_dict['x0'], params_dict['k'], float(params_dict['type'])
        ])
        branch = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).repeat(len(coords), 1).to(device)
        
        with torch.no_grad():
            u_re, u_im = model(branch, coords)
            u_cplx = (u_re + 1j * u_im).cpu().numpy().flatten()
            
        U_pred = u_cplx.reshape(X_grid.shape)

    # --- 2. Configuration du Plot ---
    fig, (ax_mod, ax_re, ax_im) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # Titre global
    title = fig.suptitle(f"t = 0.00s", fontsize=16)

    # Calcul des limites Y (Global Max pour que ça ne saute pas)
    y_max_abs = np.max(np.abs(U_true)) * 1.2
    y_max_re  = np.max(np.abs(np.real(U_true))) * 1.2
    y_max_im  = np.max(np.abs(np.imag(U_true))) * 1.2
    
    if U_pred is not None:
        y_max_abs = max(y_max_abs, np.max(np.abs(U_pred)) * 1.2)
        y_max_re  = max(y_max_re, np.max(np.abs(np.real(U_pred))) * 1.2)
        y_max_im  = max(y_max_im, np.max(np.abs(np.imag(U_pred))) * 1.2)

    # Init des axes
    ax_mod.set_xlim(x_min, x_max)
    ax_mod.set_ylim(-0.1, y_max_abs)
    ax_mod.set_ylabel("|u| (Module)", fontsize=12)
    ax_mod.grid(True, alpha=0.3)
    
    ax_re.set_xlim(x_min, x_max)
    ax_re.set_ylim(-y_max_re, y_max_re)
    ax_re.set_ylabel("Re(u)", fontsize=12)
    ax_re.grid(True, alpha=0.3)
    
    ax_im.set_xlim(x_min, x_max)
    ax_im.set_ylim(-y_max_im, y_max_im)
    ax_im.set_ylabel("Im(u)", fontsize=12)
    ax_im.set_xlabel("x", fontsize=12)
    ax_im.grid(True, alpha=0.3)

    # Init des lignes (vides au début)
    # Lignes noires pleines pour Exact, Rouges pointillées pour Pred
    line_true_mod, = ax_mod.plot([], [], 'k-', lw=2, label='Exact')
    line_pred_mod, = ax_mod.plot([], [], 'r--', lw=2, label='DeepONet')
    
    line_true_re, = ax_re.plot([], [], 'k-', lw=1.5)
    line_pred_re, = ax_re.plot([], [], 'r--', lw=1.5)
    
    line_true_im, = ax_im.plot([], [], 'k-', lw=1.5)
    line_pred_im, = ax_im.plot([], [], 'r--', lw=1.5)

    # Légende unique sur le premier graphe
    if U_pred is not None:
        ax_mod.legend(loc='upper right')
    else:
        line_true_mod.set_label('Exact')
        ax_mod.legend(loc='upper right')

    # --- 3. Fonction d'Animation ---
    # Sélection des indices temporels pour ne pas faire 1000 frames
    idx_frames = np.linspace(0, len(t_full)-1, frames).astype(int)

    def update(frame_idx):
        t_idx = idx_frames[frame_idx]
        current_t = t_full[t_idx]
        
        # Update Titre
        title.set_text(f"CGL Simulation | t = {current_t:.2f}s")
        
        # Données Exactes
        u_t = U_true[:, t_idx]
        
        line_true_mod.set_data(x, np.abs(u_t))
        line_true_re.set_data(x, np.real(u_t))
        line_true_im.set_data(x, np.imag(u_t))
        
        # Données Modèle
        if U_pred is not None:
            u_p = U_pred[:, t_idx]
            line_pred_mod.set_data(x, np.abs(u_p))
            line_pred_re.set_data(x, np.real(u_p))
            line_pred_im.set_data(x, np.imag(u_p))
            return line_true_mod, line_pred_mod, line_true_re, line_pred_re, line_true_im, line_pred_im
        
        return line_true_mod, line_true_re, line_true_im

    # --- 4. Génération et Sauvegarde ---
    print(f"🎥 Génération de l'animation ({frames} frames)...")
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    # Création du dossier si nécessaire
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Choix du writer (Pillow pour GIF est le plus sûr sans installer ffmpeg)
    if save_path.endswith('.gif'):
        anim.save(save_path, writer='pillow', fps=20)
    else:
        # Pour .mp4 il faut ffmpeg installé sur la machine
        try:
            anim.save(save_path, writer='ffmpeg', fps=30)
        except Exception as e:
            print(f"⚠️ Erreur ffmpeg (pas installé ?), fallback sur .gif")
            save_path = save_path.replace(".mp4", ".gif")
            anim.save(save_path, writer='pillow', fps=20)

    print(f"✅ Animation sauvegardée : {save_path}")
    plt.close(fig)