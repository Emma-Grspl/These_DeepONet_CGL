import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# Ajout du path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.solver_nlse import NLSESolver

class SimpleConfig:
    def __init__(self):
        self.physics = {
            'k': 7853.98,
            'r_max': 20.0,
            'z_max': 2000.0,
            'coefficients': {
                'K': 8,
                'C_kerr': 1.0186,
                'C_plasma': 1.0552,
                'C_abs': 3.4925e-06
            }
        }

def run_visu():
    # 1. Configuration & Simulation
    print("⚙️  Initialisation NLSE...")
    cfg = SimpleConfig()
    solver = NLSESolver(cfg, nr=1024, nz=1500) # 1500 pas pour être fluide

    # Paramètres qui "cassent" (Filamentation assurée)
    A_val = 0.95  
    w0_val = 10.0
    f_val = 500.0

    print(f"🚀 Calcul en cours (Split-Step)...")
    z, r, E = solver.solve(A_val, w0_val, f_val)
    print("✅ Données prêtes.")

    # 2. Préparation des Données pour l'affichage
    # On limite l'affichage à r < 4 mm (inutile de voir le vide à 20mm)
    mask_r = r < 4.0
    r_crop = r[mask_r]
    E_crop = E[:, mask_r]

    # Calcul des max globaux pour fixer les axes (Anti-Jumping)
    max_mod = np.max(np.abs(E_crop))
    max_re = np.max(np.abs(np.real(E_crop)))
    
    print(f"📊 Info Max: Le pic montera jusqu'à {max_mod:.2f} (départ à {np.sqrt(A_val):.2f})")

    # 3. Setup Graphique "Style Grille"
    # Style propre
    plt.style.use('bmh') 
    
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # --- Plot 1 : Module |E| (Le Filament) ---
    ax1 = fig.add_subplot(gs[0, 0])
    line1, = ax1.plot([], [], 'r-', lw=2)
    ax1.set_xlim(0, 4.0)
    ax1.set_ylim(0, max_mod * 1.1) # Marge de 10% au dessus du max historique
    ax1.set_xlabel("Rayon r (mm)")
    ax1.set_ylabel("|E| (Amplitude)")
    ax1.set_title("Module (Intensité)")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2 : Phase (Front d'onde) ---
    ax2 = fig.add_subplot(gs[0, 1])
    line2, = ax2.plot([], [], 'g-', lw=1.5)
    ax2.set_xlim(0, 4.0)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_xlabel("Rayon r (mm)")
    ax2.set_ylabel("Rad")
    ax2.set_title("Phase (Argument)")
    ax2.grid(True, alpha=0.3)

    # --- Plot 3 : Partie Réelle (Oscillations) ---
    ax3 = fig.add_subplot(gs[0, 2])
    line3, = ax3.plot([], [], 'b-', lw=1, alpha=0.6)
    ax3.set_xlim(0, 4.0)
    ax3.set_ylim(-max_re, max_re)
    ax3.set_xlabel("Rayon r (mm)")
    ax3.set_title("Partie Réelle Re(E)")
    ax3.grid(True, alpha=0.3)

    # Titre Global animé
    title_text = fig.suptitle('', fontsize=16)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        title_text.set_text('')
        return line1, line2, line3, title_text

    # On saute des frames pour que le GIF ne fasse pas 500 Mo
    step = 5 
    frames = range(0, len(z), step)

    def update(frame_idx):
        z_curr = z[frame_idx]
        field = E_crop[frame_idx, :]

        # Données
        mod = np.abs(field)
        re = np.real(field)
        
        # Astuce Phase : On met NaN là où c'est noir pour éviter le bruit
        # Seuls les points avec au moins 5% de l'intensité max du moment sont affichés
        ph = np.angle(field)
        mask_clean = mod > (0.05 * np.max(mod)) 
        ph_clean = np.where(mask_clean, ph, np.nan)

        line1.set_data(r_crop, mod)
        line2.set_data(r_crop, ph_clean)
        line3.set_data(r_crop, re)
        
        title_text.set_text(f'Propagation Z = {z_curr:.1f} mm')
        
        return line1, line2, line3, title_text

    print("🎥 Génération du GIF...")
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=30)
    
    save_path = "outputs/nlse_grid_f_500_w0_10.gif"
    os.makedirs("outputs", exist_ok=True)
    anim.save(save_path, writer='pillow', fps=20)
    print(f"💾 Sauvegardé : {save_path}")

if __name__ == "__main__":
    run_visu()