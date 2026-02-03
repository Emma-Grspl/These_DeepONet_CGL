import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.solver_nlse import NLSESolver

class SimpleConfig:
    def __init__(self):
        self.physics = {
            'k': 7853.98,
            'r_max': 20.0,
            'z_max': 2000.0,
            'coefficients': {'K': 8, 'C_kerr': 1.0186, 'C_plasma': 1.0552, 'C_abs': 3.4925e-06}
        }

def analyze_physics_robust(f_val, w0_val, A_val=0.95):
    print(f"\n🔬 Analyse Labo ROBUSTE : f={f_val}mm, w0={w0_val}mm...")
    
    # 1. Simulation
    cfg = SimpleConfig()
    # On garde une haute résolution pour capturer les pics fins
    solver = NLSESolver(cfg, nr=2000, nz=2000) 
    z, r, E = solver.solve(A_val, w0_val, f_val)

    # Z cibles
    target_zs = [0, 400, 600, 800, 1000, 1500]
    
    # 2. Setup Graphique
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, len(target_zs), height_ratios=[1, 1], hspace=0.2, wspace=0.3)
    
    # Zoom spatial (cœur du filament)
    mask_r = r < 1.5 
    r_crop = r[mask_r]
    max_amp = np.max(np.abs(E[:, mask_r])) # Echelle globale

    print("📊 Génération des plots...")

    for i, z_target in enumerate(target_zs):
        idx = (np.abs(z - z_target)).argmin()
        field = E[idx, mask_r]
        
        # Données
        mod = np.abs(field)
        phase = np.angle(field) # Phase brute (-pi à +pi)

        # Filtre de propreté : on n'affiche la phase que si l'intensité est > 1% du pic LOCAL
        # Ça évite de voir le bruit de fond numérique
        mask_valid = mod > (0.01 * np.max(mod))
        
        r_phase = r_crop[mask_valid]
        phase_clean = phase[mask_valid]

        # --- Plot Haut : Amplitude ---
        ax_amp = fig.add_subplot(gs[0, i])
        ax_amp.plot(r_crop, mod, 'k-', lw=1)
        ax_amp.fill_between(r_crop, mod, color='red', alpha=0.3)
        
        ax_amp.set_title(f"z = {z[idx]:.0f} mm", fontsize=11, fontweight='bold')
        ax_amp.set_ylim(0, max_amp * 1.1)
        ax_amp.set_xlim(0, 1.5)
        ax_amp.grid(True, alpha=0.2)
        
        if i == 0: ax_amp.set_ylabel("|E| (Amplitude)")
        else: ax_amp.set_yticklabels([])

        # --- Plot Bas : Phase (WRAPPED) ---
        ax_ph = fig.add_subplot(gs[1, i])
        
        # On utilise SCATTER (points) au lieu de PLOT (ligne)
        # pour éviter les traits verticaux moches quand ça passe de -pi à +pi
        ax_ph.scatter(r_phase, phase_clean, s=1, c='blue', alpha=0.5)
        
        ax_ph.set_ylim(-3.5, 3.5) # Fixé entre -Pi et +Pi (avec petite marge)
        ax_ph.set_xlim(0, 1.5)
        ax_ph.set_xlabel("r (mm)")
        ax_ph.grid(True, alpha=0.2)
        
        # Lignes guides
        ax_ph.axhline(np.pi, color='gray', linestyle='--', linewidth=0.5)
        ax_ph.axhline(-np.pi, color='gray', linestyle='--', linewidth=0.5)
        ax_ph.axhline(0, color='k', linestyle='-', linewidth=0.5)

        if i == 0: 
            ax_ph.set_ylabel("Phase (rad)\n[-π, +π]")
            ax_ph.text(0.1, -2.5, "Plat = Plane\nRayures = Courbe", fontsize=8, color='gray')
        else:
            ax_ph.set_yticklabels([])

    plt.suptitle(f"Dynamique du Filament (Phase Enroulée) | f={f_val}mm", fontsize=16)
    
    filename = f"outputs/labo_robust_f{int(f_val)}.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"💾 Sauvegardé : {filename}")
    plt.close()

if __name__ == "__main__":
    analyze_physics_robust(f_val=500.0, w0_val=1.0)