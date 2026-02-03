import sys
import os
import argparse
import yaml
import torch
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.Diffractive_PI_DeepOnet import PI_DeepONet_Robust
from src.data.generators import get_ic_batch_sobolev, get_pde_batch_z_limited
# --- IMPORT NLSE ---
from src.physics.nlse import pde_residual_nlse
from src.utils.metrics_nlse import evaluate_nlse_metrics

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, value) 
            else:
                setattr(self, key, value)
        self.__dict__.update(dictionary)

def main(config_path, start_z):
    print(f"📖 Chargement de {config_path}...")
    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    cfg = Config(yaml_data)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device : {device}")

    print("🏗️ Création du modèle NLSE...")
    model = PI_DeepONet_Robust(cfg).to(device)

    # =========================================================
    # ♻️ REPRISE 
    # =========================================================
    z_current = float(start_z)
    ckpt_path = f"outputs/checkpoints_nlse/ckpt_nlse_z{int(z_current)}.pth"
    
    print(f"\n⏩ MODE REPRISE NLSE à Z={z_current} mm")
    
    if os.path.exists(ckpt_path):
        print(f"✅ Chargement des poids depuis : {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        # Fallback : essaye le dossier checkpoints normal si jamais
        fallback_path = f"outputs/checkpoints/ckpt_nlse_z{int(z_current)}.pth"
        if os.path.exists(fallback_path):
             print(f"✅ Chargement depuis fallback : {fallback_path}")
             model.load_state_dict(torch.load(fallback_path, map_location=device))
        else:
             raise FileNotFoundError(f"❌ CRITIQUE : Checkpoint introuvable pour z={z_current} !")
    # =========================================================

    z_max = cfg.physics['z_max']
    batch_size_pde = int(cfg.training['batch_size_pde'])
    
    save_interval = 20.0
    last_save_z = z_current

    print("\n🚀 DÉMARRAGE CURRICULUM ZONES (NLSE)...")

    while z_current < z_max:
        zones = cfg.training['zones']
        
        # 1. Zone d'Approche (0 -> 250 mm)
        if z_current < zones.get('zone_1', {}).get('limit', 0):
             zone_cfg = zones['zone_1']; name = "ZONE 1 (Approche)"

        # 2. Zone de Collapse (250 -> 600 mm) <--- C'est celle-ci qu'il faut ajouter !
        elif z_current < zones.get('zone_collapse', {}).get('limit', 0):
             zone_cfg = zones['zone_collapse']; name = "ZONE 2 (Collapse 💥)"

        # 3. Zone de Filamentation (600 -> 2000 mm)
        elif z_current < zones.get('zone_filamentation', {}).get('limit', 0):
             zone_cfg = zones['zone_filamentation']; name = "ZONE 3 (Filament)"

        # 4. Sécurité (au cas où on dépasse z_max ou qu'il y a une zone finale non nommée)
        else:
             keys = list(zones.keys())
             # On prend la dernière clé du dictionnaire YAML comme fallback
             last_key = keys[-1]
             zone_cfg = zones[last_key]; name = "ZONE FINALE"
        z_next = min(z_current + zone_cfg['step_size'], z_max)
        print(f"\n🌍 {name} : {z_current:.1f} -> {z_next:.1f} mm")

        current_lr = float(zone_cfg.get('first_learning_rate', 2e-4))
        max_retries = int(zone_cfg.get('max_retries', 3))
        target_err = float(zone_cfg.get('target_error', 0.05))
        iterations = int(zone_cfg['iterations'])
        success = False

        for retry in range(max_retries):
            print(f"  🔄 Tentative {retry+1}/{max_retries} | Z={z_next:.1f} | LR={current_lr:.2e}")
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            model.train()
            curr_iters = iterations + (retry * 1000)
            
            for it in range(curr_iters):
                optimizer.zero_grad(set_to_none=True)
                br_ic, co_ic, t_re, t_im, _, _ = get_ic_batch_sobolev(1024, cfg, device)
                br_pde, co_pde = get_pde_batch_z_limited(batch_size_pde, cfg, device, z_next)

                # Loss IC
                p_re, p_im = model(br_ic, co_ic)
                mod_pred = torch.sqrt(p_re**2 + p_im**2 + 1e-12)
                mod_true = torch.sqrt(t_re**2 + t_im**2 + 1e-12)
                l_ic = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2) + torch.mean((mod_pred - mod_true)**2)

                # --- LOSS PDE NLSE ---
                r_re, r_im = pde_residual_nlse(model, br_pde, co_pde, cfg)
                l_pde = torch.mean(r_re**2) + torch.mean(r_im**2)

                loss = cfg.training['weights']['ic_loss'] * l_ic + (l_pde / cfg.training['weights']['pde_loss_divisor'])
                loss.backward()
                optimizer.step()

                if it % 1000 == 0:
                    print(f"    It {it}/{curr_iters} | Loss: {loss.item():.2e}")

            err, _ = evaluate_nlse_metrics(model, cfg, n_samples=500, z_eval=z_next)
            print(f"  📊 Audit : {err*100:.2f}% (Cible < {target_err*100}%)")

            if err < target_err:
                print("  ✅ Succès Adam !")
                success = True
                break
            else:
                if retry < max_retries - 1:
                    print("  ⚠️ Échec Adam. Division LR.")
                    current_lr /= 2.0

        if not success and zone_cfg.get('use_lbfgs', False):
            print("  🚀 Sauvetage L-BFGS...")
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                bi, ci, tr, ti, _, _ = get_ic_batch_sobolev(1024, cfg, device)
                bp, cp = get_pde_batch_z_limited(batch_size_pde, cfg, device, z_next)
                pr, pi = model(bi, ci)
                m_p = torch.sqrt(pr**2+pi**2+1e-12); m_t = torch.sqrt(tr**2+ti**2+1e-12)
                li = torch.mean((pr-tr)**2) + torch.mean((pi-ti)**2) + torch.mean((m_p-m_t)**2)
                rr, ri = pde_residual_nlse(model, bp, cp, cfg)
                lp = torch.mean(rr**2) + torch.mean(ri**2)
                loss = cfg.training['weights']['ic_loss']*li + (lp/cfg.training['weights']['pde_loss_divisor'])
                loss.backward()
                return loss
            lbfgs.step(closure)
            err, _ = evaluate_nlse_metrics(model, cfg, n_samples=500, z_eval=z_next)
            print(f"  ✅ Fin L-BFGS. Erreur : {err*100:.2f}%")
            success = True

        z_current = z_next
        if (z_current - last_save_z) >= save_interval or int(z_current) % 50 == 0:
            os.makedirs("outputs/checkpoints_nlse", exist_ok=True)
            ckpt_name = f"ckpt_nlse_z{int(z_current)}.pth"
            save_path = os.path.join("outputs/checkpoints_nlse", ckpt_name)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Sauvegarde NLSE : {save_path}")
            last_save_z = z_current

    print("\n💾 Sauvegarde finale NLSE...")
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/nlse_final.pth")
    print("🏁 Entraînement NLSE terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/nlse.yaml")
    parser.add_argument("--start_z", type=float, required=True, help="Point de reprise (ex: 500.0)")
    args = parser.parse_args()
    main(args.config, args.start_z)