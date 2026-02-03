import sys
import os
import argparse
import yaml
import torch
import torch.optim as optim
import numpy as np

# Ajout du dossier racine au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.Diffractive_PI_DeepOnet import PI_DeepONet_Robust
from src.data.generators import get_ic_batch_sobolev, get_pde_batch_z_limited
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

def main(config_path):
    print(f"📖 Chargement de {config_path}...")
    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    cfg = Config(yaml_data)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device : {device}")

    print("🏗️ Création du modèle NLSE...")
    model = PI_DeepONet_Robust(cfg).to(device)

    # --- 3. Entraînement IC ---
    print("\n🔥 DÉMARRAGE IC (Condition Initiale)...")
    ic_cfg = cfg.training['ic_phase']
    optimizer_ic = optim.Adam(model.parameters(), lr=float(ic_cfg['learning_rate']))
    scheduler_ic = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ic, factor=0.5, patience=500)

    model.train()
    batch_size_ic = int(cfg.training['batch_size_ic'])

    for it in range(int(ic_cfg['iterations'])):
        optimizer_ic.zero_grad(set_to_none=True)
        branch, coords, t_re, t_im, dt_re, dt_im = get_ic_batch_sobolev(batch_size_ic, cfg, device)
        p_re, p_im = model(branch, coords)

        # Loss IC (Module + Complexe)
        mod_pred = torch.sqrt(p_re**2 + p_im**2 + 1e-12)
        mod_true = torch.sqrt(t_re**2 + t_im**2 + 1e-12)
        loss_val = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2) + torch.mean((mod_pred - mod_true)**2)
        
        # Loss Gradient
        grad_re = torch.autograd.grad(p_re, coords, torch.ones_like(p_re), create_graph=True)[0][:, 0:1]
        grad_im = torch.autograd.grad(p_im, coords, torch.ones_like(p_im), create_graph=True)[0][:, 0:1]
        loss_grad = torch.mean((grad_re - dt_re)**2) + torch.mean((grad_im - dt_im)**2)

        loss = loss_val + 0.1 * loss_grad
        loss.backward()
        optimizer_ic.step()
        scheduler_ic.step(loss.detach())

        if it % 1000 == 0:
            print(f"IC It {it} | Loss: {loss.item():.2e}")

    ic_err, _ = evaluate_nlse_metrics(model, cfg, n_samples=1000, z_eval=0.0)
    print(f"✅ IC Validée avec erreur L2: {ic_err*100:.2f}%")

    # --- 4. Curriculum Loop ---
    z_current = 0.0
    z_max = cfg.physics['z_max']
    batch_size_pde = int(cfg.training['batch_size_pde'])
    
    # Safety Net
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

        # Config Training
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

                # Loss IC (Rappel)
                p_re, p_im = model(br_ic, co_ic)
                # ... (Calcul Loss IC simplifié pour lisibilité, idem haut) ...
                mod_pred = torch.sqrt(p_re**2 + p_im**2 + 1e-12)
                mod_true = torch.sqrt(t_re**2 + t_im**2 + 1e-12)
                l_ic = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2) + torch.mean((mod_pred - mod_true)**2)

                # --- LOSS PDE NLSE ---
                # C'est ici que ça change !
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

        # L-BFGS si besoin
        if not success and zone_cfg.get('use_lbfgs', False):
            print("  🚀 Sauvetage L-BFGS...")
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                bi, ci, tr, ti, _, _ = get_ic_batch_sobolev(1024, cfg, device)
                bp, cp = get_pde_batch_z_limited(batch_size_pde, cfg, device, z_next)
                pr, pi = model(bi, ci)
                # ... Loss IC ...
                m_p = torch.sqrt(pr**2+pi**2+1e-12); m_t = torch.sqrt(tr**2+ti**2+1e-12)
                li = torch.mean((pr-tr)**2) + torch.mean((pi-ti)**2) + torch.mean((m_p-m_t)**2)
                # ... Loss PDE NLSE ...
                rr, ri = pde_residual_nlse(model, bp, cp, cfg)
                lp = torch.mean(rr**2) + torch.mean(ri**2)
                loss = cfg.training['weights']['ic_loss']*li + (lp/cfg.training['weights']['pde_loss_divisor'])
                loss.backward()
                return loss
            lbfgs.step(closure)
            err, _ = evaluate_nlse_metrics(model, cfg, n_samples=500, z_eval=z_next)
            print(f"  ✅ Fin L-BFGS. Erreur : {err*100:.2f}%")
            success = True # On force le passage

        # Validation étape
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
    args = parser.parse_args()
    main(args.config)