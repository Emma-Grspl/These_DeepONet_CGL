import sys
import os
import argparse
import yaml
import torch
import torch.optim as optim

# Ajout du dossier racine au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import avec le nom exact de ton fichier
from src.models.Diffractive_PI_DeepOnet import PI_DeepONet_Robust
from src.data.generators import get_ic_batch_sobolev, get_pde_batch_z_limited
from src.physics.diffractive import pde_residual_corrected
from src.utils.metrics import evaluate_robust_metrics_smart

# --- Helper pour supporter la syntaxe cfg.physics['k'] ---
class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def main(config_path):
    # 1. Chargement Config
    print(f"📖 Chargement de {config_path}...")
    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Transformation en objet compatible avec ton code (cfg.section['key'])
    cfg = Config(yaml_data)

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device : {device}")

    # 2. Modèle
    print("🏗️ Création du modèle...")
    model = PI_DeepONet_Robust(cfg).to(device)

    # 3. Entraînement IC
    print("\n🔥 DÉMARRAGE IC (Condition Initiale)...")
    ic_iters = int(cfg.training['ic_phase']['iterations'])
    ic_lr = float(cfg.training['ic_phase']['learning_rate'])

    optimizer_ic = optim.Adam(model.parameters(), lr=ic_lr)
    scheduler_ic = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ic, factor=0.5, patience=500, verbose=True)

    model.train()
    for it in range(ic_iters):
        optimizer_ic.zero_grad()
        branch, coords, t_re, t_im, dt_re, dt_im = get_ic_batch_sobolev(cfg.training['batch_size_ic'], cfg, device)

        p_re, p_im = model(branch, coords)
        loss_val = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2)

        # Sobolev (Gradients)
        grad_re = torch.autograd.grad(p_re, coords, torch.ones_like(p_re), create_graph=True)[0][:, 0:1]
        grad_im = torch.autograd.grad(p_im, coords, torch.ones_like(p_im), create_graph=True)[0][:, 0:1]
        loss_grad = torch.mean((grad_re - dt_re)**2) + torch.mean((grad_im - dt_im)**2)

        loss = loss_val + 0.1 * loss_grad
        loss.backward()
        optimizer_ic.step()
        scheduler_ic.step(loss)

        if it % 1000 == 0:
            print(f"IC It {it} | Loss: {loss.item():.2e}")

    # Audit IC
    ic_err, _ = evaluate_robust_metrics_smart(model, cfg, n_samples=1000, z_eval=0.0)
    print(f"✅ IC Validée avec erreur L2: {ic_err*100:.2f}%")

    # 4. Curriculum Loop (Les Zones)
    z_current = 0.0
    z_max = cfg.physics['z_max']

    print("\n🚀 DÉMARRAGE CURRICULUM ZONES...")

    while z_current < z_max:
        # Détermination de la zone active via la config
        if z_current < cfg.training['zones']['zone_1']['limit']:
            zone_cfg = cfg.training['zones']['zone_1']
            name = "ZONE 1 (Chauffe)"
        elif z_current < cfg.training['zones']['zone_2']['limit']:
            zone_cfg = cfg.training['zones']['zone_2']
            name = "ZONE 2 (Critique)"
        else:
            zone_cfg = cfg.training['zones']['zone_3']
            name = "ZONE 3 (Sortie)"

        z_next = min(z_current + zone_cfg['step_size'], z_max)
        print(f"\n🌍 {name} : {z_current:.1f} -> {z_next:.1f} mm")

        # --- Appel de la logique train_step_zone ---
        success = False

        # Phase Adam
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        model.train()
        for it in range(int(zone_cfg['iterations'])):
            optimizer.zero_grad()
            # Batchs
            br_ic, co_ic, t_re, t_im, _, _ = get_ic_batch_sobolev(128, cfg, device)
            br_pde, co_pde = get_pde_batch_z_limited(cfg.training['batch_size_pde'], cfg, device, z_next)

            # Loss
            p_re, p_im = model(br_ic, co_ic)
            l_ic = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2)
            r_re, r_im = pde_residual_corrected(model, br_pde, co_pde, cfg)
            l_pde = torch.mean(r_re**2) + torch.mean(r_im**2)

            loss = cfg.training['weights']['ic_loss'] * l_ic + (l_pde / cfg.training['weights']['pde_loss_divisor'])
            loss.backward()
            optimizer.step()

        # Check
        err, _ = evaluate_robust_metrics_smart(model, cfg, n_samples=500, z_eval=z_next)
        if err < zone_cfg['target_error']:
            success = True
        else:
            # Retries logic...
            print(f"⚠️ Warning: Erreur {err*100:.2f}% > {zone_cfg['target_error']*100}%.")
            if zone_cfg['use_lbfgs']:
                print("   🚀 L-BFGS activé...")
                lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
                def closure():
                    lbfgs.zero_grad()
                    br_ic, co_ic, t_re, t_im, _, _ = get_ic_batch_sobolev(256, cfg, device)
                    br_pde, co_pde = get_pde_batch_z_limited(cfg.training['batch_size_pde'], cfg, device, z_next)
                    p_re, p_im = model(br_ic, co_ic); l_ic = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2)
                    r_re, r_im = pde_residual_corrected(model, br_pde, co_pde, cfg)
                    l_pde = torch.mean(r_re**2) + torch.mean(r_im**2)
                    ls = cfg.training['weights']['ic_loss'] * l_ic + (l_pde / cfg.training['weights']['pde_loss_divisor'])
                    ls.backward()
                    return ls
                lbfgs.step(closure)

            # Re-check final
            err, _ = evaluate_robust_metrics_smart(model, cfg, n_samples=500, z_eval=z_next)
            if err < 0.08: # Tolérance ultime
                success = True

        if success:
            print(f"✅ Étape validée. ({err*100:.2f}%)")
            z_current = z_next
            # Sauvegarde Checkpoint intermédiaire
            os.makedirs("outputs/checkpoints", exist_ok=True)
            if int(z_current) % 100 == 0:
                torch.save(model.state_dict(), f"outputs/checkpoints/ckpt_z{int(z_current)}.pth")
        else:
            print("❌ Échec critique. Arrêt.")
            break

    print("💾 Sauvegarde finale...")
    torch.save(model.state_dict(), "outputs/diffractive_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffractive.yaml")
    args = parser.parse_args()
    main(args.config)
