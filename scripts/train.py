import sys
import os
import argparse
import yaml
import torch
import torch.optim as optim

# Ajout du dossier racine au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.Diffractive_PI_DeepOnet import PI_DeepONet_Robust
from src.data.generators import get_ic_batch_sobolev, get_pde_batch_z_limited
from src.physics.diffractive import pde_residual_corrected
from src.utils.metrics import evaluate_robust_metrics_smart

class Config:
    """Helper simple pour accéder à la config."""
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def main(config_path):
    # 1. Chargement Config
    print(f"📖 Chargement de {config_path}...")
    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    cfg = Config(yaml_data)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device : {device}")

    # 2. Modèle
    print("🏗️ Création du modèle...")
    model = PI_DeepONet_Robust(cfg).to(device)

    # 3. Entraînement IC (Condition Initiale)
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

        # --- MODIFICATION LOSS IC ---
        # 1. Loss sur le Module (Guide l'enveloppe)
        mod_pred = torch.sqrt(p_re**2 + p_im**2 + 1e-12)
        mod_true = torch.sqrt(t_re**2 + t_im**2 + 1e-12)
        loss_mod = torch.mean((mod_pred - mod_true)**2)

        # 2. Loss Classique (Complexe)
        loss_complex = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2)

        # 3. Combinaison
        loss_val = loss_complex + loss_mod
        # ----------------------------

        # Loss Gradient (Sobolev) - On garde tel quel
        grad_re = torch.autograd.grad(p_re, coords, torch.ones_like(p_re), create_graph=True)[0][:, 0:1]
        grad_im = torch.autograd.grad(p_im, coords, torch.ones_like(p_im), create_graph=True)[0][:, 0:1]
        loss_grad = torch.mean((grad_re - dt_re)**2) + torch.mean((grad_im - dt_im)**2)

        loss = loss_val + 0.1 * loss_grad
        loss.backward()
        optimizer_ic.step()
        scheduler_ic.step(loss.detach())

        if it % 500 == 0:
            print(f"IC It {it} | Loss: {loss.item():.2e} (Mod: {loss_mod.item():.2e})")

    ic_err, _ = evaluate_robust_metrics_smart(model, cfg, n_samples=1000, z_eval=0.0)
    print(f"✅ IC Validée avec erreur L2: {ic_err*100:.2f}%")

    # 4. Curriculum Loop (Les Zones)
    z_current = 0.0
    z_max = cfg.physics['z_max']
    batch_size_pde = int(cfg.training['batch_size_pde'])

    print("\n🚀 DÉMARRAGE CURRICULUM ZONES...")

    while z_current < z_max:
        # --- A. Préparation Zone ---
        zones = cfg.training['zones']
        if z_current < zones['zone_1']['limit']:
            zone_cfg = zones['zone_1']; name = "ZONE 1 (Chauffe)"
        elif z_current < zones['zone_2']['limit']:
            zone_cfg = zones['zone_2']; name = "ZONE 2 (Critique)"
        else:
            zone_cfg = zones['zone_3']; name = "ZONE 3 (Sortie)"

        z_next = min(z_current + zone_cfg['step_size'], z_max)
        print(f"\n🌍 {name} : {z_current:.1f} -> {z_next:.1f} mm")

        # --- B. Config Adam ---
        first_lr = float(zone_cfg.get('first_learning_rate', 5e-4))
        current_lr = first_lr
        max_retries = int(zone_cfg.get('max_retries', 3))
        target_err = float(zone_cfg.get('target_error', 0.03))
        iterations = int(zone_cfg['iterations'])

        success = False

        # --- C. Boucle Retries ---
        for retry in range(max_retries):
            print(f"  🔄 Tentative {retry+1}/{max_retries} | Z={z_next:.1f} | LR={current_lr:.2e}")

            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            model.train()
            base_iters = int(zone_cfg['iterations'])
            current_iterations = base_iters + (retry * 2000)
            for it in range(current_iterations):
                optimizer.zero_grad(set_to_none=True)

                br_ic, co_ic, t_re, t_im, _, _ = get_ic_batch_sobolev(1024, cfg, device)
                br_pde, co_pde = get_pde_batch_z_limited(batch_size_pde, cfg, device, z_next)

                # --- 1. Calcul l_ic AVEC MODULE ---
                p_re, p_im = model(br_ic, co_ic)
                
                # Module
                mod_pred = torch.sqrt(p_re**2 + p_im**2 + 1e-12)
                mod_true = torch.sqrt(t_re**2 + t_im**2 + 1e-12)
                l_mod = torch.mean((mod_pred - mod_true)**2)
                
                # Complexe
                l_complex = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2)
                
                # Total IC
                l_ic = l_complex + l_mod
                # ----------------------------------

                # --- 2. Calcul l_pde (Classique) ---
                r_re, r_im = pde_residual_corrected(model, br_pde, co_pde, cfg)
                l_pde = torch.mean(r_re**2) + torch.mean(r_im**2)

                # Loss Totale
                loss = cfg.training['weights']['ic_loss'] * l_ic + (l_pde / cfg.training['weights']['pde_loss_divisor'])
                loss.backward()
                optimizer.step()

                if it % 500 == 0:
                    print(f"    It {it}/{iterations} | Loss: {loss.item():.2e}")

            err, _ = evaluate_robust_metrics_smart(model, cfg, n_samples=500, z_eval=z_next)
            print(f"  📊 Audit fin tentative {retry+1}: Erreur = {err*100:.2f}% (Cible < {target_err*100}%)")

            if err < target_err:
                print("  ✅ Succès Adam ! On passe à la suite.")
                success = True
                break 
            else:
                if retry < max_retries - 1:
                    print("  ⚠️ Échec. On divise le Learning Rate par 2 et on recommence.")
                    current_lr /= 2.0
                else:
                    print("  ❌ Échec final Adam après tous les retries.")

        # --- D. Fallback L-BFGS ---
        if not success and zone_cfg.get('use_lbfgs', False):
            print(f"  🚀 Tentative de sauvetage au L-BFGS...")
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                
                # Batchs
                bi, ci, tr, ti, _, _ = get_ic_batch_sobolev(1024, cfg, device)
                bp, cp = get_pde_batch_z_limited(batch_size_pde, cfg, device, z_next)
                
                # Loss IC Modifiée
                pr, pi = model(bi, ci)
                m_pred = torch.sqrt(pr**2 + pi**2 + 1e-12)
                m_true = torch.sqrt(tr**2 + ti**2 + 1e-12)
                li = (torch.mean((pr - tr)**2) + torch.mean((pi - ti)**2)) + torch.mean((m_pred - m_true)**2)
                
                # Loss PDE
                rr, ri = pde_residual_corrected(model, bp, cp, cfg)
                lp = torch.mean(rr**2) + torch.mean(ri**2)
                
                ls = cfg.training['weights']['ic_loss'] * li + (lp / cfg.training['weights']['pde_loss_divisor'])
                ls.backward()
                return ls

            lbfgs.step(closure)
            err, _ = evaluate_robust_metrics_smart(model, cfg, n_samples=500, z_eval=z_next)

            if err < (target_err * 1.5): 
                print(f"  ✅ Sauvetage réussi ! Erreur : {err*100:.2f}%")
                success = True

        # --- E. Décision Finale pour l'étape ---
        if success:
            z_current = z_next
            os.makedirs("outputs/checkpoints", exist_ok=True)
            if int(z_current) % 50 == 0:
                torch.save(model.state_dict(), f"outputs/checkpoints/ckpt_z{int(z_current)}.pth")
        else:
            print(f"⚠️  ATTENTION : Z={z_next} non validé (Err={err*100:.2f}%). On avance quand même...")
            z_current = z_next 

    # --- F. SAUVEGARDE FINALE ---
    print("\n💾 Sauvegarde finale...")
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/diffractive_final.pth")
    print("🏁 Entraînement terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffractive.yaml")
    args = parser.parse_args()
    main(args.config)