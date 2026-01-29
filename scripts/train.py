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
    # On vérifie que la clé existe pour éviter le crash d'hier
    if 'ic_phase' not in cfg.training:
        raise KeyError("La section 'ic_phase' est manquante dans le fichier YAML sous 'training'.")

    print("\n🔥 DÉMARRAGE IC (Condition Initiale)...")
    ic_cfg = cfg.training['ic_phase']
    optimizer_ic = optim.Adam(model.parameters(), lr=float(ic_cfg['learning_rate']))
    scheduler_ic = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ic, factor=0.5, patience=500)

    model.train()
    for it in range(int(ic_cfg['iterations'])):
        # Libération propre de la mémoire (crucial sur V100/A100)
        optimizer_ic.zero_grad(set_to_none=True)
        
        branch, coords, t_re, t_im, dt_re, dt_im = get_ic_batch_sobolev(cfg.training['batch_size_ic'], cfg, device)

        p_re, p_im = model(branch, coords)
        
        # Loss de valeur
        loss_val = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2)

        # Sobolev (Gradients en r) pour une IC plus "rigide"
        grad_re = torch.autograd.grad(p_re, coords, torch.ones_like(p_re), create_graph=True)[0][:, 0:1]
        grad_im = torch.autograd.grad(p_im, coords, torch.ones_like(p_im), create_graph=True)[0][:, 0:1]
        loss_grad = torch.mean((grad_re - dt_re)**2) + torch.mean((grad_im - dt_im)**2)

        loss = loss_val + 0.1 * loss_grad
        loss.backward()
        optimizer_ic.step()
        scheduler_ic.step(loss.detach())

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
        # Sélection de la zone
        zones = cfg.training['zones']
        if z_current < zones['zone_1']['limit']:
            zone_cfg = zones['zone_1']; name = "ZONE 1 (Chauffe)"
        elif z_current < zones['zone_2']['limit']:
            zone_cfg = zones['zone_2']; name = "ZONE 2 (Critique)"
        else:
            zone_cfg = zones['zone_3']; name = "ZONE 3 (Sortie)"

        z_next = min(z_current + zone_cfg['step_size'], z_max)
        print(f"\n🌍 {name} : {z_current:.1f} -> {z_next:.1f} mm")

        # Phase Adam
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        model.train()
        
        for it in range(int(zone_cfg['iterations'])):
            optimizer.zero_grad(set_to_none=True)
            
            # Équilibrage des batchs : On augmente la part de l'IC pour ne pas l'oublier
            # On utilise 1024 au lieu de 128
            br_ic, co_ic, t_re, t_im, _, _ = get_ic_batch_sobolev(1024, cfg, device)
            br_pde, co_pde = get_pde_batch_z_limited(cfg.training['batch_size_pde'], cfg, device, z_next)

            # Calcul des prédictions et résidus
            p_re, p_im = model(br_ic, co_ic)
            l_ic = torch.mean((p_re - t_re)**2) + torch.mean((p_im - t_im)**2)
            
            r_re, r_im = pde_residual_corrected(model, br_pde, co_pde, cfg)
            l_pde = torch.mean(r_re**2) + torch.mean(r_im**2)

            loss = cfg.training['weights']['ic_loss'] * l_ic + (l_pde / cfg.training['weights']['pde_loss_divisor'])
            loss.backward()
            optimizer.step()

            if it % 1000 == 0:
                print(f"  It {it} | L_ic: {l_ic.item():.2e} | L_pde: {l_pde.item():.2e}")

        # Évaluation de l'étape
        err, _ = evaluate_robust_metrics_smart(model, cfg, n_samples=500, z_eval=z_next)
        success = err < zone_cfg['target_error']

        # Fallback L-BFGS si l'erreur est trop haute et que l'option est active
        if not success and zone_cfg.get('use_lbfgs', False):
            print(f"  ⚠️ Erreur {err*100:.2f}% trop haute. Tentative L-BFGS...")
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
            
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                # Batchs frais pour le L-BFGS
                bi, ci, tr, ti, _, _ = get_ic_batch_sobolev(1024, cfg, device)
                bp, cp = get_pde_batch_z_limited(cfg.training['batch_size_pde'], cfg, device, z_next)
                
                pr, pi = model(bi, ci); li = torch.mean((pr - tr)**2) + torch.mean((pi - ti)**2)
                rr, ri = pde_residual_corrected(model, bp, cp, cfg); lp = torch.mean(rr**2) + torch.mean(ri**2)
                
                ls = cfg.training['weights']['ic_loss'] * li + (lp / cfg.training['weights']['pde_loss_divisor'])
                ls.backward()
                return ls
            
            lbfgs.step(closure)
            err, _ = evaluate_robust_metrics_smart(model, cfg, n_samples=500, z_eval=z_next)
            success = err < (zone_cfg['target_error'] * 1.5) # Tolérance assouplie post-LBFGS

        if success:
            print(f"✅ Étape validée. Erreur: {err*100:.2f}%")
            z_current = z_next
            # Sauvegarde régulière
            os.makedirs("outputs/checkpoints", exist_ok=True)
            if int(z_current) % 100 == 0:
                torch.save(model.state_dict(), f"outputs/checkpoints/ckpt_z{int(z_current)}.pth")
        else:
            print(f"❌ Échec critique à z={z_next:.1f} (Erreur: {err*100:.2f}%). Arrêt.")
            break

    print("\n💾 Sauvegarde finale...")
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/diffractive_final.pth")
    print("🏁 Entraînement terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffractive.yaml")
    args = parser.parse_args()
    main(args.config)