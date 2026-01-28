import torch
import numpy as np

def evaluate_robust_metrics_smart(model, cfg, n_samples=500, z_eval=0.0, chunk_size=50):
    model.eval()
    device = next(model.parameters()).device

    total_l2 = 0.0
    total_peak_err = 0.0
    total_samples = 0
    n_chunks = int(np.ceil(n_samples / chunk_size))

    with torch.no_grad():
        for i in range(n_chunks):
            current_n = min(chunk_size, n_samples - i*chunk_size)
            if current_n == 0: break

            # Accès via dictionnaire (adaptation pour le YAML)
            b = cfg.physics['bounds']

            A = torch.rand(current_n, 1, device=device) * (b['A'][1] - b['A'][0]) + b['A'][0]
            w0 = 10 ** (torch.rand(current_n, 1, device=device) * np.log10(b['w0'][1]/b['w0'][0]) + np.log10(b['w0'][0]))
            f = torch.rand(current_n, 1, device=device) * (b['f'][1] - b['f'][0]) + b['f'][0]

            # --- VRAIE PHYSIQUE ---
            z_R = (cfg.physics['k'] * w0**2) / 2.0
            factor_geom = (1.0 - z_eval / f)
            factor_diff = (z_eval / z_R)
            w_z = w0 * torch.sqrt( factor_geom**2 + factor_diff**2 )

            # Scan Spatial
            n_pts = 100
            r_scan = torch.linspace(0, 1, n_pts, device=device).view(1, -1)
            r_phys = r_scan * 3.0 * w_z 

            r_flat = r_phys.view(-1, 1)
            z_flat = torch.ones_like(r_flat) * z_eval
            coords = torch.cat([r_flat, z_flat], dim=1)
            branch = torch.cat([A, w0, f], dim=1).repeat_interleave(n_pts, dim=0)

            p_re, p_im = model(branch, coords)
            p_mod = torch.sqrt(p_re**2 + p_im**2).view(current_n, n_pts)

            w0_ext = branch[:, 1:2].view(current_n, n_pts)
            wz_ext = w_z.repeat_interleave(n_pts, dim=0).view(current_n, n_pts)
            A_ext = branch[:, 0:1].view(current_n, n_pts)

            amp_axis = torch.sqrt(A_ext) * (w0_ext / wz_ext)
            arg = -(r_phys**2)/(wz_ext**2)
            t_mod = amp_axis * torch.exp(arg)

            # Peak Error
            p_max, _ = torch.max(p_mod, dim=1)
            t_max, _ = torch.max(t_mod, dim=1)
            batch_peak_err = torch.sum(torch.abs(p_max - t_max) / (t_max + 1e-9))

            # L2 Relative
            num = torch.sum((p_mod - t_mod)**2, dim=1)
            den = torch.sum(t_mod**2, dim=1)
            batch_l2 = torch.sum(torch.sqrt(num / (den + 1e-9)))

            total_l2 += batch_l2.item()
            total_peak_err += batch_peak_err.item()
            total_samples += current_n

    avg_l2 = total_l2 / total_samples
    avg_peak = total_peak_err / total_samples

    return avg_l2, avg_peak
