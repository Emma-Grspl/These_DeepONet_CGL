import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import trange

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.cgl_local_operator import CGLLocalOperator
from src.utils.solver_cgl import get_ground_truth_CGL


class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return self._dict[item]

    def get(self, key, default=None):
        return self._dict.get(key, default)


def _atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp-{os.getpid()}"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def build_case_params(cfg):
    physics_cfg = cfg["physics"] if isinstance(cfg, dict) else cfg.physics
    bounds = physics_cfg["bounds"]
    eq = physics_cfg["equation_params"]
    return {
        "alpha": float(eq["alpha"][0]),
        "beta": float(eq["beta"][0]),
        "mu": float(eq["mu"][0]),
        "V": float(eq["V"][0]),
        "A": float(bounds["A"][0]),
        "w0": float(bounds["w0"][0]),
        "x0": float(bounds["x0"][0]),
        "k": float(bounds["k"][0]),
        "type": float(physics_cfg.get("initial_conditions", [0])[0]),
    }


def build_trajectory(cfg):
    physics_cfg = cfg["physics"] if isinstance(cfg, dict) else cfg.physics
    op_cfg = cfg["operator"] if isinstance(cfg, dict) else cfg.operator
    dt_local = float(op_cfg["dt_local"])
    t_max = float(physics_cfg["t_max"])
    nx = int(op_cfg["nx"])
    nt = int(round(t_max / dt_local)) + 1
    params = build_case_params(cfg)
    x_grid, t_grid, u_grid = get_ground_truth_CGL(
        params_dict=params,
        x_min=float(physics_cfg["x_domain"][0]),
        x_max=float(physics_cfg["x_domain"][1]),
        T_max=t_max,
        Nx=nx,
        Nt=nt,
    )
    x = x_grid[:, 0].astype(np.float32)
    t = t_grid[0, :].astype(np.float32)
    u = u_grid.astype(np.complex64)
    return params, x, t, u


def build_sensor_indices(nx, sensor_nx):
    if sensor_nx >= nx:
        return np.arange(nx, dtype=np.int64)
    return np.linspace(0, nx - 1, sensor_nx, dtype=np.int64)


def build_branch_vector(u_curr, params_vec, dt_local, sensor_idx):
    sensors = u_curr[sensor_idx]
    branch = np.concatenate(
        [
            sensors.real.astype(np.float32),
            sensors.imag.astype(np.float32),
            params_vec.astype(np.float32),
            np.array([dt_local], dtype=np.float32),
        ],
        axis=0,
    )
    return branch


def sample_batch(u_hist, x_grid, params_vec, dt_local, sensor_idx, batch_size, device):
    nx, nt = u_hist.shape
    time_idx = np.random.randint(0, nt - 1, size=batch_size)
    x_idx = np.random.randint(0, nx, size=batch_size)

    branch_np = np.stack(
        [build_branch_vector(u_hist[:, ti], params_vec, dt_local, sensor_idx) for ti in time_idx],
        axis=0,
    )
    x_np = x_grid[x_idx][:, None].astype(np.float32)
    target = u_hist[x_idx, time_idx + 1]
    target_re = target.real.astype(np.float32)[:, None]
    target_im = target.imag.astype(np.float32)[:, None]

    branch_t = torch.tensor(branch_np, dtype=torch.float32, device=device)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_re = torch.tensor(target_re, dtype=torch.float32, device=device)
    y_im = torch.tensor(target_im, dtype=torch.float32, device=device)
    return branch_t, x_t, y_re, y_im


def rollout_operator(model, u0, x_grid, params_vec, dt_local, sensor_idx, device, chunk_size=512):
    model.eval()
    nx = len(x_grid)
    states = [u0.astype(np.complex64).copy()]
    x_t = torch.tensor(x_grid[:, None], dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(1, len(states) + 10**9):
            break

    return states


def rollout_full(model, u_hist, x_grid, params_vec, dt_local, sensor_idx, device, chunk_size=512):
    model.eval()
    nx, nt = u_hist.shape
    pred = np.zeros_like(u_hist)
    pred[:, 0] = u_hist[:, 0]
    x_t = torch.tensor(x_grid[:, None], dtype=torch.float32, device=device)

    with torch.no_grad():
        for ti in range(nt - 1):
            branch_np = build_branch_vector(pred[:, ti], params_vec, dt_local, sensor_idx)[None, :]
            branch_t_full = torch.tensor(branch_np, dtype=torch.float32, device=device).repeat(nx, 1)

            chunks_re = []
            chunks_im = []
            for start in range(0, nx, chunk_size):
                stop = min(start + chunk_size, nx)
                out_re, out_im = model(branch_t_full[start:stop], x_t[start:stop])
                chunks_re.append(out_re.cpu().numpy().reshape(-1))
                chunks_im.append(out_im.cpu().numpy().reshape(-1))

            pred[:, ti + 1] = np.concatenate(chunks_re) + 1j * np.concatenate(chunks_im)

    return pred


def relative_l2(y_pred, y_true):
    denom = np.linalg.norm(y_true)
    if denom < 1e-12:
        denom = 1e-12
    return float(np.linalg.norm(y_pred - y_true) / denom)


def evaluate_rollout(model, u_hist, x_grid, t_grid, params_vec, dt_local, sensor_idx, eval_times, device):
    pred_hist = rollout_full(model, u_hist, x_grid, params_vec, dt_local, sensor_idx, device)
    metrics = []
    for t_eval in eval_times:
        idx = int(np.argmin(np.abs(t_grid - float(t_eval))))
        metrics.append(
            {
                "t_eval": float(t_grid[idx]),
                "l2_complex": relative_l2(pred_hist[:, idx], u_hist[:, idx]),
                "l2_amplitude": relative_l2(np.abs(pred_hist[:, idx]), np.abs(u_hist[:, idx])),
            }
        )
    return metrics, pred_hist


def save_checkpoint(model, optimizer, step, save_dir, name):
    state = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    _atomic_torch_save(state, os.path.join(save_dir, name))
    _atomic_torch_save(state, os.path.join(save_dir, "model_latest.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_local_single_case.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = ConfigObj(cfg_dict)

    training_cfg = cfg["training"]
    save_root = training_cfg.get("save_dir", "results/CGL_Local")
    os.makedirs(save_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(save_root, f"CGL_Local_Run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")
    print(f"📂 Save dir : {run_dir}")

    seed = int(training_cfg.get("seed", 1234))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    params, x_grid, t_grid, u_hist = build_trajectory(cfg)
    params_vec = np.array(
        [params[k] for k in ["alpha", "beta", "mu", "V", "A", "w0", "x0", "k", "type"]],
        dtype=np.float32,
    )
    dt_local = float(cfg["operator"]["dt_local"])
    sensor_idx = build_sensor_indices(len(x_grid), int(cfg["operator"]["sensor_nx"]))

    model = CGLLocalOperator(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training_cfg.get("lr_decay_step", 5000)),
        gamma=float(training_cfg.get("lr_decay_gamma", 0.9)),
    )

    steps = int(training_cfg["train_steps"])
    batch_size = int(training_cfg["batch_size"])
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = int(training_cfg.get("log_every", 1000))
    eval_every = int(training_cfg.get("eval_every", 2000))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 5000))
    eval_times = list(cfg["operator"]["eval_times"])
    primary_eval_time = float(cfg["operator"]["primary_eval_time"])

    best_primary = math.inf
    best_step = 0

    for step in trange(1, steps + 1, desc="CGL local"):
        model.train()
        branch_t, x_t, y_re, y_im = sample_batch(
            u_hist, x_grid, params_vec, dt_local, sensor_idx, batch_size, device
        )
        pred_re, pred_im = model(branch_t, x_t)
        loss = F.mse_loss(pred_re, y_re) + F.mse_loss(pred_im, y_im)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            print(f"📊 [It {step}] Loss: {loss.item():.2e} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if step % eval_every == 0 or step == 1:
            metrics, _ = evaluate_rollout(
                model, u_hist, x_grid, t_grid, params_vec, dt_local, sensor_idx, eval_times, device
            )
            primary = min(metrics, key=lambda row: abs(row["t_eval"] - primary_eval_time))
            metrics_path = os.path.join(run_dir, "metrics_latest.yaml")
            with open(metrics_path, "w") as f:
                yaml.safe_dump({"step": int(step), "metrics": metrics}, f, sort_keys=False)

            metric_str = " | ".join([f"t={m['t_eval']:.3f}: {100.0 * m['l2_complex']:.2f}%" for m in metrics])
            print(f"🌍 Rollout @ step {step} | {metric_str}")

            if primary["l2_complex"] < best_primary:
                best_primary = primary["l2_complex"]
                best_step = step
                save_checkpoint(model, optimizer, step, ckpt_dir, "model_best.pth")
                print(
                    f"🏆 Nouveau meilleur rollout à t={primary['t_eval']:.3f}: "
                    f"{100.0 * best_primary:.2f}% (step {best_step})"
                )

        if step % checkpoint_every == 0:
            save_checkpoint(model, optimizer, step, ckpt_dir, f"ckpt_step{step}.pth")

    save_checkpoint(model, optimizer, steps, ckpt_dir, "model_final.pth")
    print("✅ Entraînement local terminé.")


if __name__ == "__main__":
    main()
