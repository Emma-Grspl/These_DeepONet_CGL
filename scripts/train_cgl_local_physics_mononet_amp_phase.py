import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.data.local_physics_single_case import (
    analytical_initial_field,
    build_branch_features,
    build_single_case_params,
    build_window_schedule,
    interp_complex_field,
    is_periodic_case,
    make_spatial_grid,
    prepare_reference_trajectory,
)
from src.models.cgl_local_physics_deeponet_amp_phase import CGL_LocalPhysics_DeepONet_AmpPhase
from src.plot import postprocess_single_case as single_case_postprocess
from src.utils.solver_cgl import get_ground_truth_CGL


def _relative_l2_curve_on_mask_fallback(u_pred, u_true, spatial_mask):
    mask = np.asarray(spatial_mask, dtype=bool)
    rel_l2 = np.zeros(u_true.shape[1], dtype=np.float64)
    for idx in range(u_true.shape[1]):
        u_true_slice = u_true[mask, idx]
        u_pred_slice = u_pred[mask, idx]
        denom = np.linalg.norm(u_true_slice) + 1.0e-12
        rel_l2[idx] = np.linalg.norm(u_pred_slice - u_true_slice) / denom
    return rel_l2


benchmark_inference = single_case_postprocess.benchmark_inference
plot_error_heatmap = single_case_postprocess.plot_error_heatmap
plot_l2_curve = single_case_postprocess.plot_l2_curve
plot_snapshots = single_case_postprocess.plot_snapshots
relative_l2_curve_on_mask = getattr(
    single_case_postprocess,
    "relative_l2_curve_on_mask",
    _relative_l2_curve_on_mask_fallback,
)
save_comparison_gif = single_case_postprocess.save_comparison_gif
save_rel_l2_csv = single_case_postprocess.save_rel_l2_csv
spatial_mask_from_bounds = single_case_postprocess.spatial_mask_from_bounds
write_rollout_summary = single_case_postprocess.write_rollout_summary


def find_latest_run_dir(base_results_dir):
    all_runs = [path for path in glob.glob(os.path.join(base_results_dir, "run_*")) if os.path.isdir(path)]
    if not all_runs:
        return None
    return max(all_runs, key=os.path.getmtime)


def build_run_dir(project_root, cfg_dict, resume_dir=None):
    configured = cfg_dict["training"]["save_dir"]
    output_root = configured if os.path.isabs(configured) else os.path.join(project_root, configured)
    os.makedirs(output_root, exist_ok=True)
    if resume_dir is not None:
        if resume_dir == "latest":
            latest = find_latest_run_dir(output_root)
            if latest is None:
                raise ValueError(f"Aucun run_* trouve dans {output_root}")
            return latest
        return resume_dir if os.path.isabs(resume_dir) else os.path.join(project_root, resume_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    run_name = f"run_{timestamp}" if not job_id else f"run_{timestamp}_{job_id}"
    return os.path.join(output_root, run_name)


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def stage_name(stage_idx, t_start, t_end):
    return f"stage_{stage_idx:02d}_t{t_start:.2f}_{t_end:.2f}"


def write_timing_summary(run_dir, start_iso, start_perf, status, stage_rows):
    end_dt = datetime.now()
    elapsed_seconds = max(0.0, time.perf_counter() - start_perf)
    timing_path = os.path.join(run_dir, "timing_summary.txt")
    with open(timing_path, "w", encoding="utf-8") as handle:
        handle.write(f"status={status}\n")
        handle.write(f"start_time={start_iso}\n")
        handle.write(f"end_time={end_dt.isoformat(timespec='seconds')}\n")
        handle.write(f"total_wall_seconds={elapsed_seconds:.6f}\n")
        handle.write(f"total_wall_hours={elapsed_seconds / 3600.0:.6f}\n")
        handle.write(f"stage_count={len(stage_rows)}\n")
        for row in stage_rows:
            prefix = f"stage_{int(row['stage_idx']):02d}"
            handle.write(f"{prefix}_label={row['stage_label']}\n")
            handle.write(f"{prefix}_t_start={float(row['t_start']):.10f}\n")
            handle.write(f"{prefix}_t_end={float(row['t_end']):.10f}\n")
            handle.write(f"{prefix}_seconds={float(row['wall_seconds']):.6f}\n")
            handle.write(f"{prefix}_best_proxy={float(row['best_proxy']):.10e}\n")

    csv_path = os.path.join(run_dir, "timing_stages.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("stage_idx,stage_label,t_start,t_end,wall_seconds,best_proxy\n")
        for row in stage_rows:
            handle.write(
                f"{int(row['stage_idx'])},{row['stage_label']},{float(row['t_start']):.10f},"
                f"{float(row['t_end']):.10f},{float(row['wall_seconds']):.6f},{float(row['best_proxy']):.10e}\n"
            )


def save_yaml_copy(run_dir, cfg_dict, source_config_path):
    payload = dict(cfg_dict)
    payload["_meta"] = {"source_config": source_config_path}
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def state_bank_to_payload(state_bank):
    return {
        "times": np.asarray([float(item["t_start"]) for item in state_bank], dtype=np.float32),
        "sensor_re": np.stack([np.real(item["sensor"]).astype(np.float32) for item in state_bank], axis=0),
        "sensor_im": np.stack([np.imag(item["sensor"]).astype(np.float32) for item in state_bank], axis=0),
    }


def state_bank_from_payload(payload):
    times = np.asarray(payload["times"], dtype=np.float32)
    sensor_re = np.asarray(payload["sensor_re"], dtype=np.float32)
    sensor_im = np.asarray(payload["sensor_im"], dtype=np.float32)
    state_bank = []
    for idx in range(len(times)):
        state_bank.append(
            {
                "t_start": float(times[idx]),
                "sensor": (sensor_re[idx] + 1j * sensor_im[idx]).astype(np.complex64),
            }
        )
    return state_bank


def save_stage_input_bank(stage_dir, state_bank):
    payload = state_bank_to_payload(state_bank)
    np.savez(os.path.join(stage_dir, "state_bank_input.npz"), **payload)


def load_stage_input_bank(stage_dir):
    path = os.path.join(stage_dir, "state_bank_input.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return state_bank_from_payload(data)


def save_run_state(run_dir, next_stage_idx, state_bank, stage_rows, model, optimizer):
    path = os.path.join(run_dir, "run_state.pth")
    payload = {
        "next_stage_idx": int(next_stage_idx),
        "state_bank": state_bank_to_payload(state_bank),
        "stage_rows": list(stage_rows),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    atomic_torch_save(payload, path)


def load_run_state(run_dir, device):
    path = os.path.join(run_dir, "run_state.pth")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device)


def save_stage_checkpoint(model, optimizer, epoch, best_proxy, stage_dir, name):
    ckpt_dir = os.path.join(stage_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_proxy": float(best_proxy),
        },
        os.path.join(ckpt_dir, name),
    )


def load_stage_checkpoint_if_available(model, optimizer, stage_dir, device):
    ckpt_path = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
    if not os.path.exists(ckpt_path):
        return 0, float("inf")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_proxy", float("inf")))


def fixed_case_setup(cfg_dict):
    params = build_single_case_params(cfg_dict)
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    local_cfg = cfg_dict["local_physics"]
    periodic = is_periodic_case(params)
    x_sensor = make_spatial_grid(x_min, x_max, int(local_cfg["sensor_nx"]), periodic)
    u0_sensor = analytical_initial_field(params, x_sensor)
    return params, periodic, x_sensor, u0_sensor


def state_sampling_weights(n_states, latest_bias):
    if n_states == 1:
        return np.ones(1, dtype=np.float64)
    uniform = np.ones(n_states, dtype=np.float64) / float(n_states)
    decay = np.exp(np.linspace(-2.5, 0.0, n_states))
    decay /= decay.sum()
    weights = latest_bias * decay + (1.0 - latest_bias) * uniform
    return weights / weights.sum()


def sanitize_sensor_state(sensor_state, fallback_state=None):
    sensor = np.asarray(sensor_state, dtype=np.complex64).copy()
    finite_mask = np.isfinite(np.real(sensor)) & np.isfinite(np.imag(sensor))
    if np.all(finite_mask):
        return sensor

    if fallback_state is not None:
        fallback = np.asarray(fallback_state, dtype=np.complex64)
        if fallback.shape == sensor.shape:
            sensor[~finite_mask] = fallback[~finite_mask]
            finite_mask = np.isfinite(np.real(sensor)) & np.isfinite(np.imag(sensor))

    if not np.all(finite_mask):
        sensor[~finite_mask] = np.complex64(0.0 + 0.0j)
    return sensor


def safe_sampling_probabilities(sensor_state):
    amp = np.abs(sensor_state).astype(np.float64)
    amp = np.where(np.isfinite(amp), np.maximum(amp, 0.0), 0.0)
    if amp.size == 0:
        raise ValueError("Etat capteur vide: impossible d'echantillonner les points spatiaux.")

    total_amp = float(np.sum(amp))
    if not np.isfinite(total_amp) or total_amp <= 0.0:
        return np.full(amp.shape[0], 1.0 / float(amp.shape[0]), dtype=np.float64)

    scale = max(float(np.max(amp)), 1.0)
    probs = amp + 1.0e-4 * scale + 1.0e-8
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(amp.shape[0], 1.0 / float(amp.shape[0]), dtype=np.float64)
    return probs / total


def sample_spatial_points(sensor_state, x_sensor, periodic, n_points, focus_mix):
    n_points = int(n_points)
    focus_mix = float(focus_mix)
    dx = float(x_sensor[1] - x_sensor[0])
    x_min = float(x_sensor[0])
    x_max = x_min + dx * len(x_sensor) if periodic else float(x_sensor[-1])

    n_focus = int(round(focus_mix * n_points))
    n_uniform = n_points - n_focus
    sensor_state = sanitize_sensor_state(sensor_state)
    probs = safe_sampling_probabilities(sensor_state)

    values = []
    if n_focus > 0:
        idx = np.random.choice(len(x_sensor), size=n_focus, replace=True, p=probs)
        x_focus = x_sensor[idx] + np.random.uniform(-0.5 * dx, 0.5 * dx, size=n_focus)
        values.append(x_focus)
    if n_uniform > 0:
        high = x_max if periodic else float(x_sensor[-1])
        x_uniform = np.random.uniform(x_min, high, size=n_uniform)
        values.append(x_uniform)

    x = np.concatenate(values, axis=0).astype(np.float32)
    if periodic:
        period = x_max - x_min
        x = ((x - x_min) % period) + x_min
    else:
        x = np.clip(x, x_min, float(x_sensor[-1]))
    np.random.shuffle(x)
    return x


def build_branch_tensor(cfg_dict, sensor_state, params, window_dt, repeat_count, device):
    sensor_state = sanitize_sensor_state(sensor_state)
    branch_vec = build_branch_features(cfg_dict, sensor_state, window_dt, params)
    branch_tensor = torch.tensor(branch_vec[None, :], dtype=torch.float32, device=device)
    return branch_tensor.repeat(int(repeat_count), 1)


def build_pde_batch(state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device):
    local_cfg = cfg_dict["local_physics"]
    train_cfg = cfg_dict["training"]
    batch_states = int(train_cfg["batch_states"])
    points_per_state = int(train_cfg["pde_points_per_state"])
    tau_eps = float(local_cfg.get("tau_epsilon", 1.0e-4))
    focus_mix = float(local_cfg.get("focus_mix", 0.8))

    weights = state_sampling_weights(len(state_bank), float(train_cfg.get("latest_bias", 0.7)))
    selected_idx = np.random.choice(len(state_bank), size=batch_states, replace=True, p=weights)
    branch_rows = []
    coords_rows = []
    for idx in selected_idx:
        entry = state_bank[int(idx)]
        sensor_state = sanitize_sensor_state(entry["sensor"])
        x_vals = sample_spatial_points(sensor_state, x_sensor, periodic, points_per_state, focus_mix)
        tau_vals = np.random.uniform(tau_eps, window_dt, size=points_per_state).astype(np.float32)
        branch_rows.append(build_branch_tensor(cfg_dict, sensor_state, params, window_dt, points_per_state, device))
        coords_rows.append(np.stack([x_vals, tau_vals], axis=1))

    branch = torch.cat(branch_rows, dim=0)
    coords = torch.tensor(np.concatenate(coords_rows, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
    return branch, coords


def build_bc_batch(state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device):
    train_cfg = cfg_dict["training"]
    batch_states = int(train_cfg["bc_states"])
    points_per_state = int(train_cfg["bc_points_per_state"])
    weights = state_sampling_weights(len(state_bank), float(train_cfg.get("latest_bias", 0.7)))
    selected_idx = np.random.choice(len(state_bank), size=batch_states, replace=True, p=weights)

    branch_rows = []
    tau_rows = []
    left_targets = []
    right_targets = []
    for idx in selected_idx:
        entry = state_bank[int(idx)]
        sensor_state = sanitize_sensor_state(entry["sensor"])
        tau_vals = np.random.uniform(0.0, window_dt, size=points_per_state).astype(np.float32)
        branch_rows.append(build_branch_tensor(cfg_dict, sensor_state, params, window_dt, points_per_state, device))
        tau_rows.append(tau_vals[:, None])
        left_targets.append(np.repeat(sensor_state[0:1], points_per_state).astype(np.complex64))
        right_targets.append(np.repeat(sensor_state[-1:], points_per_state).astype(np.complex64))

    branch = torch.cat(branch_rows, dim=0)
    tau = np.concatenate(tau_rows, axis=0).astype(np.float32)
    left_target = np.concatenate(left_targets, axis=0).astype(np.complex64)
    right_target = np.concatenate(right_targets, axis=0).astype(np.complex64)

    x_left = float(cfg_dict["physics"]["x_domain"][0])
    x_right = float(cfg_dict["physics"]["x_domain"][1])
    coords_left = torch.tensor(
        np.concatenate([np.full_like(tau, x_left), tau], axis=1),
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)
    coords_right = torch.tensor(
        np.concatenate([np.full_like(tau, x_right), tau], axis=1),
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)
    targets = {
        "left_re": torch.tensor(np.real(left_target)[:, None], dtype=torch.float32, device=device),
        "left_im": torch.tensor(np.imag(left_target)[:, None], dtype=torch.float32, device=device),
        "right_re": torch.tensor(np.real(right_target)[:, None], dtype=torch.float32, device=device),
        "right_im": torch.tensor(np.imag(right_target)[:, None], dtype=torch.float32, device=device),
    }
    return branch, coords_left, coords_right, targets


def build_ic_batch(state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device):
    train_cfg = cfg_dict["training"]
    batch_states = int(train_cfg["ic_states"])
    points_per_state = int(train_cfg["ic_points_per_state"])
    focus_mix = float(cfg_dict["local_physics"].get("focus_mix", 0.8))
    weights = state_sampling_weights(len(state_bank), float(train_cfg.get("latest_bias", 0.7)))
    selected_idx = np.random.choice(len(state_bank), size=batch_states, replace=True, p=weights)

    branch_rows = []
    coords_rows = []
    target_rows = []
    for idx in selected_idx:
        entry = state_bank[int(idx)]
        sensor_state = sanitize_sensor_state(entry["sensor"])
        x_vals = sample_spatial_points(sensor_state, x_sensor, periodic, points_per_state, focus_mix)
        targets = interp_complex_field(x_sensor, sensor_state, x_vals, periodic)
        branch_rows.append(build_branch_tensor(cfg_dict, sensor_state, params, window_dt, points_per_state, device))
        coords_rows.append(np.stack([x_vals, np.zeros(points_per_state, dtype=np.float32)], axis=1))
        target_rows.append(targets)

    branch = torch.cat(branch_rows, dim=0)
    coords = torch.tensor(np.concatenate(coords_rows, axis=0), dtype=torch.float32, device=device)
    targets = np.concatenate(target_rows, axis=0).astype(np.complex64)
    target_dict = {
        "re": torch.tensor(np.real(targets)[:, None], dtype=torch.float32, device=device),
        "im": torch.tensor(np.imag(targets)[:, None], dtype=torch.float32, device=device),
    }
    return branch, coords, target_dict


def local_pde_residual(model, branch, coords, params):
    alpha = torch.full_like(coords[:, 0:1], float(params["alpha"]))
    beta = torch.full_like(coords[:, 0:1], float(params["beta"]))
    mu = torch.full_like(coords[:, 0:1], float(params["mu"]))
    V = torch.full_like(coords[:, 0:1], float(params["V"]))

    u_re, u_im = model(branch, coords)

    grads_re = torch.autograd.grad(u_re, coords, torch.ones_like(u_re), create_graph=True)[0]
    grads_im = torch.autograd.grad(u_im, coords, torch.ones_like(u_im), create_graph=True)[0]
    du_dx_re = grads_re[:, 0:1]
    du_dt_re = grads_re[:, 1:2]
    du_dx_im = grads_im[:, 0:1]
    du_dt_im = grads_im[:, 1:2]

    grads2_re = torch.autograd.grad(du_dx_re, coords, torch.ones_like(du_dx_re), create_graph=True)[0]
    grads2_im = torch.autograd.grad(du_dx_im, coords, torch.ones_like(du_dx_im), create_graph=True)[0]
    d2u_dx2_re = grads2_re[:, 0:1]
    d2u_dx2_im = grads2_im[:, 0:1]

    diff_re = d2u_dx2_re - alpha * d2u_dx2_im
    diff_im = d2u_dx2_im + alpha * d2u_dx2_re
    lin_re = mu * u_re
    lin_im = mu * u_im

    intensity = u_re ** 2 + u_im ** 2
    nl_re = -intensity * (u_re - beta * u_im)
    nl_im = -intensity * (u_im + beta * u_re)
    adv_re = -V * du_dx_re
    adv_im = -V * du_dx_im

    res_re = du_dt_re - (diff_re + lin_re + nl_re + adv_re)
    res_im = du_dt_im - (diff_im + lin_im + nl_im + adv_im)
    return res_re, res_im


def compute_loss_components(model, state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device):
    weights = cfg_dict["training"]["loss_weights"]

    branch_pde, coords_pde = build_pde_batch(state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device)
    res_re, res_im = local_pde_residual(model, branch_pde, coords_pde, params)
    loss_pde = torch.mean(res_re ** 2 + res_im ** 2)

    branch_bc, coords_left, coords_right, bc_targets = build_bc_batch(
        state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device
    )
    left_re, left_im = model(branch_bc, coords_left)
    right_re, right_im = model(branch_bc, coords_right)

    if periodic:
        left_grad_re = torch.autograd.grad(left_re, coords_left, torch.ones_like(left_re), create_graph=True)[0][:, 0:1]
        right_grad_re = torch.autograd.grad(right_re, coords_right, torch.ones_like(right_re), create_graph=True)[0][:, 0:1]
        left_grad_im = torch.autograd.grad(left_im, coords_left, torch.ones_like(left_im), create_graph=True)[0][:, 0:1]
        right_grad_im = torch.autograd.grad(right_im, coords_right, torch.ones_like(right_im), create_graph=True)[0][:, 0:1]
        loss_bc = torch.mean(
            (left_re - right_re) ** 2
            + (left_im - right_im) ** 2
            + (left_grad_re - right_grad_re) ** 2
            + (left_grad_im - right_grad_im) ** 2
        )
    else:
        loss_bc = torch.mean(
            (left_re - bc_targets["left_re"]) ** 2
            + (left_im - bc_targets["left_im"]) ** 2
            + (right_re - bc_targets["right_re"]) ** 2
            + (right_im - bc_targets["right_im"]) ** 2
        )

    if float(weights.get("ic", 0.0)) > 0.0:
        branch_ic, coords_ic, ic_targets = build_ic_batch(state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device)
        ic_re, ic_im = model(branch_ic, coords_ic)
        loss_ic = torch.mean((ic_re - ic_targets["re"]) ** 2 + (ic_im - ic_targets["im"]) ** 2)
    else:
        loss_ic = torch.zeros((), dtype=torch.float32, device=device)

    total = (
        float(weights["pde"]) * loss_pde
        + float(weights["bc"]) * loss_bc
        + float(weights.get("ic", 0.0)) * loss_ic
    )
    return total, {
        "pde": loss_pde,
        "bc": loss_bc,
        "ic": loss_ic,
    }


def evaluate_proxy_loss(model, state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device):
    eval_cfg = cfg_dict["training"].get("proxy_eval", {})
    n_batches = int(eval_cfg.get("batches", 3))
    values = []
    components = {"pde": [], "bc": [], "ic": []}
    model.eval()
    for _ in range(n_batches):
        with torch.enable_grad():
            total, losses = compute_loss_components(model, state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device)
        values.append(float(total.detach().item()))
        for key in components:
            components[key].append(float(losses[key].detach().item()))
    means = {key: float(np.mean(vals)) for key, vals in components.items()}
    return float(np.mean(values)), means


def predict_sensor_state(model, cfg_dict, params, sensor_state, x_sensor, window_dt, device):
    sensor_state = sanitize_sensor_state(sensor_state)
    branch = build_branch_tensor(cfg_dict, sensor_state, params, window_dt, len(x_sensor), device)
    coords = torch.tensor(
        np.stack([x_sensor, np.full(len(x_sensor), float(window_dt), dtype=np.float32)], axis=1),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        pred_re, pred_im = model(branch, coords)
    predicted = (pred_re[:, 0].cpu().numpy() + 1j * pred_im[:, 0].cpu().numpy()).astype(np.complex64)
    return sanitize_sensor_state(predicted, fallback_state=sensor_state)


def train_one_stage(model, optimizer, state_bank, cfg_dict, params, x_sensor, periodic, stage_dir, stage_window, device):
    train_cfg = cfg_dict["training"]
    window_dt = float(stage_window[1] - stage_window[0])
    max_iters = int(train_cfg["stage_num_iters"])
    log_every = int(train_cfg["log_every"])
    eval_every = int(train_cfg["eval_every"])
    snapshot_every = int(train_cfg["snapshot_every"])
    grad_clip = float(train_cfg["grad_clip"])
    early_cfg = train_cfg.get("early_stop", {})

    save_stage_input_bank(stage_dir, state_bank)
    start_iter, best_proxy = load_stage_checkpoint_if_available(model, optimizer, stage_dir, device)
    patience_count = 0
    stage_start_perf = time.perf_counter()
    best_proxy_components = {"pde": float("nan"), "bc": float("nan"), "ic": float("nan")}

    print(
        f"🔁 Stage={os.path.basename(stage_dir)} | resume_iter={start_iter} | bank_size={len(state_bank)} | "
        f"window_dt={window_dt:.4f}"
    )

    for step in range(start_iter + 1, max_iters + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, losses = compute_loss_components(model, state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            print(
                f"[{os.path.basename(stage_dir)} | iter {step}] total={float(total.item()):.3e} "
                f"| pde={float(losses['pde'].item()):.3e} | bc={float(losses['bc'].item()):.3e} | ic={float(losses['ic'].item()):.3e}"
            )

        if step % eval_every == 0 or step == max_iters:
            proxy, proxy_components = evaluate_proxy_loss(model, state_bank, cfg_dict, params, x_sensor, periodic, window_dt, device)
            print(
                f"    📏 proxy={proxy:.3e} | pde={proxy_components['pde']:.3e} "
                f"| bc={proxy_components['bc']:.3e} | ic={proxy_components['ic']:.3e}"
            )
            save_stage_checkpoint(model, optimizer, step, best_proxy, stage_dir, "model_latest.pth")
            min_delta = float(early_cfg.get("min_delta", 0.01))
            improved = proxy < best_proxy * (1.0 - min_delta)
            if improved or not np.isfinite(best_proxy):
                best_proxy = proxy
                best_proxy_components = dict(proxy_components)
                save_stage_checkpoint(model, optimizer, step, best_proxy, stage_dir, "model_best.pth")
                patience_count = 0
                print(f"    ✅ Nouveau meilleur proxy : {best_proxy:.3e}")
            else:
                patience_count += 1

            if (
                bool(early_cfg.get("enabled", True))
                and step >= int(early_cfg.get("min_iters", 1500))
                and patience_count >= int(early_cfg.get("patience_evals", 5))
            ):
                print("    ⏹️ Early stop.")
                break

        if step % snapshot_every == 0:
            save_stage_checkpoint(model, optimizer, step, best_proxy, stage_dir, f"ckpt_iter_{step:06d}.pth")

    best_path = os.path.join(stage_dir, "checkpoints", "model_best.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"], strict=True)

    save_stage_checkpoint(model, optimizer, step, best_proxy, stage_dir, "model_final.pth")
    with open(os.path.join(stage_dir, "stage_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "stage_name": os.path.basename(stage_dir),
                "best_proxy": float(best_proxy),
                "best_proxy_components": best_proxy_components,
                "completed_iters": int(step),
                "bank_size": int(len(state_bank)),
            },
            handle,
            indent=2,
        )
    return {
        "wall_seconds": max(0.0, time.perf_counter() - stage_start_perf),
        "best_proxy": float(best_proxy),
    }


def predict_window_grid(model, cfg_dict, params, sensor_state, x_eval, local_times, window_dt, device):
    if len(local_times) == 0:
        return np.zeros((len(x_eval), 0), dtype=np.complex64)

    branch_vec = build_branch_features(cfg_dict, sensor_state, window_dt, params)
    total = len(x_eval) * len(local_times)
    coords = np.stack(
        [
            np.tile(x_eval, len(local_times)),
            np.repeat(np.asarray(local_times, dtype=np.float32), len(x_eval)),
        ],
        axis=1,
    ).astype(np.float32)

    chunk_size = int(cfg_dict["benchmark"].get("prediction_chunk_size", 32768))
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            branch = torch.tensor(branch_vec[None, :], dtype=torch.float32, device=device).repeat(stop - start, 1)
            coords_t = torch.tensor(coords[start:stop], dtype=torch.float32, device=device)
            pred_re, pred_im = model(branch, coords_t)
            outputs.append((pred_re[:, 0].cpu().numpy() + 1j * pred_im[:, 0].cpu().numpy()).astype(np.complex64))
    values = np.concatenate(outputs, axis=0)
    return values.reshape(len(local_times), len(x_eval)).T


def rollout_shared_model(model, cfg_dict, params, initial_sensor, x_sensor, x_eval, t_eval, device):
    window_schedule = build_window_schedule(cfg_dict["physics"]["t_max"], cfg_dict["local_physics"]["window_dt"])
    start_states = [sanitize_sensor_state(initial_sensor.astype(np.complex64))]
    for stage_idx, (t_start, t_end) in enumerate(window_schedule[:-1]):
        next_sensor = predict_sensor_state(
            model,
            cfg_dict,
            params,
            start_states[stage_idx],
            x_sensor,
            float(t_end - t_start),
            device,
        )
        start_states.append(next_sensor)

    u_pred = np.zeros((len(x_eval), len(t_eval)), dtype=np.complex64)
    for stage_idx, (t_start, t_end) in enumerate(window_schedule):
        if stage_idx < len(window_schedule) - 1:
            mask = (t_eval >= t_start - 1.0e-10) & (t_eval < t_end - 1.0e-10)
        else:
            mask = (t_eval >= t_start - 1.0e-10) & (t_eval <= t_end + 1.0e-10)
        local_times = t_eval[mask] - float(t_start)
        if len(local_times) == 0:
            continue
        u_pred[:, mask] = predict_window_grid(
            model,
            cfg_dict,
            params,
            start_states[stage_idx],
            x_eval,
            local_times,
            float(t_end - t_start),
            device,
        )
    return u_pred, start_states, window_schedule


def relative_l2_curve(u_pred, u_true):
    rel_l2 = np.zeros(u_true.shape[1], dtype=np.float64)
    for idx in range(u_true.shape[1]):
        denom = np.linalg.norm(u_true[:, idx]) + 1.0e-12
        rel_l2[idx] = np.linalg.norm(u_pred[:, idx] - u_true[:, idx]) / denom
    return rel_l2


def save_state_bank_manifest(run_dir, start_states, window_schedule):
    path = os.path.join(run_dir, "state_bank.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("stage_idx,t_start,t_end,state_file\n")
        for idx, sensor in enumerate(start_states):
            if idx < len(window_schedule):
                t_start, t_end = window_schedule[idx]
            else:
                t_start = window_schedule[-1][1]
                t_end = window_schedule[-1][1]
            state_path = os.path.join(run_dir, "state_bank", f"state_{idx:02d}.npz")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            np.savez(state_path, sensor_re=np.real(sensor).astype(np.float32), sensor_im=np.imag(sensor).astype(np.float32))
            handle.write(f"{idx},{float(t_start):.10f},{float(t_end):.10f},{state_path}\n")


def run_benchmark_and_postprocess(model, cfg_dict, params, periodic, x_sensor, initial_sensor, run_dir, device):
    bench_cfg = cfg_dict.get("benchmark", {})
    if not bool(bench_cfg.get("enabled", True)):
        return

    reference = prepare_reference_trajectory(cfg_dict, nx_override=int(bench_cfg.get("solver_nx", 256)))
    x_eval = reference["x"]
    t_eval = reference["t"]
    u_true = reference["u"]
    u_pred, start_states, window_schedule = rollout_shared_model(
        model,
        cfg_dict,
        params,
        initial_sensor,
        x_sensor,
        x_eval,
        t_eval,
        device,
    )
    rel_l2 = relative_l2_curve(u_pred, u_true)
    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    save_rel_l2_csv(os.path.join(eval_dir, "rollout_metrics.csv"), t_eval, rel_l2)
    center_mask = spatial_mask_from_bounds(x_eval, -10.0, 10.0)
    rel_l2_center = relative_l2_curve_on_mask(u_pred, u_true, center_mask)
    save_rel_l2_csv(os.path.join(eval_dir, "rollout_metrics_center_xm10_xp10.csv"), t_eval, rel_l2_center)
    plot_l2_curve(
        t_eval,
        rel_l2,
        "CGL local monoreseau physics-only : erreur relative",
        os.path.join(eval_dir, "rollout_rel_l2.png"),
        stage_markers=[t_end for _, t_end in window_schedule[:-1]],
    )
    plot_l2_curve(
        t_eval,
        rel_l2_center,
        "CGL local monoreseau physics-only : erreur relative au centre x in [-10, 10]",
        os.path.join(eval_dir, "rollout_rel_l2_center_xm10_xp10.png"),
        stage_markers=[t_end for _, t_end in window_schedule[:-1]],
    )
    plot_error_heatmap(
        x_eval,
        t_eval,
        u_true,
        u_pred,
        "CGL local monoreseau physics-only : heatmap erreur",
        os.path.join(eval_dir, "error_heatmap.png"),
        stage_markers=[t_end for _, t_end in window_schedule[:-1]],
    )
    plot_snapshots(
        x_eval,
        t_eval,
        u_true,
        u_pred,
        "CGL local monoreseau physics-only : snapshots",
        os.path.join(eval_dir, "snapshots.png"),
        snapshot_times=list(bench_cfg.get("snapshot_times", [0.1, 0.2, 0.5, 1.0])),
    )
    save_comparison_gif(
        x_eval,
        t_eval,
        u_true,
        u_pred,
        "CGL local monoreseau physics-only",
        os.path.join(eval_dir, "comparison_animation.gif"),
    )
    write_rollout_summary(
        os.path.join(eval_dir, "summary.txt"),
        rel_l2,
        t_eval,
        extra={
            "n_windows": len(window_schedule),
            "window_dt_nominal": float(cfg_dict["local_physics"]["window_dt"]),
            "periodic_case": periodic,
            "final_rel_l2_center_xm10_xp10": float(rel_l2_center[-1]),
            "max_rel_l2_center_xm10_xp10": float(np.max(rel_l2_center)),
            "mean_rel_l2_center_xm10_xp10": float(np.mean(rel_l2_center)),
        },
    )

    save_state_bank_manifest(run_dir, start_states, window_schedule)

    t_max = float(cfg_dict["physics"]["t_max"])
    benchmark_inference(
        "LocalMononet",
        solver_callable=lambda: get_ground_truth_CGL(params, cfg_dict["physics"]["x_domain"][0], cfg_dict["physics"]["x_domain"][1], t_max, Nx=128, Nt=None),
        model_callable=lambda: rollout_shared_model(
            model,
            cfg_dict,
            params,
            initial_sensor,
            x_sensor,
            x_eval,
            t_eval,
            device,
        )[0],
        output_dir=eval_dir,
        repeats=int(bench_cfg.get("timing_repeats", 4)),
        warmup=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = build_run_dir(PROJECT_DIR, cfg_dict, resume_dir=args.resume)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "slurm"), exist_ok=True)
    save_yaml_copy(run_dir, cfg_dict, args.config)

    params, periodic, x_sensor, u0_sensor = fixed_case_setup(cfg_dict)
    model = CGL_LocalPhysics_DeepONet_AmpPhase(cfg_dict).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_dict["training"]["learning_rate"]),
        weight_decay=float(cfg_dict["training"]["weight_decay"]),
    )

    state_bank = [{"t_start": 0.0, "sensor": u0_sensor}]
    stage_rows = []
    next_stage_idx = 0

    saved_state = load_run_state(run_dir, device)
    if saved_state is not None:
        model.load_state_dict(saved_state["model_state"], strict=True)
        optimizer.load_state_dict(saved_state["optimizer_state"])
        state_bank = state_bank_from_payload(saved_state["state_bank"])
        stage_rows = list(saved_state.get("stage_rows", []))
        next_stage_idx = int(saved_state.get("next_stage_idx", 0))
        print(f"🔄 Resume run state loaded | next_stage_idx={next_stage_idx} | bank_size={len(state_bank)}")

    start_dt = datetime.now()
    start_perf = time.perf_counter()
    print(f"📱 Device : {device}")
    print(f"📂 Run dir : {run_dir}")
    print(f"🧾 Config : {args.config}")
    print(f"📚 State bank size initial : {len(state_bank)}")

    try:
        windows = build_window_schedule(cfg_dict["physics"]["t_max"], cfg_dict["local_physics"]["window_dt"])
        for stage_idx in range(next_stage_idx, len(windows)):
            t_start, t_end = windows[stage_idx]
            stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
            os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)

            resumed_bank = load_stage_input_bank(stage_dir) if stage_idx == next_stage_idx else None
            active_bank = resumed_bank if resumed_bank is not None else state_bank
            print(
                f"\n🚧 Stage {stage_idx + 1}/{len(windows)} | bloc=[{t_start:.2f}, {t_end:.2f}] "
                f"| bank_size={len(active_bank)}"
            )
            metrics = train_one_stage(
                model,
                optimizer,
                active_bank,
                cfg_dict,
                params,
                x_sensor,
                periodic,
                stage_dir,
                (t_start, t_end),
                device,
            )

            latest_state = active_bank[-1]["sensor"]
            latest_state = sanitize_sensor_state(active_bank[-1]["sensor"])
            next_sensor = predict_sensor_state(model, cfg_dict, params, latest_state, x_sensor, float(t_end - t_start), device)
            np.savez(
                os.path.join(stage_dir, "state_transition.npz"),
                state_start_re=np.real(latest_state).astype(np.float32),
                state_start_im=np.imag(latest_state).astype(np.float32),
                state_end_re=np.real(next_sensor).astype(np.float32),
                state_end_im=np.imag(next_sensor).astype(np.float32),
            )

            state_bank = list(active_bank) + [{"t_start": float(t_end), "sensor": next_sensor}]
            stage_rows.append(
                {
                    "stage_idx": stage_idx,
                    "stage_label": f"{t_start:.2f}_{t_end:.2f}",
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "wall_seconds": float(metrics["wall_seconds"]),
                    "best_proxy": float(metrics["best_proxy"]),
                }
            )
            save_run_state(run_dir, stage_idx + 1, state_bank, stage_rows, model, optimizer)

        atomic_torch_save({"model_state": model.state_dict()}, os.path.join(run_dir, "model_final_local_physics_mononet_amp_phase.pth"))
        run_benchmark_and_postprocess(model, cfg_dict, params, periodic, x_sensor, u0_sensor, run_dir, device)
        write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "completed", stage_rows)
        print("\n🏁 Local monoreseau physics-only termine.")
    except Exception:
        write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "failed", stage_rows)
        raise


if __name__ == "__main__":
    main()
