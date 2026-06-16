import argparse
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
    interp_complex_field,
    is_periodic_case,
    make_spatial_grid,
    prepare_reference_trajectory,
)
from src.models.cgl_local_physics_deeponet_amp_phase import CGL_LocalPhysics_DeepONet_AmpPhase
from src.plot.postprocess_single_case import (
    benchmark_inference,
    plot_error_heatmap,
    plot_l2_curve,
    plot_snapshots,
    save_comparison_gif,
    save_rel_l2_csv,
    write_rollout_summary,
)
from src.utils.solver_cgl import get_ground_truth_CGL


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def find_latest_run_dir(base_results_dir):
    all_runs = [path for path in os.listdir(base_results_dir) if path.startswith("run_")]
    if not all_runs:
        return None
    full_paths = [os.path.join(base_results_dir, path) for path in all_runs]
    full_paths = [path for path in full_paths if os.path.isdir(path)]
    if not full_paths:
        return None
    return max(full_paths, key=os.path.getmtime)


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


def save_yaml_copy(run_dir, cfg_dict, source_config_path):
    payload = dict(cfg_dict)
    payload["_meta"] = {"source_config": source_config_path}
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_timing_summary(run_dir, start_iso, start_perf, status, history_rows):
    end_dt = datetime.now()
    elapsed_seconds = max(0.0, time.perf_counter() - start_perf)
    timing_path = os.path.join(run_dir, "timing_summary.txt")
    with open(timing_path, "w", encoding="utf-8") as handle:
        handle.write(f"status={status}\n")
        handle.write(f"start_time={start_iso}\n")
        handle.write(f"end_time={end_dt.isoformat(timespec='seconds')}\n")
        handle.write(f"total_wall_seconds={elapsed_seconds:.6f}\n")
        handle.write(f"total_wall_hours={elapsed_seconds / 3600.0:.6f}\n")
        handle.write(f"history_count={len(history_rows)}\n")
        for row in history_rows:
            prefix = f"pass_{int(row['pass_idx']):02d}_stage_{int(row['stage_idx']):02d}"
            handle.write(f"{prefix}_label={row['stage_label']}\n")
            handle.write(f"{prefix}_proxy={float(row['best_proxy']):.10e}\n")
            handle.write(f"{prefix}_wall_seconds={float(row['wall_seconds']):.6f}\n")

    csv_path = os.path.join(run_dir, "timing_history.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("pass_idx,stage_idx,stage_label,best_proxy,wall_seconds\n")
        for row in history_rows:
            handle.write(
                f"{int(row['pass_idx'])},{int(row['stage_idx'])},{row['stage_label']},"
                f"{float(row['best_proxy']):.10e},{float(row['wall_seconds']):.6f}\n"
            )


def fixed_case_setup(cfg_dict):
    params = build_single_case_params(cfg_dict)
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    local_cfg = cfg_dict["local_physics"]
    periodic = is_periodic_case(params)
    x_sensor = make_spatial_grid(x_min, x_max, int(local_cfg["sensor_nx"]), periodic)
    u0_sensor = analytical_initial_field(params, x_sensor)
    return params, periodic, x_sensor, u0_sensor


def load_time_blocks(cfg_dict):
    return [tuple(map(float, block)) for block in cfg_dict["multinet"]["time_blocks"]]


def stage_name(stage_idx, t_start, t_end):
    return f"stage_{stage_idx:02d}_t{t_start:.2f}_{t_end:.2f}"


def state_sampling_weights(n_states, latest_bias):
    if n_states == 1:
        return np.ones(1, dtype=np.float64)
    uniform = np.ones(n_states, dtype=np.float64) / float(n_states)
    decay = np.exp(np.linspace(-2.5, 0.0, n_states))
    decay /= decay.sum()
    weights = latest_bias * decay + (1.0 - latest_bias) * uniform
    return weights / weights.sum()


def sample_spatial_points(sensor_state, x_sensor, periodic, n_points, focus_mix):
    n_points = int(n_points)
    focus_mix = float(focus_mix)
    dx = float(x_sensor[1] - x_sensor[0])
    x_min = float(x_sensor[0])
    x_max = x_min + dx * len(x_sensor) if periodic else float(x_sensor[-1])

    n_focus = int(round(focus_mix * n_points))
    n_uniform = n_points - n_focus
    amp = np.abs(sensor_state).astype(np.float64)
    probs = amp + 1.0e-4 * max(float(np.max(amp)), 1.0) + 1.0e-8
    probs /= probs.sum()

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
    branch_vec = build_branch_features(cfg_dict, sensor_state, window_dt, params)
    branch_tensor = torch.tensor(branch_vec[None, :], dtype=torch.float32, device=device)
    return branch_tensor.repeat(int(repeat_count), 1)


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


def select_bank_entries_for_block(state_bank, t_start, t_end):
    selected = [entry for entry in state_bank if float(t_start) - 1.0e-10 <= entry["t_start"] < float(t_end) - 1.0e-10]
    if not selected and state_bank:
        closest = min(state_bank, key=lambda row: abs(row["t_start"] - float(t_start)))
        selected = [closest]
    return selected


def compute_overlap_interval(block_a, block_b):
    start = max(float(block_a[0]), float(block_b[0]))
    end = min(float(block_a[1]), float(block_b[1]))
    if end <= start + 1.0e-10:
        return None
    return (start, end)


def build_pde_batch_from_entries(entries, cfg_dict, params, x_sensor, periodic, window_dt, device):
    local_cfg = cfg_dict["local_physics"]
    train_cfg = cfg_dict["training"]
    batch_states = min(len(entries), int(train_cfg["batch_states"]))
    points_per_state = int(train_cfg["pde_points_per_state"])
    tau_eps = float(local_cfg.get("tau_epsilon", 1.0e-4))
    focus_mix = float(local_cfg.get("focus_mix", 0.8))
    weights = state_sampling_weights(len(entries), float(train_cfg.get("latest_bias", 0.7)))
    chosen_idx = np.random.choice(len(entries), size=batch_states, replace=True, p=weights)

    branch_rows = []
    coords_rows = []
    for idx in chosen_idx:
        entry = entries[int(idx)]
        x_vals = sample_spatial_points(entry["sensor"], x_sensor, periodic, points_per_state, focus_mix)
        tau_vals = np.random.uniform(tau_eps, window_dt, size=points_per_state).astype(np.float32)
        branch_rows.append(build_branch_tensor(cfg_dict, entry["sensor"], params, window_dt, points_per_state, device))
        coords_rows.append(np.stack([x_vals, tau_vals], axis=1))

    branch = torch.cat(branch_rows, dim=0)
    coords = torch.tensor(np.concatenate(coords_rows, axis=0), dtype=torch.float32, device=device).requires_grad_(True)
    return branch, coords


def build_bc_batch_from_entries(entries, cfg_dict, params, window_dt, device):
    train_cfg = cfg_dict["training"]
    batch_states = min(len(entries), int(train_cfg["bc_states"]))
    points_per_state = int(train_cfg["bc_points_per_state"])
    weights = state_sampling_weights(len(entries), float(train_cfg.get("latest_bias", 0.7)))
    chosen_idx = np.random.choice(len(entries), size=batch_states, replace=True, p=weights)

    branch_rows = []
    tau_rows = []
    for idx in chosen_idx:
        entry = entries[int(idx)]
        tau_vals = np.random.uniform(0.0, window_dt, size=points_per_state).astype(np.float32)
        branch_rows.append(build_branch_tensor(cfg_dict, entry["sensor"], params, window_dt, points_per_state, device))
        tau_rows.append(tau_vals[:, None])

    branch = torch.cat(branch_rows, dim=0)
    tau = np.concatenate(tau_rows, axis=0).astype(np.float32)
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
    return branch, coords_left, coords_right


def build_ic_batch_from_entries(entries, cfg_dict, params, x_sensor, periodic, window_dt, device):
    train_cfg = cfg_dict["training"]
    batch_states = min(len(entries), int(train_cfg["ic_states"]))
    points_per_state = int(train_cfg["ic_points_per_state"])
    focus_mix = float(cfg_dict["local_physics"].get("focus_mix", 0.8))
    weights = state_sampling_weights(len(entries), float(train_cfg.get("latest_bias", 0.7)))
    chosen_idx = np.random.choice(len(entries), size=batch_states, replace=True, p=weights)

    branch_rows = []
    coords_rows = []
    target_rows = []
    for idx in chosen_idx:
        entry = entries[int(idx)]
        x_vals = sample_spatial_points(entry["sensor"], x_sensor, periodic, points_per_state, focus_mix)
        targets = interp_complex_field(x_sensor, entry["sensor"], x_vals, periodic)
        branch_rows.append(build_branch_tensor(cfg_dict, entry["sensor"], params, window_dt, points_per_state, device))
        coords_rows.append(np.stack([x_vals, np.zeros(points_per_state, dtype=np.float32)], axis=1))
        target_rows.append(targets)

    branch = torch.cat(branch_rows, dim=0)
    coords = torch.tensor(np.concatenate(coords_rows, axis=0), dtype=torch.float32, device=device)
    targets = np.concatenate(target_rows, axis=0).astype(np.complex64)
    return branch, coords, {
        "re": torch.tensor(np.real(targets)[:, None], dtype=torch.float32, device=device),
        "im": torch.tensor(np.imag(targets)[:, None], dtype=torch.float32, device=device),
    }


def compute_overlap_consistency_loss(student_model, teacher_models, overlap_entries, cfg_dict, params, x_sensor, periodic, window_dt, device):
    if not teacher_models or not overlap_entries:
        return torch.zeros((), dtype=torch.float32, device=device)

    train_cfg = cfg_dict["training"]
    n_states = min(len(overlap_entries), int(train_cfg["overlap_states"]))
    points_per_state = int(train_cfg["overlap_points_per_state"])
    focus_mix = float(cfg_dict["local_physics"].get("focus_mix", 0.8))
    tau_eps = float(cfg_dict["local_physics"].get("tau_epsilon", 1.0e-4))
    weights = state_sampling_weights(len(overlap_entries), float(train_cfg.get("latest_bias", 0.7)))
    chosen_idx = np.random.choice(len(overlap_entries), size=n_states, replace=True, p=weights)

    branch_rows = []
    coords_rows = []
    for idx in chosen_idx:
        entry = overlap_entries[int(idx)]
        x_vals = sample_spatial_points(entry["sensor"], x_sensor, periodic, points_per_state, focus_mix)
        tau_vals = np.random.uniform(tau_eps, window_dt, size=points_per_state).astype(np.float32)
        branch_rows.append(build_branch_tensor(cfg_dict, entry["sensor"], params, window_dt, points_per_state, device))
        coords_rows.append(np.stack([x_vals, tau_vals], axis=1))

    branch = torch.cat(branch_rows, dim=0)
    coords = torch.tensor(np.concatenate(coords_rows, axis=0), dtype=torch.float32, device=device)
    student_re, student_im = student_model(branch, coords)

    losses = []
    for teacher_model in teacher_models:
        with torch.no_grad():
            teacher_re, teacher_im = teacher_model(branch, coords)
        diff_sq = (student_re - teacher_re) ** 2 + (student_im - teacher_im) ** 2
        ref_sq = teacher_re ** 2 + teacher_im ** 2
        losses.append(torch.mean(diff_sq / (ref_sq + 1.0e-6)))
    return torch.stack(losses).mean()


def compute_proxy_components(model, block_entries, overlap_entries, teacher_models, cfg_dict, params, x_sensor, periodic, window_dt, device):
    loss_weights = cfg_dict["training"]["loss_weights"]
    branch_pde, coords_pde = build_pde_batch_from_entries(block_entries, cfg_dict, params, x_sensor, periodic, window_dt, device)
    res_re, res_im = local_pde_residual(model, branch_pde, coords_pde, params)
    loss_pde = torch.mean(res_re ** 2 + res_im ** 2)

    branch_bc, coords_left, coords_right = build_bc_batch_from_entries(block_entries, cfg_dict, params, window_dt, device)
    pred_re_left, pred_im_left = model(branch_bc, coords_left)
    pred_re_right, pred_im_right = model(branch_bc, coords_right)
    grad_re_left = torch.autograd.grad(pred_re_left.sum(), coords_left, create_graph=True)[0][:, 0:1]
    grad_re_right = torch.autograd.grad(pred_re_right.sum(), coords_right, create_graph=True)[0][:, 0:1]
    grad_im_left = torch.autograd.grad(pred_im_left.sum(), coords_left, create_graph=True)[0][:, 0:1]
    grad_im_right = torch.autograd.grad(pred_im_right.sum(), coords_right, create_graph=True)[0][:, 0:1]
    loss_bc = torch.mean(
        (pred_re_left - pred_re_right) ** 2
        + (pred_im_left - pred_im_right) ** 2
        + (grad_re_left - grad_re_right) ** 2
        + (grad_im_left - grad_im_right) ** 2
    )

    ic_weight = float(loss_weights.get("ic", 0.0))
    if ic_weight > 0.0:
        branch_ic, coords_ic, targets_ic = build_ic_batch_from_entries(block_entries, cfg_dict, params, x_sensor, periodic, window_dt, device)
        pred_re_ic, pred_im_ic = model(branch_ic, coords_ic)
        loss_ic = torch.mean((pred_re_ic - targets_ic["re"]) ** 2 + (pred_im_ic - targets_ic["im"]) ** 2)
    else:
        loss_ic = torch.zeros((), dtype=torch.float32, device=device)

    overlap_weight = float(loss_weights.get("overlap", 0.0))
    if overlap_weight > 0.0:
        loss_overlap = compute_overlap_consistency_loss(
            model,
            teacher_models,
            overlap_entries,
            cfg_dict,
            params,
            x_sensor,
            periodic,
            window_dt,
            device,
        )
    else:
        loss_overlap = torch.zeros((), dtype=torch.float32, device=device)

    total = (
        float(loss_weights["pde"]) * loss_pde
        + float(loss_weights["bc"]) * loss_bc
        + ic_weight * loss_ic
        + overlap_weight * loss_overlap
    )
    return total, {
        "pde": loss_pde,
        "bc": loss_bc,
        "ic": loss_ic,
        "overlap": loss_overlap,
    }


def compute_proxy_loss(model, block_entries, overlap_entries, teacher_models, cfg_dict, params, x_sensor, periodic, window_dt, device):
    eval_batches = int(cfg_dict["training"].get("proxy_eval_batches", 2))
    values = []
    components = {"pde": [], "bc": [], "ic": [], "overlap": []}
    model.eval()
    for _ in range(eval_batches):
        with torch.enable_grad():
            total, losses = compute_proxy_components(
                model,
                block_entries,
                overlap_entries,
                teacher_models,
                cfg_dict,
                params,
                x_sensor,
                periodic,
                window_dt,
                device,
            )
        values.append(float(total.detach().item()))
        for key in components:
            components[key].append(float(losses[key].detach().item()))
    return float(np.mean(values)), {key: float(np.mean(vals)) for key, vals in components.items()}


def save_stage_checkpoint(model, optimizer, iteration, best_proxy, stage_dir, name):
    ckpt_dir = os.path.join(stage_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": int(iteration),
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
    return int(ckpt.get("iteration", 0)), float(ckpt.get("best_proxy", float("inf")))


def load_model_state_only(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)


def stage_models_file(run_dir):
    return os.path.join(run_dir, "run_state.json")


def save_run_state(run_dir, next_pass_idx, next_stage_idx, history_rows):
    payload = {
        "next_pass_idx": int(next_pass_idx),
        "next_stage_idx": int(next_stage_idx),
        "history_rows": list(history_rows),
    }
    with open(stage_models_file(run_dir), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_run_state(run_dir):
    path = stage_models_file(run_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def active_stage_indices(time_blocks, t_current):
    active = []
    for idx, (t_start, t_end) in enumerate(time_blocks):
        if float(t_start) - 1.0e-10 <= float(t_current) < float(t_end) - 1.0e-10:
            active.append(idx)
    if not active:
        if float(t_current) <= float(time_blocks[0][0]):
            return [0]
        return [len(time_blocks) - 1]
    return active


def blend_weights_for_time(time_blocks, active_indices, t_current):
    if len(active_indices) == 1:
        return {active_indices[0]: 1.0}
    if len(active_indices) == 2:
        left_idx, right_idx = active_indices
        overlap = compute_overlap_interval(time_blocks[left_idx], time_blocks[right_idx])
        if overlap is None:
            return {left_idx: 0.5, right_idx: 0.5}
        overlap_start, overlap_end = overlap
        alpha = (float(t_current) - overlap_start) / max(overlap_end - overlap_start, 1.0e-12)
        alpha = min(max(alpha, 0.0), 1.0)
        return {left_idx: 1.0 - alpha, right_idx: alpha}
    weight = 1.0 / float(len(active_indices))
    return {idx: weight for idx in active_indices}


def predict_local_field(model, cfg_dict, params, sensor_state, x_query, local_times, window_dt, device):
    if len(local_times) == 0:
        return np.zeros((len(x_query), 0), dtype=np.complex64)
    branch_vec = build_branch_features(cfg_dict, sensor_state, window_dt, params)
    total = len(x_query) * len(local_times)
    coords_np = np.stack(
        [
            np.tile(x_query, len(local_times)),
            np.repeat(np.asarray(local_times, dtype=np.float32), len(x_query)),
        ],
        axis=1,
    ).astype(np.float32)
    chunk_size = int(cfg_dict["evaluation"].get("prediction_chunk_size", 32768))
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            branch = torch.tensor(branch_vec[None, :], dtype=torch.float32, device=device).repeat(stop - start, 1)
            coords = torch.tensor(coords_np[start:stop], dtype=torch.float32, device=device)
            pred_re, pred_im = model(branch, coords)
            outputs.append((pred_re[:, 0].cpu().numpy() + 1j * pred_im[:, 0].cpu().numpy()).astype(np.complex64))
    values = np.concatenate(outputs, axis=0)
    return values.reshape(len(local_times), len(x_query)).T


def predict_blended_window(models, time_blocks, cfg_dict, params, sensor_state, x_query, local_times, window_dt, t_current, device):
    active = active_stage_indices(time_blocks, t_current)
    weights = blend_weights_for_time(time_blocks, active, t_current)
    pred = np.zeros((len(x_query), len(local_times)), dtype=np.complex64)
    for idx in active:
        pred += float(weights[idx]) * predict_local_field(models[idx], cfg_dict, params, sensor_state, x_query, local_times, window_dt, device)
    return pred


def build_state_bank(models, time_blocks, cfg_dict, params, initial_sensor, x_sensor, device):
    t_max = float(cfg_dict["physics"]["t_max"])
    window_dt = float(cfg_dict["local_physics"]["window_dt"])
    entries = [{"t_start": 0.0, "sensor": initial_sensor.astype(np.complex64)}]
    current_sensor = initial_sensor.astype(np.complex64)
    current_t = 0.0

    while current_t + window_dt <= t_max + 1.0e-10:
        next_sensor_grid = predict_blended_window(
            models,
            time_blocks,
            cfg_dict,
            params,
            current_sensor,
            x_sensor,
            np.asarray([window_dt], dtype=np.float32),
            window_dt,
            current_t,
            device,
        )[:, 0]
        current_t = round(current_t + window_dt, 10)
        if current_t < t_max - 1.0e-10:
            entries.append({"t_start": current_t, "sensor": next_sensor_grid.astype(np.complex64)})
        current_sensor = next_sensor_grid.astype(np.complex64)
    return entries


def save_state_bank(path, state_bank):
    np.savez(
        path,
        times=np.asarray([float(entry["t_start"]) for entry in state_bank], dtype=np.float32),
        sensor_re=np.stack([np.real(entry["sensor"]).astype(np.float32) for entry in state_bank], axis=0),
        sensor_im=np.stack([np.imag(entry["sensor"]).astype(np.float32) for entry in state_bank], axis=0),
    )


def load_state_bank(path):
    data = np.load(path)
    times = np.asarray(data["times"], dtype=np.float32)
    sensor_re = np.asarray(data["sensor_re"], dtype=np.float32)
    sensor_im = np.asarray(data["sensor_im"], dtype=np.float32)
    state_bank = []
    for idx in range(len(times)):
        state_bank.append(
            {
                "t_start": float(times[idx]),
                "sensor": (sensor_re[idx] + 1j * sensor_im[idx]).astype(np.complex64),
            }
        )
    return state_bank


def build_overlap_entries(state_bank, time_blocks, stage_idx):
    neighbors = []
    overlap_entries = []
    if stage_idx > 0:
        overlap_left = compute_overlap_interval(time_blocks[stage_idx], time_blocks[stage_idx - 1])
        if overlap_left is not None:
            left_entries = select_bank_entries_for_block(state_bank, overlap_left[0], overlap_left[1])
            overlap_entries.extend(left_entries)
            neighbors.append(stage_idx - 1)
    if stage_idx < len(time_blocks) - 1:
        overlap_right = compute_overlap_interval(time_blocks[stage_idx], time_blocks[stage_idx + 1])
        if overlap_right is not None:
            right_entries = select_bank_entries_for_block(state_bank, overlap_right[0], overlap_right[1])
            overlap_entries.extend(right_entries)
            neighbors.append(stage_idx + 1)
    unique = {}
    for entry in overlap_entries:
        unique[round(entry["t_start"], 8)] = entry
    return list(unique.values()), sorted(set(neighbors))


def train_one_stage(model, optimizer, block_entries, overlap_entries, teacher_models, cfg_dict, params, x_sensor, periodic, stage_dir, device):
    train_cfg = cfg_dict["training"]
    window_dt = float(cfg_dict["local_physics"]["window_dt"])
    max_iters = int(train_cfg["stage_num_iters"])
    log_every = int(train_cfg["log_every"])
    eval_every = int(train_cfg["eval_every"])
    snapshot_every = int(train_cfg["snapshot_every"])
    grad_clip = float(train_cfg["grad_clip"])
    early_cfg = train_cfg.get("early_stop", {})

    start_iter, best_proxy = load_stage_checkpoint_if_available(model, optimizer, stage_dir, device)
    patience_count = 0
    stage_start_perf = time.perf_counter()
    best_proxy_components = {"pde": float("nan"), "bc": float("nan"), "ic": float("nan"), "overlap": float("nan")}
    iteration = start_iter

    print(
        f"🔁 Stage={os.path.basename(stage_dir)} | resume_iter={start_iter} | "
        f"bank_block={len(block_entries)} | bank_overlap={len(overlap_entries)}"
    )

    for iteration in range(start_iter + 1, max_iters + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, losses = compute_proxy_components(
            model,
            block_entries,
            overlap_entries,
            teacher_models,
            cfg_dict,
            params,
            x_sensor,
            periodic,
            window_dt,
            device,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if iteration % log_every == 0 or iteration == 1:
            print(
                f"[{os.path.basename(stage_dir)} | iter {iteration}] total={float(total.item()):.3e} "
                f"| pde={float(losses['pde'].item()):.3e} | bc={float(losses['bc'].item()):.3e} "
                f"| ic={float(losses['ic'].item()):.3e} | overlap={float(losses['overlap'].item()):.3e}"
            )

        if iteration % eval_every == 0 or iteration == max_iters:
            proxy, proxy_components = compute_proxy_loss(
                model,
                block_entries,
                overlap_entries,
                teacher_models,
                cfg_dict,
                params,
                x_sensor,
                periodic,
                window_dt,
                device,
            )
            print(
                f"    📏 proxy={proxy:.3e} | pde={proxy_components['pde']:.3e} | bc={proxy_components['bc']:.3e} "
                f"| ic={proxy_components['ic']:.3e} | overlap={proxy_components['overlap']:.3e}"
            )
            save_stage_checkpoint(model, optimizer, iteration, best_proxy, stage_dir, "model_latest.pth")
            improved = proxy < best_proxy * (1.0 - float(early_cfg.get("min_delta", 0.01))) or not np.isfinite(best_proxy)
            if improved:
                best_proxy = proxy
                best_proxy_components = dict(proxy_components)
                save_stage_checkpoint(model, optimizer, iteration, best_proxy, stage_dir, "model_best.pth")
                patience_count = 0
                print(f"    ✅ Nouveau meilleur proxy : {best_proxy:.3e}")
            else:
                patience_count += 1

            if (
                bool(early_cfg.get("enabled", True))
                and iteration >= int(early_cfg.get("min_iters", 1200))
                and patience_count >= int(early_cfg.get("patience_evals", 4))
            ):
                print("    ⏹️ Early stop.")
                break

        if iteration % snapshot_every == 0:
            save_stage_checkpoint(model, optimizer, iteration, best_proxy, stage_dir, f"ckpt_iter_{iteration:06d}.pth")

    best_path = os.path.join(stage_dir, "checkpoints", "model_best.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"], strict=True)

    save_stage_checkpoint(model, optimizer, iteration, best_proxy, stage_dir, "model_final.pth")
    with open(os.path.join(stage_dir, "stage_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_proxy": float(best_proxy),
                "best_proxy_components": best_proxy_components,
                "completed_iters": int(iteration),
                "block_entries": int(len(block_entries)),
                "overlap_entries": int(len(overlap_entries)),
            },
            handle,
            indent=2,
        )
    return {
        "best_proxy": float(best_proxy),
        "wall_seconds": max(0.0, time.perf_counter() - stage_start_perf),
    }


def load_or_init_models(cfg_dict, run_dir, device):
    time_blocks = load_time_blocks(cfg_dict)
    models = []
    bootstrap_path = cfg_dict.get("bootstrap", {}).get("mononet_checkpoint")
    bootstrap_path_resolved = None
    if bootstrap_path:
        bootstrap_path_resolved = bootstrap_path if os.path.isabs(bootstrap_path) else os.path.join(PROJECT_DIR, bootstrap_path)
        if not os.path.exists(bootstrap_path_resolved):
            raise FileNotFoundError(f"bootstrap.mononet_checkpoint introuvable: {bootstrap_path_resolved}")

    for stage_idx, (t_start, t_end) in enumerate(time_blocks):
        model = CGL_LocalPhysics_DeepONet_AmpPhase(cfg_dict).to(device)
        stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
        latest_ckpt = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
        final_ckpt = os.path.join(stage_dir, "checkpoints", "model_final.pth")
        if os.path.exists(latest_ckpt):
            load_model_state_only(model, latest_ckpt, device)
        elif os.path.exists(final_ckpt):
            load_model_state_only(model, final_ckpt, device)
        elif bootstrap_path_resolved is not None:
            load_model_state_only(model, bootstrap_path_resolved, device)
        elif stage_idx > 0 and bool(cfg_dict["multinet"].get("warm_start_interstage", True)):
            model.load_state_dict(models[-1].state_dict(), strict=True)
        else:
            print(f"⚠️ Stage {stage_idx:02d} initialise aleatoirement.")
        models.append(model)
    return models


def save_stage_manifest(run_dir, history_rows):
    path = os.path.join(run_dir, "history_manifest.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("pass_idx,stage_idx,stage_label,best_proxy,wall_seconds\n")
        for row in history_rows:
            handle.write(
                f"{int(row['pass_idx'])},{int(row['stage_idx'])},{row['stage_label']},"
                f"{float(row['best_proxy']):.10e},{float(row['wall_seconds']):.6f}\n"
            )


def relative_l2_curve(u_pred, u_true):
    rel_l2 = np.zeros(u_true.shape[1], dtype=np.float64)
    for idx in range(u_true.shape[1]):
        denom = np.linalg.norm(u_true[:, idx]) + 1.0e-12
        rel_l2[idx] = np.linalg.norm(u_pred[:, idx] - u_true[:, idx]) / denom
    return rel_l2


def rollout_multinet(models, time_blocks, cfg_dict, params, initial_sensor, x_sensor, x_eval, t_eval, device):
    window_dt = float(cfg_dict["local_physics"]["window_dt"])
    u_pred = np.zeros((len(x_eval), len(t_eval)), dtype=np.complex64)
    current_sensor = initial_sensor.astype(np.complex64)
    current_t = 0.0

    while current_t < float(cfg_dict["physics"]["t_max"]) - 1.0e-10:
        step_end = min(current_t + window_dt, float(cfg_dict["physics"]["t_max"]))
        if step_end >= float(cfg_dict["physics"]["t_max"]) - 1.0e-10:
            mask = (t_eval >= current_t - 1.0e-10) & (t_eval <= step_end + 1.0e-10)
        else:
            mask = (t_eval >= current_t - 1.0e-10) & (t_eval < step_end - 1.0e-10)
        local_times = t_eval[mask] - float(current_t)
        if len(local_times) > 0:
            u_pred[:, mask] = predict_blended_window(
                models,
                time_blocks,
                cfg_dict,
                params,
                current_sensor,
                x_eval,
                local_times,
                window_dt,
                current_t,
                device,
            )
        next_sensor = predict_blended_window(
            models,
            time_blocks,
            cfg_dict,
            params,
            current_sensor,
            x_sensor,
            np.asarray([window_dt], dtype=np.float32),
            window_dt,
            current_t,
            device,
        )[:, 0]
        current_sensor = next_sensor.astype(np.complex64)
        current_t = round(current_t + window_dt, 10)

    return u_pred


def evaluate_and_save(models, time_blocks, cfg_dict, params, periodic, x_sensor, initial_sensor, run_dir, device, label):
    reference = prepare_reference_trajectory(cfg_dict, nx_override=int(cfg_dict["evaluation"].get("solver_nx", 256)))
    x_eval = reference["x"]
    t_eval = reference["t"]
    u_true = reference["u"]
    u_pred = rollout_multinet(models, time_blocks, cfg_dict, params, initial_sensor, x_sensor, x_eval, t_eval, device)
    rel_l2 = relative_l2_curve(u_pred, u_true)

    eval_dir = os.path.join(run_dir, label)
    os.makedirs(eval_dir, exist_ok=True)
    save_rel_l2_csv(os.path.join(eval_dir, "rollout_metrics.csv"), t_eval, rel_l2)
    plot_l2_curve(
        t_eval,
        rel_l2,
        "CGL local multireseau physics-only : erreur relative",
        os.path.join(eval_dir, "rollout_rel_l2.png"),
        stage_markers=[block[1] for block in time_blocks[:-1]],
    )
    plot_error_heatmap(
        x_eval,
        t_eval,
        u_true,
        u_pred,
        "CGL local multireseau physics-only : heatmap erreur",
        os.path.join(eval_dir, "error_heatmap.png"),
        stage_markers=[block[1] for block in time_blocks[:-1]],
    )
    plot_snapshots(
        x_eval,
        t_eval,
        u_true,
        u_pred,
        "CGL local multireseau physics-only : snapshots",
        os.path.join(eval_dir, "snapshots.png"),
        snapshot_times=list(cfg_dict["evaluation"].get("snapshot_times", [0.1, 0.2, 0.5, 1.0])),
    )
    save_comparison_gif(
        x_eval,
        t_eval,
        u_true,
        u_pred,
        "CGL local multireseau physics-only",
        os.path.join(eval_dir, "comparison_animation.gif"),
    )
    write_rollout_summary(
        os.path.join(eval_dir, "summary.txt"),
        rel_l2,
        t_eval,
        extra={
            "n_models": len(models),
            "window_dt": float(cfg_dict["local_physics"]["window_dt"]),
            "periodic_case": periodic,
        },
    )
    params_ref = reference["params"]
    t_max = float(cfg_dict["physics"]["t_max"])
    benchmark_inference(
        "LocalMultinet",
        solver_callable=lambda: get_ground_truth_CGL(params_ref, cfg_dict["physics"]["x_domain"][0], cfg_dict["physics"]["x_domain"][1], t_max, Nx=128, Nt=None),
        model_callable=lambda: rollout_multinet(models, time_blocks, cfg_dict, params, initial_sensor, x_sensor, x_eval, t_eval, device),
        output_dir=eval_dir,
        repeats=int(cfg_dict["evaluation"].get("timing_repeats", 4)),
        warmup=1,
    )
    return float(rel_l2[-1]), float(np.max(rel_l2))


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
    save_yaml_copy(run_dir, cfg_dict, args.config)

    params, periodic, x_sensor, u0_sensor = fixed_case_setup(cfg_dict)
    time_blocks = load_time_blocks(cfg_dict)
    max_passes = int(cfg_dict["multinet"]["max_passes"])
    history_rows = []
    next_pass_idx = 0
    next_stage_idx = 0

    saved_state = load_run_state(run_dir)
    if saved_state is not None:
        history_rows = list(saved_state.get("history_rows", []))
        next_pass_idx = int(saved_state.get("next_pass_idx", 0))
        next_stage_idx = int(saved_state.get("next_stage_idx", 0))
        print(f"🔄 Resume state loaded | next_pass={next_pass_idx} | next_stage={next_stage_idx}")

    models = load_or_init_models(cfg_dict, run_dir, device)
    optimizers = [
        torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg_dict["training"]["learning_rate"]),
            weight_decay=float(cfg_dict["training"]["weight_decay"]),
        )
        for model in models
    ]
    for stage_idx, (t_start, t_end) in enumerate(time_blocks):
        stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
        os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
        load_stage_checkpoint_if_available(models[stage_idx], optimizers[stage_idx], stage_dir, device)

    start_dt = datetime.now()
    start_perf = time.perf_counter()
    print(f"📱 Device : {device}")
    print(f"📂 Run dir : {run_dir}")
    print(f"🧾 Config : {args.config}")

    try:
        for pass_idx in range(next_pass_idx, max_passes):
            pass_dir = os.path.join(run_dir, f"pass_{pass_idx:02d}")
            os.makedirs(pass_dir, exist_ok=True)

            state_bank_path = os.path.join(pass_dir, "state_bank.npz")
            if pass_idx == next_pass_idx and next_stage_idx > 0 and os.path.exists(state_bank_path):
                state_bank = load_state_bank(state_bank_path)
            else:
                state_bank = build_state_bank(models, time_blocks, cfg_dict, params, u0_sensor, x_sensor, device)
                save_state_bank(state_bank_path, state_bank)

            print(f"\n🧭 Pass {pass_idx + 1}/{max_passes} | state_bank={len(state_bank)}")

            stage_start_idx = next_stage_idx if pass_idx == next_pass_idx else 0
            for stage_idx in range(stage_start_idx, len(time_blocks)):
                t_start, t_end = time_blocks[stage_idx]
                stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
                block_entries = select_bank_entries_for_block(state_bank, t_start, t_end)
                overlap_entries, neighbor_indices = build_overlap_entries(state_bank, time_blocks, stage_idx)
                teacher_models = []
                for neighbor_idx in neighbor_indices:
                    teacher = CGL_LocalPhysics_DeepONet_AmpPhase(cfg_dict).to(device)
                    teacher.load_state_dict(models[neighbor_idx].state_dict(), strict=True)
                    teacher.eval()
                    for param in teacher.parameters():
                        param.requires_grad_(False)
                    teacher_models.append(teacher)

                print(
                    f"\n🚧 Pass {pass_idx + 1} | Stage {stage_idx + 1}/{len(time_blocks)} "
                    f"| bloc=[{t_start:.2f}, {t_end:.2f}] | bank={len(block_entries)} | overlap={len(overlap_entries)}"
                )
                metrics = train_one_stage(
                    models[stage_idx],
                    optimizers[stage_idx],
                    block_entries,
                    overlap_entries,
                    teacher_models,
                    cfg_dict,
                    params,
                    x_sensor,
                    periodic,
                    stage_dir,
                    device,
                )
                history_rows.append(
                    {
                        "pass_idx": pass_idx,
                        "stage_idx": stage_idx,
                        "stage_label": f"{t_start:.2f}_{t_end:.2f}",
                        "best_proxy": float(metrics["best_proxy"]),
                        "wall_seconds": float(metrics["wall_seconds"]),
                    }
                )
                save_stage_manifest(run_dir, history_rows)
                save_run_state(run_dir, pass_idx, stage_idx + 1, history_rows)

            next_stage_idx = 0
            save_run_state(run_dir, pass_idx + 1, 0, history_rows)
            final_rel_l2, max_rel_l2 = evaluate_and_save(
                models,
                time_blocks,
                cfg_dict,
                params,
                periodic,
                x_sensor,
                u0_sensor,
                run_dir,
                device,
                label=f"pass_{pass_idx:02d}_evaluation",
            )
            print(f"    📊 Pass {pass_idx + 1} evaluation | final_rel_l2={final_rel_l2:.3%} | max_rel_l2={max_rel_l2:.3%}")

        final_rel_l2, max_rel_l2 = evaluate_and_save(
            models,
            time_blocks,
            cfg_dict,
            params,
            periodic,
            x_sensor,
            u0_sensor,
            run_dir,
            device,
            label="evaluation",
        )
        write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "completed", history_rows)
        print(f"\n🏁 Local multireseau physics-only termine | final_rel_l2={final_rel_l2:.3%} | max_rel_l2={max_rel_l2:.3%}")
    except Exception:
        save_stage_manifest(run_dir, history_rows)
        write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "failed", history_rows)
        raise


if __name__ == "__main__":
    main()
