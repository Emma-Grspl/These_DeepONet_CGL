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

from src.data.generators import get_pde_batch_cgle_global
from src.models.cgl_deeponet_amp_phase import CGL_PI_DeepONet_AmpPhase
from src.physics.pde_cgl import pde_residual_cgle
from src.plot import postprocess_single_case as single_case_postprocess
from src.training.trainer_CGL_modern import (
    _compute_continuity_loss,
    _compute_mass_balance_loss,
    _compute_relative_pde_loss,
    _compute_weak_pde_loss,
    _get_early_stop_cfg,
    _get_physics_loss_cfg,
    run_audit,
)
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


def _spatial_mask_from_bounds_fallback(x, x_min=None, x_max=None):
    x = np.asarray(x, dtype=np.float64)
    mask = np.ones(x.shape, dtype=bool)
    if x_min is not None:
        mask &= x >= float(x_min)
    if x_max is not None:
        mask &= x <= float(x_max)
    if not np.any(mask):
        raise ValueError(f"No spatial samples found inside window [{x_min}, {x_max}].")
    return mask


def _save_rel_l2_csv_fallback(path, t_values, rel_l2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("time,rel_l2\n")
        for t_val, err in zip(t_values, rel_l2):
            handle.write(f"{float(t_val):.8f},{float(err):.10f}\n")


def _first_above_threshold(t_values, rel_l2, threshold):
    mask = np.asarray(rel_l2) > float(threshold)
    if not np.any(mask):
        return np.nan
    return float(np.asarray(t_values)[int(np.argmax(mask))])


def _write_rollout_summary_fallback(path, rel_l2, t_values, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"final_rel_l2={float(rel_l2[-1]):.10f}\n")
        handle.write(f"max_rel_l2={float(np.max(rel_l2)):.10f}\n")
        handle.write(f"mean_rel_l2={float(np.mean(rel_l2)):.10f}\n")
        handle.write(f"first_t_gt_5pct={_first_above_threshold(t_values, rel_l2, 0.05)}\n")
        if extra:
            for key, value in extra.items():
                handle.write(f"{key}={value}\n")


def _save_comparison_gif_fallback(*args, **kwargs):
    return None


benchmark_inference = getattr(single_case_postprocess, "benchmark_inference")
plot_error_heatmap = getattr(single_case_postprocess, "plot_error_heatmap")
plot_l2_curve = getattr(single_case_postprocess, "plot_l2_curve")
plot_snapshots = getattr(single_case_postprocess, "plot_snapshots")
relative_l2_curve_on_mask = getattr(
    single_case_postprocess,
    "relative_l2_curve_on_mask",
    _relative_l2_curve_on_mask_fallback,
)
save_comparison_gif = getattr(single_case_postprocess, "save_comparison_gif", _save_comparison_gif_fallback)
save_rel_l2_csv = getattr(single_case_postprocess, "save_rel_l2_csv", _save_rel_l2_csv_fallback)
spatial_mask_from_bounds = getattr(
    single_case_postprocess,
    "spatial_mask_from_bounds",
    _spatial_mask_from_bounds_fallback,
)
write_rollout_summary = getattr(
    single_case_postprocess,
    "write_rollout_summary",
    _write_rollout_summary_fallback,
)


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


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


def save_yaml_copy(run_dir, cfg_dict, source_config_path):
    payload = dict(cfg_dict)
    payload["_meta"] = {"source_config": source_config_path}
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


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
            handle.write(f"{prefix}_best_score={float(row['best_score']):.10e}\n")
            handle.write(f"{prefix}_final_score={float(row['final_score']):.10e}\n")
            if row.get("best_historical_mean") is not None:
                handle.write(f"{prefix}_best_historical_mean={float(row['best_historical_mean']):.10e}\n")
            if row.get("best_historical_max") is not None:
                handle.write(f"{prefix}_best_historical_max={float(row['best_historical_max']):.10e}\n")
            if row.get("final_historical_mean") is not None:
                handle.write(f"{prefix}_final_historical_mean={float(row['final_historical_mean']):.10e}\n")
            if row.get("final_historical_max") is not None:
                handle.write(f"{prefix}_final_historical_max={float(row['final_historical_max']):.10e}\n")
            handle.write(f"{prefix}_wall_seconds={float(row['wall_seconds']):.6f}\n")

    csv_path = os.path.join(run_dir, "timing_stages.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("stage_idx,stage_label,t_start,t_end,best_score,final_score,wall_seconds\n")
        for row in stage_rows:
            handle.write(
                f"{int(row['stage_idx'])},{row['stage_label']},{float(row['t_start']):.10f},"
                f"{float(row['t_end']):.10f},{float(row['best_score']):.10e},{float(row['final_score']):.10e},"
                f"{float(row['wall_seconds']):.6f}\n"
            )


def stage_name(stage_idx, t_start, t_end):
    return f"stage_{stage_idx:02d}_t{t_start:.2f}_{t_end:.2f}"


def fixed_case_params(cfg_dict):
    eq = cfg_dict["physics"]["equation_params"]
    bounds = cfg_dict["physics"]["bounds"]
    return {
        "alpha": float(eq["alpha"][0]),
        "beta": float(eq["beta"][0]),
        "mu": float(eq["mu"][0]),
        "V": float(eq["V"][0]),
        "A": float(bounds["A"][0]),
        "w0": float(bounds["w0"][0]),
        "x0": float(bounds["x0"][0]),
        "k": float(bounds["k"][0]),
        "type": int(cfg_dict["physics"]["initial_conditions"][0]),
    }


def prepare_reference_trajectory(cfg_dict, t_max_override=None, nx_override=None, nt_override=None):
    params = fixed_case_params(cfg_dict)
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    t_max = float(cfg_dict["physics"]["t_max"] if t_max_override is None else t_max_override)
    nx = int(cfg_dict.get("benchmark", {}).get("solver_nx", 256) if nx_override is None else nx_override)
    X, T, U = get_ground_truth_CGL(params, x_min, x_max, t_max, Nx=nx, Nt=nt_override)
    return {
        "params": params,
        "x": X[:, 0].astype(np.float32),
        "t": T[0, :].astype(np.float32),
        "u": U.astype(np.complex64),
    }


def load_time_blocks(cfg_dict):
    return [tuple(map(float, block)) for block in cfg_dict["multistage"]["time_blocks"]]


def stage_markers_from_blocks(time_blocks):
    if len(time_blocks) <= 1:
        return []
    return [float(t_end) for _, t_end in time_blocks[:-1]]


def stage_target_error(cfg_dict, stage_idx):
    stage_cfg = cfg_dict["multistage_training"]
    by_stage = stage_cfg.get("target_error_by_stage")
    if by_stage is not None:
        return float(by_stage[stage_idx])
    return float(stage_cfg.get("default_target_error", cfg_dict["training"].get("target_error_global", 0.055)))


def stage_num_iters(cfg_dict, stage_idx):
    stage_cfg = cfg_dict["multistage_training"]
    by_stage = stage_cfg.get("stage_num_iters_by_stage")
    if by_stage is not None:
        return int(by_stage[stage_idx])
    return int(stage_cfg["stage_num_iters"])


def stage_eval_cases(cfg_dict):
    return int(cfg_dict["multistage_training"].get("audit_cases", 1))


def stage_allow_relaxation(cfg_dict):
    return bool(cfg_dict["multistage_training"].get("allow_relaxation", False))


def internal_curriculum_cfg(cfg_dict):
    return dict(cfg_dict["multistage_training"].get("internal_curriculum", {}))


def internal_curriculum_enabled(cfg_dict):
    return bool(internal_curriculum_cfg(cfg_dict).get("enabled", False))


def internal_curriculum_fractions(cfg_dict):
    cur_cfg = internal_curriculum_cfg(cfg_dict)
    fractions = [float(v) for v in cur_cfg.get("fractions", [1.0])]
    fractions = sorted({min(1.0, max(1.0e-6, value)) for value in fractions})
    if not fractions or fractions[-1] < 1.0:
        fractions.append(1.0)
    return fractions


def internal_curriculum_target_scale(cfg_dict, phase_idx):
    cur_cfg = internal_curriculum_cfg(cfg_dict)
    scales = cur_cfg.get("target_scale_by_phase")
    if scales is None:
        return 1.0
    if phase_idx < len(scales):
        return float(scales[phase_idx])
    return float(scales[-1])


def stage_curriculum_state(cfg_dict, t_start, t_end, iteration, num_iters):
    if not internal_curriculum_enabled(cfg_dict):
        return float(t_end), 1.0, 0

    fractions = internal_curriculum_fractions(cfg_dict)
    phase_len = max(1, int(np.ceil(float(num_iters) / float(len(fractions)))))
    phase_idx = min((max(1, int(iteration)) - 1) // phase_len, len(fractions) - 1)
    fraction = float(fractions[phase_idx])
    active_t_end = float(t_start) + fraction * (float(t_end) - float(t_start))
    return active_t_end, fraction, int(phase_idx)


def historical_validation_cfg(cfg_dict):
    return dict(cfg_dict["multistage_training"].get("historical_validation", {}))


def historical_validation_enabled(cfg_dict):
    return bool(historical_validation_cfg(cfg_dict).get("enabled", False))


def historical_time_stride(cfg_dict):
    return max(1, int(historical_validation_cfg(cfg_dict).get("time_stride", 1)))


def historical_solver_nx(cfg_dict):
    hist_cfg = historical_validation_cfg(cfg_dict)
    if "solver_nx" in hist_cfg:
        return int(hist_cfg["solver_nx"])
    return int(cfg_dict.get("benchmark", {}).get("solver_nx", 256))


def stage_historical_target(cfg_dict, stage_idx):
    hist_cfg = historical_validation_cfg(cfg_dict)
    by_stage = hist_cfg.get("target_by_stage")
    if by_stage is not None:
        return float(by_stage[stage_idx])
    return float(hist_cfg.get("target", cfg_dict["training"].get("target_error_global", 0.055)))


def stage_historical_max_target(cfg_dict, stage_idx):
    hist_cfg = historical_validation_cfg(cfg_dict)
    by_stage = hist_cfg.get("max_target_by_stage")
    if by_stage is not None:
        return float(by_stage[stage_idx])
    value = hist_cfg.get("max_target")
    return None if value is None else float(value)


def historical_selection_metric(cfg_dict):
    hist_cfg = historical_validation_cfg(cfg_dict)
    return str(hist_cfg.get("selection_metric", "historical_mean_rel_l2"))


def historical_selection_score(cfg_dict, hist_stats):
    metric_name = historical_selection_metric(cfg_dict)
    if metric_name == "historical_max_rel_l2":
        return float(hist_stats["max_rel_l2"])
    if metric_name == "historical_final_rel_l2":
        return float(hist_stats["final_rel_l2"])
    return float(hist_stats["mean_rel_l2"])


def slice_reference_trajectory(reference, t_end, time_stride=1):
    indices = np.flatnonzero(reference["t"] <= float(t_end) + 1.0e-10)
    if len(indices) == 0:
        indices = np.asarray([0], dtype=np.int64)
    stride = max(1, int(time_stride))
    if stride > 1 and len(indices) > 2:
        indices = indices[::stride]
        last_idx = np.flatnonzero(reference["t"] <= float(t_end) + 1.0e-10)[-1]
        if indices[-1] != last_idx:
            indices = np.concatenate([indices, np.asarray([last_idx], dtype=np.int64)], axis=0)

    return {
        "params": dict(reference["params"]),
        "x": reference["x"],
        "t": reference["t"][indices].astype(np.float32),
        "u": reference["u"][:, indices].astype(np.complex64),
    }


def evaluate_historical_rollout(
    prefix_models,
    current_model,
    time_blocks,
    reference_full,
    stage_idx,
    cfg_dict,
    device,
    t_end_override=None,
):
    if reference_full is None:
        return None

    active_t_end = time_blocks[stage_idx][1] if t_end_override is None else float(t_end_override)
    ref = slice_reference_trajectory(
        reference_full,
        active_t_end,
        time_stride=historical_time_stride(cfg_dict),
    )
    models = list(prefix_models) + [current_model]
    active_blocks = time_blocks[: stage_idx + 1]

    was_training = current_model.training
    current_model.eval()
    try:
        rollout = rollout_multistage_models(models, active_blocks, ref, device)
    finally:
        if was_training:
            current_model.train()

    rel = np.asarray(rollout["rel_l2"], dtype=np.float64)
    return {
        "mean_rel_l2": float(np.mean(rel)),
        "max_rel_l2": float(np.max(rel)),
        "final_rel_l2": float(rel[-1]),
        "n_times": int(len(rel)),
        "t_end": float(active_t_end),
    }


def sample_stage_pde_batch(n_samples, cfg_dict, device, t_start, t_end):
    local_tmax = float(t_end - t_start)
    branch, coords, params_dict = get_pde_batch_cgle_global(int(n_samples), cfg_dict, device, local_tmax)
    coords = coords.clone()
    coords[:, 1:2] = coords[:, 1:2] + float(t_start)
    coords.requires_grad_(True)
    return branch, coords, params_dict


def compute_stage_loss_components(model, cfg_dict, t_start, t_end, device, teacher_model=None):
    bs_pde = int(cfg_dict["training"]["batch_size_pde"])
    loss_cfg = _get_physics_loss_cfg(cfg_dict)
    branch, coords, params_dict = sample_stage_pde_batch(bs_pde, cfg_dict, device, t_start, t_end)
    components = pde_residual_cgle(model, branch, coords, params_dict, cfg_dict, return_components=True)

    loss_pde_abs = torch.mean(components["res_re"] ** 2 + components["res_im"] ** 2)
    loss_pde_rel = _compute_relative_pde_loss(components)
    loss_pde_weak = _compute_weak_pde_loss(components, coords, cfg_dict)
    loss_pde = (
        loss_pde_abs
        + float(loss_cfg["pde_relative_weight"]) * loss_pde_rel
        + float(loss_cfg["weak_weight"]) * loss_pde_weak
    )

    idx_bc = torch.randperm(branch.size(0), device=device)[: max(1, int(branch.size(0) * 0.25))]
    branch_bc = branch[idx_bc]
    coords_bc_base = coords[idx_bc].detach().clone()
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    coords_left = coords_bc_base.clone()
    coords_right = coords_bc_base.clone()
    coords_left[:, 0] = float(x_min)
    coords_right[:, 0] = float(x_max)
    coords_all_bc = torch.cat([coords_left, coords_right], dim=0).requires_grad_(True)
    branch_all_bc = torch.cat([branch_bc, branch_bc], dim=0)

    pred_re_bc, pred_im_bc = model(branch_all_bc, coords_all_bc)
    grads_re_bc = torch.autograd.grad(pred_re_bc.sum(), coords_all_bc, create_graph=True)[0]
    grads_im_bc = torch.autograd.grad(pred_im_bc.sum(), coords_all_bc, create_graph=True)[0]
    n_bc = branch_bc.size(0)
    pred_re_left, pred_re_right = pred_re_bc[:n_bc], pred_re_bc[n_bc:]
    pred_im_left, pred_im_right = pred_im_bc[:n_bc], pred_im_bc[n_bc:]
    grad_re_left, grad_re_right = grads_re_bc[:n_bc, 0:1], grads_re_bc[n_bc:, 0:1]
    grad_im_left, grad_im_right = grads_im_bc[:n_bc, 0:1], grads_im_bc[n_bc:, 0:1]

    loss_bc = torch.mean(
        (pred_re_left - pred_re_right) ** 2
        + (pred_im_left - pred_im_right) ** 2
        + (grad_re_left - grad_re_right) ** 2
        + (grad_im_left - grad_im_right) ** 2
    )
    loss_mass = _compute_mass_balance_loss(model, cfg_dict, device, t_start, t_end, loss_cfg)
    loss_continuity = _compute_continuity_loss(model, teacher_model, cfg_dict, device, t_start, loss_cfg)
    aux_loss = loss_bc + float(loss_cfg["mass_weight"]) * loss_mass + float(loss_cfg["continuity_weight"]) * loss_continuity
    return {
        "loss_pde_abs": loss_pde_abs,
        "loss_pde_rel": loss_pde_rel,
        "loss_pde_weak": loss_pde_weak,
        "loss_pde": loss_pde,
        "loss_bc": loss_bc,
        "loss_mass": loss_mass,
        "loss_continuity": loss_continuity,
        "aux_loss": aux_loss,
    }


def save_stage_checkpoint(model, optimizer, iteration, best_score, stage_dir, name):
    ckpt_dir = os.path.join(stage_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": int(iteration),
            "best_score": float(best_score),
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
    return int(ckpt.get("iteration", 0)), float(ckpt.get("best_score", float("inf")))


def load_best_stage_model(cfg_dict, stage_dir, device):
    ckpt_path = os.path.join(stage_dir, "checkpoints", "model_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(stage_dir, "checkpoints", "model_final.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def train_one_stage(
    model,
    optimizer,
    cfg_dict,
    stage_idx,
    t_start,
    t_end,
    stage_dir,
    device,
    teacher_model=None,
    prefix_models=None,
    time_blocks=None,
    historical_reference=None,
):
    loss_cfg = _get_physics_loss_cfg(cfg_dict)
    early_cfg = _get_early_stop_cfg(cfg_dict)
    stage_cfg = cfg_dict["multistage_training"]
    hist_cfg = historical_validation_cfg(cfg_dict)
    num_iters = stage_num_iters(cfg_dict, stage_idx)
    log_every = int(stage_cfg.get("log_every", 100))
    eval_every = int(stage_cfg.get("eval_every", 1000))
    snapshot_every = int(stage_cfg.get("snapshot_every", 4000))
    grad_clip = float(stage_cfg.get("grad_clip", 1.0))
    base_lr = float(cfg_dict["time_marching"].get("learning_rate", cfg_dict["training"].get("learning_rate", 2e-4)))
    target_error = stage_target_error(cfg_dict, stage_idx)
    audit_cases = stage_eval_cases(cfg_dict)
    allow_relaxation = stage_allow_relaxation(cfg_dict)
    hist_enabled = historical_validation_enabled(cfg_dict) and historical_reference is not None and time_blocks is not None
    hist_target = stage_historical_target(cfg_dict, stage_idx) if hist_enabled else None
    hist_max_target = stage_historical_max_target(cfg_dict, stage_idx) if hist_enabled else None

    start_iter, best_score = load_stage_checkpoint_if_available(model, optimizer, stage_dir, device)
    print(
        f"🔁 Reprise stage={os.path.basename(stage_dir)} iter={start_iter} "
        f"best_score={best_score if np.isfinite(best_score) else float('nan'):.6e}"
    )

    stage_start_perf = time.perf_counter()
    ema_alpha = float(stage_cfg.get("ema_alpha", 0.999))
    w_pde = 1.0
    w_bc = 1.0
    loss_pde_ema = None
    loss_aux_ema = None
    plateau_count = 0
    final_score = float("inf")
    best_score = float(best_score)
    relaxed_target = float(target_error) + float(stage_cfg.get("relaxation_delta", 0.005))
    current_target = float(target_error)
    relax_iter = num_iters // 2
    best_metrics = {}
    best_stage_score = float("inf")
    best_hist_stats = None
    final_hist_stats = None
    iteration = start_iter
    prefix_models = [] if prefix_models is None else list(prefix_models)
    curriculum_enabled = internal_curriculum_enabled(cfg_dict)
    last_curriculum_phase_idx = None

    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg_dict["time_marching"].get("lr_decay_step", 4000)),
        gamma=float(cfg_dict["time_marching"].get("lr_decay_gamma", 0.85)),
    )

    for iteration in range(start_iter + 1, num_iters + 1):
        if allow_relaxation and iteration == relax_iter:
            current_target = relaxed_target
            print(f"    ⚠️ Cible relaxee a {current_target:.2%} a mi-stage.")

        active_t_end, curriculum_fraction, curriculum_phase_idx = stage_curriculum_state(
            cfg_dict,
            t_start,
            t_end,
            iteration,
            num_iters,
        )
        active_target = current_target * internal_curriculum_target_scale(cfg_dict, curriculum_phase_idx)
        can_select_best = (not curriculum_enabled) or (active_t_end >= float(t_end) - 1.0e-10)
        if curriculum_enabled and curriculum_phase_idx != last_curriculum_phase_idx:
            print(
                f"    🧭 Curriculum interne phase {curriculum_phase_idx + 1}/"
                f"{len(internal_curriculum_fractions(cfg_dict))} | "
                f"fraction={curriculum_fraction:.2f} | t_end_actif={active_t_end:.4f}"
            )
            last_curriculum_phase_idx = curriculum_phase_idx

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = compute_stage_loss_components(model, cfg_dict, t_start, active_t_end, device, teacher_model=teacher_model)

        with torch.no_grad():
            if loss_pde_ema is None:
                loss_pde_ema = float(losses["loss_pde"].item())
                loss_aux_ema = float(losses["aux_loss"].item())
            else:
                loss_pde_ema = ema_alpha * loss_pde_ema + (1.0 - ema_alpha) * float(losses["loss_pde"].item())
                loss_aux_ema = ema_alpha * loss_aux_ema + (1.0 - ema_alpha) * float(losses["aux_loss"].item())
            tot_ema = loss_pde_ema + loss_aux_ema + 1.0e-9
            target_w_pde = min(tot_ema / (2.0 * loss_pde_ema + 1.0e-9), 5.0)
            target_w_bc = min(tot_ema / (2.0 * loss_aux_ema + 1.0e-9), 5.0)
            w_pde = ema_alpha * w_pde + (1.0 - ema_alpha) * target_w_pde
            w_bc = ema_alpha * w_bc + (1.0 - ema_alpha) * target_w_bc

        total_loss = w_pde * losses["loss_pde"] + w_bc * losses["aux_loss"]
        if torch.isnan(total_loss) or torch.isinf(total_loss) or float(total_loss.item()) > 1.0e4:
            print("    💥 Loss invalide ou explosive. Arret du stage.")
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        for param_group in optimizer.param_groups:
            if param_group["lr"] < 5.0e-6:
                param_group["lr"] = 5.0e-6

        if iteration % log_every == 0 or iteration == 1:
            print(
                f"[{os.path.basename(stage_dir)} | iter {iteration}] total={float(total_loss.item()):.3e} "
                f"| pde={float(losses['loss_pde'].item()):.3e} "
                f"| bc={float(losses['loss_bc'].item()):.3e} "
                f"| mass={float(losses['loss_mass'].item()):.3e} "
                f"| cont={float(losses['loss_continuity'].item()):.3e}"
            )

        if iteration % eval_every == 0 or iteration == num_iters:
            _, stage_score = run_audit(
                model,
                cfg_dict,
                active_t_end,
                threshold=active_target,
                n_global=audit_cases,
                verbose=False,
                historical=False,
            )
            final_score = float(stage_score)
            selection_score = final_score
            hist_stats = None
            if hist_enabled:
                hist_stats = evaluate_historical_rollout(
                    prefix_models,
                    model,
                    time_blocks,
                    historical_reference,
                    stage_idx,
                    cfg_dict,
                    device,
                    t_end_override=active_t_end,
                )
                selection_score = historical_selection_score(cfg_dict, hist_stats)
                print(
                    f"    📏 stage_score(t={active_t_end:.2f})={final_score:.3%} "
                    f"| hist_mean={hist_stats['mean_rel_l2']:.3%} "
                    f"| hist_max={hist_stats['max_rel_l2']:.3%} "
                    f"| hist_final={hist_stats['final_rel_l2']:.3%}"
                )
            else:
                print(f"    📏 stage_score(t={active_t_end:.2f})={final_score:.3%}")
            save_stage_checkpoint(model, optimizer, iteration, best_score, stage_dir, "model_latest.pth")

            improved = can_select_best and selection_score < best_score
            if improved or not np.isfinite(best_score):
                if can_select_best:
                    best_score = selection_score
                    best_stage_score = final_score
                    best_hist_stats = dict(hist_stats) if hist_stats is not None else None
                    best_metrics = {
                        "loss_pde": float(losses["loss_pde"].detach().item()),
                        "loss_bc": float(losses["loss_bc"].detach().item()),
                        "loss_mass": float(losses["loss_mass"].detach().item()),
                        "loss_continuity": float(losses["loss_continuity"].detach().item()),
                    }
                    if best_hist_stats is not None:
                        best_metrics["historical_mean_rel_l2"] = float(best_hist_stats["mean_rel_l2"])
                        best_metrics["historical_max_rel_l2"] = float(best_hist_stats["max_rel_l2"])
                        best_metrics["historical_final_rel_l2"] = float(best_hist_stats["final_rel_l2"])
                    save_stage_checkpoint(model, optimizer, iteration, best_score, stage_dir, "model_best.pth")
                    plateau_count = 0
                    print(f"    ✅ Nouveau meilleur score de selection : {best_score:.3%}")
                else:
                    print("    ℹ️ Audit curriculum hors intervalle final : checkpoint non eligible comme best final.")
            else:
                if can_select_best:
                    plateau_count += 1

            local_target_ok = can_select_best and final_score < active_target
            hist_target_ok = True
            hist_max_ok = True
            if hist_stats is not None:
                hist_target_ok = hist_stats["mean_rel_l2"] < hist_target
                if hist_max_target is not None and bool(hist_cfg.get("require_max_target", True)):
                    hist_max_ok = hist_stats["max_rel_l2"] < hist_max_target
            if local_target_ok and hist_target_ok and hist_max_ok:
                if hist_stats is not None:
                    print(
                        f"    🎯 Cibles stage+historique atteintes "
                        f"(local={final_score:.3%}, hist_mean={hist_stats['mean_rel_l2']:.3%}, "
                        f"hist_max={hist_stats['max_rel_l2']:.3%})."
                    )
                else:
                    print(f"    🎯 Cible de stage atteinte ({final_score:.3%} < {current_target:.3%}).")
                break

            if (
                bool(early_cfg.get("enabled", True))
                and iteration >= int(early_cfg.get("min_iters", 8000))
                and plateau_count >= int(early_cfg.get("patience_audits", 4))
            ):
                print("    ⏹️ Plateau detecte, arret anticipe du stage.")
                break

        if iteration % snapshot_every == 0:
            save_stage_checkpoint(model, optimizer, iteration, best_score, stage_dir, f"ckpt_iter_{iteration:06d}.pth")

    best_path = os.path.join(stage_dir, "checkpoints", "model_best.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"], strict=True)

    _, final_score = run_audit(
        model,
        cfg_dict,
        t_end,
        threshold=target_error,
        n_global=audit_cases,
        verbose=True,
        historical=False,
    )
    if hist_enabled:
        final_hist_stats = evaluate_historical_rollout(
            prefix_models,
            model,
            time_blocks,
            historical_reference,
            stage_idx,
            cfg_dict,
            device,
            t_end_override=t_end,
        )
    save_stage_checkpoint(model, optimizer, iteration, best_score, stage_dir, "model_final.pth")
    with open(os.path.join(stage_dir, "stage_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "stage_idx": int(stage_idx),
                "t_start": float(t_start),
                "t_end": float(t_end),
                "best_score": float(best_score),
                "best_selection_score": float(best_score),
                "best_stage_score": float(best_stage_score),
                "best_historical_mean": None if best_hist_stats is None else float(best_hist_stats["mean_rel_l2"]),
                "best_historical_max": None if best_hist_stats is None else float(best_hist_stats["max_rel_l2"]),
                "best_historical_final": None if best_hist_stats is None else float(best_hist_stats["final_rel_l2"]),
                "final_score": float(final_score),
                "final_stage_score": float(final_score),
                "final_historical_mean": None if final_hist_stats is None else float(final_hist_stats["mean_rel_l2"]),
                "final_historical_max": None if final_hist_stats is None else float(final_hist_stats["max_rel_l2"]),
                "final_historical_final": None if final_hist_stats is None else float(final_hist_stats["final_rel_l2"]),
                "completed_iters": int(iteration),
                "best_metrics": best_metrics,
            },
            handle,
            indent=2,
        )
    return {
        "best_score": float(best_score),
        "best_selection_score": float(best_score),
        "best_stage_score": float(best_stage_score),
        "best_historical_mean": None if best_hist_stats is None else float(best_hist_stats["mean_rel_l2"]),
        "best_historical_max": None if best_hist_stats is None else float(best_hist_stats["max_rel_l2"]),
        "final_score": float(final_score),
        "final_stage_score": float(final_score),
        "final_historical_mean": None if final_hist_stats is None else float(final_hist_stats["mean_rel_l2"]),
        "final_historical_max": None if final_hist_stats is None else float(final_hist_stats["max_rel_l2"]),
        "wall_seconds": max(0.0, time.perf_counter() - stage_start_perf),
    }


def load_stage_summary(stage_dir):
    path = os.path.join(stage_dir, "stage_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_stage_model(models, time_blocks, t_current):
    for (t_start, t_end), model in zip(time_blocks, models):
        if float(t_current) <= float(t_end) + 1.0e-10:
            return model
    return models[-1]


def rollout_multistage_models(models, time_blocks, reference, device):
    x = reference["x"]
    t_values = reference["t"]
    u_true = reference["u"]
    params = reference["params"]
    p_vec = np.array(
        [
            params["alpha"],
            params["beta"],
            params["mu"],
            params["V"],
            params["A"],
            params["w0"],
            params["x0"],
            params["k"],
            float(params["type"]),
        ],
        dtype=np.float32,
    )

    u_pred = np.zeros_like(u_true)
    for idx_t, t_val in enumerate(t_values):
        model = select_stage_model(models, time_blocks, float(t_val))
        coords = torch.tensor(np.stack([x, np.full_like(x, t_val)], axis=1), dtype=torch.float32, device=device)
        branch = torch.tensor(np.repeat(p_vec[None, :], len(x), axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_re, pred_im = model(branch, coords)
        u_pred[:, idx_t] = (pred_re[:, 0] + 1j * pred_im[:, 0]).cpu().numpy().astype(np.complex64)

    rel_l2 = np.zeros(len(t_values), dtype=np.float64)
    for idx_t in range(len(t_values)):
        denom = np.linalg.norm(u_true[:, idx_t]) + 1.0e-12
        rel_l2[idx_t] = np.linalg.norm(u_pred[:, idx_t] - u_true[:, idx_t]) / denom
    return {
        "x": x,
        "t_values": t_values,
        "u_true": u_true,
        "u_pred": u_pred,
        "rel_l2": rel_l2,
    }


def write_stage_manifest(run_dir, stage_rows):
    path = os.path.join(run_dir, "stage_manifest.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "stage_idx,t_start,t_end,best_score,final_score,best_historical_mean,"
            "best_historical_max,final_historical_mean,final_historical_max,wall_seconds\n"
        )
        for row in stage_rows:
            best_hist_mean = "" if row.get("best_historical_mean") is None else f"{float(row['best_historical_mean']):.10e}"
            best_hist_max = "" if row.get("best_historical_max") is None else f"{float(row['best_historical_max']):.10e}"
            final_hist_mean = "" if row.get("final_historical_mean") is None else f"{float(row['final_historical_mean']):.10e}"
            final_hist_max = "" if row.get("final_historical_max") is None else f"{float(row['final_historical_max']):.10e}"
            handle.write(
                f"{int(row['stage_idx'])},{float(row['t_start']):.10f},{float(row['t_end']):.10f},"
                f"{float(row['best_score']):.10e},{float(row['final_score']):.10e},"
                f"{best_hist_mean},{best_hist_max},{final_hist_mean},{final_hist_max},"
                f"{float(row['wall_seconds']):.6f}\n"
            )


def run_postprocess(cfg_dict, run_dir, stage_rows, device):
    time_blocks = load_time_blocks(cfg_dict)
    stage_models = [
        load_best_stage_model(cfg_dict, os.path.join(run_dir, stage_name(idx, t_start, t_end)), device)
        for idx, (t_start, t_end) in enumerate(time_blocks)
    ]
    reference = prepare_reference_trajectory(cfg_dict, nx_override=int(cfg_dict.get("benchmark", {}).get("solver_nx", 256)))
    rollout = rollout_multistage_models(stage_models, time_blocks, reference, device)

    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    save_rel_l2_csv(os.path.join(eval_dir, "rollout_metrics.csv"), rollout["t_values"], rollout["rel_l2"])
    center_mask = spatial_mask_from_bounds(rollout["x"], -10.0, 10.0)
    rel_l2_center = relative_l2_curve_on_mask(rollout["u_pred"], rollout["u_true"], center_mask)
    save_rel_l2_csv(os.path.join(eval_dir, "rollout_metrics_center_xm10_xp10.csv"), rollout["t_values"], rel_l2_center)
    plot_l2_curve(
        rollout["t_values"],
        rollout["rel_l2"],
        "CGL global multireseau physics-only : erreur relative",
        os.path.join(eval_dir, "rollout_rel_l2.png"),
        stage_markers=stage_markers_from_blocks(time_blocks),
    )
    plot_l2_curve(
        rollout["t_values"],
        rel_l2_center,
        "CGL global multireseau physics-only : erreur relative au centre x in [-10, 10]",
        os.path.join(eval_dir, "rollout_rel_l2_center_xm10_xp10.png"),
        stage_markers=stage_markers_from_blocks(time_blocks),
    )
    plot_error_heatmap(
        rollout["x"],
        rollout["t_values"],
        rollout["u_true"],
        rollout["u_pred"],
        "CGL global multireseau physics-only : heatmap erreur",
        os.path.join(eval_dir, "error_heatmap.png"),
        stage_markers=stage_markers_from_blocks(time_blocks),
    )
    plot_snapshots(
        rollout["x"],
        rollout["t_values"],
        rollout["u_true"],
        rollout["u_pred"],
        "CGL global multireseau physics-only : snapshots",
        os.path.join(eval_dir, "snapshots.png"),
        snapshot_times=list(cfg_dict.get("benchmark", {}).get("eval_times", [0.2, 0.5, 1.0, 2.0, 3.0, 5.0])),
    )
    if os.environ.get("CGL_SKIP_GIF", "0") != "1":
        save_comparison_gif(
            rollout["x"],
            rollout["t_values"],
            rollout["u_true"],
            rollout["u_pred"],
            "CGL global multireseau physics-only",
            os.path.join(eval_dir, "comparison_animation.gif"),
        )
    write_rollout_summary(
        os.path.join(eval_dir, "summary.txt"),
        rollout["rel_l2"],
        rollout["t_values"],
        extra={
            "n_stages": len(time_blocks),
            "stage_markers": ",".join(str(float(x)) for x in stage_markers_from_blocks(time_blocks)),
            "final_stage_score": stage_rows[-1]["final_score"] if stage_rows else float("nan"),
            "final_stage_historical_mean": stage_rows[-1].get("final_historical_mean", float("nan")) if stage_rows else float("nan"),
            "final_stage_historical_max": stage_rows[-1].get("final_historical_max", float("nan")) if stage_rows else float("nan"),
            "final_rel_l2_center_xm10_xp10": float(rel_l2_center[-1]),
            "max_rel_l2_center_xm10_xp10": float(np.max(rel_l2_center)),
            "mean_rel_l2_center_xm10_xp10": float(np.mean(rel_l2_center)),
        },
    )
    params = reference["params"]
    t_max = float(cfg_dict["physics"]["t_max"])
    if os.environ.get("CGL_SKIP_BENCHMARK", "0") != "1":
        benchmark_inference(
            "GlobalMultinet",
            solver_callable=lambda: get_ground_truth_CGL(params, cfg_dict["physics"]["x_domain"][0], cfg_dict["physics"]["x_domain"][1], t_max, Nx=128, Nt=None),
            model_callable=lambda: rollout_multistage_models(stage_models, time_blocks, reference, device)["u_pred"],
            output_dir=eval_dir,
            repeats=int(cfg_dict.get("benchmark", {}).get("timing_repeats", 4)),
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
    save_yaml_copy(run_dir, cfg_dict, args.config)

    start_dt = datetime.now()
    start_perf = time.perf_counter()
    print(f"📱 Device : {device}")
    print(f"📂 Run dir : {run_dir}")
    print(f"🧾 Config : {args.config}")

    time_blocks = load_time_blocks(cfg_dict)
    historical_reference = None
    if historical_validation_enabled(cfg_dict):
        hist_nx = historical_solver_nx(cfg_dict)
        print(f"🕰️ Validation historique active | solver_nx={hist_nx} | time_stride={historical_time_stride(cfg_dict)}")
        historical_reference = prepare_reference_trajectory(cfg_dict, nx_override=hist_nx)
    stage_rows = []
    completed_stage_models = []

    try:
        for stage_idx, (t_start, t_end) in enumerate(time_blocks):
            stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
            os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)

            summary = load_stage_summary(stage_dir)
            final_ckpt = os.path.join(stage_dir, "checkpoints", "model_final.pth")
            if summary is not None and os.path.exists(final_ckpt):
                print(f"\n⏭️ Stage {stage_idx + 1}/{len(time_blocks)} deja termine, on passe au suivant.")
                stage_rows.append(
                    {
                        "stage_idx": stage_idx,
                        "stage_label": f"{t_start:.2f}_{t_end:.2f}",
                        "t_start": float(t_start),
                        "t_end": float(t_end),
                        "best_score": float(summary["best_score"]),
                        "final_score": float(summary["final_score"]),
                        "best_historical_mean": summary.get("best_historical_mean"),
                        "best_historical_max": summary.get("best_historical_max"),
                        "final_historical_mean": summary.get("final_historical_mean"),
                        "final_historical_max": summary.get("final_historical_max"),
                        "wall_seconds": 0.0,
                    }
                )
                completed_stage_models.append(load_best_stage_model(cfg_dict, stage_dir, device))
                continue

            teacher_model = completed_stage_models[-1] if completed_stage_models else None

            model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg_dict["time_marching"].get("learning_rate", cfg_dict["training"].get("learning_rate", 2e-4))),
                weight_decay=float(cfg_dict["training"].get("weight_decay", 1.0e-6)),
            )

            latest_ckpt = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
            if not os.path.exists(latest_ckpt) and teacher_model is not None and bool(cfg_dict["multistage"].get("warm_start_interstage", True)):
                model.load_state_dict(teacher_model.state_dict(), strict=True)
                print(f"🔗 Warm-start stage_{stage_idx:02d} depuis le meilleur checkpoint du stage precedent.")

            print(f"\n🚧 Stage {stage_idx + 1}/{len(time_blocks)} | bloc=[{t_start:.2f}, {t_end:.2f}]")
            metrics = train_one_stage(
                model,
                optimizer,
                cfg_dict,
                stage_idx,
                t_start,
                t_end,
                stage_dir,
                device,
                teacher_model=teacher_model,
                prefix_models=completed_stage_models,
                time_blocks=time_blocks,
                historical_reference=historical_reference,
            )
            stage_rows.append(
                {
                    "stage_idx": stage_idx,
                    "stage_label": f"{t_start:.2f}_{t_end:.2f}",
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "best_score": float(metrics["best_score"]),
                    "final_score": float(metrics["final_score"]),
                    "best_historical_mean": metrics.get("best_historical_mean"),
                    "best_historical_max": metrics.get("best_historical_max"),
                    "final_historical_mean": metrics.get("final_historical_mean"),
                    "final_historical_max": metrics.get("final_historical_max"),
                    "wall_seconds": float(metrics["wall_seconds"]),
                }
            )
            completed_stage_models.append(load_best_stage_model(cfg_dict, stage_dir, device))

        write_stage_manifest(run_dir, stage_rows)
        run_postprocess(cfg_dict, run_dir, stage_rows, device)
        write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "completed", stage_rows)
        print("\n🏁 Global multireseau physics-only termine.")
    except Exception:
        write_stage_manifest(run_dir, stage_rows)
        write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "failed", stage_rows)
        raise


if __name__ == "__main__":
    main()
