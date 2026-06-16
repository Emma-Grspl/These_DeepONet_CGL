import argparse
import csv
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

from src.models.cgl_deeponet_amp_phase import CGL_PI_DeepONet_AmpPhase
from src.utils.solver_cgl import get_ground_truth_CGL


PARAM_KEYS = ["alpha", "beta", "mu", "V", "A", "w0", "x0", "k", "type"]
PARAM_INDEX = {key: idx for idx, key in enumerate(PARAM_KEYS)}
EQ_PARAM_KEYS = {"alpha", "beta", "mu", "V"}
BOUND_PARAM_KEYS = {"A", "w0", "x0", "k"}


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def resolve_project_path(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


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
            handle.write(f"{prefix}_best_valid_loss={float(row['best_valid_loss']):.10e}\n")
            handle.write(f"{prefix}_final_valid_loss={float(row['final_valid_loss']):.10e}\n")
            handle.write(f"{prefix}_wall_seconds={float(row['wall_seconds']):.6f}\n")

    csv_path = os.path.join(run_dir, "timing_stages.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("stage_idx,stage_label,t_start,t_end,best_valid_loss,final_valid_loss,wall_seconds\n")
        for row in stage_rows:
            handle.write(
                f"{int(row['stage_idx'])},{row['stage_label']},{float(row['t_start']):.10f},"
                f"{float(row['t_end']):.10f},{float(row['best_valid_loss']):.10e},"
                f"{float(row['final_valid_loss']):.10e},{float(row['wall_seconds']):.6f}\n"
            )


def stage_name(stage_idx, t_start, t_end):
    return f"stage_{stage_idx:02d}_t{t_start:.2f}_{t_end:.2f}"


def write_stage_manifest(run_dir, stage_rows):
    path = os.path.join(run_dir, "stage_manifest.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("stage_idx,t_start,t_end,best_valid_loss,final_valid_loss,wall_seconds\n")
        for row in stage_rows:
            handle.write(
                f"{int(row['stage_idx'])},{float(row['t_start']):.10f},{float(row['t_end']):.10f},"
                f"{float(row['best_valid_loss']):.10e},{float(row['final_valid_loss']):.10e},"
                f"{float(row['wall_seconds']):.6f}\n"
            )


def _sample_uniform(low, high, rng):
    low = float(low)
    high = float(high)
    if abs(high - low) < 1.0e-12:
        return low
    return float(rng.uniform(low, high))


def branch_vector_from_params(params):
    return np.array([float(params[key]) for key in PARAM_KEYS], dtype=np.float32)


def params_from_branch_row(branch_row):
    return {key: float(branch_row[idx]) for idx, key in enumerate(PARAM_KEYS)}


def base_case_defaults(cfg_dict):
    physics = cfg_dict["physics"]
    eq = physics["equation_params"]
    bounds = physics["bounds"]
    return {
        "alpha": float(eq["alpha"][0]),
        "beta": float(eq["beta"][0]),
        "mu": float(eq["mu"][0]),
        "V": float(eq["V"][0]),
        "A": float(bounds["A"][0]),
        "w0": float(bounds["w0"][0]),
        "x0": float(bounds["x0"][0]),
        "k": float(bounds["k"][0]),
        "type": int(physics["initial_conditions"][0]),
    }


def apply_physics_overrides(base_cfg, physics_overrides):
    cfg_copy = json.loads(json.dumps(base_cfg))
    if not physics_overrides:
        return cfg_copy
    physics = cfg_copy["physics"]
    for family_name in ("equation_params", "bounds"):
        overrides = physics_overrides.get(family_name, {})
        target = physics[family_name]
        for key, value in overrides.items():
            if key not in target:
                raise KeyError(f"Override inconnu: {family_name}.{key}")
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"Override invalide pour {family_name}.{key}: {value}")
            target[key] = [float(value[0]), float(value[1])]
    return cfg_copy


def sample_case_params(cfg_dict, rng):
    physics = cfg_dict["physics"]
    eq = physics["equation_params"]
    bounds = physics["bounds"]
    return {
        "alpha": _sample_uniform(eq["alpha"][0], eq["alpha"][1], rng),
        "beta": _sample_uniform(eq["beta"][0], eq["beta"][1], rng),
        "mu": _sample_uniform(eq["mu"][0], eq["mu"][1], rng),
        "V": _sample_uniform(eq["V"][0], eq["V"][1], rng),
        "A": _sample_uniform(bounds["A"][0], bounds["A"][1], rng),
        "w0": _sample_uniform(bounds["w0"][0], bounds["w0"][1], rng),
        "x0": _sample_uniform(bounds["x0"][0], bounds["x0"][1], rng),
        "k": _sample_uniform(bounds["k"][0], bounds["k"][1], rng),
        "type": int(physics["initial_conditions"][0]),
    }


def build_param_pool(cfg_dict, n_cases, seed, physics_overrides=None):
    rng = np.random.default_rng(int(seed))
    sampler_cfg = apply_physics_overrides(cfg_dict, physics_overrides)
    rows = []
    for _ in range(int(n_cases)):
        params = sample_case_params(sampler_cfg, rng)
        rows.append({"params": params, "branch_vec": branch_vector_from_params(params)})
    return rows


def save_case_pool_csv(path, pool):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_idx"] + PARAM_KEYS)
        writer.writeheader()
        for idx, row in enumerate(pool):
            payload = {"case_idx": idx}
            payload.update({key: row["params"][key] for key in PARAM_KEYS})
            writer.writerow(payload)


def teacher_cfg(cfg_dict):
    return cfg_dict["teacher_distillation"]


def free_variable_name(cfg_dict):
    return str(teacher_cfg(cfg_dict)["variable"])


def free_variable_index(cfg_dict):
    variable = free_variable_name(cfg_dict)
    if variable not in PARAM_INDEX:
        raise KeyError(f"Variable libre inconnue: {variable}")
    return PARAM_INDEX[variable]


def focus_sampling_cfg(cfg_dict):
    return cfg_dict.get("focus_sampling", {})


def focus_sampling_enabled(cfg_dict):
    cfg = focus_sampling_cfg(cfg_dict)
    return bool(cfg.get("enabled", False) and cfg.get("physics"))


def build_focus_pool(cfg_dict):
    if not focus_sampling_enabled(cfg_dict):
        return []
    ds_cfg = cfg_dict["parametric_dataset"]
    fs_cfg = focus_sampling_cfg(cfg_dict)
    focus_cases = int(fs_cfg.get("focus_cases", max(32, int(ds_cfg["train_cases"]) // 2)))
    seed = int(ds_cfg["seed"]) + int(fs_cfg.get("focus_seed_offset", 3000))
    return build_param_pool(cfg_dict, focus_cases, seed, physics_overrides=fs_cfg["physics"])


def _normalize_anchor_params(cfg_dict, params):
    base = base_case_defaults(cfg_dict)
    merged = dict(base)
    merged.update(params)
    return {
        "alpha": float(merged["alpha"]),
        "beta": float(merged["beta"]),
        "mu": float(merged["mu"]),
        "V": float(merged["V"]),
        "A": float(merged["A"]),
        "w0": float(merged["w0"]),
        "x0": float(merged["x0"]),
        "k": float(merged["k"]),
        "type": int(merged["type"]),
    }


def _extract_model_state(checkpoint_payload):
    if isinstance(checkpoint_payload, dict):
        if "model_state" in checkpoint_payload:
            return checkpoint_payload["model_state"]
        if "state_dict" in checkpoint_payload:
            return checkpoint_payload["state_dict"]
        if all(torch.is_tensor(value) for value in checkpoint_payload.values()):
            return checkpoint_payload
    raise ValueError("Format de checkpoint teacher non supporte.")


def load_anchor_teachers(cfg_dict, device):
    anchors_cfg = teacher_cfg(cfg_dict).get("anchors", [])
    if len(anchors_cfg) < 2:
        raise ValueError("teacher_distillation.anchors doit contenir au moins 2 anchors.")

    variable = free_variable_name(cfg_dict)
    anchors = []
    for anchor_cfg in anchors_cfg:
        params = _normalize_anchor_params(cfg_dict, anchor_cfg.get("params", {}))
        checkpoint = resolve_project_path(anchor_cfg.get("teacher_checkpoint"))
        if checkpoint is None or not os.path.exists(checkpoint):
            raise FileNotFoundError(
                "Checkpoint teacher introuvable. "
                f"Compléter teacher_checkpoint pour {anchor_cfg.get('label', '<sans_label>')}."
            )
        payload = torch.load(checkpoint, map_location=device)
        model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
        model.load_state_dict(_extract_model_state(payload), strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        anchors.append(
            {
                "label": str(anchor_cfg.get("label", f"anchor_{len(anchors)}")),
                "value": float(anchor_cfg.get("variable_value", params[variable])),
                "params": params,
                "branch_vec": branch_vector_from_params(params),
                "checkpoint": checkpoint,
                "model": model,
            }
        )

    anchors.sort(key=lambda row: row["value"])
    values = [row["value"] for row in anchors]
    if any(values[idx] >= values[idx + 1] for idx in range(len(values) - 1)):
        raise ValueError("Les variable_value des teachers doivent etre strictement croissants.")
    return anchors


def save_teacher_manifest(run_dir, anchor_teachers):
    path = os.path.join(run_dir, "teacher_manifest.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "variable_value", "checkpoint"] + PARAM_KEYS)
        writer.writeheader()
        for anchor in anchor_teachers:
            row = {
                "label": anchor["label"],
                "variable_value": float(anchor["value"]),
                "checkpoint": anchor["checkpoint"],
            }
            row.update(anchor["params"])
            writer.writerow(row)


def sample_case_rows(train_pool, focus_pool, anchor_teachers, cfg_dict, n_queries, rng):
    td_cfg = teacher_cfg(cfg_dict)
    anchor_fraction = float(td_cfg.get("anchor_fraction", 0.35))
    focus_fraction = float(td_cfg.get("focus_fraction", 0.25)) if focus_pool else 0.0
    anchor_fraction = max(0.0, min(1.0, anchor_fraction))
    focus_fraction = max(0.0, min(1.0, focus_fraction))
    if anchor_fraction + focus_fraction > 1.0:
        scale = 1.0 / (anchor_fraction + focus_fraction)
        anchor_fraction *= scale
        focus_fraction *= scale

    n_anchor = int(round(int(n_queries) * anchor_fraction))
    n_focus = int(round(int(n_queries) * focus_fraction))
    n_uniform = int(n_queries) - n_anchor - n_focus
    if n_uniform < 0:
        overflow = -n_uniform
        take_from_focus = min(overflow, n_focus)
        n_focus -= take_from_focus
        overflow -= take_from_focus
        if overflow > 0:
            n_anchor = max(0, n_anchor - overflow)
        n_uniform = int(n_queries) - n_anchor - n_focus

    rows = []
    if n_anchor > 0:
        anchor_ids = rng.integers(0, len(anchor_teachers), size=n_anchor)
        rows.extend(anchor_teachers[int(idx)]["branch_vec"].copy() for idx in anchor_ids)
    if n_uniform > 0:
        uniform_ids = rng.integers(0, len(train_pool), size=n_uniform)
        rows.extend(train_pool[int(idx)]["branch_vec"].copy() for idx in uniform_ids)
    if n_focus > 0:
        focus_ids = rng.integers(0, len(focus_pool), size=n_focus)
        rows.extend(focus_pool[int(idx)]["branch_vec"].copy() for idx in focus_ids)

    if not rows:
        raise ValueError("Aucun cas disponible pour construire le batch.")
    branch_np = np.stack(rows, axis=0).astype(np.float32)
    rng.shuffle(branch_np)
    return branch_np


def spatial_focus_cfg(cfg_dict):
    return cfg_dict.get("spatial_focus_sampling", {})


def temporal_focus_cfg(cfg_dict):
    return cfg_dict.get("temporal_focus_sampling", {})


def sample_time_values(n_queries, t_start, t_end, cfg_dict, rng):
    t_start = float(t_start)
    t_end = float(t_end)
    if t_end <= t_start + 1.0e-12:
        return np.full(int(n_queries), t_start, dtype=np.float32)

    tf_cfg = temporal_focus_cfg(cfg_dict)
    if not tf_cfg.get("enabled", False):
        return rng.uniform(t_start, t_end, size=int(n_queries)).astype(np.float32)

    start_fraction = float(tf_cfg.get("start_fraction", 0.6))
    start_window_fraction = float(tf_cfg.get("start_window_fraction", 0.25))
    start_fraction = max(0.0, min(1.0, start_fraction))
    start_window_fraction = max(0.0, min(1.0, start_window_fraction))
    early_end = t_start + start_window_fraction * (t_end - t_start)
    mask = rng.random(int(n_queries)) < start_fraction
    t_vals = rng.uniform(t_start, t_end, size=int(n_queries)).astype(np.float32)
    if early_end > t_start + 1.0e-12:
        t_vals[mask] = rng.uniform(t_start, early_end, size=int(np.sum(mask))).astype(np.float32)
    return t_vals


def sample_x_values(branch_np, t_values, cfg_dict, rng):
    x_min, x_max = map(float, cfg_dict["physics"]["x_domain"])
    sf_cfg = spatial_focus_cfg(cfg_dict)
    center_fraction = float(sf_cfg.get("center_fraction", 0.8)) if sf_cfg.get("enabled", False) else 0.8
    center_fraction = max(0.0, min(1.0, center_fraction))

    x0 = branch_np[:, PARAM_INDEX["x0"]]
    w0 = branch_np[:, PARAM_INDEX["w0"]]
    width = w0 * np.sqrt(1.0 + (2.0 * t_values) ** 2)
    center_mask = rng.random(len(branch_np)) < center_fraction

    x_vals = rng.uniform(x_min, x_max, size=len(branch_np)).astype(np.float32)
    n_center = int(np.sum(center_mask))
    if n_center > 0:
        center_noise = rng.normal(loc=0.0, scale=1.0, size=n_center).astype(np.float32)
        x_vals[center_mask] = x0[center_mask] + center_noise * width[center_mask] * 1.5
    np.clip(x_vals, x_min, x_max, out=x_vals)
    return x_vals.astype(np.float32)


def build_query_batch(branch_np, cfg_dict, t_start, t_end, device, rng):
    t_vals = sample_time_values(len(branch_np), t_start, t_end, cfg_dict, rng)
    x_vals = sample_x_values(branch_np, t_vals, cfg_dict, rng)
    coords_np = np.stack([x_vals, t_vals], axis=1).astype(np.float32)
    return (
        torch.tensor(branch_np, dtype=torch.float32, device=device),
        torch.tensor(coords_np, dtype=torch.float32, device=device),
    )


def build_boundary_batch(branch_np, cfg_dict, t_start, t_end, device, rng):
    t_vals = sample_time_values(len(branch_np), t_start, t_end, cfg_dict, rng)
    x_min, x_max = map(float, cfg_dict["physics"]["x_domain"])
    left_np = np.stack([np.full(len(branch_np), x_min, dtype=np.float32), t_vals], axis=1)
    right_np = np.stack([np.full(len(branch_np), x_max, dtype=np.float32), t_vals], axis=1)
    branch = torch.tensor(branch_np, dtype=torch.float32, device=device)
    coords_left = torch.tensor(left_np, dtype=torch.float32, device=device).requires_grad_(True)
    coords_right = torch.tensor(right_np, dtype=torch.float32, device=device).requires_grad_(True)
    return branch, coords_left, coords_right


def build_interface_batch(branch_np, cfg_dict, t_fixed, device, rng):
    t_vals = np.full(len(branch_np), float(t_fixed), dtype=np.float32)
    x_vals = sample_x_values(branch_np, t_vals, cfg_dict, rng)
    coords_np = np.stack([x_vals, t_vals], axis=1).astype(np.float32)
    return (
        torch.tensor(branch_np, dtype=torch.float32, device=device),
        torch.tensor(coords_np, dtype=torch.float32, device=device),
    )


def compute_interp_indices(anchor_values, values):
    anchor_values = np.asarray(anchor_values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    left = np.zeros(len(values), dtype=np.int64)
    right = np.zeros(len(values), dtype=np.int64)
    weight_right = np.zeros(len(values), dtype=np.float32)
    for idx, value in enumerate(values):
        if value <= anchor_values[0]:
            left[idx] = 0
            right[idx] = 0
            continue
        if value >= anchor_values[-1]:
            left[idx] = len(anchor_values) - 1
            right[idx] = len(anchor_values) - 1
            continue
        ridx = int(np.searchsorted(anchor_values, value, side="right"))
        lidx = ridx - 1
        left[idx] = lidx
        right[idx] = ridx
        denom = float(anchor_values[ridx] - anchor_values[lidx])
        weight_right[idx] = float((value - anchor_values[lidx]) / max(denom, 1.0e-12))
    return left, right, weight_right


def interpolate_teacher_targets(anchor_teachers, variable_values, coords, device):
    n_samples = int(coords.shape[0])
    anchor_values = np.asarray([anchor["value"] for anchor in anchor_teachers], dtype=np.float64)
    preds_re = []
    preds_im = []
    with torch.no_grad():
        for anchor in anchor_teachers:
            branch = torch.tensor(
                np.repeat(anchor["branch_vec"][None, :], n_samples, axis=0),
                dtype=torch.float32,
                device=device,
            )
            pred_re, pred_im = anchor["model"](branch, coords)
            preds_re.append(pred_re[:, 0])
            preds_im.append(pred_im[:, 0])

    pred_re_stack = torch.stack(preds_re, dim=0)
    pred_im_stack = torch.stack(preds_im, dim=0)
    left_idx, right_idx, weight_right = compute_interp_indices(anchor_values, variable_values)
    left_t = torch.tensor(left_idx, dtype=torch.long, device=device)
    right_t = torch.tensor(right_idx, dtype=torch.long, device=device)
    sample_ids = torch.arange(n_samples, device=device)
    w_right = torch.tensor(weight_right, dtype=torch.float32, device=device)
    w_left = 1.0 - w_right

    left_re = pred_re_stack[left_t, sample_ids]
    left_im = pred_im_stack[left_t, sample_ids]
    right_re = pred_re_stack[right_t, sample_ids]
    right_im = pred_im_stack[right_t, sample_ids]

    target_re = (w_left * left_re + w_right * right_re).unsqueeze(1)
    target_im = (w_left * left_im + w_right * right_im).unsqueeze(1)
    return target_re, target_im


def compute_anchor_distillation_loss(model, anchor_teachers, branch, coords, variable_values, device):
    pred_re, pred_im = model(branch, coords)
    target_re, target_im = interpolate_teacher_targets(anchor_teachers, variable_values, coords, device)
    loss = torch.mean((pred_re - target_re) ** 2 + (pred_im - target_im) ** 2)
    return loss, {
        "pred_re": pred_re,
        "pred_im": pred_im,
        "target_re": target_re,
        "target_im": target_im,
    }


def compute_boundary_loss(model, branch, coords_left, coords_right):
    pred_re_left, pred_im_left = model(branch, coords_left)
    pred_re_right, pred_im_right = model(branch, coords_right)
    grad_re_left = torch.autograd.grad(pred_re_left.sum(), coords_left, create_graph=True)[0][:, 0:1]
    grad_re_right = torch.autograd.grad(pred_re_right.sum(), coords_right, create_graph=True)[0][:, 0:1]
    grad_im_left = torch.autograd.grad(pred_im_left.sum(), coords_left, create_graph=True)[0][:, 0:1]
    grad_im_right = torch.autograd.grad(pred_im_right.sum(), coords_right, create_graph=True)[0][:, 0:1]
    return torch.mean(
        (pred_re_left - pred_re_right) ** 2
        + (pred_im_left - pred_im_right) ** 2
        + (grad_re_left - grad_re_right) ** 2
        + (grad_im_left - grad_im_right) ** 2
    )


def compute_interface_alignment_loss(model, prev_stage_model, branch, coords):
    if prev_stage_model is None:
        return torch.zeros((), dtype=torch.float32, device=branch.device)
    pred_re, pred_im = model(branch, coords)
    with torch.no_grad():
        prev_re, prev_im = prev_stage_model(branch, coords)
    return torch.mean((pred_re - prev_re) ** 2 + (pred_im - prev_im) ** 2)


def compute_param_smoothness_loss(model, branch_np, coords, cfg_dict, device):
    variable = free_variable_name(cfg_dict)
    variable_idx = PARAM_INDEX[variable]
    if len(branch_np) == 0:
        return torch.zeros((), dtype=torch.float32, device=device)

    if variable in EQ_PARAM_KEYS:
        low, high = cfg_dict["physics"]["equation_params"][variable]
    elif variable in BOUND_PARAM_KEYS:
        low, high = cfg_dict["physics"]["bounds"][variable]
    else:
        raise KeyError(f"Variable lissee non supportee: {variable}")
    low = float(low)
    high = float(high)
    if high <= low + 1.0e-12:
        return torch.zeros((), dtype=torch.float32, device=device)

    delta_fraction = float(cfg_dict["training"].get("smooth_delta_fraction", 0.05))
    delta_abs = max((high - low) * delta_fraction, 1.0e-4)
    signs = np.where(np.random.random(len(branch_np)) < 0.5, -1.0, 1.0).astype(np.float32)
    perturbed_np = np.array(branch_np, copy=True)
    perturbed_np[:, variable_idx] = np.clip(perturbed_np[:, variable_idx] + signs * delta_abs, low, high)
    delta = perturbed_np[:, variable_idx] - branch_np[:, variable_idx]
    delta = np.where(np.abs(delta) < 1.0e-6, np.sign(signs) * 1.0e-6, delta).astype(np.float32)

    branch = torch.tensor(branch_np, dtype=torch.float32, device=device)
    branch_perturbed = torch.tensor(perturbed_np, dtype=torch.float32, device=device)
    pred_re, pred_im = model(branch, coords)
    pred_re_pert, pred_im_pert = model(branch_perturbed, coords)
    delta_t = torch.tensor(delta[:, None], dtype=torch.float32, device=device)
    return torch.mean(((pred_re_pert - pred_re) ** 2 + (pred_im_pert - pred_im) ** 2) / (delta_t ** 2 + 1.0e-6))


def build_stage_query_batches(train_pool, focus_pool, anchor_teachers, cfg_dict, t_start, t_end, device, rng):
    train_queries = int(cfg_dict["training"]["train_queries"])
    boundary_queries = int(cfg_dict["training"]["boundary_queries"])
    continuity_queries = int(cfg_dict["training"].get("continuity_queries", max(512, train_queries // 4)))
    overlap_queries = int(cfg_dict["training"].get("overlap_queries", max(512, train_queries // 4)))
    smooth_queries = int(cfg_dict["training"].get("smooth_queries", max(512, train_queries // 8)))

    branch_main_np = sample_case_rows(train_pool, focus_pool, anchor_teachers, cfg_dict, train_queries, rng)
    branch_boundary_np = sample_case_rows(train_pool, focus_pool, anchor_teachers, cfg_dict, boundary_queries, rng)
    branch_interface_np = sample_case_rows(train_pool, focus_pool, anchor_teachers, cfg_dict, continuity_queries, rng)
    branch_overlap_np = sample_case_rows(train_pool, focus_pool, anchor_teachers, cfg_dict, overlap_queries, rng)
    branch_smooth_np = sample_case_rows(train_pool, focus_pool, anchor_teachers, cfg_dict, smooth_queries, rng)

    branch_main, coords_main = build_query_batch(branch_main_np, cfg_dict, t_start, t_end, device, rng)
    branch_boundary, coords_left, coords_right = build_boundary_batch(branch_boundary_np, cfg_dict, t_start, t_end, device, rng)
    branch_interface, coords_interface = build_interface_batch(branch_interface_np, cfg_dict, t_start, device, rng)

    overlap_width = float(teacher_cfg(cfg_dict).get("overlap_width", 0.0))
    overlap_end = min(float(t_end), float(t_start) + max(overlap_width, 0.0))
    if overlap_end <= float(t_start) + 1.0e-12:
        branch_overlap, coords_overlap = branch_interface, coords_interface
    else:
        branch_overlap, coords_overlap = build_query_batch(branch_overlap_np, cfg_dict, t_start, overlap_end, device, rng)

    branch_smooth, coords_smooth = build_query_batch(branch_smooth_np, cfg_dict, t_start, t_end, device, rng)
    return {
        "branch_main_np": branch_main_np,
        "branch_main": branch_main,
        "coords_main": coords_main,
        "branch_boundary_np": branch_boundary_np,
        "branch_boundary": branch_boundary,
        "coords_left": coords_left,
        "coords_right": coords_right,
        "branch_interface_np": branch_interface_np,
        "branch_interface": branch_interface,
        "coords_interface": coords_interface,
        "branch_overlap_np": branch_overlap_np,
        "branch_overlap": branch_overlap,
        "coords_overlap": coords_overlap,
        "branch_smooth_np": branch_smooth_np,
        "branch_smooth": branch_smooth,
        "coords_smooth": coords_smooth,
    }


def compute_weighted_losses(model, prev_stage_model, anchor_teachers, batches, cfg_dict, device):
    variable_idx = free_variable_index(cfg_dict)
    weights = cfg_dict["training"]["loss_weights"]

    loss_anchor, _ = compute_anchor_distillation_loss(
        model,
        anchor_teachers,
        batches["branch_main"],
        batches["coords_main"],
        batches["branch_main_np"][:, variable_idx],
        device,
    )
    loss_boundary = compute_boundary_loss(
        model,
        batches["branch_boundary"],
        batches["coords_left"],
        batches["coords_right"],
    )
    loss_continuity = compute_interface_alignment_loss(
        model,
        prev_stage_model,
        batches["branch_interface"],
        batches["coords_interface"],
    )
    loss_overlap = compute_interface_alignment_loss(
        model,
        prev_stage_model,
        batches["branch_overlap"],
        batches["coords_overlap"],
    )
    param_smooth_weight = float(weights.get("param_smooth", 0.0))
    if param_smooth_weight > 0.0:
        loss_param_smooth = compute_param_smoothness_loss(
            model,
            batches["branch_smooth_np"],
            batches["coords_smooth"],
            cfg_dict,
            device,
        )
    else:
        loss_param_smooth = torch.zeros((), dtype=torch.float32, device=device)

    total = (
        float(weights.get("anchor", 1.0)) * loss_anchor
        + float(weights.get("boundary", 0.0)) * loss_boundary
        + float(weights.get("continuity", 0.0)) * loss_continuity
        + float(weights.get("overlap", 0.0)) * loss_overlap
        + param_smooth_weight * loss_param_smooth
    )
    return total, {
        "anchor": loss_anchor,
        "boundary": loss_boundary,
        "continuity": loss_continuity,
        "overlap": loss_overlap,
        "param_smooth": loss_param_smooth,
    }


def save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name):
    ckpt_dir = os.path.join(stage_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_valid_loss": float(best_valid_loss),
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
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_valid_loss", float("inf")))


def load_best_stage_model(cfg_dict, stage_dir, device):
    for name in ("model_best.pth", "model_final.pth", "model_latest.pth"):
        ckpt_path = os.path.join(stage_dir, "checkpoints", name)
        if os.path.exists(ckpt_path):
            payload = torch.load(ckpt_path, map_location=device)
            model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            return model
    raise FileNotFoundError(f"Aucun checkpoint stage dans {stage_dir}")


def maybe_warm_start_stage(model, cfg_dict, prev_stage_dir, device):
    if prev_stage_dir is None or not bool(cfg_dict.get("warm_start_interstage", {}).get("enabled", True)):
        return False
    for name in ("model_best.pth", "model_final.pth", "model_latest.pth"):
        ckpt_path = os.path.join(prev_stage_dir, "checkpoints", name)
        if os.path.exists(ckpt_path):
            payload = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(payload["model_state"], strict=True)
            print(f"    ♻️ Warm-start depuis {ckpt_path}")
            return True
    return False


def load_prev_stage_model(cfg_dict, prev_stage_dir, device):
    if prev_stage_dir is None:
        return None
    try:
        return load_best_stage_model(cfg_dict, prev_stage_dir, device)
    except FileNotFoundError:
        return None


def eval_stage_valid_loss(model, prev_stage_model, valid_pool, focus_pool, anchor_teachers, cfg_dict, t_start, t_end, device):
    rng = np.random.default_rng(4242)
    values = []
    model.eval()
    for _ in range(int(cfg_dict["training"].get("valid_eval_batches", 3))):
        batches = build_stage_query_batches(valid_pool, focus_pool, anchor_teachers, cfg_dict, t_start, t_end, device, rng)
        with torch.enable_grad():
            total, _ = compute_weighted_losses(model, prev_stage_model, anchor_teachers, batches, cfg_dict, device)
        values.append(float(total.detach().item()))
    return float(np.mean(values))


def train_one_stage(
    model,
    optimizer,
    train_pool,
    focus_pool,
    valid_pool,
    anchor_teachers,
    cfg_dict,
    t_start,
    t_end,
    stage_dir,
    device,
    stage_idx,
    prev_stage_dir=None,
):
    training_cfg = cfg_dict["training"]
    stage_epoch_overrides = training_cfg.get("stage_epoch_overrides", [])
    num_epochs = int(stage_epoch_overrides[stage_idx] if stage_idx < len(stage_epoch_overrides) else training_cfg["stage_num_epochs"])
    log_every = int(training_cfg["log_every"])
    eval_every = int(training_cfg["eval_every"])
    snapshot_every = int(training_cfg["snapshot_every"])
    grad_clip = float(training_cfg["grad_clip"])
    early_cfg = training_cfg.get("early_stop", {})
    start_epoch, best_valid_loss = load_stage_checkpoint_if_available(model, optimizer, stage_dir, device)
    if start_epoch == 0:
        maybe_warm_start_stage(model, cfg_dict, prev_stage_dir, device)
    prev_stage_model = load_prev_stage_model(cfg_dict, prev_stage_dir, device)

    stage_start_perf = time.perf_counter()
    stage_rng = np.random.default_rng(int(training_cfg.get("seed_offset", 0)) + 1000 * stage_idx + 17)
    lr_decay_step = int(training_cfg.get("lr_decay_step", 4000))
    lr_decay_gamma = float(training_cfg.get("lr_decay_gamma", 0.85))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, lr_decay_step), gamma=lr_decay_gamma)
    plateau_count = 0
    final_valid_loss = float("inf")
    min_delta_rel = float(early_cfg.get("min_delta_rel", 0.005))
    patience_evals = int(early_cfg.get("patience_evals", 4))
    min_epochs = int(early_cfg.get("min_epochs", max(1000, eval_every)))

    print(f"🔁 Reprise stage={os.path.basename(stage_dir)} epoch={start_epoch} best_valid={best_valid_loss:.6e}")
    current_epoch = start_epoch
    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        batches = build_stage_query_batches(train_pool, focus_pool, anchor_teachers, cfg_dict, t_start, t_end, device, stage_rng)
        optimizer.zero_grad(set_to_none=True)
        total, losses = compute_weighted_losses(model, prev_stage_model, anchor_teachers, batches, cfg_dict, device)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        current_epoch = epoch

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"[{os.path.basename(stage_dir)} | epoch {epoch}] total={float(total.item()):.3e} "
                f"| anchor={float(losses['anchor'].item()):.3e} "
                f"| bc={float(losses['boundary'].item()):.3e} "
                f"| cont={float(losses['continuity'].item()):.3e} "
                f"| overlap={float(losses['overlap'].item()):.3e} "
                f"| smooth={float(losses['param_smooth'].item()):.3e}"
            )

        if epoch % eval_every == 0 or epoch == num_epochs:
            final_valid_loss = eval_stage_valid_loss(
                model,
                prev_stage_model,
                valid_pool,
                focus_pool,
                anchor_teachers,
                cfg_dict,
                t_start,
                t_end,
                device,
            )
            print(f"    📏 valid_loss={final_valid_loss:.3e}")
            save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, "model_latest.pth")
            improved = final_valid_loss < best_valid_loss * (1.0 - min_delta_rel) or not np.isfinite(best_valid_loss)
            if improved:
                best_valid_loss = final_valid_loss
                save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, "model_best.pth")
                plateau_count = 0
                print(f"    ✅ Nouveau meilleur valid_loss : {best_valid_loss:.3e}")
            else:
                plateau_count += 1

            if (
                bool(early_cfg.get("enabled", True))
                and epoch >= min_epochs
                and plateau_count >= patience_evals
            ):
                print("    ⏹️ Plateau detecte, arret anticipe du stage.")
                break

        if epoch % snapshot_every == 0:
            save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, f"ckpt_epoch_{epoch:06d}.pth")

    if os.path.exists(os.path.join(stage_dir, "checkpoints", "model_best.pth")):
        best_payload = torch.load(os.path.join(stage_dir, "checkpoints", "model_best.pth"), map_location=device)
        model.load_state_dict(best_payload["model_state"], strict=True)

    final_valid_loss = eval_stage_valid_loss(
        model,
        prev_stage_model,
        valid_pool,
        focus_pool,
        anchor_teachers,
        cfg_dict,
        t_start,
        t_end,
        device,
    )
    save_stage_checkpoint(model, optimizer, current_epoch, best_valid_loss, stage_dir, "model_final.pth")
    with open(os.path.join(stage_dir, "stage_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "stage_idx": int(stage_idx),
                "t_start": float(t_start),
                "t_end": float(t_end),
                "best_valid_loss": float(best_valid_loss),
                "final_valid_loss": float(final_valid_loss),
                "completed_epochs": int(current_epoch),
            },
            handle,
            indent=2,
        )
    return {
        "wall_seconds": max(0.0, time.perf_counter() - stage_start_perf),
        "best_valid_loss": float(best_valid_loss),
        "final_valid_loss": float(final_valid_loss),
    }


def build_eval_cases(cfg_dict):
    base = base_case_defaults(cfg_dict)
    cases = []
    for entry in cfg_dict.get("evaluation", {}).get("cases", []):
        params = dict(base)
        params.update({key: entry[key] for key in PARAM_KEYS if key in entry})
        label = entry.get("label")
        if label is None:
            raise ValueError("Chaque evaluation case doit definir label.")
        cases.append({"label": str(label), "params": params})
    return cases


def solve_case(cfg_dict, params):
    x_min, x_max = map(float, cfg_dict["physics"]["x_domain"])
    t_max = float(cfg_dict["physics"]["t_max"])
    nx = int(cfg_dict["data"].get("solver_nx", 256))
    dt = float(cfg_dict["data"].get("solver_dt", 0.025))
    n_steps = int(round(t_max / dt))
    X, T, U = get_ground_truth_CGL(params, x_min, x_max, t_max, Nx=nx, Nt=n_steps + 1)
    return {
        "params": params,
        "branch_vec": branch_vector_from_params(params),
        "x": X[:, 0].astype(np.float32),
        "t_values": T[0, :].astype(np.float32),
        "u": U.astype(np.complex64),
    }


def select_stage_model(models, time_blocks, t_current):
    for (t_start, t_end), model in zip(time_blocks, models):
        if float(t_current) <= float(t_end) + 1.0e-10:
            return model
    return models[-1]


def rollout_multistage_models(models, time_blocks, case_ref, device):
    x = case_ref["x"]
    t_values = case_ref["t_values"]
    u_true = case_ref["u"]
    branch_vec = case_ref["branch_vec"]
    u_pred = np.zeros_like(u_true)

    for idx_t, t_val in enumerate(t_values):
        model = select_stage_model(models, time_blocks, float(t_val))
        coords = torch.tensor(np.stack([x, np.full_like(x, t_val)], axis=1), dtype=torch.float32, device=device)
        branch = torch.tensor(np.repeat(branch_vec[None, :], len(x), axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_re, pred_im = model(branch, coords)
        u_pred[:, idx_t] = (pred_re[:, 0] + 1j * pred_im[:, 0]).cpu().numpy().astype(np.complex64)

    rel_l2 = np.zeros(len(t_values), dtype=np.float64)
    for idx_t in range(len(t_values)):
        denom = np.linalg.norm(u_true[:, idx_t]) + 1.0e-12
        rel_l2[idx_t] = np.linalg.norm(u_pred[:, idx_t] - u_true[:, idx_t]) / denom
    return {"t_values": t_values, "rel_l2": rel_l2}


def evaluate_case_suite(run_dir, cfg_dict, models, time_blocks, device):
    eval_cases = build_eval_cases(cfg_dict)
    if not eval_cases:
        return
    out_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for case in eval_cases:
        label = case["label"]
        reference = solve_case(cfg_dict, case["params"])
        rollout = rollout_multistage_models(models, time_blocks, reference, device)
        case_dir = os.path.join(out_dir, label)
        os.makedirs(case_dir, exist_ok=True)
        csv_path = os.path.join(case_dir, "rollout_metrics.csv")
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("time,rel_l2\n")
            for t_val, err in zip(rollout["t_values"], rollout["rel_l2"]):
                handle.write(f"{float(t_val):.8f},{float(err):.10f}\n")

        first_over = np.where(np.asarray(rollout["rel_l2"]) > 0.05)[0]
        first_t = "none" if len(first_over) == 0 else float(rollout["t_values"][int(first_over[0])])
        with open(os.path.join(case_dir, "summary.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"final_rel_l2={float(rollout['rel_l2'][-1]):.10f}\n")
            handle.write(f"max_rel_l2={float(np.max(rollout['rel_l2'])):.10f}\n")
            handle.write(f"mean_rel_l2={float(np.mean(rollout['rel_l2'])):.10f}\n")
            handle.write(f"first_t_gt_5pct={first_t}\n")
            handle.write(f"metrics_csv={csv_path}\n")

        rows.append(
            {
                "label": label,
                "final_rel_l2": float(rollout["rel_l2"][-1]),
                "max_rel_l2": float(np.max(rollout["rel_l2"])),
                "mean_rel_l2": float(np.mean(rollout["rel_l2"])),
                "first_t_gt_5pct": first_t,
            }
        )

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "final_rel_l2", "max_rel_l2", "mean_rel_l2", "first_t_gt_5pct"])
        writer.writeheader()
        writer.writerows(rows)


def load_time_blocks(cfg_dict):
    return [tuple(map(float, block)) for block in cfg_dict["multistage"]["time_blocks"]]


def load_stage_summary(stage_dir):
    path = os.path.join(stage_dir, "stage_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stage_is_complete(stage_dir):
    return os.path.exists(os.path.join(stage_dir, "checkpoints", "model_final.pth")) and os.path.exists(
        os.path.join(stage_dir, "stage_summary.json")
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

    anchor_teachers = load_anchor_teachers(cfg_dict, device)
    save_teacher_manifest(run_dir, anchor_teachers)

    ds_cfg = cfg_dict["parametric_dataset"]
    train_pool = build_param_pool(cfg_dict, int(ds_cfg["train_cases"]), int(ds_cfg["seed"]))
    focus_pool = build_focus_pool(cfg_dict)
    valid_pool = build_param_pool(cfg_dict, int(ds_cfg["valid_cases"]), int(ds_cfg["seed"]) + 1000)
    save_case_pool_csv(os.path.join(run_dir, "train_cases.csv"), train_pool)
    if focus_pool:
        save_case_pool_csv(os.path.join(run_dir, "focus_cases.csv"), focus_pool)
    save_case_pool_csv(os.path.join(run_dir, "valid_cases.csv"), valid_pool)

    time_blocks = load_time_blocks(cfg_dict)
    force_retrain_stage_indices = {int(value) for value in cfg_dict["training"].get("force_retrain_stage_indices", [])}
    stage_rows = []

    try:
        for stage_idx, (t_start, t_end) in enumerate(time_blocks):
            stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
            os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)

            if stage_is_complete(stage_dir) and stage_idx not in force_retrain_stage_indices:
                summary = load_stage_summary(stage_dir)
                print(f"\n⏭️ Stage {stage_idx + 1}/{len(time_blocks)} deja termine.")
                stage_rows.append(
                    {
                        "stage_idx": stage_idx,
                        "stage_label": f"{t_start:.2f}_{t_end:.2f}",
                        "t_start": float(t_start),
                        "t_end": float(t_end),
                        "best_valid_loss": float(summary["best_valid_loss"]),
                        "final_valid_loss": float(summary["final_valid_loss"]),
                        "wall_seconds": 0.0,
                    }
                )
                continue

            model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg_dict["training"]["learning_rate"]),
                weight_decay=float(cfg_dict["training"]["weight_decay"]),
            )
            prev_stage_dir = None
            if stage_idx > 0:
                prev_stage_dir = os.path.join(run_dir, stage_name(stage_idx - 1, *time_blocks[stage_idx - 1]))

            print(f"\n🚧 Stage {stage_idx + 1}/{len(time_blocks)} | bloc=[{t_start:.2f}, {t_end:.2f}]")
            metrics = train_one_stage(
                model,
                optimizer,
                train_pool,
                focus_pool,
                valid_pool,
                anchor_teachers,
                cfg_dict,
                t_start,
                t_end,
                stage_dir,
                device,
                stage_idx,
                prev_stage_dir=prev_stage_dir,
            )
            stage_rows.append(
                {
                    "stage_idx": stage_idx,
                    "stage_label": f"{t_start:.2f}_{t_end:.2f}",
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "best_valid_loss": float(metrics["best_valid_loss"]),
                    "final_valid_loss": float(metrics["final_valid_loss"]),
                    "wall_seconds": float(metrics["wall_seconds"]),
                }
            )
            write_stage_manifest(run_dir, stage_rows)

        stage_models = [
            load_best_stage_model(cfg_dict, os.path.join(run_dir, stage_name(stage_idx, t_start, t_end)), device)
            for stage_idx, (t_start, t_end) in enumerate(time_blocks)
        ]
        evaluate_case_suite(run_dir, cfg_dict, stage_models, time_blocks, device)
        write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "completed", stage_rows)
        print("\n🏁 Global multireseau parametrique distille termine")
    except Exception:
        write_stage_manifest(run_dir, stage_rows)
        write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "failed", stage_rows)
        raise


if __name__ == "__main__":
    main()
