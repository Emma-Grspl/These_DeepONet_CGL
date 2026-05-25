import argparse
import csv
import os
import sys
import time
import copy
from datetime import datetime

import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.models.cgl_deeponet_amp_phase import CGL_PI_DeepONet_AmpPhase
from src.utils.solver_cgl import get_ground_truth_CGL


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def build_run_dir(project_root, cfg_dict, resume_dir=None):
    configured = cfg_dict["training"]["save_dir"]
    output_root = configured if os.path.isabs(configured) else os.path.join(project_root, configured)
    os.makedirs(output_root, exist_ok=True)
    if resume_dir is not None:
        return resume_dir if os.path.isabs(resume_dir) else os.path.join(project_root, resume_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    run_name = f"run_{timestamp}" if not job_id else f"run_{timestamp}_{job_id}"
    return os.path.join(output_root, run_name)


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
            handle.write(f"{prefix}_seconds={row['wall_seconds']:.6f}\n")
            handle.write(f"{prefix}_hours={row['wall_seconds'] / 3600.0:.6f}\n")
            handle.write(f"{prefix}_best_valid_loss={row['best_valid_loss']:.10e}\n")

    csv_path = os.path.join(run_dir, "timing_stages.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("stage_idx,stage_label,wall_seconds,wall_hours,best_valid_loss\n")
        for row in stage_rows:
            handle.write(
                f"{int(row['stage_idx'])},{row['stage_label']},{row['wall_seconds']:.6f},"
                f"{row['wall_seconds'] / 3600.0:.6f},{row['best_valid_loss']:.10e}\n"
            )


def stage_name(stage_idx, t_start, t_end):
    return f"stage_{stage_idx:02d}_t{t_start:.1f}_{t_end:.1f}"


def _sample_uniform(low, high, rng):
    if abs(float(high) - float(low)) < 1e-12:
        return float(low)
    return float(rng.uniform(float(low), float(high)))


def _base_case_defaults(cfg_dict):
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


def sample_case_params(cfg_dict, rng):
    physics = cfg_dict["physics"]
    eq = physics["equation_params"]
    bounds = physics["bounds"]
    params = {
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
    return params


def curriculum_cfg(cfg_dict):
    return cfg_dict.get("parametric_curriculum", {})


def curriculum_enabled(cfg_dict):
    return bool(curriculum_cfg(cfg_dict).get("enabled", False))


def apply_param_overrides(cfg_dict, phase_params):
    cfg_copy = copy.deepcopy(cfg_dict)
    physics = cfg_copy["physics"]
    for family_name in ("equation_params", "bounds"):
        overrides = phase_params.get(family_name, {})
        target = physics[family_name]
        for key, value in overrides.items():
            if key not in target:
                raise KeyError(f"Parametre de curriculum inconnu: {family_name}.{key}")
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"Chaque override de curriculum doit etre une paire [min, max] pour {family_name}.{key}")
            target[key] = [float(value[0]), float(value[1])]
    return cfg_copy


def build_curriculum_train_pools(cfg_dict):
    ds_cfg = cfg_dict["parametric_dataset"]
    base_seed = int(ds_cfg["seed"])
    n_cases = int(ds_cfg["train_cases"])
    cur_cfg = curriculum_cfg(cfg_dict)
    phases = cur_cfg.get("phases", [])
    if not curriculum_enabled(cfg_dict) or not phases:
        base_pool = build_case_pool(cfg_dict, n_cases, base_seed)
        return [{"end_epoch": int(cfg_dict["training"]["stage_num_epochs"]), "pool": base_pool, "phase_idx": 0}]

    built = []
    for phase_idx, phase in enumerate(phases):
        end_epoch = int(phase["end_epoch"])
        phase_params = phase.get("physics", {})
        phase_cfg = apply_param_overrides(cfg_dict, phase_params)
        phase_pool = build_case_pool(phase_cfg, n_cases, base_seed + 100 * phase_idx)
        built.append({"end_epoch": end_epoch, "pool": phase_pool, "phase_idx": phase_idx})
    return built


def active_curriculum_pool(curriculum_pools, epoch):
    for phase in curriculum_pools:
        if int(epoch) <= int(phase["end_epoch"]):
            return phase
    return curriculum_pools[-1]


def make_branch_vector(params):
    return np.array(
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


def solve_case(cfg_dict, params):
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    t_max = float(cfg_dict["physics"]["t_max"])
    nx = int(cfg_dict["data"]["solver_nx"])
    dt = float(cfg_dict.get("data", {}).get("solver_dt", 0.025))
    n_steps = int(round(t_max / dt))
    X, T, U = get_ground_truth_CGL(params, x_min, x_max, t_max, Nx=nx, Nt=n_steps + 1)
    return {
        "params": params,
        "branch_vec": make_branch_vector(params),
        "x": X[:, 0].astype(np.float32),
        "t_values": T[0, :].astype(np.float32),
        "u": U.astype(np.complex64),
    }


def build_case_pool(cfg_dict, n_cases, seed):
    rng = np.random.default_rng(int(seed))
    return [solve_case(cfg_dict, sample_case_params(cfg_dict, rng)) for _ in range(int(n_cases))]


def save_case_pool_csv(path, pool):
    fieldnames = ["case_idx", "alpha", "beta", "mu", "V", "A", "w0", "x0", "k", "type"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, case in enumerate(pool):
            row = {"case_idx": idx}
            row.update(case["params"])
            writer.writerow(row)


def spatial_focus_cfg(cfg_dict):
    return cfg_dict.get("spatial_focus_sampling", {})


def spatial_focus_enabled(cfg_dict):
    cfg = spatial_focus_cfg(cfg_dict)
    return bool(cfg.get("enabled", False))


def temporal_focus_cfg(cfg_dict):
    return cfg_dict.get("temporal_focus_sampling", {})


def temporal_focus_enabled(cfg_dict):
    cfg = temporal_focus_cfg(cfg_dict)
    return bool(cfg.get("enabled", False))


def sample_spatial_index(x, spatial_cfg=None):
    if not spatial_cfg or not spatial_cfg.get("enabled", False):
        return int(np.random.randint(0, len(x)))

    center_fraction = float(spatial_cfg.get("center_fraction", 0.8))
    center_fraction = max(0.0, min(1.0, center_fraction))
    center_window = spatial_cfg.get("center_window", [-10.0, 10.0])
    x_min = float(center_window[0])
    x_max = float(center_window[1])

    center_ids = np.nonzero((x >= x_min) & (x <= x_max))[0]
    tail_ids = np.nonzero((x < x_min) | (x > x_max))[0]

    if len(center_ids) == 0 or len(tail_ids) == 0:
        return int(np.random.randint(0, len(x)))

    if float(np.random.random()) < center_fraction:
        return int(np.random.choice(center_ids))
    return int(np.random.choice(tail_ids))


def sample_time_index(valid_t_idx, temporal_cfg=None):
    if len(valid_t_idx) == 0:
        raise ValueError("valid_t_idx ne doit pas etre vide")
    if not temporal_cfg or not temporal_cfg.get("enabled", False) or len(valid_t_idx) == 1:
        return int(np.random.choice(valid_t_idx))

    start_fraction = float(temporal_cfg.get("start_fraction", 0.7))
    start_fraction = max(0.0, min(1.0, start_fraction))
    window_fraction = float(temporal_cfg.get("start_window_fraction", 0.25))
    window_fraction = max(0.0, min(1.0, window_fraction))

    n_early = max(1, int(np.ceil(len(valid_t_idx) * window_fraction)))
    early_ids = valid_t_idx[:n_early]
    late_ids = valid_t_idx[n_early:]
    if len(late_ids) == 0:
        return int(np.random.choice(early_ids))
    if float(np.random.random()) < start_fraction:
        return int(np.random.choice(early_ids))
    return int(np.random.choice(late_ids))


def sample_block_batch(case_pool, t_start, t_end, n_queries, device, case_ids=None, spatial_cfg=None, temporal_cfg=None):
    if case_ids is None:
        case_ids = np.random.randint(0, len(case_pool), size=int(n_queries))
    else:
        case_ids = np.asarray(case_ids, dtype=np.int64)
    coords = np.zeros((int(n_queries), 2), dtype=np.float32)
    target_re = np.zeros((int(n_queries), 1), dtype=np.float32)
    target_im = np.zeros((int(n_queries), 1), dtype=np.float32)
    branch = np.zeros((int(n_queries), 9), dtype=np.float32)

    for i, case_idx in enumerate(case_ids):
        case = case_pool[int(case_idx)]
        t_values = case["t_values"]
        x = case["x"]
        U = case["u"]
        valid_t_idx = np.nonzero((t_values >= t_start - 1e-10) & (t_values <= t_end + 1e-10))[0]
        tidx = sample_time_index(valid_t_idx, temporal_cfg=temporal_cfg)
        xidx = sample_spatial_index(x, spatial_cfg=spatial_cfg)
        coords[i, 0] = x[xidx]
        coords[i, 1] = t_values[tidx]
        target = U[xidx, tidx]
        target_re[i, 0] = float(np.real(target))
        target_im[i, 0] = float(np.imag(target))
        branch[i, :] = case["branch_vec"]

    return {
        "branch": torch.tensor(branch, dtype=torch.float32, device=device),
        "coords": torch.tensor(coords, dtype=torch.float32, device=device),
        "target_re": torch.tensor(target_re, dtype=torch.float32, device=device),
        "target_im": torch.tensor(target_im, dtype=torch.float32, device=device),
    }


def compute_supervised_loss(model, batch):
    pred_re, pred_im = model(batch["branch"], batch["coords"])
    return torch.mean((pred_re - batch["target_re"]) ** 2 + (pred_im - batch["target_im"]) ** 2)


def hard_case_cfg(cfg_dict):
    return cfg_dict.get("hard_case_sampling", {})


def hard_case_enabled(cfg_dict):
    return bool(hard_case_cfg(cfg_dict).get("enabled", False))


def adaptive_refinement_cfg(cfg_dict):
    return cfg_dict.get("adaptive_stage_refinement", {})


def adaptive_refinement_enabled(cfg_dict):
    return bool(adaptive_refinement_cfg(cfg_dict).get("enabled", False))


def focus_sampling_cfg(cfg_dict):
    return cfg_dict.get("parametric_focus_sampling", {})


def focus_sampling_enabled(cfg_dict):
    cfg = focus_sampling_cfg(cfg_dict)
    return bool(cfg.get("enabled", False) and cfg.get("physics"))


def build_focus_pool(cfg_dict):
    if not focus_sampling_enabled(cfg_dict):
        return []
    fs_cfg = focus_sampling_cfg(cfg_dict)
    ds_cfg = cfg_dict["parametric_dataset"]
    n_cases = int(fs_cfg.get("focus_cases", max(16, int(ds_cfg["train_cases"]) // 2)))
    seed = int(ds_cfg["seed"]) + int(fs_cfg.get("focus_seed_offset", 3000))
    focus_cfg_dict = apply_param_overrides(cfg_dict, fs_cfg["physics"])
    return build_case_pool(focus_cfg_dict, n_cases, seed)


def _concat_batches(batches):
    valid_batches = [batch for batch in batches if batch is not None]
    if not valid_batches:
        return None
    merged = {
        key: torch.cat([batch[key] for batch in valid_batches], dim=0)
        for key in valid_batches[0].keys()
    }
    perm = torch.randperm(merged["branch"].shape[0], device=merged["branch"].device)
    return {key: value[perm] for key, value in merged.items()}


def _sample_pool_batch(case_pool, t_start, t_end, n_queries, device, case_ids=None, spatial_cfg=None, temporal_cfg=None):
    if int(n_queries) <= 0:
        return None
    return sample_block_batch(
        case_pool,
        t_start,
        t_end,
        n_queries,
        device,
        case_ids=case_ids,
        spatial_cfg=spatial_cfg,
        temporal_cfg=temporal_cfg,
    )


def sample_training_batch(train_pool, focus_pool, audit_pool, sampler_state, cfg_dict, t_start, t_end, n_queries, device):
    train_spatial_cfg = spatial_focus_cfg(cfg_dict) if spatial_focus_enabled(cfg_dict) else None
    train_temporal_cfg = temporal_focus_cfg(cfg_dict) if temporal_focus_enabled(cfg_dict) else None
    if not focus_pool:
        use_audit_pool = bool(sampler_state is not None and sampler_state.get("active", False))
        case_ids = sample_case_ids_for_training(
            len(audit_pool) if use_audit_pool else len(train_pool),
            n_queries,
            sampler_state,
        )
        active_pool = audit_pool if use_audit_pool else train_pool
        return sample_block_batch(
            active_pool,
            t_start,
            t_end,
            n_queries,
            device,
            case_ids=case_ids,
            spatial_cfg=train_spatial_cfg,
            temporal_cfg=train_temporal_cfg,
        )

    fs_cfg = focus_sampling_cfg(cfg_dict)
    focus_fraction = float(fs_cfg.get("focus_fraction", 0.3))
    active_hard_fraction = float(fs_cfg.get("active_hard_fraction", 0.2))

    use_hard = bool(
        sampler_state is not None
        and sampler_state.get("active", False)
        and len(sampler_state.get("hard_case_ids", [])) > 0
    )
    hard_fraction = active_hard_fraction if use_hard else 0.0
    focus_fraction = max(0.0, min(1.0, focus_fraction))
    hard_fraction = max(0.0, min(1.0, hard_fraction))
    if focus_fraction + hard_fraction > 1.0:
        scale = 1.0 / (focus_fraction + hard_fraction)
        focus_fraction *= scale
        hard_fraction *= scale

    n_focus = int(round(int(n_queries) * focus_fraction))
    n_hard = int(round(int(n_queries) * hard_fraction))
    n_uniform = int(n_queries) - n_focus - n_hard
    if n_uniform < 0:
        overflow = -n_uniform
        take_from_focus = min(overflow, n_focus)
        n_focus -= take_from_focus
        overflow -= take_from_focus
        if overflow > 0:
            n_hard = max(0, n_hard - overflow)
        n_uniform = int(n_queries) - n_focus - n_hard

    batches = []
    batches.append(_sample_pool_batch(train_pool, t_start, t_end, n_uniform, device, spatial_cfg=train_spatial_cfg, temporal_cfg=train_temporal_cfg))
    batches.append(_sample_pool_batch(focus_pool, t_start, t_end, n_focus, device, spatial_cfg=train_spatial_cfg, temporal_cfg=train_temporal_cfg))
    if use_hard and n_hard > 0:
        hard_ids = np.random.choice(sampler_state["hard_case_ids"], size=n_hard, replace=True)
        batches.append(
            _sample_pool_batch(
                audit_pool,
                t_start,
                t_end,
                n_hard,
                device,
                case_ids=hard_ids,
                spatial_cfg=train_spatial_cfg,
                temporal_cfg=train_temporal_cfg,
            )
        )
    return _concat_batches(batches)


def sample_case_ids_for_training(case_pool_size, n_queries, sampler_state):
    if sampler_state is None:
        return np.random.randint(0, case_pool_size, size=int(n_queries))

    hard_ratio = float(sampler_state["mix_hard_ratio"])
    hard_ids = sampler_state["hard_case_ids"]
    easy_ids = sampler_state["easy_case_ids"]

    if len(hard_ids) == 0 and len(easy_ids) == 0:
        return np.random.randint(0, case_pool_size, size=int(n_queries))
    if len(hard_ids) == 0:
        return np.random.choice(easy_ids, size=int(n_queries), replace=True)
    if len(easy_ids) == 0:
        return np.random.choice(hard_ids, size=int(n_queries), replace=True)

    n_hard = int(round(int(n_queries) * hard_ratio))
    n_hard = max(0, min(int(n_queries), n_hard))
    n_easy = int(n_queries) - n_hard
    picked_hard = np.random.choice(hard_ids, size=n_hard, replace=True)
    picked_easy = np.random.choice(easy_ids, size=n_easy, replace=True)
    merged = np.concatenate([picked_hard, picked_easy], axis=0)
    np.random.shuffle(merged)
    return merged


def block_case_rel_l2(model, case, t_start, t_end, device):
    x = case["x"]
    t_values = case["t_values"]
    u = case["u"]
    branch_vec = case["branch_vec"]
    valid_t_idx = np.nonzero((t_values >= t_start - 1e-10) & (t_values <= t_end + 1e-10))[0]
    t_block = t_values[valid_t_idx]
    u_block = u[:, valid_t_idx]
    xx = np.tile(x, len(t_block))
    tt = np.repeat(t_block, len(x))
    coords = torch.tensor(np.stack([xx, tt], axis=1), dtype=torch.float32, device=device)
    branch = torch.tensor(np.repeat(branch_vec[None, :], len(coords), axis=0), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_re, pred_im = model(branch, coords)
    pred = (pred_re + 1j * pred_im).cpu().numpy().reshape(len(t_block), len(x)).T
    denom = np.linalg.norm(u_block) + 1e-12
    return float(np.linalg.norm(pred - u_block) / denom)


def refresh_hard_case_sampler(model, audit_pool, cfg_dict, t_start, t_end, device, stage_dir, epoch):
    hc_cfg = hard_case_cfg(cfg_dict)
    threshold = float(hc_cfg.get("success_threshold", 0.05))
    activation_mean_threshold = float(hc_cfg.get("activation_mean_threshold", 0.05))
    activation_bad_fraction_threshold = float(hc_cfg.get("activation_bad_fraction_threshold", 0.20))
    activation_max_threshold = float(hc_cfg.get("activation_max_threshold", 0.10))
    hard_fraction = float(hc_cfg.get("hard_fraction", 0.4))
    mix_hard_ratio = float(hc_cfg.get("mix_hard_ratio", 0.7))

    model.eval()
    scores = []
    for case_idx, case in enumerate(audit_pool):
        score = block_case_rel_l2(model, case, t_start, t_end, device)
        scores.append({"case_idx": case_idx, "block_rel_l2": score})
    scores.sort(key=lambda item: item["block_rel_l2"], reverse=True)
    mean_case_rel_l2 = float(np.mean([row["block_rel_l2"] for row in scores]))
    bad_fraction = float(np.mean([row["block_rel_l2"] > threshold for row in scores]))
    max_case_rel_l2 = float(scores[0]["block_rel_l2"])
    hard_sampling_active = bool(
        (mean_case_rel_l2 > activation_mean_threshold)
        or (bad_fraction > activation_bad_fraction_threshold)
        or (max_case_rel_l2 > activation_max_threshold)
    )

    n_hard = max(1, int(round(len(scores) * hard_fraction)))
    hard_ids = np.array([row["case_idx"] for row in scores[:n_hard]], dtype=np.int64)
    easy_ids = np.array([row["case_idx"] for row in scores if row["block_rel_l2"] < threshold], dtype=np.int64)

    audit_dir = os.path.join(stage_dir, "hard_case_sampling")
    os.makedirs(audit_dir, exist_ok=True)
    audit_csv = os.path.join(audit_dir, f"audit_epoch_{int(epoch):06d}.csv")
    with open(audit_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_idx", "block_rel_l2"])
        writer.writeheader()
        writer.writerows(scores)

    summary_path = os.path.join(audit_dir, "latest_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"epoch={int(epoch)}\n")
        handle.write(f"threshold={threshold:.6f}\n")
        handle.write(f"activation_mean_threshold={activation_mean_threshold:.6f}\n")
        handle.write(f"activation_bad_fraction_threshold={activation_bad_fraction_threshold:.6f}\n")
        handle.write(f"activation_max_threshold={activation_max_threshold:.6f}\n")
        handle.write(f"hard_fraction={hard_fraction:.6f}\n")
        handle.write(f"mix_hard_ratio={mix_hard_ratio:.6f}\n")
        handle.write(f"audit_cases={len(scores)}\n")
        handle.write(f"hard_sampling_active={int(hard_sampling_active)}\n")
        handle.write(f"n_validated={len(easy_ids)}\n")
        handle.write(f"n_hard={len(hard_ids)}\n")
        handle.write(f"best_case_rel_l2={scores[-1]['block_rel_l2']:.10f}\n")
        handle.write(f"worst_case_rel_l2={scores[0]['block_rel_l2']:.10f}\n")
        handle.write(f"mean_case_rel_l2={mean_case_rel_l2:.10f}\n")
        handle.write(f"bad_case_fraction={bad_fraction:.10f}\n")
        handle.write(f"max_case_rel_l2={max_case_rel_l2:.10f}\n")
        handle.write(f"audit_csv={audit_csv}\n")

    print(
        "    🎯 hard-case audit "
        f"(epoch {epoch}) | mean={mean_case_rel_l2:.2%} | bad_frac={bad_fraction:.2%} | "
        f"max={max_case_rel_l2:.2%} | active={hard_sampling_active} | "
        f"validated<{threshold:.2%}: {len(easy_ids)}/{len(scores)} | "
        f"hard set: {len(hard_ids)} | worst={scores[0]['block_rel_l2']:.3e} | "
        f"best={scores[-1]['block_rel_l2']:.3e}"
    )
    return {
        "active": hard_sampling_active,
        "hard_case_ids": hard_ids if hard_sampling_active else np.array([], dtype=np.int64),
        "easy_case_ids": easy_ids if hard_sampling_active else np.array([], dtype=np.int64),
        "mix_hard_ratio": mix_hard_ratio if hard_sampling_active else 0.0,
        "mean_case_rel_l2": mean_case_rel_l2,
        "bad_case_fraction": bad_fraction,
        "max_case_rel_l2": max_case_rel_l2,
    }


def refinement_should_trigger(cfg_dict, sampler_state):
    if sampler_state is None:
        return False
    ref_cfg = adaptive_refinement_cfg(cfg_dict)
    hc_cfg = hard_case_cfg(cfg_dict)
    mean_threshold = float(ref_cfg.get("trigger_mean_threshold", hc_cfg.get("activation_mean_threshold", 0.05)))
    bad_fraction_threshold = float(
        ref_cfg.get("trigger_bad_fraction_threshold", hc_cfg.get("activation_bad_fraction_threshold", 0.20))
    )
    max_threshold = float(ref_cfg.get("trigger_max_threshold", hc_cfg.get("activation_max_threshold", 0.10)))
    return bool(
        (float(sampler_state.get("mean_case_rel_l2", 0.0)) > mean_threshold)
        or (float(sampler_state.get("bad_case_fraction", 0.0)) > bad_fraction_threshold)
        or (float(sampler_state.get("max_case_rel_l2", 0.0)) > max_threshold)
    )


def save_refinement_summary(stage_dir, rounds, used_extra_epochs, last_sampler_state):
    out_dir = os.path.join(stage_dir, "adaptive_refinement")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "latest_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"rounds={int(rounds)}\n")
        handle.write(f"used_extra_epochs={int(used_extra_epochs)}\n")
        if last_sampler_state is not None:
            handle.write(f"mean_case_rel_l2={float(last_sampler_state.get('mean_case_rel_l2', 0.0)):.10f}\n")
            handle.write(f"bad_case_fraction={float(last_sampler_state.get('bad_case_fraction', 0.0)):.10f}\n")
            handle.write(f"max_case_rel_l2={float(last_sampler_state.get('max_case_rel_l2', 0.0)):.10f}\n")
            handle.write(f"hard_case_count={len(last_sampler_state.get('hard_case_ids', []))}\n")
            handle.write(f"easy_case_count={len(last_sampler_state.get('easy_case_ids', []))}\n")


def save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name):
    ckpt_dir = os.path.join(stage_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
    }
    atomic_torch_save(state, os.path.join(ckpt_dir, name))


def load_stage_checkpoint_if_available(model, optimizer, stage_dir, device):
    ckpt_path = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
    if not os.path.exists(ckpt_path):
        return 0, float("inf")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_valid_loss", float("inf")))


def eval_block_valid_loss(model, case_pool, t_start, t_end, n_queries, device):
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(4):
            batch = sample_block_batch(case_pool, t_start, t_end, n_queries, device)
            vals.append(float(compute_supervised_loss(model, batch).item()))
    return float(np.mean(vals))


def load_best_stage_model(cfg_dict, stage_dir, device):
    ckpt_path = os.path.join(stage_dir, "checkpoints", "model_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def select_stage_model(models, time_blocks, t_current):
    for (t_start, t_end), model in zip(time_blocks, models):
        if t_current <= t_end + 1e-10:
            return model
    return models[-1]


def rollout_multistage_models(models, time_blocks, case_ref, device):
    x = case_ref["x"]
    t_values = case_ref["t_values"]
    U_true = case_ref["u"]
    branch_vec = case_ref["branch_vec"]
    pred = np.zeros_like(U_true)
    for j, t_val in enumerate(t_values):
        model = select_stage_model(models, time_blocks, float(t_val))
        coords = torch.tensor(np.stack([x, np.full_like(x, t_val)], axis=1), dtype=torch.float32, device=device)
        branch = torch.tensor(np.repeat(branch_vec[None, :], len(x), axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_re, pred_im = model(branch, coords)
        pred[:, j] = (pred_re + 1j * pred_im).cpu().numpy().reshape(-1)

    rel_l2 = np.zeros(len(t_values), dtype=np.float64)
    for j in range(len(t_values)):
        denom = np.linalg.norm(U_true[:, j]) + 1e-12
        rel_l2[j] = np.linalg.norm(pred[:, j] - U_true[:, j]) / denom
    return {"t_values": t_values, "rel_l2": rel_l2}


def train_one_stage(model, optimizer, train_pool, focus_pool, valid_pool, audit_pool, cfg_dict, t_start, t_end, stage_dir, device):
    num_epochs = int(cfg_dict["training"]["stage_num_epochs"])
    log_every = int(cfg_dict["training"]["log_every"])
    eval_every = int(cfg_dict["training"]["eval_every"])
    snapshot_every = int(cfg_dict["training"]["snapshot_every"])
    grad_clip = float(cfg_dict["training"]["grad_clip"])
    n_queries = int(cfg_dict["data"]["train_queries"])
    n_valid_queries = int(cfg_dict["data"]["valid_queries"])
    hc_cfg = hard_case_cfg(cfg_dict)
    use_hard_cases = hard_case_enabled(cfg_dict) and len(audit_pool) > 0
    warmup_epochs = int(hc_cfg.get("warmup_epochs", 5000))
    refresh_every = int(hc_cfg.get("refresh_every", eval_every))
    use_refinement = adaptive_refinement_enabled(cfg_dict) and use_hard_cases
    ref_cfg = adaptive_refinement_cfg(cfg_dict)
    refinement_extra_epochs = int(ref_cfg.get("extra_epochs", 10000))
    refinement_rounds = int(ref_cfg.get("max_refinement_rounds", 1))
    refinement_mix_ratio = float(ref_cfg.get("hard_mix_ratio", hc_cfg.get("mix_hard_ratio", 0.7)))
    refinement_refresh_every = int(ref_cfg.get("refresh_every", refresh_every))

    start_epoch, best_valid_loss = load_stage_checkpoint_if_available(model, optimizer, stage_dir, device)
    print(f"🔁 Reprise stage={os.path.basename(stage_dir)} epoch={start_epoch} best_valid={best_valid_loss:.6e}")
    stage_start_perf = time.perf_counter()
    sampler_state = None
    current_epoch = start_epoch
    refinement_used_epochs = 0
    refinement_used_rounds = 0

    def maybe_run_eval(epoch, force=False):
        nonlocal best_valid_loss
        if not force and epoch % eval_every != 0:
            return
        valid_loss = eval_block_valid_loss(model, valid_pool, t_start, t_end, n_valid_queries, device)
        print(f"    📏 valid_loss={valid_loss:.3e}")
        save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name="model_latest.pth")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name="model_best.pth")
            print(f"    ✅ Nouveau meilleur valid_loss : {best_valid_loss:.3e}")

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        if use_hard_cases and epoch > warmup_epochs and ((epoch - warmup_epochs - 1) % max(1, refresh_every) == 0):
            sampler_state = refresh_hard_case_sampler(model, audit_pool, cfg_dict, t_start, t_end, device, stage_dir, epoch)

        batch = sample_training_batch(train_pool, focus_pool, audit_pool, sampler_state, cfg_dict, t_start, t_end, n_queries, device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_supervised_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            print(f"[{os.path.basename(stage_dir)} | Epoch {epoch}] loss={loss.item():.3e}")

        if epoch % eval_every == 0 or epoch == num_epochs:
            maybe_run_eval(epoch, force=True)

        if epoch % snapshot_every == 0:
            save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name=f"ckpt_epoch_{epoch:06d}.pth")
        current_epoch = epoch

    if use_refinement:
        sampler_state = refresh_hard_case_sampler(
            model,
            audit_pool,
            cfg_dict,
            t_start,
            t_end,
            device,
            stage_dir,
            max(1, current_epoch),
        )
        for round_idx in range(refinement_rounds):
            if not refinement_should_trigger(cfg_dict, sampler_state):
                break
            if len(sampler_state.get("hard_case_ids", [])) == 0:
                break
            refinement_used_rounds += 1
            print(
                f"    🔧 Raffinement adaptatif round {round_idx + 1}/{refinement_rounds} "
                f"| mean={sampler_state['mean_case_rel_l2']:.2%} "
                f"| bad_frac={sampler_state['bad_case_fraction']:.2%} "
                f"| max={sampler_state['max_case_rel_l2']:.2%}"
            )
            refinement_state = {
                "active": True,
                "hard_case_ids": sampler_state["hard_case_ids"],
                "easy_case_ids": sampler_state["easy_case_ids"],
                "mix_hard_ratio": refinement_mix_ratio,
                "mean_case_rel_l2": sampler_state["mean_case_rel_l2"],
                "bad_case_fraction": sampler_state["bad_case_fraction"],
                "max_case_rel_l2": sampler_state["max_case_rel_l2"],
            }
            round_start_epoch = current_epoch
            for epoch in range(round_start_epoch + 1, round_start_epoch + refinement_extra_epochs + 1):
                model.train()
                if (epoch - round_start_epoch - 1) % max(1, refinement_refresh_every) == 0:
                    sampler_state = refresh_hard_case_sampler(
                        model, audit_pool, cfg_dict, t_start, t_end, device, stage_dir, epoch
                    )
                    if len(sampler_state.get("hard_case_ids", [])) > 0:
                        refinement_state["hard_case_ids"] = sampler_state["hard_case_ids"]
                        refinement_state["easy_case_ids"] = sampler_state["easy_case_ids"]
                        refinement_state["mean_case_rel_l2"] = sampler_state["mean_case_rel_l2"]
                        refinement_state["bad_case_fraction"] = sampler_state["bad_case_fraction"]
                        refinement_state["max_case_rel_l2"] = sampler_state["max_case_rel_l2"]

                batch = sample_training_batch(
                    train_pool,
                    focus_pool,
                    audit_pool,
                    refinement_state,
                    cfg_dict,
                    t_start,
                    t_end,
                    n_queries,
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = compute_supervised_loss(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if epoch % log_every == 0 or epoch == round_start_epoch + 1:
                    print(f"[{os.path.basename(stage_dir)} | Refinement Epoch {epoch}] loss={loss.item():.3e}")

                if epoch % eval_every == 0 or epoch == round_start_epoch + refinement_extra_epochs:
                    maybe_run_eval(epoch, force=True)

                if epoch % snapshot_every == 0:
                    save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name=f"ckpt_epoch_{epoch:06d}.pth")
                current_epoch = epoch

            refinement_used_epochs += refinement_extra_epochs
            sampler_state = refresh_hard_case_sampler(
                model, audit_pool, cfg_dict, t_start, t_end, device, stage_dir, max(1, current_epoch)
            )

    save_refinement_summary(stage_dir, refinement_used_rounds, refinement_used_epochs, sampler_state)

    save_stage_checkpoint(model, optimizer, current_epoch, best_valid_loss, stage_dir, name="model_final.pth")
    return {
        "wall_seconds": max(0.0, time.perf_counter() - stage_start_perf),
        "best_valid_loss": float(best_valid_loss),
    }


def build_eval_cases(cfg_dict):
    base = _base_case_defaults(cfg_dict)
    eval_cases = []
    for case_cfg in cfg_dict.get("evaluation", {}).get("cases", []):
        params = dict(base)
        params.update({k: case_cfg[k] for k in base.keys() if k in case_cfg})
        label = case_cfg.get("label")
        if label is None:
            raise ValueError("Chaque evaluation.cases doit definir un champ 'label'.")
        eval_cases.append({"label": label, "params": params})
    return eval_cases


def evaluate_case_suite(run_dir, cfg_dict, models, time_blocks, device):
    eval_cases = build_eval_cases(cfg_dict)
    if not eval_cases:
        return
    out_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for case in eval_cases:
        label = case["label"]
        ref = solve_case(cfg_dict, case["params"])
        rollout = rollout_multistage_models(models, time_blocks, ref, device)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    run_dir = build_run_dir(PROJECT_DIR, cfg_dict, resume_dir=args.resume)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_dt = datetime.now()
    start_perf = time.perf_counter()
    print(f"📱 Device : {device}")
    print(f"📂 Run dir : {run_dir}")

    ds_cfg = cfg_dict["parametric_dataset"]
    train_pool = build_case_pool(cfg_dict, int(ds_cfg["train_cases"]), int(ds_cfg["seed"]))
    focus_pool = build_focus_pool(cfg_dict)
    valid_pool = build_case_pool(cfg_dict, int(ds_cfg["valid_cases"]), int(ds_cfg["seed"]) + 1000)
    audit_cases = int(hard_case_cfg(cfg_dict).get("audit_cases", 0))
    audit_seed_offset = int(hard_case_cfg(cfg_dict).get("audit_seed_offset", 2000))
    audit_pool = build_case_pool(cfg_dict, audit_cases, int(ds_cfg["seed"]) + audit_seed_offset) if hard_case_enabled(cfg_dict) and audit_cases > 0 else []
    save_case_pool_csv(os.path.join(run_dir, "train_cases.csv"), train_pool)
    if focus_pool:
        save_case_pool_csv(os.path.join(run_dir, "focus_cases.csv"), focus_pool)
    save_case_pool_csv(os.path.join(run_dir, "valid_cases.csv"), valid_pool)
    if audit_pool:
        save_case_pool_csv(os.path.join(run_dir, "audit_cases.csv"), audit_pool)

    time_blocks = [tuple(map(float, block)) for block in cfg_dict["multistage"]["time_blocks"]]
    stage_rows = []
    for stage_idx, (t_start, t_end) in enumerate(time_blocks):
        stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
        os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
        model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg_dict["training"]["learning_rate"]),
            weight_decay=float(cfg_dict["training"]["weight_decay"]),
        )
        print(f"\n🚧 Stage {stage_idx + 1}/{len(time_blocks)} | bloc=[{t_start}, {t_end}]")
        stage_metrics = train_one_stage(model, optimizer, train_pool, focus_pool, valid_pool, audit_pool, cfg_dict, t_start, t_end, stage_dir, device)
        stage_rows.append(
            {
                "stage_idx": stage_idx,
                "stage_label": f"{t_start:.1f}_{t_end:.1f}",
                "wall_seconds": float(stage_metrics["wall_seconds"]),
                "best_valid_loss": float(stage_metrics["best_valid_loss"]),
            }
        )

    stage_models = [
        load_best_stage_model(cfg_dict, os.path.join(run_dir, stage_name(stage_idx, t_start, t_end)), device)
        for stage_idx, (t_start, t_end) in enumerate(time_blocks)
    ]
    evaluate_case_suite(run_dir, cfg_dict, stage_models, time_blocks, device)
    write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "completed", stage_rows)
    print("\n🏁 Global multistage parametrique termine")


if __name__ == "__main__":
    main()
