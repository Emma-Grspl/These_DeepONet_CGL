import argparse
import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.data.local_operator_amp_phase import (
    build_single_case_params,
    interp_complex_field,
    normalize_linear,
    normalize_log,
    prepare_single_case_trajectory,
    rollout_local_model,
    sample_training_batch,
    save_rollout_metrics,
    split_pair_indices,
)
from src.models.cgl_local_deeponet_amp_phase import CGL_LocalDirect_DeepONet_AmpPhase


class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return self._dict[item]

    def get(self, key, default=None):
        return self._dict.get(key, default)


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def tensor_is_finite(value):
    return bool(torch.isfinite(torch.real(value)).all().item() and torch.isfinite(torch.imag(value)).all().item())


def scalar_is_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(float(value)))


def write_failure_summary(run_dir, epoch, reason, metrics=None):
    os.makedirs(os.path.join(run_dir, "audits"), exist_ok=True)
    path = os.path.join(run_dir, "audits", "failure_summary.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"epoch={int(epoch)}\n")
        handle.write(f"reason={reason}\n")
        if metrics:
            for key, value in metrics.items():
                handle.write(f"{key}={value}\n")
    print(f"🛑 Stop non-fini epoch={epoch}: {reason} | summary={path}")


def save_checkpoint(model, optimizer, epoch, best_score, run_dir, name="model_latest.pth"):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    atomic_torch_save(state, os.path.join(ckpt_dir, name))


def load_checkpoint_if_available(model, optimizer, run_dir, device):
    ckpt_path = os.path.join(run_dir, "checkpoints", "model_latest.pth")
    if not os.path.exists(ckpt_path):
        return 0, float("inf")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_score", float("inf")))


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


def complex_to_feature_parts_torch(u_complex, amp_floor):
    amp = torch.abs(u_complex)
    phase = torch.angle(u_complex)
    log_amp = torch.log(amp + amp_floor)
    return log_amp, torch.cos(phase), torch.sin(phase)


def build_params_norm(cfg_dict):
    params = build_single_case_params(cfg_dict)
    eq = cfg_dict["physics"]["equation_params"]
    bounds = cfg_dict["physics"]["bounds"]
    values = [
        normalize_linear(params["alpha"], eq["alpha"][0], eq["alpha"][1]),
        normalize_linear(params["beta"], eq["beta"][0], eq["beta"][1]),
        normalize_linear(params["mu"], eq["mu"][0], eq["mu"][1]),
        normalize_linear(params["V"], eq["V"][0], eq["V"][1]),
        normalize_linear(params["A"], bounds["A"][0], bounds["A"][1]),
        normalize_log(params["w0"], bounds["w0"][0], bounds["w0"][1]),
        normalize_linear(params["x0"], bounds["x0"][0], bounds["x0"][1]),
        normalize_linear(params["k"], bounds["k"][0], bounds["k"][1]),
        normalize_linear(float(params["type"]), 0.0, 2.0),
    ]
    return np.asarray(values, dtype=np.float32)


def build_branch_features_torch(sensor_state, params_norm, amp_floor, dt_norm):
    log_amp, cos_phase, sin_phase = complex_to_feature_parts_torch(sensor_state, amp_floor)
    return torch.cat([log_amp, cos_phase, sin_phase, params_norm, dt_norm], dim=0)


def interp_complex_field_torch(x_src, u_src, x_dst, periodic):
    re_src = torch.real(u_src)
    im_src = torch.imag(u_src)
    dx = x_src[1] - x_src[0]
    n = x_src.numel()

    if periodic:
        x0 = x_src[0]
        xi = torch.remainder((x_dst - x0) / dx, n)
        i0 = torch.floor(xi).long()
        w = (xi - i0.to(xi.dtype)).unsqueeze(1)
        i1 = (i0 + 1) % n
    else:
        xi = torch.clamp((x_dst - x_src[0]) / dx, 0.0, n - 1 - 1.0e-6)
        i0 = torch.floor(xi).long()
        i1 = torch.clamp(i0 + 1, max=n - 1)
        w = (xi - i0.to(xi.dtype)).unsqueeze(1)

    re = (1.0 - w[:, 0]) * re_src[i0] + w[:, 0] * re_src[i1]
    im = (1.0 - w[:, 0]) * im_src[i0] + w[:, 0] * im_src[i1]
    return torch.complex(re, im)


def predict_next_state(model, sensor_state, x_coords, x_sensor, params_norm, amp_floor, dt_norm, periodic):
    branch_features = build_branch_features_torch(sensor_state, params_norm, amp_floor, dt_norm)
    branch = branch_features.unsqueeze(0).repeat(x_coords.shape[0], 1)
    delta_log_amp, delta_phase = model(branch, x_coords.unsqueeze(1))
    current_state = interp_complex_field_torch(x_sensor, sensor_state, x_coords, periodic)
    current_amp = torch.abs(current_state).unsqueeze(1)
    current_phase = torch.angle(current_state).unsqueeze(1)
    next_amp = torch.exp(torch.log(current_amp + amp_floor) + delta_log_amp) - amp_floor
    next_phase = current_phase + delta_phase
    return next_amp[:, 0] * torch.exp(1j * next_phase[:, 0])


def compute_one_step_losses(model, batch, amp_floor):
    delta_log_amp, delta_phase = model(batch["branch"], batch["x_query"])
    next_amp = torch.exp(torch.log(batch["current_amp"] + amp_floor) + delta_log_amp) - amp_floor
    next_phase = batch["current_phase"] + delta_phase
    pred_re = next_amp * torch.cos(next_phase)
    pred_im = next_amp * torch.sin(next_phase)

    loss_complex = torch.mean((pred_re - batch["target_u_re"]) ** 2 + (pred_im - batch["target_u_im"]) ** 2)
    loss_amp = torch.mean((next_amp - batch["target_amp"]) ** 2)
    loss_delta = torch.mean(
        (delta_log_amp - batch["target_delta_log_amp"]) ** 2 + (delta_phase - batch["target_delta_phase"]) ** 2
    )
    return loss_complex, loss_amp, loss_delta


def compute_multistep_rollout_loss(model, trajectory, cfg_dict, pair_indices, device, params_norm):
    train_cfg = cfg_dict["training"]
    amp_floor = float(cfg_dict["local_operator"]["amp_floor"])
    horizons = [int(v) for v in train_cfg["rollout_horizons_steps"]]
    horizon_weights = np.asarray(train_cfg["rollout_horizon_weights"], dtype=np.float32)
    horizon_weights = horizon_weights / np.clip(np.sum(horizon_weights), 1.0e-12, None)
    n_starts = int(train_cfg["rollout_batch_starts"])

    max_h = max(horizons)
    valid_starts = np.asarray([idx for idx in pair_indices if idx + max_h < trajectory["u_sensor"].shape[1]], dtype=np.int64)
    if valid_starts.size == 0:
        raise ValueError("Aucun start valide pour la rollout loss.")
    chosen_starts = np.random.choice(valid_starts, size=n_starts, replace=True)

    x_sensor = torch.tensor(trajectory["x_sensor"], dtype=torch.float32, device=device)
    periodic = bool(trajectory["periodic"])
    dt_norm = torch.tensor(
        [normalize_linear(float(trajectory["dt"]), 0.0, float(cfg_dict["local_operator"]["rollout_dt"]))],
        dtype=torch.float32,
        device=device,
    )

    loss_terms = []
    per_horizon = {int(h): [] for h in horizons}
    for start_idx in chosen_starts:
        current_sensor = torch.tensor(trajectory["u_sensor"][:, start_idx], dtype=torch.complex64, device=device)
        for step in range(1, max_h + 1):
            current_sensor = predict_next_state(
                model,
                current_sensor,
                x_sensor,
                x_sensor,
                params_norm,
                amp_floor,
                dt_norm,
                periodic,
            )
            if not tensor_is_finite(current_sensor):
                return torch.full((), float("nan"), dtype=torch.float32, device=device), per_horizon
            if step in per_horizon:
                target_sensor = torch.tensor(
                    trajectory["u_sensor"][:, start_idx + step],
                    dtype=torch.complex64,
                    device=device,
                )
                diff = current_sensor - target_sensor
                rel_num = torch.mean(torch.abs(diff) ** 2)
                rel_den = torch.mean(torch.abs(target_sensor) ** 2) + 1.0e-12
                rel_l2 = rel_num / rel_den
                amp_loss = torch.mean((torch.abs(current_sensor) - torch.abs(target_sensor)) ** 2)
                combined = rel_l2 + 0.25 * amp_loss
                if not scalar_is_finite(combined):
                    return torch.full((), float("nan"), dtype=torch.float32, device=device), per_horizon
                per_horizon[step].append(rel_l2.detach().item())
                loss_terms.append(float(horizon_weights[horizons.index(step)]) * combined)

    return torch.stack(loss_terms).mean(), per_horizon


def rollout_weight_multiplier(epoch, cfg_dict):
    schedule = cfg_dict["training"].get("rollout_schedule", {})
    if not schedule or not bool(schedule.get("enabled", False)):
        return 1.0
    start_epoch = int(schedule.get("start_epoch", 1))
    ramp_epochs = max(1, int(schedule.get("ramp_epochs", 1)))
    max_multiplier = float(schedule.get("max_multiplier", 1.0))
    if epoch < start_epoch:
        return 0.0
    progress = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
    return max_multiplier * progress


def make_audit_starts(total_pairs, max_horizon, audit_start_count):
    upper = max(1, total_pairs - max_horizon + 1)
    candidates = np.arange(0, upper, dtype=np.int64)
    if candidates.size == 0:
        return np.asarray([0], dtype=np.int64)
    if candidates.size <= audit_start_count:
        return candidates
    return np.linspace(0, candidates[-1], audit_start_count, dtype=np.int64)


def run_short_horizon_audit(model, trajectory, cfg_dict, device, params_norm):
    horizons = [int(v) for v in cfg_dict["training"]["rollout_horizons_steps"]]
    max_h = max(horizons)
    n_pairs = trajectory["u_solver"].shape[1] - 1
    audit_starts = make_audit_starts(
        n_pairs,
        max_h,
        int(cfg_dict["training"]["audit_start_count"]),
    )

    x_sensor = trajectory["x_sensor"]
    x_solver = trajectory["x_solver"]
    periodic = bool(trajectory["periodic"])
    amp_floor = float(cfg_dict["local_operator"]["amp_floor"])
    dt_norm_np = np.asarray(
        [normalize_linear(float(trajectory["dt"]), 0.0, float(cfg_dict["local_operator"]["rollout_dt"]))],
        dtype=np.float32,
    )

    one_step_errors = []
    first_step_error = np.nan
    closed_errors = {int(h): [] for h in horizons}

    model.eval()
    with torch.no_grad():
        for start_idx in audit_starts:
            true_sensor = trajectory["u_sensor"][:, start_idx]
            branch_features = np.concatenate(
                [
                    np.log(np.abs(true_sensor) + amp_floor).astype(np.float32),
                    np.cos(np.angle(true_sensor)).astype(np.float32),
                    np.sin(np.angle(true_sensor)).astype(np.float32),
                    params_norm.cpu().numpy(),
                    dt_norm_np,
                ],
                axis=0,
            )
            branch_tensor = torch.tensor(branch_features[None, :], dtype=torch.float32, device=device)

            branch_solver = branch_tensor.repeat(len(x_solver), 1)
            x_solver_tensor = torch.tensor(x_solver[:, None], dtype=torch.float32, device=device)
            delta_log_amp, delta_phase = model(branch_solver, x_solver_tensor)
            current_solver = trajectory["u_solver"][:, start_idx]
            current_amp = np.abs(current_solver)
            current_phase = np.angle(current_solver)
            next_amp = np.exp(np.log(current_amp + amp_floor) + delta_log_amp.cpu().numpy().reshape(-1)) - amp_floor
            next_phase = current_phase + delta_phase.cpu().numpy().reshape(-1)
            pred_next = next_amp * np.exp(1j * next_phase)
            target_next = trajectory["u_solver"][:, start_idx + 1]
            one_step_rel = np.linalg.norm(pred_next - target_next) / (np.linalg.norm(target_next) + 1.0e-12)
            one_step_errors.append(float(one_step_rel))
            if start_idx == 0:
                first_step_error = float(one_step_rel)

            pred_sensor = np.array(true_sensor, copy=True)
            for step in range(1, max_h + 1):
                branch_features = np.concatenate(
                    [
                        np.log(np.abs(pred_sensor) + amp_floor).astype(np.float32),
                        np.cos(np.angle(pred_sensor)).astype(np.float32),
                        np.sin(np.angle(pred_sensor)).astype(np.float32),
                        params_norm.cpu().numpy(),
                        dt_norm_np,
                    ],
                    axis=0,
                )
                branch_tensor = torch.tensor(branch_features[None, :], dtype=torch.float32, device=device)
                branch_sensor = branch_tensor.repeat(len(x_sensor), 1)
                x_sensor_tensor = torch.tensor(x_sensor[:, None], dtype=torch.float32, device=device)
                delta_log_amp_sensor, delta_phase_sensor = model(branch_sensor, x_sensor_tensor)
                curr_amp_sensor = np.abs(pred_sensor)
                curr_phase_sensor = np.angle(pred_sensor)
                next_amp_sensor = (
                    np.exp(np.log(curr_amp_sensor + amp_floor) + delta_log_amp_sensor.cpu().numpy().reshape(-1)) - amp_floor
                )
                next_phase_sensor = curr_phase_sensor + delta_phase_sensor.cpu().numpy().reshape(-1)
                pred_sensor = next_amp_sensor * np.exp(1j * next_phase_sensor)

                if step in closed_errors:
                    pred_solver = interp_complex_field(x_sensor, pred_sensor, x_solver, periodic)
                    target_solver = trajectory["u_solver"][:, start_idx + step]
                    rel = np.linalg.norm(pred_solver - target_solver) / (np.linalg.norm(target_solver) + 1.0e-12)
                    closed_errors[step].append(float(rel))

    metrics = {
        "teacher_forced_one_step_mean_rel_l2": float(np.mean(one_step_errors)),
        "teacher_forced_one_step_max_rel_l2": float(np.max(one_step_errors)),
        "teacher_forced_first_step_rel_l2": float(first_step_error),
    }
    for horizon in horizons:
        values = np.asarray(closed_errors[horizon], dtype=np.float64)
        metrics[f"closed_h{horizon}_mean_rel_l2"] = float(np.mean(values))
        metrics[f"closed_h{horizon}_max_rel_l2"] = float(np.max(values))
    return metrics


def write_audit_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_audit_history(rows, output_dir):
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    threshold = rows[-1]["l2_threshold"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epochs, [row["teacher_forced_one_step_mean_rel_l2"] for row in rows], label="one-step teacher forced", linewidth=2.0)
    for key in [k for k in rows[0].keys() if k.startswith("closed_h") and k.endswith("_mean_rel_l2")]:
        axes[0].plot(epochs, [row[key] for row in rows], label=key.replace("_mean_rel_l2", ""), linewidth=1.8)
    axes[0].axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("L2 relative")
    axes[0].set_title("Audits horizons courts")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(epochs, [row["rollout_final_rel_l2"] for row in rows], label="rollout final", linewidth=2.0)
    axes[1].plot(epochs, [row["rollout_max_rel_l2"] for row in rows], label="rollout max", linewidth=2.0)
    axes[1].plot(epochs, [row["teacher_forced_first_step_rel_l2"] for row in rows], label="first step teacher forced", linewidth=1.8)
    axes[1].axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L2 relative")
    axes[1].set_title("Rollout complet vs premier pas")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "audit_history.png"), dpi=220)
    plt.close()


def plot_rollout_curve(rollout, output_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(rollout["t_values"], rollout["rel_l2"], color="#c2185b", linewidth=2.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title("Rollout local direct harmonized : erreur relative vs solveur classique")
    plt.grid(alpha=0.25)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def rollout_first_t_over_threshold(rollout, threshold):
    above = np.where(np.asarray(rollout["rel_l2"]) > float(threshold))[0]
    if above.size == 0:
        return np.nan
    return float(np.asarray(rollout["t_values"])[above[0]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_single_case_local_direct_harmonized_amp_phase_alpha075_beta0_mu0_t5.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    run_dir = build_run_dir(PROJECT_DIR, cfg_dict, resume_dir=args.resume)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "rollout"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "audits"), exist_ok=True)

    cfg = ConfigObj(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")
    print(f"📂 Run dir : {run_dir}")

    trajectory = prepare_single_case_trajectory(cfg_dict)
    train_idx, _ = split_pair_indices(trajectory, float(cfg_dict["training"]["train_split"]))
    params_norm = torch.tensor(build_params_norm(cfg_dict), dtype=torch.float32, device=device)

    model = CGL_LocalDirect_DeepONet_AmpPhase(cfg_dict).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_dict["training"]["learning_rate"]),
        weight_decay=float(cfg_dict["training"]["weight_decay"]),
    )
    start_epoch, best_score = load_checkpoint_if_available(model, optimizer, run_dir, device)
    print(f"🔁 Reprise epoch={start_epoch} | best_score={best_score:.6e}")

    amp_floor = float(cfg_dict["local_operator"]["amp_floor"])
    weights = cfg_dict["training"]["loss_weights"]
    num_epochs = int(cfg_dict["training"]["num_epochs"])
    log_every = int(cfg_dict["training"]["log_every"])
    audit_every = int(cfg_dict["training"]["audit_every"])
    snapshot_every = int(cfg_dict["training"]["snapshot_every"])
    grad_clip = float(cfg_dict["training"]["grad_clip"])
    l2_threshold = float(cfg_dict["evaluation"]["l2_threshold"])

    audit_rows = []
    audit_csv_path = os.path.join(run_dir, "audits", "audit_history.csv")
    if os.path.exists(audit_csv_path):
        with open(audit_csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
                audit_rows.append(parsed)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        batch = sample_training_batch(
            trajectory,
            cfg_dict,
            pair_indices=train_idx,
            batch_pairs=int(cfg_dict["training"]["batch_pairs"]),
            batch_queries=int(cfg_dict["training"]["batch_queries"]),
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_complex, loss_amp, loss_delta = compute_one_step_losses(model, batch, amp_floor)
        rollout_mult = rollout_weight_multiplier(epoch, cfg_dict)
        if rollout_mult > 0.0:
            loss_rollout, rollout_per_horizon = compute_multistep_rollout_loss(
                model,
                trajectory,
                cfg_dict,
                pair_indices=train_idx,
                device=device,
                params_norm=params_norm,
            )
        else:
            loss_rollout = torch.zeros((), dtype=torch.float32, device=device)
            rollout_per_horizon = {int(v): [] for v in cfg_dict["training"]["rollout_horizons_steps"]}
        loss = (
            float(weights["complex"]) * loss_complex
            + float(weights["amplitude"]) * loss_amp
            + float(weights["delta"]) * loss_delta
            + rollout_mult * float(weights["rollout"]) * loss_rollout
        )
        if not scalar_is_finite(loss):
            write_failure_summary(
                run_dir,
                epoch,
                "non_finite_training_loss",
                {
                    "loss_complex": float(loss_complex.detach().cpu()) if scalar_is_finite(loss_complex) else "nan",
                    "loss_amp": float(loss_amp.detach().cpu()) if scalar_is_finite(loss_amp) else "nan",
                    "loss_delta": float(loss_delta.detach().cpu()) if scalar_is_finite(loss_delta) else "nan",
                    "loss_rollout": float(loss_rollout.detach().cpu()) if scalar_is_finite(loss_rollout) else "nan",
                    "rollout_multiplier": rollout_mult,
                },
            )
            save_checkpoint(model, optimizer, epoch, best_score, run_dir, name="model_failed.pth")
            return
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all().item():
                write_failure_summary(run_dir, epoch, f"non_finite_parameter:{name}")
                save_checkpoint(model, optimizer, epoch, best_score, run_dir, name="model_failed.pth")
                return

        if epoch % log_every == 0 or epoch == 1:
            rollout_msg = " | ".join(
                f"h{h}={np.mean(vals):.3e}" for h, vals in sorted(rollout_per_horizon.items()) if len(vals) > 0
            )
            print(
                f"[Epoch {epoch}] loss={loss.item():.3e}"
                f" | complex={loss_complex.item():.3e}"
                f" | amp={loss_amp.item():.3e}"
                f" | delta={loss_delta.item():.3e}"
                f" | rollout={loss_rollout.item():.3e}"
                f" | rollout_mult={rollout_mult:.3e}"
                f" | {rollout_msg}"
            )

        if epoch % audit_every == 0 or epoch == num_epochs:
            short_metrics = run_short_horizon_audit(model, trajectory, cfg_dict, device, params_norm)
            rollout = rollout_local_model(model, trajectory, cfg_dict, device)
            csv_path = save_rollout_metrics(os.path.join(run_dir, "rollout"), rollout)
            plot_rollout_curve(rollout, os.path.join(run_dir, "rollout", "rollout_rel_l2.png"))

            final_l2 = float(rollout["rel_l2"][-1])
            max_l2 = float(np.max(rollout["rel_l2"]))
            first_t_gt = rollout_first_t_over_threshold(rollout, l2_threshold)
            selection_score = final_l2 + 0.5 * short_metrics["closed_h8_mean_rel_l2"]
            if not np.isfinite(selection_score):
                row = {
                    "epoch": float(epoch),
                    "train_total_loss": float(loss.item()) if scalar_is_finite(loss) else np.nan,
                    "train_one_step_complex": float(loss_complex.item()) if scalar_is_finite(loss_complex) else np.nan,
                    "train_one_step_amp": float(loss_amp.item()) if scalar_is_finite(loss_amp) else np.nan,
                    "train_one_step_delta": float(loss_delta.item()) if scalar_is_finite(loss_delta) else np.nan,
                    "train_rollout_loss": float(loss_rollout.item()) if scalar_is_finite(loss_rollout) else np.nan,
                    "rollout_weight_multiplier": float(rollout_mult),
                    "rollout_final_rel_l2": final_l2,
                    "rollout_max_rel_l2": max_l2,
                    "rollout_first_t_gt_threshold": float(first_t_gt) if not np.isnan(first_t_gt) else np.nan,
                    "selection_score": np.nan,
                    "l2_threshold": float(l2_threshold),
                }
                row.update(short_metrics)
                audit_rows.append(row)
                write_audit_csv(audit_rows, audit_csv_path)
                write_failure_summary(run_dir, epoch, "non_finite_audit_or_rollout", row)
                save_checkpoint(model, optimizer, epoch, best_score, run_dir, name="model_failed.pth")
                return

            row = {
                "epoch": float(epoch),
                "train_total_loss": float(loss.item()),
                "train_one_step_complex": float(loss_complex.item()),
                "train_one_step_amp": float(loss_amp.item()),
                "train_one_step_delta": float(loss_delta.item()),
                "train_rollout_loss": float(loss_rollout.item()),
                "rollout_weight_multiplier": float(rollout_mult),
                "rollout_final_rel_l2": final_l2,
                "rollout_max_rel_l2": max_l2,
                "rollout_first_t_gt_threshold": float(first_t_gt) if not np.isnan(first_t_gt) else np.nan,
                "selection_score": float(selection_score),
                "l2_threshold": float(l2_threshold),
            }
            row.update(short_metrics)
            audit_rows.append(row)
            write_audit_csv(audit_rows, audit_csv_path)
            plot_audit_history(audit_rows, os.path.join(run_dir, "audits"))

            with open(os.path.join(run_dir, "audits", "latest_summary.txt"), "w", encoding="utf-8") as handle:
                for key, value in row.items():
                    handle.write(f"{key}={value}\n")
                handle.write(f"rollout_metrics_csv={csv_path}\n")

            print(
                f"    🔎 Audit epoch={epoch}"
                f" | one-step={short_metrics['teacher_forced_one_step_mean_rel_l2']:.4%}"
                f" | h8={short_metrics['closed_h8_mean_rel_l2']:.4%}"
                f" | rollout_final={final_l2:.4%}"
                f" | rollout_max={max_l2:.4%}"
                f" | first_t_gt={first_t_gt}"
            )

            save_checkpoint(model, optimizer, epoch, best_score, run_dir, name="model_latest.pth")
            if selection_score < best_score:
                best_score = selection_score
                save_checkpoint(model, optimizer, epoch, best_score, run_dir, name="model_best.pth")
                print(f"    ✅ Nouveau meilleur score audit : {best_score:.6e}")

        if epoch % snapshot_every == 0:
            save_checkpoint(model, optimizer, epoch, best_score, run_dir, name=f"ckpt_epoch_{epoch:06d}.pth")

    save_checkpoint(model, optimizer, num_epochs, best_score, run_dir, name="model_final.pth")
    print(f"🏁 Entraînement terminé | best_score={best_score:.6e}")


if __name__ == "__main__":
    main()
