import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.data.local_operator_amp_phase import (
    build_branch_features,
    interp_complex_field,
    prepare_single_case_trajectory,
    sample_training_batch,
    save_rollout_metrics,
)
from src.models.cgl_local_deeponet_amp_phase import CGL_LocalDirect_DeepONet_AmpPhase


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
    print(f"⏱️ Timing summary : {timing_path}")


def stage_name(stage_idx, t_start, t_end):
    return f"stage_{stage_idx:02d}_t{t_start:.1f}_{t_end:.1f}"


def load_time_blocks(cfg_dict):
    return [tuple(map(float, block)) for block in cfg_dict["multistage"]["time_blocks"]]


def load_rollout_windows(cfg_dict):
    windows_cfg = cfg_dict["multistage"].get("rollout_windows")
    windows = load_time_blocks(cfg_dict) if windows_cfg is None else [tuple(map(float, block)) for block in windows_cfg]
    time_blocks = load_time_blocks(cfg_dict)
    if len(windows) != len(time_blocks):
        raise ValueError("multistage.rollout_windows doit avoir la meme longueur que multistage.time_blocks")
    for idx, (rollout_window, time_block) in enumerate(zip(windows, time_blocks)):
        rs, re = rollout_window
        ts, te = time_block
        if rs < ts - 1e-10 or re > te + 1e-10:
            raise ValueError(
                f"rollout_window[{idx}]={rollout_window} doit etre inclus dans time_block[{idx}]={time_block}"
            )
    return windows


def stage_markers_from_windows(windows):
    if len(windows) <= 1:
        return []
    return [float(window[1]) for window in windows[:-1]]


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


def maybe_warm_start_stage(model, cfg_dict, stage_idx, run_dir, device):
    warm_cfg = cfg_dict.get("warm_start_interstage", {})
    if not bool(warm_cfg.get("enabled", False)):
        return None
    if stage_idx <= 0:
        return None

    source_name = str(warm_cfg.get("checkpoint_name", "model_best.pth"))
    fallback_names = [str(name) for name in warm_cfg.get("fallback_checkpoint_names", ["model_final.pth", "model_latest.pth"])]

    prev_t_start, prev_t_end = load_time_blocks(cfg_dict)[stage_idx - 1]
    prev_stage_dir = os.path.join(run_dir, stage_name(stage_idx - 1, prev_t_start, prev_t_end))
    candidate_paths = [os.path.join(prev_stage_dir, "checkpoints", source_name)]
    candidate_paths.extend(os.path.join(prev_stage_dir, "checkpoints", name) for name in fallback_names)

    ckpt_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if ckpt_path is None:
        print(f"ℹ️ Warm-start ignore pour stage_{stage_idx:02d} : aucun checkpoint precedent trouve dans {prev_stage_dir}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"🔗 Warm-start stage_{stage_idx:02d} depuis {ckpt_path}")
    return ckpt_path


def compute_losses(model, batch, amp_floor):
    delta_log_amp, delta_phase = model(batch["branch"], batch["x_query"])
    next_amp = torch.exp(torch.log(batch["current_amp"] + amp_floor) + delta_log_amp) - amp_floor
    next_phase = batch["current_phase"] + delta_phase
    pred_re = next_amp * torch.cos(next_phase)
    pred_im = next_amp * torch.sin(next_phase)

    loss_complex = torch.mean((pred_re - batch["target_u_re"]) ** 2 + (pred_im - batch["target_u_im"]) ** 2)
    loss_amp = torch.mean((next_amp - batch["target_amp"]) ** 2)
    loss_delta = torch.mean((delta_log_amp - batch["target_delta_log_amp"]) ** 2 + (delta_phase - batch["target_delta_phase"]) ** 2)
    return loss_complex, loss_amp, loss_delta


def stage_pair_indices(trajectory, t_start, t_end):
    t_values = trajectory["t_values"]
    dt = trajectory["dt"]
    pair_times = t_values[:-1]
    mask = (pair_times >= t_start - 1e-10) & (pair_times < t_end - dt / 2.0 + 1e-10)
    idx = np.nonzero(mask)[0].astype(np.int64)
    if len(idx) == 0:
        raise ValueError(f"Aucune paire trouvee pour le bloc [{t_start}, {t_end}]")
    return idx


def split_indices(indices, train_ratio):
    n_total = len(indices)
    if n_total == 1:
        return indices.copy(), indices.copy()
    n_train = max(1, min(n_total - 1, int(round(train_ratio * n_total))))
    train_idx = indices[:n_train].copy()
    valid_idx = indices[n_train:].copy()
    if len(valid_idx) == 0:
        valid_idx = train_idx[-1:].copy()
    return train_idx, valid_idx


def eval_stage_valid_loss(model, trajectory, cfg_dict, valid_idx, device, valid_batches):
    amp_floor = float(cfg_dict["local_operator"]["amp_floor"])
    weights = cfg_dict["training"]["loss_weights"]
    vals = []
    model.eval()
    with torch.no_grad():
        for _ in range(valid_batches):
            batch = sample_training_batch(
                trajectory,
                cfg_dict,
                pair_indices=valid_idx,
                batch_pairs=int(cfg_dict["training"]["batch_pairs"]),
                batch_queries=int(cfg_dict["training"]["batch_queries"]),
                device=device,
            )
            loss_complex, loss_amp, loss_delta = compute_losses(model, batch, amp_floor)
            loss = (
                float(weights["complex"]) * loss_complex
                + float(weights["amplitude"]) * loss_amp
                + float(weights["delta"]) * loss_delta
            )
            vals.append(float(loss.item()))
    return float(np.mean(vals))


def load_best_stage_model(cfg_dict, stage_dir, device):
    ckpt_path = os.path.join(stage_dir, "checkpoints", "model_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CGL_LocalDirect_DeepONet_AmpPhase(cfg_dict).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def select_stage_model(models, rollout_windows, t_current):
    for (t_start, t_end), model in zip(rollout_windows, models):
        if t_current < t_end - 1e-10:
            return model
    return models[-1]


def _predict_next_sensor(model, current_sensor, x_sensor, dt_value, params, cfg_dict, device, amp_floor):
    branch_vec = build_branch_features(cfg_dict, current_sensor, dt_value, params)
    branch_tensor = torch.tensor(branch_vec[None, :], dtype=torch.float32, device=device)
    branch_sensor = branch_tensor.repeat(len(x_sensor), 1)
    x_sensor_tensor = torch.tensor(x_sensor[:, None], dtype=torch.float32, device=device)
    delta_log_amp_sensor, delta_phase_sensor = model(branch_sensor, x_sensor_tensor)

    current_amp_sensor = np.abs(current_sensor)
    current_phase_sensor = np.angle(current_sensor)
    next_amp_sensor = np.exp(
        np.log(current_amp_sensor + amp_floor) + delta_log_amp_sensor.cpu().numpy().reshape(-1)
    ) - amp_floor
    next_phase_sensor = current_phase_sensor + delta_phase_sensor.cpu().numpy().reshape(-1)
    return next_amp_sensor * np.exp(1j * next_phase_sensor)


def _active_stage_indices(time_blocks, t_current):
    active = []
    for idx, (t_start, t_end) in enumerate(time_blocks):
        if t_start - 1e-10 <= t_current < t_end - 1e-10:
            active.append(idx)
    if not active:
        return [len(time_blocks) - 1]
    return active


def _blend_stage_predictions(models, time_blocks, current_sensor, x_sensor, dt_value, params, cfg_dict, device, amp_floor, t_current):
    active = _active_stage_indices(time_blocks, t_current)
    if len(active) == 1:
        return _predict_next_sensor(models[active[0]], current_sensor, x_sensor, dt_value, params, cfg_dict, device, amp_floor)

    if len(active) == 2:
        left_idx, right_idx = active
        left_start, left_end = time_blocks[left_idx]
        right_start, right_end = time_blocks[right_idx]
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        denom = max(overlap_end - overlap_start, 1.0e-12)
        alpha = min(max((t_current - overlap_start) / denom, 0.0), 1.0)
        left_pred = _predict_next_sensor(models[left_idx], current_sensor, x_sensor, dt_value, params, cfg_dict, device, amp_floor)
        right_pred = _predict_next_sensor(models[right_idx], current_sensor, x_sensor, dt_value, params, cfg_dict, device, amp_floor)
        return (1.0 - alpha) * left_pred + alpha * right_pred

    preds = [
        _predict_next_sensor(models[idx], current_sensor, x_sensor, dt_value, params, cfg_dict, device, amp_floor)
        for idx in active
    ]
    return np.mean(np.stack(preds, axis=0), axis=0)


def rollout_multistage_models(models, rollout_windows, trajectory, cfg_dict, device, blend_mode="switch", training_blocks=None):
    params = trajectory["params"]
    x_sensor = trajectory["x_sensor"]
    x_solver = trajectory["x_solver"]
    u_sensor_ref = trajectory["u_sensor"]
    u_solver_ref = trajectory["u_solver"]
    dt_value = trajectory["dt"]
    periodic = trajectory["periodic"]
    amp_floor = float(cfg_dict["local_operator"]["amp_floor"])

    n_steps = u_solver_ref.shape[1] - 1
    pred_sensor = np.zeros_like(u_sensor_ref)
    pred_sensor[:, 0] = u_sensor_ref[:, 0]
    pred_solver = np.zeros_like(u_solver_ref)
    pred_solver[:, 0] = u_solver_ref[:, 0]
    rel_l2 = np.zeros(n_steps + 1, dtype=np.float64)

    with torch.no_grad():
        for step in range(n_steps):
            t_current = float(trajectory["t_values"][step])
            current_sensor = pred_sensor[:, step]
            if blend_mode == "overlap_linear":
                if training_blocks is None:
                    raise ValueError("training_blocks est requis pour blend_mode='overlap_linear'")
                pred_sensor[:, step + 1] = _blend_stage_predictions(
                    models,
                    training_blocks,
                    current_sensor,
                    x_sensor,
                    dt_value,
                    params,
                    cfg_dict,
                    device,
                    amp_floor,
                    t_current,
                )
            else:
                model = select_stage_model(models, rollout_windows, t_current)
                pred_sensor[:, step + 1] = _predict_next_sensor(
                    model,
                    current_sensor,
                    x_sensor,
                    dt_value,
                    params,
                    cfg_dict,
                    device,
                    amp_floor,
                )
            pred_solver[:, step + 1] = interp_complex_field(x_sensor, pred_sensor[:, step + 1], x_solver, periodic)
            diff = pred_solver[:, step + 1] - u_solver_ref[:, step + 1]
            denom = np.linalg.norm(u_solver_ref[:, step + 1]) + 1e-12
            rel_l2[step + 1] = np.linalg.norm(diff) / denom

    return {
        "x_sensor": x_sensor,
        "x_solver": x_solver,
        "t_values": trajectory["t_values"],
        "u_sensor_pred": pred_sensor,
        "u_solver_pred": pred_solver,
        "u_solver_ref": u_solver_ref,
        "rel_l2": rel_l2,
    }


def plot_rollout_curve(rollout, output_path, stage_markers=None):
    plt.figure(figsize=(8, 4.5))
    plt.plot(rollout["t_values"], rollout["rel_l2"], color="#c2185b", linewidth=2.0)
    for marker in stage_markers or []:
        plt.axvline(marker, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title("Local multistage : rollout complet vs solveur classique")
    plt.grid(alpha=0.25)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def train_one_stage(model, optimizer, trajectory, cfg_dict, train_idx, valid_idx, stage_dir, device):
    amp_floor = float(cfg_dict["local_operator"]["amp_floor"])
    weights = cfg_dict["training"]["loss_weights"]
    num_epochs = int(cfg_dict["training"]["stage_num_epochs"])
    log_every = int(cfg_dict["training"]["log_every"])
    eval_every = int(cfg_dict["training"]["eval_every"])
    snapshot_every = int(cfg_dict["training"]["snapshot_every"])
    grad_clip = float(cfg_dict["training"]["grad_clip"])
    valid_batches = int(cfg_dict["training"]["valid_batches"])

    start_epoch, best_valid_loss = load_stage_checkpoint_if_available(model, optimizer, stage_dir, device)
    print(f"🔁 Reprise stage={os.path.basename(stage_dir)} epoch={start_epoch} best_valid={best_valid_loss:.6e}")
    stage_start_perf = time.perf_counter()

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
        loss_complex, loss_amp, loss_delta = compute_losses(model, batch, amp_floor)
        loss = (
            float(weights["complex"]) * loss_complex
            + float(weights["amplitude"]) * loss_amp
            + float(weights["delta"]) * loss_delta
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"[{os.path.basename(stage_dir)} | Epoch {epoch}] loss={loss.item():.3e} "
                f"| complex={loss_complex.item():.3e} | amp={loss_amp.item():.3e} | delta={loss_delta.item():.3e}"
            )

        if epoch % eval_every == 0 or epoch == num_epochs:
            valid_loss = eval_stage_valid_loss(model, trajectory, cfg_dict, valid_idx, device, valid_batches)
            print(f"    📏 valid_loss={valid_loss:.3e}")
            save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name="model_latest.pth")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name="model_best.pth")
                print(f"    ✅ Nouveau meilleur valid_loss : {best_valid_loss:.3e}")

        if epoch % snapshot_every == 0:
            save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name=f"ckpt_epoch_{epoch:06d}.pth")

    save_stage_checkpoint(model, optimizer, num_epochs, best_valid_loss, stage_dir, name="model_final.pth")
    return {
        "wall_seconds": max(0.0, time.perf_counter() - stage_start_perf),
        "best_valid_loss": float(best_valid_loss),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_single_case_local_multistage_amp_phase_alpha075_beta0_mu0_t5.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    run_dir = build_run_dir(PROJECT_DIR, cfg_dict, resume_dir=args.resume)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "rollout"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_dt = datetime.now()
    start_perf = time.perf_counter()

    print(f"📱 Device : {device}")
    print(f"📂 Run dir : {run_dir}")

    trajectory = prepare_single_case_trajectory(cfg_dict)
    time_blocks = load_time_blocks(cfg_dict)
    rollout_windows = load_rollout_windows(cfg_dict)
    stage_rows = []

    for stage_idx, (t_start, t_end) in enumerate(time_blocks):
        stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
        os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
        block_indices = stage_pair_indices(trajectory, t_start, t_end)
        train_idx, valid_idx = split_indices(block_indices, float(cfg_dict["training"]["train_split"]))
        model = CGL_LocalDirect_DeepONet_AmpPhase(cfg_dict).to(device)
        stage_latest_ckpt = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
        if not os.path.exists(stage_latest_ckpt):
            maybe_warm_start_stage(model, cfg_dict, stage_idx, run_dir, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg_dict["training"]["learning_rate"]),
            weight_decay=float(cfg_dict["training"]["weight_decay"]),
        )
        print(f"\n🚧 Stage {stage_idx + 1}/{len(time_blocks)} | bloc=[{t_start}, {t_end}] | n_pairs={len(block_indices)}")
        stage_metrics = train_one_stage(model, optimizer, trajectory, cfg_dict, train_idx, valid_idx, stage_dir, device)
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
    rollout = rollout_multistage_models(stage_models, rollout_windows, trajectory, cfg_dict, device)
    csv_path = save_rollout_metrics(os.path.join(run_dir, "rollout"), rollout)
    plot_rollout_curve(
        rollout,
        os.path.join(run_dir, "rollout", "rollout_rel_l2.png"),
        stage_markers=stage_markers_from_windows(rollout_windows),
    )
    with open(os.path.join(run_dir, "rollout", "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"final_rel_l2={float(rollout['rel_l2'][-1]):.10f}\n")
        handle.write(f"max_rel_l2={float(np.max(rollout['rel_l2'])):.10f}\n")
        handle.write(f"mean_rel_l2={float(np.mean(rollout['rel_l2'])):.10f}\n")
        handle.write(f"metrics_csv={csv_path}\n")
    write_timing_summary(run_dir, start_dt.isoformat(timespec="seconds"), start_perf, "completed", stage_rows)
    print(f"\n🏁 Local multistage termine | final_rel_l2={float(rollout['rel_l2'][-1]):.4%}")


if __name__ == "__main__":
    main()
