import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from scripts.train_cgl_global_multistage_amp_phase import (
    load_best_stage_model as load_best_global_stage_model,
    prepare_single_case_reference,
    rollout_multistage_models as rollout_global_multistage_models,
    stage_name as global_stage_name,
)
from scripts.train_cgl_local_multistage_amp_phase import (
    load_best_stage_model as load_best_local_stage_model,
    load_rollout_windows as load_local_rollout_windows,
    load_time_blocks as load_local_time_blocks,
    rollout_multistage_models as rollout_local_multistage_models,
    stage_markers_from_windows as local_stage_markers_from_windows,
    stage_name as local_stage_name,
)
from src.data.local_operator_amp_phase import build_branch_features, interp_complex_field, prepare_single_case_trajectory


GLOBAL_CFG = os.path.join(PROJECT_DIR, "configs", "cgl_single_case_global_multistage_amp_phase_alpha075_beta0_mu0_t5.yaml")
LOCAL_CFG = os.path.join(PROJECT_DIR, "configs", "cgl_single_case_local_multistage_amp_phase_alpha075_beta0_mu0_t5.yaml")

GLOBAL_ROOT = os.path.join(REPO_ROOT, "CGL_GlobalMultistage_AmpPhase_alpha075_beta0_mu0_t5")
LOCAL_ROOT = os.path.join(REPO_ROOT, "CGL_LocalMultistage_AmpPhase_alpha075_beta0_mu0_t5")

OUT_ROOT = os.path.join(REPO_ROOT, "analyses", "multireseau", "single_cases", "run_alpha075_beta0_mu0")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def latest_run_dir(root):
    run_dirs = sorted(path for path in os.listdir(root) if path.startswith("run_") and os.path.isdir(os.path.join(root, path)))
    if not run_dirs:
        raise FileNotFoundError(root)
    return os.path.join(root, run_dirs[-1])


def choose_snapshot_indices(t_values, n=6):
    raw = np.linspace(0, len(t_values) - 1, n)
    return np.unique(np.round(raw).astype(int)).tolist()


def plot_l2_curve(t_values, rel_l2, title, output_path, stage_markers=(1.0, 2.0, 3.0, 4.0)):
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(t_values, rel_l2, color="#c2185b", linewidth=2.2)
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="Seuil 5%")
    for marker in stage_markers:
        plt.axvline(marker, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_error_heatmap(x, t_values, u_true, u_pred, title, output_path):
    err = np.abs(u_pred - u_true)
    plt.figure(figsize=(10.0, 4.8))
    plt.imshow(
        err,
        aspect="auto",
        origin="lower",
        extent=[float(t_values[0]), float(t_values[-1]), float(x[0]), float(x[-1])],
        cmap="magma",
    )
    plt.colorbar(label=r"$|u_{\mathrm{pred}}-u_{\mathrm{ref}}|$")
    for marker in [1.0, 2.0, 3.0, 4.0]:
        plt.axvline(marker, color="white", linestyle=":", linewidth=0.8, alpha=0.7)
    plt.xlabel("Temps t")
    plt.ylabel("Position x")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_snapshots(x, t_values, u_true, u_pred, title, output_path):
    idxs = choose_snapshot_indices(t_values, n=6)
    fig, axes = plt.subplots(3, len(idxs), figsize=(3.5 * len(idxs), 9.0), sharex=True)
    transforms = [np.abs, np.real, np.imag]
    row_titles = ["Module", "Partie reelle", "Partie imaginaire"]
    colors = plt.get_cmap("RdPu")(np.linspace(0.45, 0.9, len(idxs)))

    for row, (transform, row_title) in enumerate(zip(transforms, row_titles)):
        for col, (idx, color) in enumerate(zip(idxs, colors)):
            ax = axes[row, col]
            ax.plot(x, transform(u_true[:, idx]), color="black", linestyle=":", linewidth=1.2, label="Solveur")
            ax.plot(x, transform(u_pred[:, idx]), color=color, linewidth=1.8, label="Modele")
            if row == 0:
                ax.set_title(f"t={float(t_values[idx]):.2f}")
            if col == 0:
                ax.set_ylabel(row_title)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_curve_csv(path, t_values, rel_l2):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("time,rel_l2\n")
        for t_val, err in zip(t_values, rel_l2):
            handle.write(f"{float(t_val):.8f},{float(err):.10f}\n")


def first_above_threshold(t_values, rel_l2, threshold=0.05):
    mask = rel_l2 > threshold
    if not np.any(mask):
        return np.nan
    return float(t_values[int(np.argmax(mask))])


def select_local_stage_model(models, rollout_windows, t_current):
    for (t_start, t_end), model in zip(rollout_windows, models):
        if t_current < t_end - 1e-10:
            return model
    return models[-1]


def relative_l2(u_pred, u_true):
    denom = np.linalg.norm(u_true)
    if denom < 1e-12:
        denom = 1e-12
    return float(np.linalg.norm(u_pred - u_true) / denom)


def teacher_forced_piecewise_local(models, rollout_windows, trajectory, cfg_dict, device):
    params = trajectory["params"]
    x_sensor = trajectory["x_sensor"]
    x_solver = trajectory["x_solver"]
    u_sensor = trajectory["u_sensor"]
    u_solver = trajectory["u_solver"]
    dt_value = trajectory["dt"]
    periodic = trajectory["periodic"]
    amp_floor = float(cfg_dict["local_operator"]["amp_floor"])

    n_steps = u_solver.shape[1] - 1
    pred_solver = np.zeros_like(u_solver)
    pred_solver[:, 0] = u_solver[:, 0]
    rel_l2 = np.zeros(n_steps + 1, dtype=np.float64)

    with torch.no_grad():
        for step in range(n_steps):
            t_current = float(trajectory["t_values"][step])
            model = select_local_stage_model(models, rollout_windows, t_current)
            branch_vec = build_branch_features(cfg_dict, u_sensor[:, step], dt_value, params)
            branch_tensor = torch.tensor(branch_vec[None, :], dtype=torch.float32, device=device)
            branch_sensor = branch_tensor.repeat(len(x_sensor), 1)
            x_sensor_tensor = torch.tensor(x_sensor[:, None], dtype=torch.float32, device=device)
            delta_log_amp_sensor, delta_phase_sensor = model(branch_sensor, x_sensor_tensor)

            current_sensor = u_sensor[:, step]
            current_amp_sensor = np.abs(current_sensor)
            current_phase_sensor = np.angle(current_sensor)
            next_amp_sensor = np.exp(
                np.log(current_amp_sensor + amp_floor) + delta_log_amp_sensor.cpu().numpy().reshape(-1)
            ) - amp_floor
            next_phase_sensor = current_phase_sensor + delta_phase_sensor.cpu().numpy().reshape(-1)
            next_sensor = next_amp_sensor * np.exp(1j * next_phase_sensor)
            pred_solver[:, step + 1] = interp_complex_field(x_sensor, next_sensor, x_solver, periodic)
            rel_l2[step + 1] = relative_l2(pred_solver[:, step + 1], u_solver[:, step + 1])

    return pred_solver, rel_l2


def plot_local_diagnostic(t_values, one_step_rel_l2, rollout_rel_l2, output_path, stage_markers=None):
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(t_values, one_step_rel_l2, color="#006d77", linewidth=2.0, label="One-step teacher forced")
    plt.plot(t_values, rollout_rel_l2, color="#c2185b", linewidth=2.0, label="Rollout autoregressif")
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="Seuil 5%")
    for marker in stage_markers or []:
        plt.axvline(marker, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title("Local multistage : one-step piecewise vs rollout")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_summary(path, label, rel_l2, t_values):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"label={label}\n")
        handle.write(f"final_rel_l2={float(rel_l2[-1]):.10f}\n")
        handle.write(f"max_rel_l2={float(np.max(rel_l2)):.10f}\n")
        handle.write(f"mean_rel_l2={float(np.mean(rel_l2)):.10f}\n")
        handle.write(f"first_t_gt_5pct={first_above_threshold(t_values, rel_l2, 0.05)}\n")


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(os.path.join(OUT_ROOT, "global_multistage"))
    ensure_dir(os.path.join(OUT_ROOT, "local_multistage"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(GLOBAL_CFG, "r", encoding="utf-8") as handle:
        global_cfg = yaml.safe_load(handle)
    with open(LOCAL_CFG, "r", encoding="utf-8") as handle:
        local_cfg = yaml.safe_load(handle)

    global_run = latest_run_dir(GLOBAL_ROOT)
    local_run = latest_run_dir(LOCAL_ROOT)

    global_blocks = [tuple(map(float, block)) for block in global_cfg["multistage"]["time_blocks"]]
    local_blocks = load_local_time_blocks(local_cfg)
    local_rollout_windows = load_local_rollout_windows(local_cfg)
    local_stage_markers = local_stage_markers_from_windows(local_rollout_windows)

    global_models = [
        load_best_global_stage_model(global_cfg, os.path.join(global_run, global_stage_name(i, t0, t1)), device)
        for i, (t0, t1) in enumerate(global_blocks)
    ]
    local_models = [
        load_best_local_stage_model(local_cfg, os.path.join(local_run, local_stage_name(i, t0, t1)), device)
        for i, (t0, t1) in enumerate(local_blocks)
    ]

    reference_global = prepare_single_case_reference(global_cfg)
    global_rollout = rollout_global_multistage_models(global_models, global_blocks, reference_global, device)

    trajectory_local = prepare_single_case_trajectory(local_cfg)
    local_rollout = rollout_local_multistage_models(local_models, local_rollout_windows, trajectory_local, local_cfg, device)
    local_one_step_pred, local_one_step_rel_l2 = teacher_forced_piecewise_local(local_models, local_rollout_windows, trajectory_local, local_cfg, device)

    gdir = os.path.join(OUT_ROOT, "global_multistage")
    ldir = os.path.join(OUT_ROOT, "local_multistage")

    save_curve_csv(os.path.join(gdir, "l2_vs_time.csv"), global_rollout["t_values"], global_rollout["rel_l2"])
    plot_l2_curve(global_rollout["t_values"], global_rollout["rel_l2"], "Global multistage : L2(t)", os.path.join(gdir, "l2_vs_time.png"))
    plot_error_heatmap(reference_global["x"], global_rollout["t_values"], global_rollout["u_true"], global_rollout["u_pred"], "Global multistage : heatmap erreur", os.path.join(gdir, "error_heatmap.png"))
    plot_snapshots(reference_global["x"], global_rollout["t_values"], global_rollout["u_true"], global_rollout["u_pred"], "Global multistage : snapshots vs solveur", os.path.join(gdir, "snapshots.png"))
    write_summary(os.path.join(gdir, "summary.txt"), "global_multistage", global_rollout["rel_l2"], global_rollout["t_values"])

    save_curve_csv(os.path.join(ldir, "l2_vs_time.csv"), local_rollout["t_values"], local_rollout["rel_l2"])
    plot_l2_curve(local_rollout["t_values"], local_rollout["rel_l2"], "Local multistage : L2(t)", os.path.join(ldir, "l2_vs_time.png"), stage_markers=local_stage_markers)
    plot_error_heatmap(trajectory_local["x_solver"], local_rollout["t_values"], local_rollout["u_solver_ref"], local_rollout["u_solver_pred"], "Local multistage : heatmap erreur", os.path.join(ldir, "error_heatmap.png"))
    plot_snapshots(trajectory_local["x_solver"], local_rollout["t_values"], local_rollout["u_solver_ref"], local_rollout["u_solver_pred"], "Local multistage : snapshots vs solveur", os.path.join(ldir, "snapshots.png"))
    plot_local_diagnostic(
        local_rollout["t_values"],
        local_one_step_rel_l2,
        local_rollout["rel_l2"],
        os.path.join(ldir, "one_step_vs_rollout.png"),
        stage_markers=local_stage_markers,
    )
    plot_error_heatmap(trajectory_local["x_solver"], local_rollout["t_values"], trajectory_local["u_solver"], local_one_step_pred, "Local multistage : heatmap one-step piecewise", os.path.join(ldir, "one_step_error_heatmap.png"))
    write_summary(os.path.join(ldir, "summary.txt"), "local_multistage", local_rollout["rel_l2"], local_rollout["t_values"])
    with open(os.path.join(ldir, "diagnostic_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"first_one_step_t_gt_5pct={first_above_threshold(local_rollout['t_values'], local_one_step_rel_l2, 0.05)}\n")
        handle.write(f"first_rollout_t_gt_5pct={first_above_threshold(local_rollout['t_values'], local_rollout['rel_l2'], 0.05)}\n")
        handle.write(f"final_one_step_rel_l2={float(local_one_step_rel_l2[-1]):.10f}\n")
        handle.write(f"final_rollout_rel_l2={float(local_rollout['rel_l2'][-1]):.10f}\n")
        handle.write(f"mean_one_step_rel_l2={float(np.mean(local_one_step_rel_l2)):.10f}\n")
        handle.write(f"mean_rollout_rel_l2={float(np.mean(local_rollout['rel_l2'])):.10f}\n")

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(global_rollout["t_values"], global_rollout["rel_l2"], color="#6a1b9a", linewidth=2.2, label="Global multistage")
    plt.plot(local_rollout["t_values"], local_rollout["rel_l2"], color="#c2185b", linewidth=2.2, label="Local multistage")
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="Seuil 5%")
    for marker in [1.0, 2.0, 3.0, 4.0]:
        plt.axvline(marker, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title("Comparaison multistage : global vs local")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_ROOT, "global_vs_local_multistage_l2.png"), dpi=220)
    plt.close()

    with open(os.path.join(OUT_ROOT, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"global_final_rel_l2={float(global_rollout['rel_l2'][-1]):.10f}\n")
        handle.write(f"global_first_t_gt_5pct={first_above_threshold(global_rollout['t_values'], global_rollout['rel_l2'], 0.05)}\n")
        handle.write(f"local_final_rel_l2={float(local_rollout['rel_l2'][-1]):.10f}\n")
        handle.write(f"local_first_t_gt_5pct={first_above_threshold(local_rollout['t_values'], local_rollout['rel_l2'], 0.05)}\n")
        handle.write(f"local_one_step_final_rel_l2={float(local_one_step_rel_l2[-1]):.10f}\n")
        handle.write(f"local_one_step_first_t_gt_5pct={first_above_threshold(local_rollout['t_values'], local_one_step_rel_l2, 0.05)}\n")

    print(f"✅ Analyses écrites dans {OUT_ROOT}")


if __name__ == "__main__":
    main()
