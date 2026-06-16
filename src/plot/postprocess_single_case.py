import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def save_rel_l2_csv(path, t_values, rel_l2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("time,rel_l2\n")
        for t_val, err in zip(t_values, rel_l2):
            handle.write(f"{float(t_val):.8f},{float(err):.10f}\n")


def write_rollout_summary(path, rel_l2, t_values, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"final_rel_l2={float(rel_l2[-1]):.10f}\n")
        handle.write(f"max_rel_l2={float(np.max(rel_l2)):.10f}\n")
        handle.write(f"mean_rel_l2={float(np.mean(rel_l2)):.10f}\n")
        handle.write(f"first_t_gt_5pct={first_above_threshold(t_values, rel_l2, 0.05)}\n")
        if extra:
            for key, value in extra.items():
                handle.write(f"{key}={value}\n")


def first_above_threshold(t_values, rel_l2, threshold):
    mask = np.asarray(rel_l2) > float(threshold)
    if not np.any(mask):
        return np.nan
    return float(np.asarray(t_values)[int(np.argmax(mask))])


def choose_snapshot_indices(t_values, snapshot_times=None, n=6):
    t_values = np.asarray(t_values)
    if snapshot_times is not None:
        return [int(np.argmin(np.abs(t_values - float(target_t)))) for target_t in snapshot_times]
    if len(t_values) <= n:
        return list(range(len(t_values)))
    raw = np.linspace(0, len(t_values) - 1, n)
    idxs = np.unique(np.round(raw).astype(int))
    while len(idxs) < n:
        candidates = [i for i in range(len(t_values)) if i not in idxs]
        idxs = np.sort(np.append(idxs, candidates[0]))
    return idxs.tolist()


def plot_l2_curve(t_values, rel_l2, title, output_path, stage_markers=None):
    plt.figure(figsize=(8.2, 4.6))
    plt.plot(t_values, rel_l2, color="#c2185b", linewidth=2.2)
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="Seuil 5%")
    if stage_markers:
        for marker in stage_markers:
            plt.axvline(float(marker), color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_error_heatmap(x, t_values, u_true, u_pred, title, output_path, stage_markers=None):
    err = np.abs(u_pred - u_true)
    plt.figure(figsize=(10.0, 4.8))
    plt.imshow(
        err,
        aspect="auto",
        origin="lower",
        extent=[float(t_values[0]), float(t_values[-1]), float(x[0]), float(x[-1])],
        cmap="magma",
    )
    if stage_markers:
        for marker in stage_markers:
            plt.axvline(float(marker), color="white", linestyle=":", linewidth=0.8, alpha=0.75)
    plt.colorbar(label=r"$|u_{\mathrm{pred}}-u_{\mathrm{ref}}|$")
    plt.xlabel("Temps t")
    plt.ylabel("Position x")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_snapshots(x, t_values, u_true, u_pred, title, output_path, snapshot_times=None):
    idxs = choose_snapshot_indices(t_values, snapshot_times=snapshot_times, n=6)
    fig, axes = plt.subplots(3, len(idxs), figsize=(3.4 * len(idxs), 9.0), sharex=True)
    row_titles = ["Module", "Partie reelle", "Partie imaginaire"]
    transforms = [np.abs, np.real, np.imag]
    colors = plt.get_cmap("RdPu")(np.linspace(0.45, 0.9, len(idxs)))

    for row, (row_title, transform) in enumerate(zip(row_titles, transforms)):
        for col, (idx, color) in enumerate(zip(idxs, colors)):
            ax = axes[row, col]
            ax.plot(x, transform(u_true[:, idx]), color="black", linestyle=":", linewidth=1.2, label="Solveur")
            ax.plot(x, transform(u_pred[:, idx]), color=color, linewidth=2.0, label="Modele")
            if row == 0:
                ax.set_title(f"t={float(t_values[idx]):.2f}")
            if col == 0:
                ax.set_ylabel(row_title)
            ax.grid(alpha=0.2)

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_comparison_gif(x, t_values, u_true, u_pred, title, output_path, frames=120):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t_values = np.asarray(t_values)
    idx_frames = np.unique(np.linspace(0, len(t_values) - 1, min(frames, len(t_values))).astype(int))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    line_pairs = []
    transforms = [np.abs, np.real, np.imag]
    y_ranges = []

    for transform in transforms:
        true_vals = transform(u_true)
        pred_vals = transform(u_pred)
        ymin = min(float(np.min(true_vals)), float(np.min(pred_vals)))
        ymax = max(float(np.max(true_vals)), float(np.max(pred_vals)))
        margin = 0.1 * max(1.0e-8, ymax - ymin)
        y_ranges.append((ymin - margin, ymax + margin))

    row_titles = ["Module", "Partie reelle", "Partie imaginaire"]
    for ax, (title_row, transform, yrange) in zip(axes, zip(row_titles, transforms, y_ranges)):
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(yrange[0], yrange[1])
        ax.set_ylabel(title_row)
        ax.grid(alpha=0.25)
        line_true, = ax.plot([], [], color="black", linewidth=1.8, label="Solveur")
        line_pred, = ax.plot([], [], color="deeppink", linestyle="--", linewidth=1.8, label="Modele")
        line_pairs.append((line_true, line_pred, transform))
    axes[-1].set_xlabel("x")
    axes[0].legend(frameon=False, loc="upper right")
    suptitle = fig.suptitle(f"{title}\nt={float(t_values[0]):.2f}", fontsize=14)

    def update(frame_idx):
        idx = idx_frames[frame_idx]
        suptitle.set_text(f"{title}\nt={float(t_values[idx]):.2f}")
        artists = [suptitle]
        for line_true, line_pred, transform in line_pairs:
            line_true.set_data(x, transform(u_true[:, idx]))
            line_pred.set_data(x, transform(u_pred[:, idx]))
            artists.extend([line_true, line_pred])
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(idx_frames), interval=60, blit=False)
    anim.save(output_path, writer="pillow", fps=20)
    plt.close(fig)


def benchmark_inference(model_label, solver_callable, model_callable, output_dir, repeats=12, warmup=1):
    os.makedirs(output_dir, exist_ok=True)
    for _ in range(max(0, warmup)):
        solver_callable()
        model_callable()

    solver_times = []
    model_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        solver_callable()
        solver_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        model_callable()
        model_times.append(time.perf_counter() - t1)

    mean_solver = float(np.mean(solver_times))
    mean_model = float(np.mean(model_times))
    speedup = mean_solver / max(mean_model, 1.0e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["Solveur", model_label], [mean_solver, mean_model], color=["black", "deeppink"])
    axes[0].set_title(f"Temps moyen ({repeats} repetitions)")
    axes[0].set_ylabel("Temps [s]")
    axes[1].bar(["Solveur", model_label], [float(np.sum(solver_times)), float(np.sum(model_times))], color=["black", "deeppink"])
    axes[1].set_title(f"Temps cumule ({repeats} repetitions)")
    axes[1].set_ylabel("Temps [s]")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "inference_timing.png")
    plt.savefig(fig_path, dpi=220)
    plt.close(fig)

    summary_path = os.path.join(output_dir, "inference_timing.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"repeats={int(repeats)}\n")
        handle.write(f"mean_solver_seconds={mean_solver:.10f}\n")
        handle.write(f"mean_model_seconds={mean_model:.10f}\n")
        handle.write(f"speedup_vs_solver={speedup:.10f}\n")
        handle.write(f"timing_plot={fig_path}\n")

