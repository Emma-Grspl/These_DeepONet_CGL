import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.data.local_operator_amp_phase import prepare_single_case_trajectory, rollout_local_model, save_rollout_metrics
from src.models.cgl_local_deeponet_amp_phase import CGL_LocalDirect_DeepONet_AmpPhase


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt


def plot_rollout_curve(rollout, output_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(rollout["t_values"], rollout["rel_l2"], color="#c2185b", linewidth=2.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title("Rollout local direct : erreur relative vs solveur classique")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_snapshots(rollout, output_path, snapshot_times):
    x = rollout["x_solver"]
    t_values = rollout["t_values"]
    u_ref = rollout["u_solver_ref"]
    u_pred = rollout["u_solver_pred"]
    idxs = [int(np.argmin(np.abs(t_values - target_t))) for target_t in snapshot_times]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    titles = ["Module", "Partie reelle", "Partie imaginaire"]
    transforms = [np.abs, np.real, np.imag]
    labels = [f"t={t_values[idx]:.2f}" for idx in idxs]
    colors = plt.get_cmap("RdPu")(np.linspace(0.45, 0.9, len(idxs)))

    for ax, title, transform in zip(axes, titles, transforms):
        for idx, color, label in zip(idxs, colors, labels):
            ax.plot(x, transform(u_ref[:, idx]), color="black", linestyle=":", linewidth=1.2)
            ax.plot(x, transform(u_pred[:, idx]), color=color, linewidth=2.0, label=label)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("|u|")
    axes[1].set_ylabel("Re(u)")
    axes[2].set_ylabel("Im(u)")
    axes[2].set_xlabel("x")
    axes[0].legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_single_case_local_direct_amp_phase_t5.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGL_LocalDirect_DeepONet_AmpPhase(cfg_dict).to(device)
    _ = load_checkpoint(model, args.checkpoint, device)

    rollout_tmax = float(cfg_dict["evaluation"]["rollout_tmax"])
    rollout_dt = float(cfg_dict["local_operator"]["rollout_dt"])
    trajectory = prepare_single_case_trajectory(cfg_dict, t_max_override=rollout_tmax, dt_override=rollout_dt)
    rollout = rollout_local_model(model, trajectory, cfg_dict, device)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = save_rollout_metrics(args.output_dir, rollout)
    plot_rollout_curve(rollout, os.path.join(args.output_dir, "rollout_rel_l2.png"))
    plot_snapshots(
        rollout,
        os.path.join(args.output_dir, "rollout_snapshots.png"),
        snapshot_times=list(cfg_dict["evaluation"]["snapshot_times"]),
    )
    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"checkpoint={args.checkpoint}\n")
        handle.write(f"final_rel_l2={float(rollout['rel_l2'][-1]):.10f}\n")
        handle.write(f"max_rel_l2={float(np.max(rollout['rel_l2'])):.10f}\n")
        handle.write(f"mean_rel_l2={float(np.mean(rollout['rel_l2'])):.10f}\n")
        handle.write(f"metrics_csv={csv_path}\n")

    print(f"✅ Rollout terminé | final_rel_l2={float(rollout['rel_l2'][-1]):.4%}")


if __name__ == "__main__":
    main()
