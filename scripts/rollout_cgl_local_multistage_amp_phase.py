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

from src.data.local_operator_amp_phase import prepare_single_case_trajectory, save_rollout_metrics
from scripts.train_cgl_local_multistage_amp_phase import (
    load_best_stage_model,
    load_rollout_windows,
    load_time_blocks,
    rollout_multistage_models,
    stage_markers_from_windows,
    stage_name,
)
from src.plot.postprocess_single_case import (
    benchmark_inference,
    plot_error_heatmap,
    plot_l2_curve,
    plot_snapshots,
    save_comparison_gif,
    write_rollout_summary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_single_case_local_multistage_amp_phase_alpha075_beta0_mu0_t5.yaml")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_blocks = load_time_blocks(cfg_dict)
    rollout_windows = load_rollout_windows(cfg_dict)
    stage_models = [
        load_best_stage_model(cfg_dict, os.path.join(args.run_dir, stage_name(stage_idx, t_start, t_end)), device)
        for stage_idx, (t_start, t_end) in enumerate(time_blocks)
    ]
    trajectory = prepare_single_case_trajectory(cfg_dict)
    rollout = rollout_multistage_models(stage_models, rollout_windows, trajectory, cfg_dict, device)
    markers = stage_markers_from_windows(rollout_windows)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = save_rollout_metrics(args.output_dir, rollout)
    plot_l2_curve(
        rollout["t_values"],
        rollout["rel_l2"],
        "Local multistage : erreur relative vs solveur classique",
        os.path.join(args.output_dir, "rollout_rel_l2.png"),
        stage_markers=markers,
    )
    plot_error_heatmap(
        trajectory["x_solver"],
        rollout["t_values"],
        rollout["u_solver_ref"],
        rollout["u_solver_pred"],
        "Local multistage : heatmap erreur",
        os.path.join(args.output_dir, "error_heatmap.png"),
        stage_markers=markers,
    )
    plot_snapshots(
        trajectory["x_solver"],
        rollout["t_values"],
        rollout["u_solver_ref"],
        rollout["u_solver_pred"],
        "Local multistage : snapshots vs solveur",
        os.path.join(args.output_dir, "snapshots.png"),
        snapshot_times=list(cfg_dict.get("evaluation", {}).get("snapshot_times", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
    )
    save_comparison_gif(
        trajectory["x_solver"],
        rollout["t_values"],
        rollout["u_solver_ref"],
        rollout["u_solver_pred"],
        "Local multistage : solveur vs rollout",
        os.path.join(args.output_dir, "comparison_animation.gif"),
    )
    write_rollout_summary(
        os.path.join(args.output_dir, "summary.txt"),
        rollout["rel_l2"],
        rollout["t_values"],
        extra={"run_dir": args.run_dir, "metrics_csv": csv_path},
    )
    benchmark_inference(
        "Local multistage",
        solver_callable=lambda: prepare_single_case_trajectory(cfg_dict),
        model_callable=lambda: rollout_multistage_models(stage_models, time_blocks, trajectory, cfg_dict, device),
        output_dir=args.output_dir,
        repeats=8,
        warmup=1,
    )


if __name__ == "__main__":
    main()
