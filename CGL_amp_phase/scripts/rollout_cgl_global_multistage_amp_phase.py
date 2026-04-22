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

from scripts.train_cgl_global_multistage_amp_phase import (
    load_best_stage_model,
    prepare_single_case_reference,
    rollout_multistage_models,
    stage_name,
)


def plot_rollout_curve(rollout, output_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(rollout["t_values"], rollout["rel_l2"], color="#c2185b", linewidth=2.0)
    for t_sep in [1.0, 2.0, 3.0, 4.0]:
        plt.axvline(t_sep, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title("Global multistage : erreur relative vs solveur classique")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_rollout_metrics(output_dir, rollout):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "rollout_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("time,rel_l2\n")
        for t_val, err in zip(rollout["t_values"], rollout["rel_l2"]):
            handle.write(f"{float(t_val):.8f},{float(err):.10f}\n")
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_single_case_global_multistage_amp_phase_alpha075_beta0_mu0_t5.yaml")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_blocks = [tuple(map(float, block)) for block in cfg_dict["multistage"]["time_blocks"]]
    stage_models = [
        load_best_stage_model(cfg_dict, os.path.join(args.run_dir, stage_name(stage_idx, t_start, t_end)), device)
        for stage_idx, (t_start, t_end) in enumerate(time_blocks)
    ]
    reference = prepare_single_case_reference(cfg_dict)
    rollout = rollout_multistage_models(stage_models, time_blocks, reference, device)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = save_rollout_metrics(args.output_dir, rollout)
    plot_rollout_curve(rollout, os.path.join(args.output_dir, "rollout_rel_l2.png"))
    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"run_dir={args.run_dir}\n")
        handle.write(f"final_rel_l2={float(rollout['rel_l2'][-1]):.10f}\n")
        handle.write(f"max_rel_l2={float(np.max(rollout['rel_l2'])):.10f}\n")
        handle.write(f"mean_rel_l2={float(np.mean(rollout['rel_l2'])):.10f}\n")
        handle.write(f"metrics_csv={csv_path}\n")


if __name__ == "__main__":
    main()
