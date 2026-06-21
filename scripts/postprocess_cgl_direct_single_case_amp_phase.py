import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.models.cgl_deeponet_amp_phase import CGL_PI_DeepONet_AmpPhase
from src.plot.postprocess_single_case import (
    benchmark_inference,
    plot_error_heatmap,
    plot_l2_curve,
    plot_snapshots,
    relative_l2_curve_on_mask,
    save_comparison_gif,
    save_rel_l2_csv,
    spatial_mask_from_bounds,
    write_rollout_summary,
)
from src.utils.solver_cgl import get_ground_truth_CGL


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
        "type": float(cfg_dict["physics"]["initial_conditions"][0]),
    }


def infer_arch_from_state_dict(state_dict):
    branch_layer_ids = sorted(
        int(k.split(".")[2]) for k in state_dict if k.startswith("branch_net.layers.") and k.endswith(".bias")
    )
    trunk_layer_ids = sorted(
        int(k.split(".")[2]) for k in state_dict if k.startswith("trunk_net.layers.") and k.endswith(".bias")
    )
    branch_layers = [int(state_dict[f"branch_net.layers.{idx}.bias"].shape[0]) for idx in branch_layer_ids]
    trunk_layers = [int(state_dict[f"trunk_net.layers.{idx}.bias"].shape[0]) for idx in trunk_layer_ids]
    latent_dim = int(state_dict["branch_net.output_layer.bias"].shape[0] // 2)
    return latent_dim, branch_layers, trunk_layers


def load_direct_model(checkpoint_path, cfg_dict, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    cfg_dict = dict(cfg_dict)
    cfg_dict["model"] = dict(cfg_dict["model"])
    latent_dim, branch_layers, trunk_layers = infer_arch_from_state_dict(state)
    cfg_dict["model"]["latent_dim"] = latent_dim
    cfg_dict["model"]["branch_layers"] = branch_layers
    cfg_dict["model"]["trunk_layers"] = trunk_layers
    model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt, cfg_dict


def predict_grid(model, params_dict, x, t_values, device):
    xx = np.tile(x, len(t_values))
    tt = np.repeat(t_values, len(x))
    coords = torch.tensor(np.stack([xx, tt], axis=1), dtype=torch.float32, device=device)
    p_vec = np.array(
        [
            params_dict["alpha"],
            params_dict["beta"],
            params_dict["mu"],
            params_dict["V"],
            params_dict["A"],
            params_dict["w0"],
            params_dict["x0"],
            params_dict["k"],
            float(params_dict["type"]),
        ],
        dtype=np.float32,
    )
    branch = torch.tensor(p_vec, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(coords), 1)
    with torch.no_grad():
        ur, ui = model(branch, coords)
    return (ur + 1j * ui).cpu().numpy().reshape(len(t_values), len(x)).T


def relative_l2_curve(u_pred, u_true):
    rel_l2 = np.zeros(u_true.shape[1], dtype=np.float64)
    for idx in range(u_true.shape[1]):
        denom = np.linalg.norm(u_true[:, idx]) + 1.0e-12
        rel_l2[idx] = np.linalg.norm(u_pred[:, idx] - u_true[:, idx]) / denom
    return rel_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--reached-t", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, cfg_dict = load_direct_model(args.checkpoint, cfg_dict, device)
    params = fixed_case_params(cfg_dict)
    reached_t = float(args.reached_t) if args.reached_t is not None else float(ckpt.get("t_curr", cfg_dict["physics"]["t_max"]))
    solver_nx = int(os.environ.get("CGL_EVAL_NX", "256"))
    solver_nt_env = os.environ.get("CGL_EVAL_NT")
    solver_nt = int(solver_nt_env) if solver_nt_env not in (None, "") else None

    x_min, x_max = cfg_dict["physics"]["x_domain"]
    X, T, U_true = get_ground_truth_CGL(params, x_min, x_max, reached_t, Nx=solver_nx, Nt=solver_nt)
    x = X[:, 0]
    t_values = T[0, :]
    U_pred = predict_grid(model, params, x, t_values, device)
    rel_l2 = relative_l2_curve(U_pred, U_true)
    center_mask = spatial_mask_from_bounds(x, -10.0, 10.0)
    rel_l2_center = relative_l2_curve_on_mask(U_pred, U_true, center_mask)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "rollout_metrics.csv")
    save_rel_l2_csv(csv_path, t_values, rel_l2)
    csv_center_path = os.path.join(args.output_dir, "rollout_metrics_center_xm10_xp10.csv")
    save_rel_l2_csv(csv_center_path, t_values, rel_l2_center)
    plot_l2_curve(t_values, rel_l2, "Monoreseau direct : erreur relative vs solveur classique", os.path.join(args.output_dir, "rollout_rel_l2.png"))
    plot_l2_curve(
        t_values,
        rel_l2_center,
        "Monoreseau direct : erreur relative au centre x in [-10, 10]",
        os.path.join(args.output_dir, "rollout_rel_l2_center_xm10_xp10.png"),
    )
    plot_error_heatmap(x, t_values, U_true, U_pred, "Monoreseau direct : heatmap erreur", os.path.join(args.output_dir, "error_heatmap.png"))
    plot_snapshots(
        x,
        t_values,
        U_true,
        U_pred,
        "Monoreseau direct : snapshots vs solveur",
        os.path.join(args.output_dir, "snapshots.png"),
        snapshot_times=list(cfg_dict.get("evaluation", {}).get("snapshot_times", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
    )
    if os.environ.get("CGL_SKIP_GIF", "0") != "1":
        save_comparison_gif(
            x,
            t_values,
            U_true,
            U_pred,
            "Monoreseau direct : solveur vs prediction",
            os.path.join(args.output_dir, "comparison_animation.gif"),
        )
    write_rollout_summary(
        os.path.join(args.output_dir, "summary.txt"),
        rel_l2,
        t_values,
        extra={
            "checkpoint": args.checkpoint,
            "metrics_csv": csv_path,
            "metrics_csv_center_xm10_xp10": csv_center_path,
            "reached_t": reached_t,
            "final_rel_l2_center_xm10_xp10": float(rel_l2_center[-1]),
            "max_rel_l2_center_xm10_xp10": float(np.max(rel_l2_center)),
            "mean_rel_l2_center_xm10_xp10": float(np.mean(rel_l2_center)),
        },
    )
    if os.environ.get("CGL_SKIP_BENCHMARK", "0") != "1":
        benchmark_inference(
            "Monoreseau direct",
            solver_callable=lambda: get_ground_truth_CGL(params, x_min, x_max, reached_t, Nx=128, Nt=None),
            model_callable=lambda: predict_grid(model, params, x, t_values, device),
            output_dir=args.output_dir,
            repeats=8,
            warmup=1,
        )


if __name__ == "__main__":
    main()
