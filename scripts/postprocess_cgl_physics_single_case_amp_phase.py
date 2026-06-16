import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.cgl_deeponet_amp_phase import CGL_PI_DeepONet_AmpPhase
from src.plot.postprocess_single_case import (
    benchmark_inference,
    plot_error_heatmap,
    plot_l2_curve,
    plot_snapshots,
    save_comparison_gif,
    save_rel_l2_csv,
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


def load_model_from_checkpoint(checkpoint_path, cfg_dict, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    cfg_copy = dict(cfg_dict)
    cfg_copy["model"] = dict(cfg_dict["model"])
    latent_dim, branch_layers, trunk_layers = infer_arch_from_state_dict(state)
    cfg_copy["model"]["latent_dim"] = latent_dim
    cfg_copy["model"]["branch_layers"] = branch_layers
    cfg_copy["model"]["trunk_layers"] = trunk_layers
    model = CGL_PI_DeepONet_AmpPhase(cfg_copy).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt, cfg_copy


def select_checkpoint(run_dir):
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "checkpoints" / "ckpt_FINAL.pth",
        run_dir / "checkpoints" / "model_latest.pth",
        run_dir / "model_final_cgl_amp_phase.pth",
        run_dir / "model_final_cgl.pth",
    ]
    ckpt_t = sorted((run_dir / "checkpoints").glob("ckpt_t*.pth")) if (run_dir / "checkpoints").exists() else []
    candidates[1:1] = list(reversed(ckpt_t))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No usable checkpoint found in {run_dir}")


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
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reached-t", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = select_checkpoint(args.run_dir)
    model, ckpt, cfg_dict = load_model_from_checkpoint(checkpoint_path, cfg_dict, device)
    params = fixed_case_params(cfg_dict)
    reached_t = float(args.reached_t) if args.reached_t is not None else float(ckpt.get("t_curr", cfg_dict["physics"]["t_max"]))

    x_min, x_max = cfg_dict["physics"]["x_domain"]
    X, T, U_true = get_ground_truth_CGL(params, x_min, x_max, reached_t, Nx=256, Nt=None)
    x = X[:, 0]
    t_values = T[0, :]
    U_pred = predict_grid(model, params, x, t_values, device)
    rel_l2 = relative_l2_curve(U_pred, U_true)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "rollout_metrics.csv")
    save_rel_l2_csv(csv_path, t_values, rel_l2)
    plot_l2_curve(t_values, rel_l2, f"{args.label} : erreur relative vs solveur classique", os.path.join(args.output_dir, "rollout_rel_l2.png"))
    plot_error_heatmap(x, t_values, U_true, U_pred, f"{args.label} : heatmap erreur", os.path.join(args.output_dir, "error_heatmap.png"))
    plot_snapshots(
        x,
        t_values,
        U_true,
        U_pred,
        f"{args.label} : snapshots vs solveur",
        os.path.join(args.output_dir, "snapshots.png"),
        snapshot_times=list(cfg_dict.get("benchmark", {}).get("eval_times", [0.2, 0.5, 1.0, 2.0, 3.0, 5.0])),
    )
    save_comparison_gif(
        x,
        t_values,
        U_true,
        U_pred,
        f"{args.label} : solveur vs prediction",
        os.path.join(args.output_dir, "comparison_animation.gif"),
    )
    write_rollout_summary(
        os.path.join(args.output_dir, "summary.txt"),
        rel_l2,
        t_values,
        extra={
            "checkpoint": str(checkpoint_path),
            "run_dir": str(Path(args.run_dir).resolve()),
            "metrics_csv": csv_path,
            "reached_t": reached_t,
        },
    )
    benchmark_inference(
        args.label,
        solver_callable=lambda: get_ground_truth_CGL(params, x_min, x_max, reached_t, Nx=128, Nt=None),
        model_callable=lambda: predict_grid(model, params, x, t_values, device),
        output_dir=args.output_dir,
        repeats=8,
        warmup=1,
    )


if __name__ == "__main__":
    main()
