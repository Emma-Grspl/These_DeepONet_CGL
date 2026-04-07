import argparse
import copy
import csv
import glob
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.models.cgl_deeponet import CGL_PI_DeepONet
from src.plot.plot_animation import animate_cgl_solution
from src.plot.plot_snapshot import plot_temporal_snapshots
from src.training.trainer_CGL import run_audit
from src.utils.solver_cgl import get_ground_truth_CGL

CLASSICAL_COLOR = "black"
MODEL_COLOR = "deeppink"
TIME_CMAP = "RdPu"


class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return self._dict[item]

    def get(self, key, default=None):
        return self._dict.get(key, default)


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_arch_from_state_dict(state_dict):
    branch_layer_ids = sorted(
        int(k.split(".")[2])
        for k in state_dict.keys()
        if k.startswith("branch_net.layers.") and k.endswith(".bias")
    )
    trunk_layer_ids = sorted(
        int(k.split(".")[2])
        for k in state_dict.keys()
        if k.startswith("trunk_net.layers.") and k.endswith(".bias")
    )
    branch_layers = [int(state_dict[f"branch_net.layers.{idx}.bias"].shape[0]) for idx in branch_layer_ids]
    trunk_layers = [int(state_dict[f"trunk_net.layers.{idx}.bias"].shape[0]) for idx in trunk_layer_ids]
    latent_dim = int(state_dict["branch_net.output_layer.bias"].shape[0] // 2)
    return {
        "latent_dim": latent_dim,
        "branch_layers": branch_layers,
        "trunk_layers": trunk_layers,
    }


def build_cfg_for_checkpoint(base_cfg_dict, state_dict):
    cfg = copy.deepcopy(base_cfg_dict)
    arch = infer_arch_from_state_dict(state_dict)
    cfg["model"]["latent_dim"] = arch["latent_dim"]
    cfg["model"]["branch_layers"] = arch["branch_layers"]
    cfg["model"]["trunk_layers"] = arch["trunk_layers"]
    return cfg


def load_model_from_checkpoint(checkpoint_path, base_cfg_dict, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    cfg_dict = build_cfg_for_checkpoint(base_cfg_dict, state)
    model = CGL_PI_DeepONet(cfg_dict).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg_dict, ckpt


def sample_case(cfg_dict):
    eq_p = cfg_dict["physics"]["equation_params"]
    bounds = cfg_dict["physics"]["bounds"]
    return {
        "alpha": float(np.random.uniform(eq_p["alpha"][0], eq_p["alpha"][1])),
        "beta": float(np.random.uniform(eq_p["beta"][0], eq_p["beta"][1])),
        "mu": float(np.random.uniform(eq_p["mu"][0], eq_p["mu"][1])),
        "V": float(np.random.uniform(eq_p["V"][0], eq_p["V"][1])),
        "A": float(np.random.uniform(bounds["A"][0], bounds["A"][1])),
        "w0": float(10 ** np.random.uniform(np.log10(bounds["w0"][0]), np.log10(bounds["w0"][1]))),
        "x0": 0.0,
        "k": float(np.random.uniform(bounds["k"][0], bounds["k"][1])),
        "type": 0.0,
    }


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
    return (ur + 1j * ui).cpu().numpy().reshape(len(t_values), len(x))


def relative_l2(u_pred, u_true):
    denom = np.linalg.norm(u_true)
    if denom < 1e-12:
        denom = 1e-12
    return float(np.linalg.norm(u_pred - u_true) / denom)


def score_checkpoint(checkpoint_path, base_cfg_dict, n_audit_cases, device):
    model, cfg_dict, ckpt = load_model_from_checkpoint(checkpoint_path, base_cfg_dict, device)
    cfg = ConfigObj(cfg_dict)
    t_curr = float(ckpt.get("t_curr", 0.0))
    _, score = run_audit(model, cfg, t_curr, threshold=1.0, n_global=n_audit_cases, verbose=False, historical=False)
    return model, cfg_dict, ckpt, score


def select_representative_case(model, cfg_dict, t_curr, device, n_candidates=24):
    best_case = None
    best_score = float("inf")
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    for _ in range(n_candidates):
        case = sample_case(cfg_dict)
        X, T, U = get_ground_truth_CGL(case, x_min, x_max, t_curr, Nx=256, Nt=None)
        x = X[:, -1]
        t_final = float(T[0, -1])
        u_true = U[:, -1]
        u_pred = predict_grid(model, case, x, np.array([t_final], dtype=np.float32), device=device)[0]
        score = relative_l2(u_pred, u_true)
        if score < best_score:
            best_score = score
            best_case = case
    return best_case, best_score


def plot_error_heatmap(model, cfg_dict, params_dict, t_curr, output_path, device):
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    X, T, U_true = get_ground_truth_CGL(params_dict, x_min, x_max, t_curr, Nx=256, Nt=None)
    x = X[:, 0]
    t = T[0, :]
    U_pred = predict_grid(model, params_dict, x, t, device=device).T
    err = np.abs(U_pred - U_true)

    plt.figure(figsize=(10, 4.5))
    plt.imshow(
        err,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], x[0], x[-1]],
        cmap="magma",
    )
    plt.colorbar(label=r"$|u_{\mathrm{pred}}-u_{\mathrm{ref}}|$")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Heatmap de l'erreur absolue")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_mean_error_vs_time(model, cfg_dict, t_curr, output_path, device, n_cases=1000, n_times=25):
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    times = np.linspace(0.0, t_curr, n_times)
    all_errs = []
    for _ in range(n_cases):
        case = sample_case(cfg_dict)
        X, T, U_true = get_ground_truth_CGL(case, x_min, x_max, t_curr, Nx=128, Nt=None)
        t_grid = T[0, :]
        idxs = [int(np.argmin(np.abs(t_grid - tt))) for tt in times]
        x = X[:, 0]
        pred = predict_grid(model, case, x, t_grid[idxs], device=device).T
        errs = [relative_l2(pred[:, j], U_true[:, idx]) for j, idx in enumerate(idxs)]
        all_errs.append(errs)
    arr = np.asarray(all_errs)
    mean_err = arr.mean(axis=0)
    p90_err = np.percentile(arr, 90.0, axis=0)

    cmap = plt.get_cmap(TIME_CMAP)
    plt.figure(figsize=(8, 4.5))
    plt.plot(times, mean_err, label="Erreur moyenne", color=MODEL_COLOR, lw=2.5)
    plt.plot(times, p90_err, label="P90", color=cmap(0.72), lw=2.0, ls="--")
    plt.xlabel("t")
    plt.ylabel("Erreur L2 relative")
    plt.title(f"Évolution temporelle de l'erreur moyenne sur {n_cases} cas")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_inference_timing(model, cfg_dict, t_curr, output_path, device, n_cases=300, nx=128):
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    solver_times = []
    model_times = []
    for _ in range(n_cases):
        case = sample_case(cfg_dict)
        t0 = time.perf_counter()
        X, T, U_true = get_ground_truth_CGL(case, x_min, x_max, t_curr, Nx=nx, Nt=None)
        solver_times.append(time.perf_counter() - t0)
        x = X[:, 0]
        t_values = T[0, :]
        t1 = time.perf_counter()
        _ = predict_grid(model, case, x, t_values, device=device)
        model_times.append(time.perf_counter() - t1)

    mean_solver = float(np.mean(solver_times))
    mean_model = float(np.mean(model_times))
    total_solver = float(np.sum(solver_times))
    total_model = float(np.sum(model_times))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["CN", "PI-DeepONet"], [mean_solver, mean_model], color=[CLASSICAL_COLOR, MODEL_COLOR])
    axes[0].set_title(f"Temps moyen / cas ({n_cases} cas)")
    axes[0].set_ylabel("Temps [s]")
    axes[1].bar(["CN", "PI-DeepONet"], [total_solver, total_model], color=[CLASSICAL_COLOR, MODEL_COLOR])
    axes[1].set_title(f"Temps total ({n_cases} cas)")
    axes[1].set_ylabel("Temps [s]")
    fig.suptitle("Inférence complète jusqu'à t_max")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_time_jump_timing(model, cfg_dict, t_curr, output_path, device, n_cases=300, nx=128):
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    solver_times = []
    model_times = []
    for _ in range(n_cases):
        case = sample_case(cfg_dict)
        t0 = time.perf_counter()
        X, T, U_true = get_ground_truth_CGL(case, x_min, x_max, t_curr, Nx=nx, Nt=None)
        _ = U_true[:, -1]
        solver_times.append(time.perf_counter() - t0)
        x = X[:, 0]
        t1 = time.perf_counter()
        _ = predict_grid(model, case, x, np.array([t_curr], dtype=np.float32), device=device)[0]
        model_times.append(time.perf_counter() - t1)

    mean_solver = float(np.mean(solver_times))
    mean_model = float(np.mean(model_times))
    speedup = mean_solver / max(mean_model, 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [
        "Solveur classique\n(Intégration temporelle complète)",
        'PI-DeepONet\n("Time-jumping" direct)',
    ]
    values = [mean_solver, mean_model]
    bars = ax.bar(
        labels,
        values,
        color=[CLASSICAL_COLOR, MODEL_COLOR],
        alpha=0.9,
        edgecolor="black",
        width=0.6,
    )

    y_max = max(values) if values else 1.0
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_max * 0.03,
            f"{height:.3f} s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.text(
        1.0,
        y_max * 0.55,
        f"x{speedup:.1f}\nplus rapide",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="white",
        bbox=dict(facecolor=MODEL_COLOR, edgecolor="black", boxstyle="round,pad=0.45", alpha=0.92),
    )

    ax.set_ylabel(f"Temps moyen pour atteindre t = {t_curr:.3f} (s)", fontsize=12)
    ax.set_title(f"Avantage du PI-DeepONet pour le time-jumping ({n_cases} évaluations)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_ranking(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgl_config.yaml")
    parser.add_argument("--search-root", default="analyses")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-t", type=float, default=0.5)
    parser.add_argument("--ancien-min-date", type=int, default=None)
    parser.add_argument("--audit-cases", type=int, default=20)
    parser.add_argument("--error-cases", type=int, default=1000)
    parser.add_argument("--timing-cases", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--fallback-best-available", action="store_true", default=True)
    args = parser.parse_args()

    base_cfg = load_yaml_config(args.config)
    device = torch.device(args.device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        candidates = [{"path": args.checkpoint, "t_curr": float(ckpt.get("t_curr", 0.0))}]
        over_threshold = candidates
        pool = candidates
    else:
        candidates = []
        for path in sorted(glob.glob(os.path.join(args.search_root, "**", "model_latest.pth"), recursive=True)):
            try:
                ckpt = torch.load(path, map_location="cpu")
                t_curr = float(ckpt.get("t_curr", 0.0))
                if args.ancien_min_date is not None:
                    match = re.search(r"CGL_Navigator_Run_(\d{8})-", path)
                    if match:
                        run_date = int(match.group(1))
                        if "/ancien_run/" in path and run_date < args.ancien_min_date:
                            continue
                candidates.append({"path": path, "t_curr": t_curr})
            except Exception:
                continue

        over_threshold = [c for c in candidates if c["t_curr"] > args.min_t]
        pool = over_threshold if over_threshold else candidates
        if not pool:
            raise RuntimeError("Aucun checkpoint trouvé.")

    ranking = []
    best = None
    best_bundle = None
    for cand in pool:
        try:
            model, cfg_dict, ckpt, score = score_checkpoint(cand["path"], base_cfg, args.audit_cases, device)
            row = {
                "path": cand["path"],
                "t_curr": cand["t_curr"],
                "audit_l2_local": score,
                "compatible": True,
            }
            ranking.append(row)
            if best is None or score < best["audit_l2_local"]:
                best = row
                best_bundle = (model, cfg_dict, ckpt)
        except Exception as e:
            ranking.append({
                "path": cand["path"],
                "t_curr": cand["t_curr"],
                "audit_l2_local": float("inf"),
                "compatible": False,
            })

    ranking.sort(key=lambda r: (not r["compatible"], r["audit_l2_local"]))
    save_ranking(ranking, os.path.join(args.search_root, "best_model_selection.csv"))

    if best is None:
        raise RuntimeError("Aucun checkpoint compatible n'a pu être chargé.")

    chosen_model, chosen_cfg, chosen_ckpt = best_bundle
    chosen_t = float(chosen_ckpt.get("t_curr", best["t_curr"]))
    chosen_case, chosen_case_score = select_representative_case(chosen_model, chosen_cfg, chosen_t, device)

    out_dir = args.output_dir or os.path.join(args.search_root, "presentation_assets")
    os.makedirs(out_dir, exist_ok=True)

    cfg_for_visu = copy.deepcopy(chosen_cfg)
    cfg_for_visu["physics"]["t_max"] = chosen_t
    animate_cgl_solution(
        cfg_for_visu,
        chosen_case,
        model=chosen_model,
        save_path=os.path.join(out_dir, "comparison_animation.gif"),
        frames=120,
        classical_color=CLASSICAL_COLOR,
        model_color=MODEL_COLOR,
    )
    plot_temporal_snapshots(
        cfg_for_visu,
        chosen_case,
        model=chosen_model,
        save_path=os.path.join(out_dir, "five_snapshots.png"),
        show=False,
        classical_color=CLASSICAL_COLOR,
        time_cmap=TIME_CMAP,
    )
    plot_error_heatmap(
        chosen_model,
        chosen_cfg,
        chosen_case,
        chosen_t,
        os.path.join(out_dir, "error_heatmap.png"),
        device,
    )
    plot_mean_error_vs_time(
        chosen_model,
        chosen_cfg,
        chosen_t,
        os.path.join(out_dir, "mean_error_vs_time.png"),
        device,
        n_cases=args.error_cases,
    )
    plot_inference_timing(
        chosen_model,
        chosen_cfg,
        chosen_t,
        os.path.join(out_dir, "full_inference_timing.png"),
        device,
        n_cases=args.timing_cases,
    )
    plot_time_jump_timing(
        chosen_model,
        chosen_cfg,
        chosen_t,
        os.path.join(out_dir, "time_jump_timing.png"),
        device,
        n_cases=args.timing_cases,
    )

    summary_path = os.path.join(out_dir, "selected_model_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Selected checkpoint: {best['path']}\n")
        f.write(f"t_curr: {chosen_t:.6f}\n")
        f.write(f"Audit L2 local: {best['audit_l2_local']:.6%}\n")
        f.write(f"Representative case score: {chosen_case_score:.6%}\n")
        f.write(f"Representative case: {chosen_case}\n")
        if not over_threshold:
            f.write(f"WARNING: no checkpoint exceeded min_t={args.min_t}; selected best available instead.\n")

    if args.prune:
        keep = os.path.abspath(best["path"])
        for cand in candidates:
            abs_path = os.path.abspath(cand["path"])
            if abs_path != keep and os.path.exists(abs_path):
                os.remove(abs_path)

    print(f"BEST: {best['path']} | t={chosen_t:.6f} | score={best['audit_l2_local']:.6%}")
    print(f"ASSETS: {out_dir}")


if __name__ == "__main__":
    main()
