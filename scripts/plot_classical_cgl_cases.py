import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.utils.solver_cgl import get_ground_truth_CGL


CASE_CONFIGS = [
    os.path.join(PROJECT_DIR, "configs/cgl_case_alpha075_beta0_mu0_tchar.yaml"),
    os.path.join(PROJECT_DIR, "configs/cgl_case_alpha075_beta05_mu0_tchar.yaml"),
    os.path.join(PROJECT_DIR, "configs/cgl_case_alpha075_beta0_mu1_tchar.yaml"),
    os.path.join(PROJECT_DIR, "configs/cgl_case_alpha075_beta05_mu1_tchar.yaml"),
]

TIME_RATIOS = [0.0, 0.5, 1.0]
X_VIEW = (-20.0, 20.0)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "plot_presentation", "classical_solver_cases")


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_params(cfg):
    physics = cfg["physics"]
    bounds = physics["bounds"]
    eq = physics["equation_params"]
    return {
        "alpha": float(eq["alpha"][0]),
        "beta": float(eq["beta"][0]),
        "mu": float(eq["mu"][0]),
        "V": float(eq["V"][0]),
        "A": float(bounds["A"][0]),
        "w0": float(bounds["w0"][0]),
        "x0": float(bounds["x0"][0]),
        "k": float(bounds["k"][0]),
        "type": int(physics["initial_conditions"][0]),
    }


def _case_slug(params):
    return (
        f"alpha{params['alpha']:.2f}_beta{params['beta']:.2f}_"
        f"mu{params['mu']:.2f}_V{params['V']:.2f}"
    ).replace(".", "p")


def _plot_case(cfg_path, output_dir, t_max_override=None):
    cfg = _load_yaml(cfg_path)
    params = _build_params(cfg)

    x_min, x_max = cfg["physics"]["x_domain"]
    t_max = float(t_max_override) if t_max_override is not None else float(cfg["physics"]["t_max"])
    eval_times = [ratio * t_max for ratio in TIME_RATIOS]

    x_grid, t_grid, u_true = get_ground_truth_CGL(
        params,
        x_min,
        x_max,
        t_max,
        Nx=512,
        Nt=1000,
    )

    x = x_grid[:, 0]
    t = t_grid[0, :]
    time_indices = [int(np.argmin(np.abs(t - target_t))) for target_t in eval_times]

    colors = ["#1f4e79", "#c0504d", "#76923c"]
    labels = [f"t = {t[idx]:.2f}" for idx in time_indices]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        (
            "Solveur classique CGL | "
            f"$\\alpha$={params['alpha']:.2f}, "
            f"$\\beta$={params['beta']:.2f}, "
            f"$\\mu$={params['mu']:.2f}, "
            f"$V$={params['V']:.2f}, "
            f"$t_{{max}}$={t_max:.2f}"
        ),
        fontsize=15,
        y=0.98,
    )

    panel_specs = [
        ("|u|", lambda u: np.abs(u), "Module"),
        ("Re(u)", lambda u: np.real(u), "Partie reelle"),
        ("Im(u)", lambda u: np.imag(u), "Partie imaginaire"),
    ]

    for ax, (ylabel, transform, title) in zip(axes, panel_specs):
        for idx, color, label in zip(time_indices, colors, labels):
            ax.plot(x, transform(u_true[:, idx]), color=color, linewidth=2.0, label=label)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(*X_VIEW)

    axes[-1].set_xlabel("x", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right")

    os.makedirs(output_dir, exist_ok=True)
    filename = f"classical_snapshots_{_case_slug(params)}.png"
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Generate classical CGL snapshots for fixed cases.")
    parser.add_argument("--t-max", type=float, default=None, help="Override the config t_max for all cases.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for generated figures.")
    args = parser.parse_args()

    saved_paths = []
    for cfg_path in CASE_CONFIGS:
        saved_paths.append(_plot_case(cfg_path, args.output_dir, t_max_override=args.t_max))

    print("Saved figures:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
