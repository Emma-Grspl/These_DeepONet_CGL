import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.plot.plot_animation import animate_cgl_solution
from src.plot.plot_snapshot import plot_temporal_snapshots


CASE_CONFIGS = [
    ("run_alpha075_beta0_mu0", os.path.join(PROJECT_DIR, "configs/cgl_case_alpha075_beta0_mu0_tchar_t5.yaml")),
    ("run_alpha075_beta0_mu1", os.path.join(PROJECT_DIR, "configs/cgl_case_alpha075_beta0_mu1_tchar_t5.yaml")),
    ("run_alpha075_beta05_mu0", os.path.join(PROJECT_DIR, "configs/cgl_case_alpha075_beta05_mu0_tchar_t5.yaml")),
    ("run_alpha075_beta05_mu1", os.path.join(PROJECT_DIR, "configs/cgl_case_alpha075_beta05_mu1_tchar_t5.yaml")),
]

OUTPUT_ROOT = os.path.join(REPO_DIR, "analyses", "classical_solveur")
TIME_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_params(cfg):
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


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for run_name, cfg_path in CASE_CONFIGS:
        cfg = load_yaml(cfg_path)
        params = build_params(cfg)
        out_dir = os.path.join(OUTPUT_ROOT, run_name)
        os.makedirs(out_dir, exist_ok=True)

        plot_temporal_snapshots(
            cfg,
            params,
            model=None,
            save_path=os.path.join(out_dir, "classical_snapshots_t0_t5.png"),
            show=False,
            time_ratios=TIME_RATIOS,
            x_view=tuple(cfg["physics"]["x_domain"]),
        )

        animate_cgl_solution(
            cfg,
            params,
            model=None,
            save_path=os.path.join(out_dir, "classical_animation_t0_t5.gif"),
            frames=160,
            x_view=tuple(cfg["physics"]["x_domain"]),
        )

        with open(os.path.join(out_dir, "case.txt"), "w", encoding="utf-8") as handle:
            handle.write(
                f"alpha={params['alpha']}\n"
                f"beta={params['beta']}\n"
                f"mu={params['mu']}\n"
                f"V={params['V']}\n"
                f"A={params['A']}\n"
                f"w0={params['w0']}\n"
                f"x0={params['x0']}\n"
                f"k={params['k']}\n"
                f"type={params['type']}\n"
                f"t_max={cfg['physics']['t_max']}\n"
            )

        print(f"Generated classical outputs for {run_name} -> {out_dir}")


if __name__ == "__main__":
    main()
