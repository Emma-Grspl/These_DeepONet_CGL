import argparse
import os
import sys

import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from scripts.train_cgl_global_multinet_physics_only_amp_phase import load_stage_summary, run_postprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    stage_rows = []
    time_blocks = [tuple(map(float, block)) for block in cfg_dict["multistage"]["time_blocks"]]
    for stage_idx, (t_start, t_end) in enumerate(time_blocks):
        stage_dir = os.path.join(args.run_dir, f"stage_{stage_idx:02d}_t{t_start:.2f}_{t_end:.2f}")
        summary = load_stage_summary(stage_dir)
        if summary is None:
            raise FileNotFoundError(f"Stage summary missing: {stage_dir}")
        stage_rows.append(
            {
                "stage_idx": stage_idx,
                "stage_label": f"{t_start:.2f}_{t_end:.2f}",
                "t_start": float(t_start),
                "t_end": float(t_end),
                "best_score": float(summary["best_score"]),
                "final_score": float(summary["final_score"]),
                "wall_seconds": 0.0,
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_postprocess(cfg_dict, args.run_dir, stage_rows, device)


if __name__ == "__main__":
    main()
