import argparse
import os
import sys

import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from scripts.train_cgl_local_multinet_physics_only_amp_phase import (
    evaluate_and_save,
    fixed_case_setup,
    load_or_init_models,
    load_time_blocks,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params, periodic, x_sensor, u0_sensor = fixed_case_setup(cfg_dict)
    models = load_or_init_models(cfg_dict, args.run_dir, device)
    time_blocks = load_time_blocks(cfg_dict)
    evaluate_and_save(models, time_blocks, cfg_dict, params, periodic, x_sensor, u0_sensor, args.run_dir, device, label="evaluation")


if __name__ == "__main__":
    main()
