import argparse
import os
import sys

import torch
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.cgl_deeponet import CGL_PI_DeepONet
from src.utils.cgl_benchmark import (
    _get_benchmark_cfg,
    build_fixed_benchmark_cases,
    evaluate_fixed_benchmark,
    summarize_benchmark_rows,
    write_benchmark_outputs,
)


class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return self._dict[item]

    def get(self, key, default=None):
        return self._dict.get(key, default)


def _load_checkpoint(model, checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fixed CGL benchmark.")
    parser.add_argument("--config", type=str, default="configs/cgl_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/cgl_benchmark_eval")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-rebuild-cases", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_data = yaml.safe_load(f)
    cfg = ConfigObj(yaml_data)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CGL_PI_DeepONet(cfg).to(device)
    model.eval()
    _load_checkpoint(model, args.checkpoint, device)

    build_fixed_benchmark_cases(cfg, force_rebuild=args.force_rebuild_cases)
    payload, rows = evaluate_fixed_benchmark(model, cfg, force_rebuild=False)
    bench_cfg = _get_benchmark_cfg(cfg)
    summary, overall = summarize_benchmark_rows(rows, threshold=float(bench_cfg["publish_threshold"]))
    write_benchmark_outputs(rows, summary, overall, args.output_dir)

    print(f"Benchmark: {payload['name']}")
    print(f"Cases: {payload['n_cases']} | Times: {payload['eval_times']}")
    print(f"Threshold: {100.0 * bench_cfg['publish_threshold']:.2f}%")
    print(f"Overall mean L2(profile): {100.0 * overall['l2_profile_complex_mean']:.2f}%")
    print(f"Overall median L2(profile): {100.0 * overall['l2_profile_complex_median']:.2f}%")
    print(f"Overall p90 L2(profile): {100.0 * overall['l2_profile_complex_p90']:.2f}%")
    print(f"Pass rate < threshold: {100.0 * overall['pass_rate_under_threshold']:.1f}%")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
