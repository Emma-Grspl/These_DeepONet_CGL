import csv
import json
import os

import numpy as np
import torch
import yaml

from src.utils.solver_cgl import get_ground_truth_CGL


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_benchmark_cfg(cfg):
    root = cfg["benchmark"] if isinstance(cfg, dict) else cfg.benchmark
    defaults = {
        "fixed_cases_path": "benchmarks/cgl_benchmark_v1.yaml",
        "seed": 1234,
        "n_cases": 100,
        "eval_times": [0.1, 0.2, 0.5, 1.0, 5.0],
        "primary_eval_time": None,
        "publish_threshold": 0.04,
        "solver_nx_profile": 512,
        "solver_nx_slab": 256,
        "compute_slab_metric": True,
        "prediction_chunk_size": 32768,
    }
    merged = defaults.copy()
    merged.update(dict(root))
    if merged["primary_eval_time"] is None:
        merged["primary_eval_time"] = float(max(merged["eval_times"]))
    return merged


def _sample_case(rng, cfg):
    bench_cfg = _get_benchmark_cfg(cfg)
    if isinstance(cfg, dict):
        physics_cfg = cfg["physics"]
    else:
        physics_cfg = cfg.physics

    bounds = bench_cfg.get("bounds", physics_cfg["bounds"])
    eq_p = bench_cfg.get("equation_params", physics_cfg["equation_params"])
    return {
        "alpha": float(rng.uniform(eq_p["alpha"][0], eq_p["alpha"][1])),
        "beta": float(rng.uniform(eq_p["beta"][0], eq_p["beta"][1])),
        "mu": float(rng.uniform(eq_p["mu"][0], eq_p["mu"][1])),
        "V": 0.0,
        "A": float(rng.uniform(bounds["A"][0], bounds["A"][1])),
        "w0": float(10 ** rng.uniform(np.log10(bounds["w0"][0]), np.log10(bounds["w0"][1]))),
        "x0": 0.0,
        "k": float(rng.uniform(bounds["k"][0], bounds["k"][1])),
        "type": 0,
    }


def build_fixed_benchmark_cases(cfg, force_rebuild=False):
    bench_cfg = _get_benchmark_cfg(cfg)
    out_path = bench_cfg["fixed_cases_path"]

    if os.path.exists(out_path) and not force_rebuild:
        with open(out_path, "r") as f:
            return yaml.safe_load(f)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.RandomState(int(bench_cfg["seed"]))
    cases = []
    for idx in range(int(bench_cfg["n_cases"])):
        case = _sample_case(rng, cfg)
        case["case_id"] = f"case_{idx:03d}"
        cases.append(case)

    payload = {
        "name": "cgl_benchmark_v1",
        "seed": int(bench_cfg["seed"]),
        "n_cases": len(cases),
        "eval_times": [float(t) for t in bench_cfg["eval_times"]],
        "primary_eval_time": float(bench_cfg["primary_eval_time"]),
        "publish_threshold": float(bench_cfg["publish_threshold"]),
        "cases": cases,
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return payload


def _predict_complex_field(model, params_row, coords_np, chunk_size):
    device = next(model.parameters()).device
    p_vec = np.array(
        [params_row[k] for k in ["alpha", "beta", "mu", "V", "A", "w0", "x0", "k", "type"]],
        dtype=np.float32,
    )
    preds = []
    for start in range(0, len(coords_np), chunk_size):
        stop = min(start + chunk_size, len(coords_np))
        coords_t = torch.tensor(coords_np[start:stop], dtype=torch.float32, device=device)
        params_t = torch.tensor(p_vec, dtype=torch.float32, device=device).unsqueeze(0).repeat(stop - start, 1)
        with torch.no_grad():
            ur, ui = model(params_t, coords_t)
        preds.append((ur + 1j * ui).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def _relative_l2(y_pred, y_true):
    denom = np.linalg.norm(y_true)
    if denom < 1e-12:
        denom = 1e-12
    return float(np.linalg.norm(y_pred - y_true) / denom)


def _relative_l2_amplitude(y_pred, y_true):
    return _relative_l2(np.abs(y_pred), np.abs(y_true))


def _relative_l2_phase_aligned(y_pred, y_true):
    phase = np.angle(np.vdot(y_true, y_pred))
    return _relative_l2(y_pred * np.exp(-1j * phase), y_true)


def evaluate_benchmark_case(model, cfg, case_row, eval_times=None):
    bench_cfg = _get_benchmark_cfg(cfg)
    eval_times = [float(t) for t in (eval_times or bench_cfg["eval_times"])]
    chunk_size = int(bench_cfg["prediction_chunk_size"])
    x_domain = _cfg_get(cfg, "physics")["x_domain"] if isinstance(cfg, dict) else cfg.physics["x_domain"]

    max_t = max(eval_times)
    results = []

    nx_profile = int(bench_cfg["solver_nx_profile"])
    Xp, Tp, Up = get_ground_truth_CGL(case_row, x_domain[0], x_domain[1], max_t, Nx=nx_profile, Nt=None)
    profile_times = Tp[0, :]

    if bench_cfg["compute_slab_metric"]:
        nx_slab = int(bench_cfg["solver_nx_slab"])
        Xs, Ts, Us = get_ground_truth_CGL(case_row, x_domain[0], x_domain[1], max_t, Nx=nx_slab, Nt=None)
        slab_times = Ts[0, :]
    else:
        Xs = Ts = Us = slab_times = None

    for t_eval in eval_times:
        idx_p = int(np.argmin(np.abs(profile_times - t_eval)))
        x_profile = Xp[:, idx_p]
        t_profile = np.full_like(x_profile, profile_times[idx_p], dtype=np.float64)
        coords_profile = np.stack([x_profile, t_profile], axis=1).astype(np.float32)
        u_true_profile = Up[:, idx_p]
        u_pred_profile = _predict_complex_field(model, case_row, coords_profile, chunk_size)

        row = {
            "case_id": case_row["case_id"],
            "t_eval": float(t_eval),
            "t_grid": float(profile_times[idx_p]),
            "alpha": float(case_row["alpha"]),
            "beta": float(case_row["beta"]),
            "mu": float(case_row["mu"]),
            "V": float(case_row["V"]),
            "A": float(case_row["A"]),
            "w0": float(case_row["w0"]),
            "x0": float(case_row["x0"]),
            "k": float(case_row["k"]),
            "type": int(case_row["type"]),
            "l2_profile_complex": _relative_l2(u_pred_profile, u_true_profile),
            "l2_profile_amplitude": _relative_l2_amplitude(u_pred_profile, u_true_profile),
            "l2_profile_phase_aligned": _relative_l2_phase_aligned(u_pred_profile, u_true_profile),
        }

        if bench_cfg["compute_slab_metric"]:
            idx_s = int(np.argmin(np.abs(slab_times - t_eval)))
            x_slab = Xs[:, : idx_s + 1].reshape(-1)
            t_slab = Ts[:, : idx_s + 1].reshape(-1)
            coords_slab = np.stack([x_slab, t_slab], axis=1).astype(np.float32)
            u_true_slab = Us[:, : idx_s + 1].reshape(-1)
            u_pred_slab = _predict_complex_field(model, case_row, coords_slab, chunk_size)
            row["l2_slab_complex"] = _relative_l2(u_pred_slab, u_true_slab)
        else:
            row["l2_slab_complex"] = float("nan")

        results.append(row)

    return results


def evaluate_fixed_benchmark(model, cfg, force_rebuild=False):
    payload = build_fixed_benchmark_cases(cfg, force_rebuild=force_rebuild)
    all_rows = []
    for case in payload["cases"]:
        all_rows.extend(evaluate_benchmark_case(model, cfg, case, eval_times=payload["eval_times"]))
    return payload, all_rows


def _summary_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
    }


def summarize_benchmark_rows(rows, threshold=0.05):
    groups = {}
    for row in rows:
        key = row["t_eval"]
        groups.setdefault(key, []).append(row)

    summary = []
    for key in sorted(groups):
        bucket = groups[key]
        prof = [r["l2_profile_complex"] for r in bucket]
        amp = [r["l2_profile_amplitude"] for r in bucket]
        aligned = [r["l2_profile_phase_aligned"] for r in bucket]
        slab = [r["l2_slab_complex"] for r in bucket if np.isfinite(r["l2_slab_complex"])]
        row = {
            "t_eval": float(key),
            "n_cases": len(bucket),
            "pass_rate_under_threshold": float(np.mean(np.asarray(prof) <= threshold)),
        }
        for prefix, values in [
            ("l2_profile_complex", prof),
            ("l2_profile_amplitude", amp),
            ("l2_profile_phase_aligned", aligned),
        ]:
            for stat_name, value in _summary_stats(values).items():
                row[f"{prefix}_{stat_name}"] = value
        if slab:
            for stat_name, value in _summary_stats(slab).items():
                row[f"l2_slab_complex_{stat_name}"] = value
        summary.append(row)

    overall = {
        "n_rows": len(rows),
        "threshold": float(threshold),
    }
    if rows:
        prof = [r["l2_profile_complex"] for r in rows]
        amp = [r["l2_profile_amplitude"] for r in rows]
        aligned = [r["l2_profile_phase_aligned"] for r in rows]
        slab = [r["l2_slab_complex"] for r in rows if np.isfinite(r["l2_slab_complex"])]
        overall["pass_rate_under_threshold"] = float(np.mean(np.asarray(prof) <= threshold))
        for prefix, values in [
            ("l2_profile_complex", prof),
            ("l2_profile_amplitude", amp),
            ("l2_profile_phase_aligned", aligned),
        ]:
            for stat_name, value in _summary_stats(values).items():
                overall[f"{prefix}_{stat_name}"] = value
        if slab:
            for stat_name, value in _summary_stats(slab).items():
                overall[f"l2_slab_complex_{stat_name}"] = value
    return summary, overall


def summarize_primary_time(summary_rows, primary_eval_time):
    if not summary_rows:
        return {}
    target = float(primary_eval_time)
    best_row = min(summary_rows, key=lambda row: abs(float(row["t_eval"]) - target))
    result = dict(best_row)
    result["primary_eval_time_requested"] = target
    result["primary_eval_time_used"] = float(best_row["t_eval"])
    return result


def write_benchmark_report(payload, summary, overall, primary, output_dir):
    report_path = os.path.join(output_dir, "report.md")
    lines = []
    lines.append("# CGL Benchmark Report")
    lines.append("")
    lines.append(f"- Benchmark: `{payload['name']}`")
    lines.append(f"- Cases: `{payload['n_cases']}`")
    lines.append(f"- Eval times: `{payload['eval_times']}`")
    lines.append(f"- Primary eval time: `{payload['primary_eval_time']}`")
    lines.append(f"- Publish threshold: `{100.0 * payload['publish_threshold']:.2f}%`")
    lines.append("")
    lines.append("## Primary Metric")
    lines.append("")
    if primary:
        lines.append(f"- Time used: `{primary['primary_eval_time_used']}`")
        lines.append(f"- Mean relative complex L2 on final profile: `{100.0 * primary['l2_profile_complex_mean']:.2f}%`")
        lines.append(f"- Median relative complex L2 on final profile: `{100.0 * primary['l2_profile_complex_median']:.2f}%`")
        lines.append(f"- P90 relative complex L2 on final profile: `{100.0 * primary['l2_profile_complex_p90']:.2f}%`")
        lines.append(f"- Pass rate under threshold: `{100.0 * primary['pass_rate_under_threshold']:.1f}%`")
        lines.append(f"- Mean amplitude L2: `{100.0 * primary['l2_profile_amplitude_mean']:.2f}%`")
        lines.append(f"- Mean phase-aligned complex L2: `{100.0 * primary['l2_profile_phase_aligned_mean']:.2f}%`")
        if "l2_slab_complex_mean" in primary:
            lines.append(f"- Mean slab complex L2: `{100.0 * primary['l2_slab_complex_mean']:.2f}%`")
    else:
        lines.append("- No primary summary available.")
    lines.append("")
    lines.append("## Overall Aggregate")
    lines.append("")
    if overall:
        lines.append(f"- Overall mean relative complex L2: `{100.0 * overall.get('l2_profile_complex_mean', float('nan')):.2f}%`")
        lines.append(f"- Overall median relative complex L2: `{100.0 * overall.get('l2_profile_complex_median', float('nan')):.2f}%`")
        lines.append(f"- Overall pass rate under threshold: `{100.0 * overall.get('pass_rate_under_threshold', 0.0):.1f}%`")
    lines.append("")
    lines.append("## By Time")
    lines.append("")
    lines.append("| t | mean L2 | median L2 | p90 | pass rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in summary:
        lines.append(
            f"| {row['t_eval']:.2f} | {100.0 * row['l2_profile_complex_mean']:.2f}% | "
            f"{100.0 * row['l2_profile_complex_median']:.2f}% | "
            f"{100.0 * row['l2_profile_complex_p90']:.2f}% | "
            f"{100.0 * row['pass_rate_under_threshold']:.1f}% |"
        )
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_benchmark_outputs(rows, summary, overall, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    case_csv = os.path.join(output_dir, "case_metrics.csv")
    if rows:
        with open(case_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_csv = os.path.join(output_dir, "summary_by_time.csv")
    if summary:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    overall_json = os.path.join(output_dir, "summary_overall.json")
    with open(overall_json, "w") as f:
        json.dump(overall, f, indent=2)
