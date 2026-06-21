import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
OUT_DIR = PROJECT_DIR / "analyses" / "single_case_fast"

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#e377c2",
]


def classify_experiment(name):
    if name.startswith("CGL_AmpPhase_") and "_global_direct_t5" in name:
        return "global_direct"
    if name.startswith("CGL_AmpPhase_") and "_tchar_t5" in name:
        return "global_curriculum"
    if name.startswith("CGL_LocalMononet_PhysicsOnly_AmpPhase_"):
        return "local_mononet"
    if name.startswith("CGL_GlobalMultinet_PhysicsOnly_AmpPhase_"):
        return "global_multinet"
    if name.startswith("CGL_GlobalMultinet_PhysicsOnly_Historical_AmpPhase_"):
        return "global_multinet_historical"
    if name.startswith("CGL_LocalMultinet_PhysicsOnly_AmpPhase_"):
        return "local_multinet"
    return None


def family_label(family):
    labels = {
        "global_direct": "Global direct",
        "global_curriculum": "Global curriculum",
        "local_mononet": "Local mononet",
        "global_multinet": "Global multinet",
        "global_multinet_historical": "Global multinet historical",
        "local_multinet": "Local multinet",
    }
    return labels.get(family, family)


def case_label(name):
    tokens = name.split("_")
    for idx, token in enumerate(tokens):
        if token.startswith("alpha") and idx + 2 < len(tokens):
            return "_".join(tokens[idx : idx + 3])
    return name


def horizon_label(name):
    if name.endswith("_t1"):
        return "t1"
    if name.endswith("_t5"):
        return "t5"
    return "unknown"


def read_metric_csv(path):
    times = []
    values = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                time = float(row["time"])
                value = float(row["rel_l2"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(time) and math.isfinite(value)):
                continue
            times.append(time)
            values.append(value)
    return times, values


def first_above(times, values, threshold=0.05):
    for time, value in zip(times, values):
        if value > threshold:
            return time
    return math.nan


def stats(times, values):
    if not values:
        return {
            "reached_t": math.nan,
            "final": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "first_t_gt_5pct": math.nan,
        }
    return {
        "reached_t": times[-1],
        "final": values[-1],
        "max": max(values),
        "mean": sum(values) / len(values),
        "first_t_gt_5pct": first_above(times, values, 0.05),
    }


def discover_runs(results_dir):
    inventory = []
    metrics = []
    for exp_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        family = classify_experiment(exp_dir.name)
        if family is None:
            continue
        case = case_label(exp_dir.name)
        horizon = horizon_label(exp_dir.name)
        case_horizon = f"{case}_{horizon}"
        for run_dir in sorted(path for path in exp_dir.iterdir() if path.is_dir() and path.name.startswith("run_")):
            eval_dir = run_dir / "evaluation"
            csv_global = eval_dir / "rollout_metrics.csv"
            csv_center = eval_dir / "rollout_metrics_center_xm10_xp10.csv"
            pth_count = sum(1 for _ in run_dir.rglob("*.pth"))
            snapshot = eval_dir / "snapshots.png"
            has_metrics = csv_global.exists()
            if has_metrics:
                status = "evaluated"
            elif pth_count > 0:
                status = "checkpoint_only"
            elif (run_dir / "timing_summary.txt").exists():
                status = "partial"
            else:
                status = "empty_or_missing"
            inventory.append(
                {
                    "experiment": exp_dir.name,
                    "family": family,
                    "case": case,
                    "horizon": horizon,
                    "case_horizon": case_horizon,
                    "run_dir": str(run_dir),
                    "status": status,
                    "pth_count": pth_count,
                    "has_snapshot": snapshot.exists(),
                }
            )
            if not has_metrics:
                continue
            times, values = read_metric_csv(csv_global)
            global_stats = stats(times, values)
            if csv_center.exists():
                center_times, center_values = read_metric_csv(csv_center)
                center_stats = stats(center_times, center_values)
            else:
                center_times = []
                center_values = []
                center_stats = stats([], [])
            metrics.append(
                {
                    "experiment": exp_dir.name,
                    "family": family,
                    "case": case,
                    "horizon": horizon,
                    "case_horizon": case_horizon,
                    "run_dir": str(run_dir),
                    "times": times,
                    "values": values,
                    "center_times": center_times,
                    "center_values": center_values,
                    "snapshots_png": str(snapshot) if snapshot.exists() else "",
                    **{f"global_{key}": value for key, value in global_stats.items()},
                    **{f"center_{key}": value for key, value in center_stats.items()},
                }
            )
    return inventory, metrics


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt_pct(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def slug(text):
    safe = []
    for char in text:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


def downsample_xy(xs, ys, max_points=900):
    if len(xs) <= max_points:
        return xs, ys
    step = max(1, math.ceil(len(xs) / max_points))
    sampled_x = xs[::step]
    sampled_y = ys[::step]
    if sampled_x[-1] != xs[-1]:
        sampled_x.append(xs[-1])
        sampled_y.append(ys[-1])
    return sampled_x, sampled_y


def svg_polyline(xs, ys, x_min, x_max, y_min, y_max, width, height, pad):
    if not xs:
        return ""
    denom_x = max(1.0e-12, x_max - x_min)
    denom_y = max(1.0e-12, y_max - y_min)
    points = []
    for x_val, y_val in zip(xs, ys):
        px = pad + (x_val - x_min) / denom_x * (width - 2 * pad)
        py = height - pad - (y_val - y_min) / denom_y * (height - 2 * pad)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def make_svg_plot(series, title, path):
    width = 960
    height = 540
    pad = 64
    valid = []
    for row in series:
        points = [
            (time, value)
            for time, value in zip(row["times"], row["values"])
            if math.isfinite(time) and math.isfinite(value)
        ]
        if not points:
            continue
        times, values = zip(*points)
        valid.append({**row, "times": list(times), "values": list(values)})
    if not valid:
        return
    x_min = min(min(row["times"]) for row in valid)
    x_max = max(max(row["times"]) for row in valid)
    y_min = 0.0
    y_max = max(max(row["values"]) for row in valid)
    y_max = max(0.055, y_max * 1.05)
    threshold_y = height - pad - (0.05 - y_min) / max(1.0e-12, y_max - y_min) * (height - 2 * pad)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="32" font-family="Arial, sans-serif" font-size="22" fill="#111">{html.escape(title)}</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{pad}" y1="{threshold_y:.2f}" x2="{width-pad}" y2="{threshold_y:.2f}" stroke="#111" stroke-dasharray="5 5" stroke-width="1"/>',
        f'<text x="{width-pad-70}" y="{threshold_y-6:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#111">5%</text>',
        f'<text x="{width/2 - 40:.2f}" y="{height-18}" font-family="Arial, sans-serif" font-size="14" fill="#111">Temps t</text>',
        f'<text x="16" y="{height/2:.2f}" font-family="Arial, sans-serif" font-size="14" fill="#111" transform="rotate(-90 16 {height/2:.2f})">Erreur L2 relative</text>',
    ]

    for idx in range(6):
        frac = idx / 5.0
        x_val = x_min + frac * (x_max - x_min)
        px = pad + frac * (width - 2 * pad)
        lines.append(f'<line x1="{px:.2f}" y1="{pad}" x2="{px:.2f}" y2="{height-pad}" stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="{px-12:.2f}" y="{height-pad+20}" font-family="Arial, sans-serif" font-size="12" fill="#333">{x_val:.2g}</text>')
    for idx in range(6):
        frac = idx / 5.0
        y_val = y_min + frac * (y_max - y_min)
        py = height - pad - frac * (height - 2 * pad)
        lines.append(f'<line x1="{pad}" y1="{py:.2f}" x2="{width-pad}" y2="{py:.2f}" stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="{pad-54}" y="{py+4:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#333">{100*y_val:.0f}%</text>')

    legend_x = width - pad - 260
    legend_y = pad + 10
    for idx, row in enumerate(valid):
        xs, ys = downsample_xy(list(row["times"]), list(row["values"]))
        color = COLORS[idx % len(COLORS)]
        points = svg_polyline(xs, ys, x_min, x_max, y_min, y_max, width, height, pad)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        label = row["label"]
        y = legend_y + idx * 20
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
        lines.append(f'<text x="{legend_x+30}" y="{y+4}" font-family="Arial, sans-serif" font-size="12" fill="#111">{html.escape(label)}</text>')

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(metrics, output_dir):
    by_family = defaultdict(list)
    by_case_horizon = defaultdict(list)
    for row in metrics:
        by_family[row["family"]].append(row)
        by_case_horizon[row["case_horizon"]].append(row)

    for family, rows in by_family.items():
        make_svg_plot(
            [
                {"label": f"{row['case']}_{row['horizon']}", "times": row["times"], "values": row["values"]}
                for row in rows
            ],
            f"{family_label(family)} - L2 global",
            output_dir / f"{family}_l2_global.svg",
        )
        make_svg_plot(
            [
                {
                    "label": f"{row['case']}_{row['horizon']}",
                    "times": row["center_times"],
                    "values": row["center_values"],
                }
                for row in rows
                if row["center_times"]
            ],
            f"{family_label(family)} - L2 centre x in [-10,10]",
            output_dir / f"{family}_l2_center_xm10_xp10.svg",
        )

    for case_horizon, rows in by_case_horizon.items():
        make_svg_plot(
            [
                {"label": family_label(row["family"]), "times": row["times"], "values": row["values"]}
                for row in rows
            ],
            f"{case_horizon} - comparaison familles, L2 global",
            output_dir / f"{slug(case_horizon)}_families_l2_global.svg",
        )
        make_svg_plot(
            [
                {
                    "label": family_label(row["family"]),
                    "times": row["center_times"],
                    "values": row["center_values"],
                }
                for row in rows
                if row["center_times"]
            ],
            f"{case_horizon} - comparaison familles, L2 centre",
            output_dir / f"{slug(case_horizon)}_families_l2_center_xm10_xp10.svg",
        )


def best_rows(metrics):
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["family"], row["case"], row["horizon"])].append(row)
    rows = []
    for (family, case, horizon), values in sorted(grouped.items()):
        best = sorted(values, key=lambda item: (-item["global_reached_t"], item["global_mean"]))[0]
        rows.append(
            {
                "family": family,
                "case": case,
                "horizon": horizon,
                "n_runs": len(values),
                "best_run_dir": best["run_dir"],
                "reached_t": best["global_reached_t"],
                "mean_rel_l2": best["global_mean"],
                "max_rel_l2": best["global_max"],
                "final_rel_l2": best["global_final"],
                "mean_rel_l2_center": best["center_mean"],
                "max_rel_l2_center": best["center_max"],
                "final_rel_l2_center": best["center_final"],
                "snapshots_png": best["snapshots_png"],
            }
        )
    return rows


def write_report(inventory, metrics, best, output_dir):
    inventory_counts = defaultdict(int)
    for row in inventory:
        inventory_counts[row["status"]] += 1

    by_family = defaultdict(list)
    for row in metrics:
        by_family[row["family"]].append(row)

    path = output_dir / "README.md"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Analyse CGL single case\n\n")
        handle.write("## Inventaire\n\n")
        for key, value in sorted(inventory_counts.items()):
            handle.write(f"- {key}: {value}\n")
        handle.write("\n")
        handle.write("## Synthese famille\n\n")
        for family, rows in sorted(by_family.items()):
            mean_val = sum(row["global_mean"] for row in rows) / len(rows)
            max_val = sum(row["global_max"] for row in rows) / len(rows)
            reached = max(row["global_reached_t"] for row in rows)
            handle.write(
                f"- {family_label(family)}: {len(rows)} run(s) evalues, "
                f"mean L2 moyen={fmt_pct(mean_val)}, max L2 moyen={fmt_pct(max_val)}, "
                f"t max atteint={reached:.3g}\n"
            )
        handle.write("\n")
        handle.write("## Diagnostic rapide\n\n")
        handle.write(
            "- Le curriculum mono-reseau global est le seul protocole precis dans ce lot "
            "(mean L2 autour de 3%), mais les runs evalues ne couvrent pas encore t=5 sur tous les cas.\n"
        )
        handle.write(
            "- Le mono-reseau global direct couvre t=5, mais l'erreur est beaucoup trop grande "
            "(mean L2 autour de 71%). Il ne doit pas servir de configuration finale.\n"
        )
        handle.write(
            "- Le multireseau global non historique couvre t=5 sur plusieurs cas, mais reste loin du curriculum "
            "(mean L2 t5 autour de 60%). Le protocole historique relance ensuite n'est pas encore evaluable "
            "localement: il y a seulement des checkpoints ou des dossiers vides.\n"
        )
        handle.write(
            "- Les protocoles locaux sont instables dans l'etat actuel: erreurs deja fortes a t=1 et explosions "
            "a t=5, surtout beta05_mu1. Il faut diagnostiquer separement l'operateur local et le rollout.\n"
        )
        handle.write(
            "- Les courbes centre x in [-10,10] sont filtrees contre les nan. Local mononet n'a pas de metrique centre "
            "exploitable dans les CSV locaux.\n"
        )
        handle.write("\n")
        handle.write("## Meilleur run par famille/cas/horizon\n\n")
        handle.write("| Famille | Cas | Horizon | t atteint | mean L2 | max L2 | final L2 | run |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in best:
            handle.write(
                f"| {family_label(row['family'])} | {row['case']} | {row['horizon']} | "
                f"{float(row['reached_t']):.3g} | {fmt_pct(row['mean_rel_l2'])} | "
                f"{fmt_pct(row['max_rel_l2'])} | {fmt_pct(row['final_rel_l2'])} | "
                f"{Path(row['best_run_dir']).name} |\n"
            )
        handle.write("\n")
        handle.write("## Assets\n\n")
        handle.write("- `single_case_run_inventory.csv`: etat local des runs.\n")
        handle.write("- `single_case_metrics_summary.csv`: metriques par run evalue.\n")
        handle.write("- `single_case_best_by_family_case.csv`: meilleur run par famille/cas/horizon.\n")
        handle.write("- `*.svg`: overlays L2 global et centre.\n")
        handle.write("- Les courbes d'amplitude sont les `snapshots.png` dans chaque dossier `evaluation`; premiere ligne = module `|u|`.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory, metrics = discover_runs(results_dir)
    best = best_rows(metrics)

    inventory_fields = [
        "experiment",
        "family",
        "case",
        "horizon",
        "case_horizon",
        "run_dir",
        "status",
        "pth_count",
        "has_snapshot",
    ]
    metric_fields = [
        "experiment",
        "family",
        "case",
        "horizon",
        "case_horizon",
        "run_dir",
        "global_reached_t",
        "global_final",
        "global_max",
        "global_mean",
        "global_first_t_gt_5pct",
        "center_reached_t",
        "center_final",
        "center_max",
        "center_mean",
        "center_first_t_gt_5pct",
        "snapshots_png",
    ]
    best_fields = [
        "family",
        "case",
        "horizon",
        "n_runs",
        "best_run_dir",
        "reached_t",
        "mean_rel_l2",
        "max_rel_l2",
        "final_rel_l2",
        "mean_rel_l2_center",
        "max_rel_l2_center",
        "final_rel_l2_center",
        "snapshots_png",
    ]

    write_csv(output_dir / "single_case_run_inventory.csv", inventory, inventory_fields)
    write_csv(output_dir / "single_case_metrics_summary.csv", metrics, metric_fields)
    write_csv(output_dir / "single_case_best_by_family_case.csv", best, best_fields)
    write_plots(metrics, output_dir)
    write_report(inventory, metrics, best, output_dir)
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
