import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CASES = [
    ("run_alpha075_beta0_mu0", "alpha075_beta0_mu0"),
    ("run_alpha075_beta0_mu1", "alpha075_beta0_mu1"),
    ("run_alpha075_beta05_mu0", "alpha075_beta05_mu0"),
    ("run_alpha075_beta05_mu1", "alpha075_beta05_mu1"),
]

FAMILIES = {
    "global_direct": {
        "label": "Monoreseau global direct",
        "analysis_subdir": "global_direct",
        "run_assets_subdir": "global_direct",
        "config_template": "configs/cgl_case_{slug}_global_direct_t5.yaml",
        "slurm_template": "launch/jz_submit_CGL_amp_phase_{slug}_global_direct_t5_20h.slurm",
    },
    "global_curriculum": {
        "label": "Monoreseau global curriculum",
        "analysis_subdir": "global_curriculum",
        "run_assets_subdir": "global_curriculum",
        "config_template": "configs/cgl_case_{slug}_tchar_t5.yaml",
        "slurm_template": "launch/jz_submit_CGL_amp_phase_{slug}_t5_20h.slurm",
    },
}


def latest_run_dir(root_dir):
    root = Path(root_dir)
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    return runs[-1] if runs else None


def read_rollout_summary(path):
    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            values[key] = value
    return values


def read_rollout_csv(path):
    times = []
    rel_l2 = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(float(row["time"]))
            rel_l2.append(float(row["rel_l2"]))
    return np.asarray(times), np.asarray(rel_l2)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True


def build_case_label(case_name):
    return case_name.replace("run_", "").replace("_", " | ")


def plot_family_overlay(rows, output_dir, family_label):
    plt.figure(figsize=(8.4, 4.8))
    for row in rows:
        times, rel_l2 = read_rollout_csv(row["metrics_csv"])
        plt.plot(times, rel_l2, linewidth=2.0, label=row["case"])
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="Seuil 5%")
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title(f"{family_label} : comparaison des 4 cas")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "family_l2_overlay.png", dpi=220)
    plt.close()


def write_family_summary(rows, output_dir):
    fieldnames = [
        "case",
        "config",
        "slurm",
        "run_dir",
        "checkpoint",
        "reached_t",
        "final_rel_l2",
        "max_rel_l2",
        "mean_rel_l2",
        "first_t_gt_5pct",
        "analysis_dir",
        "metrics_csv",
    ]
    with open(Path(output_dir) / "family_summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(rows, family_key):
    registry_path = PROJECT_DIR / "run_registry" / "single_case_physics_only_runs.csv"
    ensure_dir(registry_path.parent)
    exists = registry_path.exists()
    fieldnames = ["family", "case", "run_dir", "config", "slurm", "analysis_dir", "status", "notes"]
    with open(registry_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "family": family_key,
                    "case": row["case"],
                    "run_dir": row["run_dir"],
                    "config": row["config"],
                    "slurm": row["slurm"],
                    "analysis_dir": row["analysis_dir"],
                    "status": "frozen",
                    "notes": "physics-only benchmark against solver outside loss",
                }
            )


def build_assets(rows, family_key):
    family_dir = PROJECT_DIR / "run_assets" / "single_case_physics_only" / FAMILIES[family_key]["run_assets_subdir"]
    for row in rows:
        case_dir = family_dir / row["case"]
        ensure_dir(case_dir / "analysis")
        ensure_dir(case_dir / "configs")
        ensure_dir(case_dir / "launch")
        copy_if_exists(row["analysis_dir"], case_dir / "analysis")
        copy_if_exists(row["config"], case_dir / "configs" / Path(row["config"]).name)
        copy_if_exists(row["slurm"], case_dir / "launch" / Path(row["slurm"]).name)
        with open(case_dir / "run_reference.txt", "w", encoding="utf-8") as handle:
            handle.write(f"run_dir={row['run_dir']}\n")
            handle.write(f"checkpoint={row['checkpoint']}\n")


def process_family(family_key, dry_run=False):
    meta = FAMILIES[family_key]
    family_analysis_dir = PROJECT_DIR / "analyses" / "single_case" / meta["analysis_subdir"]
    ensure_dir(family_analysis_dir)

    rows = []
    missing = []
    for case_name, slug in CASES:
        config_path = PROJECT_DIR / meta["config_template"].format(slug=slug)
        slurm_path = PROJECT_DIR / meta["slurm_template"].format(slug=slug)
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        run_root = PROJECT_DIR / cfg["training"]["save_dir"]
        run_dir = latest_run_dir(run_root)
        if run_dir is None:
            missing.append(case_name)
            continue
        analysis_dir = family_analysis_dir / case_name
        if not dry_run:
            cmd = [
                sys.executable,
                str(PROJECT_DIR / "scripts" / "postprocess_cgl_physics_single_case_amp_phase.py"),
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(analysis_dir),
                "--label",
                f"{meta['label']} | {build_case_label(case_name)}",
            ]
            subprocess.run(cmd, check=True, cwd=str(PROJECT_DIR))
            summary = read_rollout_summary(analysis_dir / "summary.txt")
        else:
            summary = {
                "checkpoint": "",
                "reached_t": "",
                "final_rel_l2": "",
                "max_rel_l2": "",
                "mean_rel_l2": "",
                "first_t_gt_5pct": "",
            }
        rows.append(
            {
                "case": case_name,
                "config": str(config_path),
                "slurm": str(slurm_path),
                "run_dir": str(run_dir),
                "checkpoint": summary.get("checkpoint", ""),
                "reached_t": summary.get("reached_t", ""),
                "final_rel_l2": summary.get("final_rel_l2", ""),
                "max_rel_l2": summary.get("max_rel_l2", ""),
                "mean_rel_l2": summary.get("mean_rel_l2", ""),
                "first_t_gt_5pct": summary.get("first_t_gt_5pct", ""),
                "analysis_dir": str(analysis_dir),
                "metrics_csv": str(analysis_dir / "rollout_metrics.csv"),
            }
        )

    if missing:
        print(f"[{family_key}] missing runs for: {', '.join(missing)}")
    if dry_run or missing:
        return rows, missing

    write_family_summary(rows, family_analysis_dir)
    plot_family_overlay(rows, family_analysis_dir, meta["label"])
    append_registry(rows, family_key)
    build_assets(rows, family_key)
    return rows, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES.keys()), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows, missing = process_family(args.family, dry_run=args.dry_run)
    print(f"family={args.family} rows={len(rows)} missing={len(missing)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
