import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent

CASES = [
    ("alpha075_beta0_mu0", "run_alpha075_beta0_mu0"),
    ("alpha075_beta0_mu1", "run_alpha075_beta0_mu1"),
    ("alpha075_beta05_mu0", "run_alpha075_beta05_mu0"),
    ("alpha075_beta05_mu1", "run_alpha075_beta05_mu1"),
]

GLOBAL_FAMILIES = {
    "global_direct": {
        "label": "Monoreseau global direct",
        "config_template": "configs/cgl_case_{slug}_global_direct_t5.yaml",
        "slurm_template": "launch/jz_submit_CGL_amp_phase_{slug}_global_direct_t5_20h.slurm",
        "analysis_subdir": "global_direct",
        "asset_subdir": "global_direct",
        "scripts": [
            "scripts/train_cgl_amp_phase.py",
            "scripts/postprocess_cgl_direct_single_case_amp_phase.py",
        ],
    },
    "global_curriculum": {
        "label": "Monoreseau global curriculum",
        "config_template": "configs/cgl_case_{slug}_tchar_t5.yaml",
        "slurm_template": "launch/jz_submit_CGL_amp_phase_{slug}_t5_20h.slurm",
        "analysis_subdir": "global_curriculum",
        "asset_subdir": "global_curriculum",
        "scripts": [
            "scripts/train_cgl_amp_phase.py",
            "scripts/postprocess_cgl_direct_single_case_amp_phase.py",
        ],
    },
}

LOCAL_MONONET = {
    "asset_subdir": "local_mononet",
    "analysis_subdir": "local_physics_only",
    "scripts": [
        "scripts/train_cgl_local_physics_mononet_amp_phase.py",
        "scripts/postprocess_cgl_physics_single_case_amp_phase.py",
    ],
    "docs": [
        "docs/protocol/local_monoreseau_single_case_physics_only_2026-06-16.md",
    ],
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def copy_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def copy_tree(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def latest_run_dir(root_dir):
    root = Path(root_dir)
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    return runs[-1] if runs else None


def read_summary(path):
    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            values[key] = value
    return values


def read_metrics_csv(path):
    times = []
    rel_l2 = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(float(row["time"]))
            rel_l2.append(float(row["rel_l2"]))
    return times, rel_l2


def normalize_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def useful_checkpoints(run_dir, summary):
    selected = []
    checkpoint = summary.get("checkpoint")
    if checkpoint:
        selected.append(normalize_path(checkpoint))

    model_latest = run_dir / "checkpoints" / "model_latest.pth"
    if model_latest.exists():
        selected.append(model_latest)

    final_state = run_dir / "model_final_cgl_amp_phase.pth"
    if final_state.exists():
        selected.append(final_state)

    dedup = []
    seen = set()
    for path in selected:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return dedup


def plot_family_overlay(rows, output_dir, title, metric_name, output_name):
    plt.figure(figsize=(8.4, 4.8))
    for row in rows:
        times, rel_l2 = read_metrics_csv(row[metric_name])
        plt.plot(times, rel_l2, linewidth=2.0, label=row["case_slug"])
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="Seuil 5%")
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / output_name, dpi=220)
    plt.close()


def write_csv(rows, path, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def upsert_registry(rows):
    registry_path = PROJECT_DIR / "run_registry" / "single_case_physics_only_runs.csv"
    ensure_dir(registry_path.parent)
    fieldnames = ["family", "case", "run_dir", "config", "slurm", "analysis_dir", "status", "notes"]
    existing = []
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))

    replace_keys = {(row["family"], row["case"]) for row in rows}
    kept = [row for row in existing if (row.get("family"), row.get("case")) not in replace_keys]
    kept.extend(rows)

    with open(registry_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def freeze_global_family(family_key):
    meta = GLOBAL_FAMILIES[family_key]
    analysis_root = PROJECT_DIR / "analyses" / "single_case" / meta["analysis_subdir"]
    asset_root = PROJECT_DIR / "run_assets" / "single_case_physics_only" / meta["asset_subdir"]
    ensure_dir(analysis_root)
    ensure_dir(asset_root)

    rows = []
    registry_rows = []

    for slug, case_name in CASES:
        config_path = PROJECT_DIR / meta["config_template"].format(slug=slug)
        slurm_path = PROJECT_DIR / meta["slurm_template"].format(slug=slug)
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        run_root = PROJECT_DIR / cfg["training"]["save_dir"]
        run_dir = latest_run_dir(run_root)
        if run_dir is None:
            continue

        eval_dir = run_dir / "evaluation"
        summary = read_summary(eval_dir / "summary.txt")
        summary_center_csv = normalize_path(summary["metrics_csv_center_xm10_xp10"])
        summary_csv = normalize_path(summary["metrics_csv"])

        case_analysis_dir = analysis_root / slug
        case_asset_dir = asset_root / slug
        ensure_dir(case_analysis_dir)
        ensure_dir(case_asset_dir / "checkpoints")
        ensure_dir(case_asset_dir / "configs")
        ensure_dir(case_asset_dir / "launch")
        ensure_dir(case_asset_dir / "scripts")

        copy_tree(eval_dir, case_analysis_dir / "evaluation")
        copy_tree(eval_dir, case_asset_dir / "evaluation")
        copy_file(run_dir / "timing_summary.txt", case_analysis_dir / "timing_summary.txt")
        copy_file(run_dir / "timing_summary.txt", case_asset_dir / "timing_summary.txt")
        copy_file(config_path, case_asset_dir / "configs" / config_path.name)
        copy_file(slurm_path, case_asset_dir / "launch" / slurm_path.name)

        for script_rel in meta["scripts"]:
            script_path = PROJECT_DIR / script_rel
            copy_file(script_path, case_asset_dir / "scripts" / script_path.name)

        kept_checkpoints = []
        for ckpt_path in useful_checkpoints(run_dir, summary):
            if copy_file(ckpt_path, case_asset_dir / "checkpoints" / ckpt_path.name):
                kept_checkpoints.append(ckpt_path.name)

        for audit_name in ["hard_audit_cases.csv", "hard_audit_persistent.csv", "hard_audit_summary.csv"]:
            audit_path = run_dir / "checkpoints" / audit_name
            copy_file(audit_path, case_asset_dir / "checkpoints" / audit_name)

        with open(case_asset_dir / "run_reference.txt", "w", encoding="utf-8") as handle:
            handle.write(f"family={family_key}\n")
            handle.write(f"case={slug}\n")
            handle.write(f"run_dir={run_dir}\n")
            handle.write(f"config={config_path}\n")
            handle.write(f"slurm={slurm_path}\n")
            handle.write(f"reached_t={summary.get('reached_t', '')}\n")
            handle.write(f"checkpoint={summary.get('checkpoint', '')}\n")
            handle.write(f"kept_checkpoints={','.join(kept_checkpoints)}\n")

        row = {
            "family": family_key,
            "case_slug": slug,
            "case_name": case_name,
            "run_dir": str(run_dir),
            "config": str(config_path),
            "slurm": str(slurm_path),
            "checkpoint": summary.get("checkpoint", ""),
            "reached_t": summary.get("reached_t", ""),
            "final_rel_l2": summary.get("final_rel_l2", ""),
            "max_rel_l2": summary.get("max_rel_l2", ""),
            "mean_rel_l2": summary.get("mean_rel_l2", ""),
            "first_t_gt_5pct": summary.get("first_t_gt_5pct", ""),
            "final_rel_l2_center_xm10_xp10": summary.get("final_rel_l2_center_xm10_xp10", ""),
            "max_rel_l2_center_xm10_xp10": summary.get("max_rel_l2_center_xm10_xp10", ""),
            "mean_rel_l2_center_xm10_xp10": summary.get("mean_rel_l2_center_xm10_xp10", ""),
            "metrics_csv": str(summary_csv),
            "metrics_csv_center_xm10_xp10": str(summary_center_csv),
            "analysis_dir": str(case_analysis_dir),
            "asset_dir": str(case_asset_dir),
            "kept_checkpoints": ",".join(kept_checkpoints),
        }
        rows.append(row)

        registry_rows.append(
            {
                "family": family_key,
                "case": slug,
                "run_dir": str(run_dir),
                "config": str(config_path),
                "slurm": str(slurm_path),
                "analysis_dir": str(case_analysis_dir),
                "status": "frozen",
                "notes": "physics-only single-case mononet; solver used only for evaluation",
            }
        )

    if not rows:
        return

    fieldnames = [
        "family",
        "case_slug",
        "case_name",
        "run_dir",
        "config",
        "slurm",
        "checkpoint",
        "reached_t",
        "final_rel_l2",
        "max_rel_l2",
        "mean_rel_l2",
        "first_t_gt_5pct",
        "final_rel_l2_center_xm10_xp10",
        "max_rel_l2_center_xm10_xp10",
        "mean_rel_l2_center_xm10_xp10",
        "kept_checkpoints",
        "analysis_dir",
        "asset_dir",
    ]
    write_csv(rows, analysis_root / "family_summary.csv", fieldnames)
    write_csv(rows, asset_root / "family_summary.csv", fieldnames)

    plot_family_overlay(
        rows,
        analysis_root,
        f"{meta['label']} : comparaison des 4 cas",
        "metrics_csv",
        "family_l2_overlay_global.png",
    )
    plot_family_overlay(
        rows,
        analysis_root,
        f"{meta['label']} : comparaison au centre x in [-10, 10]",
        "metrics_csv_center_xm10_xp10",
        "family_l2_overlay_center_xm10_xp10.png",
    )
    copy_file(analysis_root / "family_l2_overlay_global.png", asset_root / "family_l2_overlay_global.png")
    copy_file(
        analysis_root / "family_l2_overlay_center_xm10_xp10.png",
        asset_root / "family_l2_overlay_center_xm10_xp10.png",
    )

    with open(asset_root / "FREEZE_NOTES.txt", "w", encoding="utf-8") as handle:
        handle.write(f"family={family_key}\n")
        handle.write(f"label={meta['label']}\n")
        handle.write("scope=single_case_mononet_physics_only\n")
        handle.write("kept_artifacts=selected_pth, configs, launchers, evaluation_assets, timing_summary\n")

    upsert_registry(registry_rows)


def freeze_local_mononet_protocol():
    asset_root = PROJECT_DIR / "run_assets" / "single_case_physics_only" / LOCAL_MONONET["asset_subdir"]
    analysis_root = PROJECT_DIR / "analyses" / "single_case" / LOCAL_MONONET["analysis_subdir"]
    ensure_dir(asset_root / "configs")
    ensure_dir(asset_root / "launch")
    ensure_dir(asset_root / "scripts")
    ensure_dir(asset_root / "docs")
    ensure_dir(analysis_root)

    for cfg_path in sorted((PROJECT_DIR / "configs").glob("cgl_single_case_local_physics_mononet_amp_phase_*.yaml")):
        copy_file(cfg_path, asset_root / "configs" / cfg_path.name)

    for slurm_path in sorted((PROJECT_DIR / "launch").glob("jz_submit_CGL_local_physics_mononet_amp_phase*.slurm")):
        copy_file(slurm_path, asset_root / "launch" / slurm_path.name)

    for script_rel in LOCAL_MONONET["scripts"]:
        script_path = PROJECT_DIR / script_rel
        copy_file(script_path, asset_root / "scripts" / script_path.name)

    for doc_rel in LOCAL_MONONET["docs"]:
        doc_path = PROJECT_DIR / doc_rel
        copy_file(doc_path, asset_root / "docs" / doc_path.name)

    inventory_path = PROJECT_DIR / "analyses" / "single_case_fast" / "single_case_run_inventory.csv"
    metrics_path = PROJECT_DIR / "analyses" / "single_case_fast" / "single_case_metrics_summary.csv"

    metrics_by_run = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                metrics_by_run[row["run_dir"]] = row

    frozen_rows = []
    registry_rows = []
    if inventory_path.exists():
        with open(inventory_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("family") != "local_mononet" or row.get("status") != "evaluated":
                    continue
                case_horizon = row["case_horizon"]
                run_dir = Path(row["run_dir"])
                config_path = PROJECT_DIR / "configs" / f"cgl_single_case_local_physics_mononet_amp_phase_{case_horizon}.yaml"
                slurm_path = PROJECT_DIR / "launch" / "jz_submit_CGL_local_physics_mononet_amp_phase_case_20h.slurm"
                case_analysis_dir = analysis_root / case_horizon
                case_asset_dir = asset_root / case_horizon

                ensure_dir(case_analysis_dir)
                ensure_dir(case_asset_dir / "checkpoints")
                ensure_dir(case_asset_dir / "configs")
                ensure_dir(case_asset_dir / "launch")
                ensure_dir(case_asset_dir / "scripts")

                eval_dir = run_dir / "evaluation"
                copy_tree(eval_dir, case_analysis_dir / "evaluation")
                copy_tree(eval_dir, case_asset_dir / "evaluation")
                for filename in [
                    "timing_summary.txt",
                    "timing_stages.csv",
                    "state_bank.csv",
                    "resolved_config.yaml",
                ]:
                    copy_file(run_dir / filename, case_analysis_dir / filename)
                    copy_file(run_dir / filename, case_asset_dir / filename)
                final_ckpt = run_dir / "model_final_local_physics_mononet_amp_phase.pth"
                kept_checkpoints = []
                if copy_file(final_ckpt, case_asset_dir / "checkpoints" / final_ckpt.name):
                    kept_checkpoints.append(final_ckpt.name)
                copy_file(config_path, case_asset_dir / "configs" / config_path.name)
                copy_file(slurm_path, case_asset_dir / "launch" / slurm_path.name)

                for script_rel in LOCAL_MONONET["scripts"]:
                    script_path = PROJECT_DIR / script_rel
                    copy_file(script_path, case_asset_dir / "scripts" / script_path.name)

                metric_row = metrics_by_run.get(str(run_dir), {})
                frozen_row = {
                    "family": "local_mononet",
                    "case_horizon": case_horizon,
                    "run_dir": str(run_dir),
                    "config": str(config_path),
                    "slurm": str(slurm_path),
                    "checkpoint": str(final_ckpt) if final_ckpt.exists() else "",
                    "reached_t": metric_row.get("global_reached_t", ""),
                    "final_rel_l2": metric_row.get("global_final", ""),
                    "max_rel_l2": metric_row.get("global_max", ""),
                    "mean_rel_l2": metric_row.get("global_mean", ""),
                    "kept_checkpoints": ",".join(kept_checkpoints),
                    "analysis_dir": str(case_analysis_dir),
                    "asset_dir": str(case_asset_dir),
                }
                frozen_rows.append(frozen_row)

                with open(case_asset_dir / "run_reference.txt", "w", encoding="utf-8") as ref:
                    for key, value in frozen_row.items():
                        ref.write(f"{key}={value}\n")
                    ref.write("status=frozen_no_relaunch\n")
                    ref.write("notes=physics-only local mononet single-case; do not relaunch\n")

                registry_rows.append(
                    {
                        "family": "local_mononet",
                        "case": case_horizon,
                        "run_dir": str(run_dir),
                        "config": str(config_path),
                        "slurm": str(slurm_path),
                        "analysis_dir": str(case_analysis_dir),
                        "status": "frozen_no_relaunch",
                        "notes": "physics-only local mononet single-case frozen as-is; do not relaunch; solver used only for evaluation",
                    }
                )

    if frozen_rows:
        fieldnames = [
            "family",
            "case_horizon",
            "run_dir",
            "config",
            "slurm",
            "checkpoint",
            "reached_t",
            "final_rel_l2",
            "max_rel_l2",
            "mean_rel_l2",
            "kept_checkpoints",
            "analysis_dir",
            "asset_dir",
        ]
        write_csv(frozen_rows, analysis_root / "family_summary.csv", fieldnames)
        write_csv(frozen_rows, asset_root / "family_summary.csv", fieldnames)
        upsert_registry(registry_rows)
        status_text = (
            "status=frozen_no_relaunch\n"
            f"frozen_runs={len(frozen_rows)}\n"
            "kept_artifacts=final_pth, configs, launchers, scripts, evaluation_assets, timing_summary\n"
            "notes=local mononet single-case fige tel quel; ne plus relancer\n"
        )
    else:
        status_text = (
            "status=no_frozen_run\n"
            "reason=aucun run evalue local_mononet trouve dans analyses/single_case_fast/single_case_run_inventory.csv\n"
            "kept_artifacts=configs, launchers, scripts, protocol_doc\n"
        )
    with open(asset_root / "STATUS.txt", "w", encoding="utf-8") as handle:
        handle.write(status_text)
    with open(analysis_root / "README_no_frozen_run_yet.txt", "w", encoding="utf-8") as handle:
        handle.write(status_text)


def main():
    freeze_global_family("global_direct")
    freeze_global_family("global_curriculum")
    freeze_local_mononet_protocol()
    print("Freeze single-case mononet families complete.")


if __name__ == "__main__":
    main()
