import csv
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
INVENTORY = PROJECT_DIR / "analyses" / "single_case_fast" / "single_case_run_inventory.csv"
METRICS = PROJECT_DIR / "analyses" / "single_case_fast" / "single_case_metrics_summary.csv"
REGISTRY = PROJECT_DIR / "run_registry" / "single_case_physics_only_runs.csv"


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


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def upsert_registry(rows):
    fieldnames = ["family", "case", "run_dir", "config", "slurm", "analysis_dir", "status", "notes"]
    existing = read_csv(REGISTRY) if REGISTRY.exists() else []
    replace = {(row["family"], row["case"]) for row in rows}
    kept = [row for row in existing if (row.get("family"), row.get("case")) not in replace]
    kept.extend(rows)
    write_csv(REGISTRY, kept, fieldnames)


def main():
    inventory = read_csv(INVENTORY)
    metrics_by_run = {row["run_dir"]: row for row in read_csv(METRICS)}

    analysis_root = PROJECT_DIR / "analyses" / "single_case" / "local_physics_only"
    asset_root = PROJECT_DIR / "run_assets" / "single_case_physics_only" / "local_mononet"
    ensure_dir(analysis_root)
    ensure_dir(asset_root)

    scripts = [
        PROJECT_DIR / "scripts" / "train_cgl_local_physics_mononet_amp_phase.py",
        PROJECT_DIR / "scripts" / "postprocess_cgl_physics_single_case_amp_phase.py",
    ]
    docs = [
        PROJECT_DIR / "docs" / "protocol" / "local_monoreseau_single_case_physics_only_2026-06-16.md",
    ]
    launcher = PROJECT_DIR / "launch" / "jz_submit_CGL_local_physics_mononet_amp_phase_case_20h.slurm"

    frozen_rows = []
    registry_rows = []
    for row in inventory:
        if row.get("family") != "local_mononet" or row.get("status") != "evaluated":
            continue
        case_horizon = row["case_horizon"]
        run_dir = Path(row["run_dir"])
        cfg = PROJECT_DIR / "configs" / f"cgl_single_case_local_physics_mononet_amp_phase_{case_horizon}.yaml"
        final_ckpt = run_dir / "model_final_local_physics_mononet_amp_phase.pth"
        case_analysis = analysis_root / case_horizon
        case_assets = asset_root / case_horizon

        copy_tree(run_dir / "evaluation", case_analysis / "evaluation")
        copy_tree(run_dir / "evaluation", case_assets / "evaluation")
        for name in ["timing_summary.txt", "timing_stages.csv", "state_bank.csv", "resolved_config.yaml"]:
            copy_file(run_dir / name, case_analysis / name)
            copy_file(run_dir / name, case_assets / name)
        copy_file(final_ckpt, case_assets / "checkpoints" / final_ckpt.name)
        copy_file(cfg, case_assets / "configs" / cfg.name)
        copy_file(launcher, case_assets / "launch" / launcher.name)
        for script in scripts:
            copy_file(script, case_assets / "scripts" / script.name)
        for doc in docs:
            copy_file(doc, case_assets / "docs" / doc.name)

        metric = metrics_by_run.get(str(run_dir), {})
        frozen_row = {
            "family": "local_mononet",
            "case_horizon": case_horizon,
            "run_dir": str(run_dir),
            "config": str(cfg),
            "slurm": str(launcher),
            "checkpoint": str(final_ckpt) if final_ckpt.exists() else "",
            "reached_t": metric.get("global_reached_t", ""),
            "final_rel_l2": metric.get("global_final", ""),
            "max_rel_l2": metric.get("global_max", ""),
            "mean_rel_l2": metric.get("global_mean", ""),
            "analysis_dir": str(case_analysis),
            "asset_dir": str(case_assets),
        }
        frozen_rows.append(frozen_row)

        with open(case_assets / "run_reference.txt", "w", encoding="utf-8") as handle:
            for key, value in frozen_row.items():
                handle.write(f"{key}={value}\n")
            handle.write("status=frozen_no_relaunch\n")
            handle.write("notes=physics-only local mononet single-case frozen as-is; do not relaunch\n")

        registry_rows.append(
            {
                "family": "local_mononet",
                "case": case_horizon,
                "run_dir": str(run_dir),
                "config": str(cfg),
                "slurm": str(launcher),
                "analysis_dir": str(case_analysis),
                "status": "frozen_no_relaunch",
                "notes": "physics-only local mononet single-case frozen as-is; do not relaunch; solver used only for evaluation",
            }
        )

    fields = [
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
        "analysis_dir",
        "asset_dir",
    ]
    write_csv(analysis_root / "family_summary.csv", frozen_rows, fields)
    write_csv(asset_root / "family_summary.csv", frozen_rows, fields)
    upsert_registry(registry_rows)

    status = (
        "status=frozen_no_relaunch\n"
        f"frozen_runs={len(frozen_rows)}\n"
        "kept_artifacts=final_pth, configs, launcher, scripts, evaluation_assets, timing_summary\n"
        "notes=local mononet single-case fige tel quel; ne plus relancer\n"
    )
    (asset_root / "STATUS.txt").write_text(status, encoding="utf-8")
    (analysis_root / "STATUS.txt").write_text(status, encoding="utf-8")
    print(f"Frozen local mononet runs: {len(frozen_rows)}")


if __name__ == "__main__":
    main()
