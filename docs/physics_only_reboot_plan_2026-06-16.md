# Physics-Only Reboot Plan - 2026-06-16

## Goal

Rebuild the CGL workflow with no classical supervision.

## What remains valid

- `scripts/train_cgl.py`
- `scripts/train_cgl_amp_phase.py`
- `scripts/train_cgl_resume.py`
- `src/training/trainer_CGL_modern.py`
- `src/training/trainer_CGL_legacy.py`
- physics-only single-case configs such as:
  - `configs/cgl_case_*_global_direct_t5.yaml`
  - `configs/cgl_case_*_tchar_t5.yaml`
  - `configs/cgl_single_case.yaml`
  - `configs/cgl_single_case_amp_phase.yaml`

## Immediate plan

1. rerun a clean single-case `global_direct` physics-only baseline
2. rerun a clean single-case `global_causal` physics-only baseline
3. compare these two only
4. decide whether multistage physics-only already exists or must be implemented from the physics-only trainer
5. only after that, design a new parametric pipeline with no direct solver supervision

## Freeze scripts

Scripts now available for the two monoreseau global families:

- `scripts/postprocess_cgl_physics_single_case_amp_phase.py`
- `scripts/freeze_physics_single_case_family.py`

Expected usage after runs exist:

- `python scripts/freeze_physics_single_case_family.py --family global_direct`
- `python scripts/freeze_physics_single_case_family.py --family global_curriculum`

## Important warning

There is currently no validated parametric `physics-only` pipeline left in the repository.

That part has to be rebuilt, not resumed.
