# Audit Supervision - 2026-06-16

## Conclusion courte

Le depot contenait deux familles de pipelines distinctes :

- une branche `physics-only` basee sur residu PDE, BC, continuity et audits
- une branche `supervisee` qui apprend explicitement sur des cibles du solveur de reference

Les campagnes parametriques recentes et une grande partie des campagnes multistage/locales venaient de la branche supervisee.

## Branches physics-only

### Scripts

- `scripts/train_cgl.py`
- `scripts/train_cgl_amp_phase.py`

### Backend de training

- `src/training/trainer_CGL_modern.py`
- `src/training/trainer_CGL_legacy.py`
- `src/physics/pde_cgl.py`

### Signes distinctifs

- pas de `target_re` / `target_im` du solveur dans la loss
- perte fondee sur :
  - residu PDE
  - contraintes aux bords
  - balance de masse
  - continuity loss
- comparaison au solveur autorisee uniquement :
  - en benchmark
  - en post-traitement
  - en suivi externe
  - jamais dans la loss

## Branches supervisees

### Single-case supervise

- `scripts/train_cgl_global_multistage_amp_phase.py`
- `scripts/train_cgl_local_amp_phase.py`
- `scripts/train_cgl_local_amp_phase_harmonized.py`
- `scripts/train_cgl_local_multistage_amp_phase.py`
- `scripts/train_cgl_local_multistage_reim.py`

### Parametrique supervise

- `scripts/train_cgl_global_multistage_parametric_amp_phase.py`

### Signes distinctifs

- construction explicite de `target_re` / `target_im` ou `target_u_*`
- losses de type MSE prediction-cible

## Configurations supervisees

### Toutes les configs parametriques recentes

Toutes les configs sous :

- `configs/parameters/`

etaient reliees au script :

- `scripts/train_cgl_global_multistage_parametric_amp_phase.py`

Donc :

- `alpha_only`
- `beta_only`
- `mu_only`
- `sigma_only`
- `alpha_beta_mu_sigma08`

dans leur forme recente, etaient supervisees.

### Single-case supervise

Toutes les configs de type :

- `configs/cgl_single_case_global_multistage_amp_phase_*.yaml`
- `configs/cgl_single_case_local_direct_amp_phase_*.yaml`
- `configs/cgl_single_case_local_direct_residual_multistep_amp_phase_*.yaml`
- `configs/cgl_single_case_local_multistage_amp_phase_*.yaml`
- `configs/cgl_single_case_local_multistage_overlap_*.yaml`
- `configs/cgl_single_case_local_multistage_overlap_warmstart_*.yaml`
- `configs/cgl_single_case_local_multistage_reim_*.yaml`

etaient supervisees.

## Single-case non supervises

Les single-case suivants relevent de la branche physics-only :

- `configs/cgl_case_*_global_direct_t5.yaml`
- `configs/cgl_case_*_tchar.yaml`
- `configs/cgl_case_*_tchar_t5.yaml`
- `configs/cgl_single_case.yaml`
- `configs/cgl_single_case_amp_phase.yaml`
- `configs/cgl_single_case_hard.yaml`
- `configs/cgl_single_case_hard_resume_t15.yaml`
- `configs/cgl_single_case_resume_1p2.yaml`
- `configs/cgl_single_case_resume_t15.yaml`

## Nettoyage applique le 2026-06-16

Ont ete supprimes car issus de runs supervises :

- tous les sous-dossiers de `results/`
- `analyses/parameters/`
- les syntheses `analyses/single_case/` fondees sur les campagnes supervisees
- les manifestes `run_assets/` et `run_registry/` derives de ces runs

Ont ete conserves :

- le code
- les configs
- `analyses/single_case/reference_solver/`

## Ce qu'il faut faire maintenant

1. repartir d'un single-case `physics-only` propre avec `train_cgl_amp_phase.py`
2. revalider explicitement `global_direct` et `global_causal` sans supervision
3. verifier si un vrai `multistage physics-only` existe deja ou doit etre recode
4. ne rouvrir le parametrique qu'apres validation single-case non supervisee
