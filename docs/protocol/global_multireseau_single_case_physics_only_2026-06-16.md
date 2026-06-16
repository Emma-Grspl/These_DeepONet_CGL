# Protocole Global Multireseau Single-Case Physics-Only - 2026-06-16

## Contrainte dure

- aucune supervision solveur dans la loss
- aucune regression directe vers des snapshots solveur
- solveur autorise uniquement pour audit et benchmark

## Idee retenue

On reprend la structure historique du global multistage :

- un reseau par bloc temporel
- warm-start du stage `k` depuis le stage `k-1`
- prediction finale piecewise : chaque reseau couvre son propre bloc

La difference decisive est la loss :

- residu PDE dans le bloc
- conditions aux bords
- contrainte de balance de masse
- contrainte de continuite au temps d'interface contre le **modele precedent**, pas contre le solveur

## Ce qui remplace la supervision

Avant :

- chaque stage apprenait directement sur des cibles solveur dans son bloc

Maintenant :

- le stage apprend uniquement la physique
- l'interface `t=t_start` est reglee par une penalisation student/teacher
- le teacher est le meilleur reseau du stage precedent

Donc :

- solveur dans la loss : non
- solveur dans l'audit : oui
- coherence inter-stage : oui

## Fichiers

- `scripts/train_cgl_global_multinet_physics_only_amp_phase.py`
- `scripts/postprocess_cgl_global_multinet_physics_only_amp_phase.py`
- `launch/jz_submit_CGL_global_multinet_physics_only_amp_phase_case_20h.slurm`

Configs deja preparees :

- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t1.yaml`
- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml`
- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu1_t5.yaml`
- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta05_mu0_t5.yaml`
- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta05_mu1_t5.yaml`

## Strategie conseillee

### Etape 1. Validation rapide sur `t=1`

Cas :

- `alpha=0.75, beta=0.0, mu=0.0`

Config :

- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t1.yaml`

Objectif :

- verifier qu'on n'a pas de rupture nette a l'interface
- verifier que la continuite inter-stage suffit sans solveur

### Etape 2. Extension a `t=5` sur le meme cas

Config :

- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml`

Objectif :

- mesurer la derive stage apres stage
- identifier si la derniere partie du temps se degrade

### Etape 3. Generalisation aux 4 fixed single cases

Configs :

- `..._alpha075_beta0_mu1_t5.yaml`
- `..._alpha075_beta05_mu0_t5.yaml`
- `..._alpha075_beta05_mu1_t5.yaml`

## Commandes Slurm

### Test `t=1`

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t1.yaml \
  launch/jz_submit_CGL_global_multinet_physics_only_amp_phase_case_20h.slurm
```

### Test `t=5`

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml \
  launch/jz_submit_CGL_global_multinet_physics_only_amp_phase_case_20h.slurm
```

### Reprise d'un run

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml,RESUME_RUN_DIR=latest \
  launch/jz_submit_CGL_global_multinet_physics_only_amp_phase_case_20h.slurm
```

## Sorties attendues

Dans chaque `run_*` :

- `stage_XX_tA_B/checkpoints/`
- `stage_XX_tA_B/stage_summary.json`
- `stage_manifest.csv`
- `evaluation/rollout_metrics.csv`
- `evaluation/rollout_rel_l2.png`
- `evaluation/error_heatmap.png`
- `evaluation/snapshots.png`
- `evaluation/comparison_animation.gif`
- `timing_summary.txt`

## Critere de decision

On garde la piste si :

- pas de saut net entre deux blocs
- erreur L2 sans pic d'interface majeur
- la continuite inter-stage controle bien la transition

On l'abandonne si :

- l'erreur explose systematiquement a chaque frontiere de stage
- le warm-start ne suffit pas a stabiliser le bloc suivant
- le dernier bloc herite trop d'erreur accumulee
