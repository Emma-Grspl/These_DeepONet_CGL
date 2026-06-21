# Protocole Local Multireseau Single-Case Physics-Only V3 Diagnostic - 2026-06-21

## Objectif

Tester si le `local multinet` peut etre stabilise avant d'aller plus loin que `t=1`.

Le constat precedent est mauvais :

- erreurs deja fortes a `t=1`
- erreurs massives a `t=5`
- incertitude sur la source du probleme : operateur local, interfaces, ou rollout

## Config creee

- `configs/cgl_single_case_local_multinet_physics_only_v3_diag_amp_phase_alpha075_beta0_mu0_t1.yaml`

Cas teste :

- `alpha=0.75`
- `beta=0.0`
- `mu=0.0`
- `t_max=1.0`

## Changements V3

### Etape A - diagnostic operateur local

Apres entrainement, le script genere :

- `one_step_metrics_smooth_overlap_blend.csv`
- `one_step_metrics_hard_switch.csv`

Ces fichiers mesurent l'erreur one-step avec etat d'entree solveur, uniquement en evaluation.

But :

- si le one-step est mauvais, l'operateur local lui-meme est mauvais
- si le one-step est acceptable mais le rollout explose, le probleme vient du rollout ou des interfaces

### Etape B - blocs fins jusqu'a `t=1`

Blocs temporels :

- `[0.000, 0.250]`
- `[0.125, 0.375]`
- `[0.250, 0.500]`
- `[0.375, 0.625]`
- `[0.500, 0.750]`
- `[0.625, 0.875]`
- `[0.750, 1.000]`

Pas local :

- `window_dt=0.05`

Pourquoi :

- reduire la difficulte one-step
- augmenter le nombre de corrections courtes
- renforcer le recouvrement entre reseaux voisins

### Etape C - comparaison hard switch / smooth overlap blend

Le script supporte maintenant :

- `rollout.mode: "smooth_overlap_blend"`
- `rollout.mode: "hard_switch"`

La config V3 genere automatiquement :

- `evaluation_smooth_overlap_blend/`
- `evaluation_hard_switch/`

But :

- si `smooth_overlap_blend` est nettement meilleur que `hard_switch`, le probleme principal est l'interface
- si les deux sont mauvais, le probleme est probablement l'operateur local ou le banc d'etats

## Diagnostics supplementaires

Chaque evaluation contient aussi :

- `interface_consistency.csv`
- `one_step_metrics_<mode>.csv`
- `rollout_metrics.csv`
- `rollout_metrics_center_xm10_xp10.csv`
- `summary.txt`
- `rollout_metrics_summary.json`

## Commande Slurm

```bash
cd $WORK/These_DeepOnet_CGL || exit 1

CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_v3_diag_amp_phase_alpha075_beta0_mu0_t1.yaml \
sbatch launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

## Reprise

```bash
cd $WORK/These_DeepOnet_CGL || exit 1

CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_v3_diag_amp_phase_alpha075_beta0_mu0_t1.yaml \
RESUME_RUN_DIR=latest \
sbatch launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

## Criteres de decision

On continue le local multinet seulement si :

- `one_step_max_rel_l2` baisse fortement par rapport aux anciennes erreurs `t=1`
- `interface_max_rel_diff` reste raisonnable
- `rollout max L2` a `t=1` baisse nettement sous les anciens niveaux de 70% et plus

Sinon :

- ne pas lancer `t=5`
- revenir au design de l'operateur local ou abandonner la piste local multinet single-case
