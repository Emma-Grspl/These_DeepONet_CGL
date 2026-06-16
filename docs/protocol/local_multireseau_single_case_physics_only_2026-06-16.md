# Protocole Local Multireseau Single-Case Physics-Only - 2026-06-16

## Contrainte dure

- aucune supervision solveur dans la loss
- aucune regression vers des snapshots solveur
- solveur autorise uniquement pour benchmark et plots

## Idee retenue

On construit plusieurs reseaux locaux, chacun specialise sur une plage de temps absolu en `single case`.

Chaque reseau :

- recoit un etat local courant sur grille capteurs
- predit l'evolution sur une petite fenetre `Delta t`
- est entraine uniquement avec des pertes physiques locales

Les blocs temporels se recouvrent partiellement.

Ce recouvrement sert a :

- specialiser chaque reseau sur une distribution d'etats plus restreinte
- imposer une coherence entre reseaux voisins sur les memes etats d'entree

## Ce qui remplace la supervision dans l'overlap

Avant :

- l'overlap etait essentiellement valide contre le solveur

Maintenant :

- on construit un banc d'etats par rollout des reseaux eux-memes
- si deux reseaux couvrent la meme zone temporelle absolue, on leur demande de produire des sorties compatibles sur ces etats

Donc :

- solveur dans la loss : non
- coherence overlap reseau-reseau : oui
- benchmark solveur en sortie : oui

## Structure du protocole

### 1. Bootstrap

Le protocole est beaucoup plus propre si on initialise tous les reseaux locaux a partir d'un checkpoint `local monoreseau physics-only`.

Pourquoi :

- sinon le banc d'etats initial est construit par des reseaux aleatoires
- ce banc devient alors rapidement inexploitable

Donc l'ordre logique est :

1. figer un `local monoreseau physics-only`
2. reutiliser son checkpoint comme `bootstrap`
3. specialiser ensuite plusieurs reseaux locaux overlap

### 2. Banc d'etats

Le banc contient des etats de depart aux temps :

- `0`
- `Delta t`
- `2 Delta t`
- etc.

Ces etats ne viennent pas du solveur.

Ils sont produits par rollout des reseaux courants.

### 3. Loss par reseau

Pour un reseau associe au bloc `B_k` :

- perte PDE locale
- perte BC locale
- petite perte IC locale a `tau=0` contre l'etat d'entree
- perte d'overlap contre les reseaux voisins sur les zones de recouvrement

## Fichiers

- `scripts/train_cgl_local_multinet_physics_only_amp_phase.py`
- `scripts/postprocess_cgl_local_multinet_physics_only_amp_phase.py`
- `launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm`

Configs preparees :

- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t1.yaml`
- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml`
- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu1_t1.yaml`
- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu1_t5.yaml`
- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta05_mu0_t1.yaml`
- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta05_mu0_t5.yaml`
- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta05_mu1_t1.yaml`
- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta05_mu1_t5.yaml`

## Strategie conseillee

### Etape 1. Validation courte `t=1`

Config :

- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t1.yaml`

Blocs :

- `[0.0, 0.4]`
- `[0.2, 0.6]`
- `[0.4, 0.8]`
- `[0.6, 1.0]`

Objectif :

- verifier que la coherence inter-reseaux stabilise bien les frontieres

### Etape 2. Extension `t=5`

Config :

- `configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml`

Blocs :

- `[0.0, 1.0]`
- `[0.5, 1.5]`
- `[1.0, 2.0]`
- ...
- `[4.0, 5.0]`

Objectif :

- voir si la specialisation locale par blocs fait mieux que le local monoreseau
- verifier si les zones de recouvrement limitent la derive accumulee

## Commandes Slurm

### Test `t=1`

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t1.yaml \
  launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

### Test `t=5`

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml \
  launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

### Test `t=5` avec bootstrap monoreseau

Dans le YAML, renseigner :

```yaml
bootstrap:
  mononet_checkpoint: "results/.../model_final_local_physics_mononet_amp_phase.pth"
```

Puis lancer :

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml \
  launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

### Reprise

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml,RESUME_RUN_DIR=latest \
  launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

## Sorties attendues

Dans chaque `run_*` :

- `stage_XX_tA_B/checkpoints/`
- `stage_XX_tA_B/stage_summary.json`
- `pass_XX/state_bank.npz`
- `history_manifest.csv`
- `evaluation/rollout_metrics.csv`
- `evaluation/rollout_rel_l2.png`
- `evaluation/error_heatmap.png`
- `evaluation/snapshots.png`
- `evaluation/comparison_animation.gif`
- `timing_summary.txt`

## Critere de decision

On garde la piste si :

- les frontieres inter-reseaux ne creent pas de pics majeurs
- le resultat final est au moins comparable au local monoreseau
- l'overlap coherence stabilise la fin d'horizon

On l'abandonne si :

- les reseaux derivent chacun dans leur coin
- l'overlap ne suffit pas a aligner les sorties
- le banc d'etats s'effondre malgre le bootstrap monoreseau
