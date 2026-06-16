# Protocole Local Monoreseau Single-Case Physics-Only - 2026-06-16

## Contrainte non negociable

- aucune supervision classique dans la loss
- aucune regression vers le solveur dans la loss
- solveur classique autorise uniquement pour benchmark et plots

## Idee retenue

On entraine un **seul reseau local partage** sur une fenetre courte de taille `Delta t`.

Le reseau recoit :

- l'etat local courant sur une grille capteurs en `x`
- les parametres physiques fixes du single case
- la taille de fenetre locale

Le reseau predit ensuite `u(x, tau)` pour `tau in [0, Delta t]`.

## Point cle pour rester physics-only

Le reseau n'utilise **jamais** la trajectoire solveur comme cible.

Le banc d'etats locaux est construit ainsi :

1. etat initial analytique a `t=0`
2. apres apprentissage sur la fenetre courante, on applique le modele a son **propre** etat de depart
3. l'etat predit a la fin de fenetre est ajoute au banc
4. le stage suivant s'entraine sur un melange :
   - etats recents
   - etats plus anciens rejoues

Donc :

- supervision solveur : non
- auto-curriculum temporel du modele : oui
- benchmark solveur en sortie : oui

## Loss

Pour chaque batch local :

- perte PDE sur des points interieurs `(x, tau)`
- perte BC periodique ou bord fixe selon le type de cas
- petite perte IC locale a `tau=0` contre l'etat d'entree du banc

Remarque :

- cette perte IC ne compare pas au solveur
- elle sert uniquement a imposer que l'operateur local reparte bien de l'etat qu'on lui fournit

## Architecture

Fichiers :

- `src/models/cgl_local_physics_deeponet_amp_phase.py`
- `src/data/local_physics_single_case.py`
- `scripts/train_cgl_local_physics_mononet_amp_phase.py`

Choix principaux :

- representation amplitude / phase
- branch = etat local capteurs + parametres physiques + `Delta t`
- trunk = `(x, tau)`
- ansatz dur pour imposer exactement l'etat initial local a `tau=0`
- phase gate optionnel pour limiter les derives de phase en faible amplitude

## Strategie en 3 etapes

### Etape 1. Faisabilite courte

Objectif :

- verifier que le local monoreseau tient jusqu'a `t=1`

Config :

- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t1.yaml`

Launcher :

- `launch/jz_submit_CGL_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t1_20h.slurm`

Ce qu'on regarde :

- erreur L2 sur tout `t in [0, 1]`
- presence ou non d'un pic de frontiere entre fenetres
- taille de la derive en fin d'horizon

### Etape 2. Extension brute a `t=5`

Objectif :

- garder exactement la meme logique et tester la derive longue

Config :

- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t5.yaml`

Launcher :

- `launch/jz_submit_CGL_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t5_20h.slurm`

Point important :

- le reseau reste unique
- ce n'est pas un multireseau
- la memoire des fenetres precedentes passe par le replay du banc d'etats

### Etape 3. Reprise si 20h ne suffisent pas

Launcher generique :

- `launch/jz_submit_CGL_local_physics_mononet_amp_phase_resume_20h.slurm`

Le script sauve :

- checkpoints de stage
- `run_state.pth`
- banc d'etats en entree de chaque stage

Donc reprise possible :

- au dernier run
- ou sur un run explicite

## Commandes Slurm prevues

### Lancer `t=1`

```bash
sbatch launch/jz_submit_CGL_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t1_20h.slurm
```

### Lancer `t=5`

```bash
sbatch launch/jz_submit_CGL_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t5_20h.slurm
```

### Reprendre le dernier `t=5`

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t5.yaml,RESUME_RUN_DIR=latest \
  launch/jz_submit_CGL_local_physics_mononet_amp_phase_resume_20h.slurm
```

### Reprendre un run explicite

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t5.yaml,RESUME_RUN_DIR=results/CGL_LocalMononet_PhysicsOnly_AmpPhase_alpha075_beta0_mu0_t5/run_YYYYMMDD-HHMMSS_JOBID \
  launch/jz_submit_CGL_local_physics_mononet_amp_phase_resume_20h.slurm
```

## Sorties attendues

Dans chaque `run_*` :

- un dossier par stage `stage_XX_tA_B`
- `run_state.pth`
- `model_final_local_physics_mononet_amp_phase.pth`
- `evaluation/rollout_metrics.csv`
- `evaluation/rollout_rel_l2.png`
- `evaluation/error_heatmap.png`
- `evaluation/snapshots.png`
- `evaluation/comparison_animation.gif`
- `state_bank.csv`
- `timing_summary.txt`

## Critere de decision

On garde ce protocole si :

- pas de solveur dans la loss
- pas de saut massif entre fenetres
- erreur finale exploitable a `t=1`
- puis comportement encore lisible a `t=5`

On l'abandonne si :

- le pic de frontiere reste structurel
- la derive finale augmente fenetre apres fenetre sans stabilisation
- le replay du banc d'etats ne suffit pas a conserver les premieres fenetres
