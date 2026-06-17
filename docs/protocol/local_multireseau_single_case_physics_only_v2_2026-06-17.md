# Protocole Local Multireseau Single-Case Physics-Only V2 - 2026-06-17

## Objectif

Corriger le mecanisme observe sur `local multinet` :

- proxy local bloc par bloc correct
- rollout global qui explose ensuite
- banc d'etats autoregressif trop vite contamine par les erreurs accumulees

## Changements de protocole

### 1. Banc d'etats reconstruit apres chaque stage

Avant :

- un `state_bank` etait construit une seule fois au debut de chaque passe
- tous les blocs de la passe s'entrainaient ensuite sur ce bank fige

Maintenant :

- apres chaque bloc entraine, on reconstruit le `state_bank` avec les modeles courants
- le bloc suivant voit donc un rollout plus coherent avec les blocs deja corriges

Effet recherche :

- reduire la propagation d'un bank obsolete
- rendre l'entrainement plus causal

### 2. Selection au niveau passe complete

Avant :

- on gardait surtout les meilleurs checkpoints locaux selon un proxy de bloc

Maintenant :

- chaque passe produit une evaluation complete du rollout
- on selectionne la meilleure passe selon `max_rel_l2`
- en cas de degradation, on rollback sur le meilleur snapshot de passe

Effet recherche :

- eviter de conserver une passe qui ameliore localement mais deteriore l'horizon complet

### 3. Overlap plus fort

Par rapport au protocole precedent :

- `overlap_states` augmente
- `overlap_points_per_state` augmente
- poids `overlap` augmente a `0.5`
- poids `ic` augmente a `0.1`

Effet recherche :

- forcer plus fort la compatibilite entre blocs voisins
- limiter la rupture au moment du passage d'un reseau au suivant

### 4. Blocs `t=5` plus fins

Le protocole `t=5` passe sur :

- `[0.0, 0.8]`
- `[0.4, 1.2]`
- `[0.8, 1.6]`
- `[1.2, 2.0]`
- `[1.6, 2.4]`
- `[2.0, 2.8]`
- `[2.4, 3.2]`
- `[2.8, 3.6]`
- `[3.2, 4.0]`
- `[3.6, 4.4]`
- `[4.0, 5.0]`

L'idee est de reduire la difficulte locale tout en gardant un recouvrement significatif.

## Fichiers

- script :
  - `scripts/train_cgl_local_multinet_physics_only_amp_phase.py`
- launcher :
  - `launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm`
- configs `v2` :
  - `configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta0_mu0_t1.yaml`
  - `configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta0_mu0_t5.yaml`
  - `configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta0_mu1_t1.yaml`
  - `configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta0_mu1_t5.yaml`
  - `configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta05_mu0_t1.yaml`
  - `configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta05_mu0_t5.yaml`
  - `configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta05_mu1_t1.yaml`
  - `configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta05_mu1_t5.yaml`

## Bootstrap conseille

Si un `local monoreseau` fiable existe, renseigner :

```yaml
bootstrap:
  mononet_checkpoint: "results/.../model_final_local_physics_mononet_amp_phase.pth"
```

Le protocole `v2` fonctionne sans bootstrap, mais il est nettement plus defensable avec un monoreseau local stable.

## Commandes type

### Validation courte `t=1`

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta0_mu0_t1.yaml \
  launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

### Run complet `t=5`

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta0_mu0_t5.yaml \
  launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

### Reprise

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgl_single_case_local_multinet_physics_only_v2_amp_phase_alpha075_beta0_mu0_t5.yaml,RESUME_RUN_DIR=latest \
  launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm
```

## Critere de decision

On garde la piste si :

- `max_rel_l2` baisse d'une passe a l'autre
- le rollback evite les degradations irreversibles
- les pics de fin d'horizon diminuent nettement par rapport au protocole precedent

On abandonne si :

- meme avec reconstruction causale et rollback, le rollout diverge encore massivement
- les blocs restent localement bons mais globalement incoherents
- le gain reste tres inferieur au monoreseau local
