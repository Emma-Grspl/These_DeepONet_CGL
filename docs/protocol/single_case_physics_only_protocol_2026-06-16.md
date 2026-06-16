# Protocole Single-Case Physics-Only - 2026-06-16

## Contrainte dure

Tout ce document suppose :

- aucune supervision classique
- aucune loss de regression vers le solveur
- evaluation contre le solveur uniquement au moment du benchmark

Reference :

- `docs/project_constraints.md`

## Etat actuel par famille

### 1. `monoreseau global direct`

Statut :

- actif
- relancable tout de suite
- `physics-only`

Code actif :

- `scripts/train_cgl_amp_phase.py`

Configs presentes :

- `configs/cgl_case_alpha075_beta0_mu0_global_direct_t5.yaml`
- `configs/cgl_case_alpha075_beta0_mu1_global_direct_t5.yaml`
- `configs/cgl_case_alpha075_beta05_mu0_global_direct_t5.yaml`
- `configs/cgl_case_alpha075_beta05_mu1_global_direct_t5.yaml`

Launchers presents :

- `launch/jz_submit_CGL_amp_phase_alpha075_beta0_mu0_global_direct_t5_20h.slurm`
- `launch/jz_submit_CGL_amp_phase_alpha075_beta0_mu1_global_direct_t5_20h.slurm`
- `launch/jz_submit_CGL_amp_phase_alpha075_beta05_mu0_global_direct_t5_20h.slurm`
- `launch/jz_submit_CGL_amp_phase_alpha075_beta05_mu1_global_direct_t5_20h.slurm`

Objectif de figement :

- relancer les 4 single cases
- conserver les runs bruts
- regenerer les figures benchmark
- figer ensuite la famille comme baseline `physics-only`

Artefacts a conserver apres relance :

- `results/CGL_AmpPhase_*_global_direct_t5/run_*`
- figures de benchmark et synthese sous `analyses/single_case/global_direct/`
- manifeste minimal dans `run_registry/single_case_physics_only_runs.csv`
- copie minimale dans `run_assets/single_case_physics_only/`

Critere de figement :

- les 4 cas tournent sans supervision
- pas d'instabilite manifeste
- comparaison exploitable avec `global curriculum`

### 2. `monoreseau global curriculum`

Statut :

- actif
- relancable tout de suite
- `physics-only`

Code actif :

- `scripts/train_cgl_amp_phase.py`

Configs presentes :

- `configs/cgl_case_alpha075_beta0_mu0_tchar_t5.yaml`
- `configs/cgl_case_alpha075_beta0_mu1_tchar_t5.yaml`
- `configs/cgl_case_alpha075_beta05_mu0_tchar_t5.yaml`
- `configs/cgl_case_alpha075_beta05_mu1_tchar_t5.yaml`

Launchers presents :

- `launch/jz_submit_CGL_amp_phase_alpha075_beta0_mu0_t5_20h.slurm`
- `launch/jz_submit_CGL_amp_phase_alpha075_beta0_mu1_t5_20h.slurm`
- `launch/jz_submit_CGL_amp_phase_alpha075_beta05_mu0_t5_20h.slurm`
- `launch/jz_submit_CGL_amp_phase_alpha075_beta05_mu1_t5_20h.slurm`

Objectif de figement :

- relancer les 4 single cases
- conserver les runs bruts
- regenerer les figures benchmark
- figer ensuite la famille comme meilleure baseline `physics-only` si elle domine `global direct`

Artefacts a conserver apres relance :

- `results/CGL_AmpPhase_*_tchar_t5/run_*`
- figures de benchmark et synthese sous `analyses/single_case/global_curriculum/`
- manifeste minimal dans `run_registry/single_case_physics_only_runs.csv`
- copie minimale dans `run_assets/single_case_physics_only/`

Critere de figement :

- les 4 cas tournent sans supervision
- la comparaison avec `global direct` est complete
- la famille est stable sur l'horizon vise

### 3. `monoreseau local`

Statut :

- protocole `physics-only` prepare
- pret a etre teste sur single case
- pas encore valide experimentalement

Constat :

- toute l'ancienne branche locale reposait sur des cibles solveur directes
- la nouvelle branche locale repart de zero sans solveur dans la loss

Code prepare :

- `scripts/train_cgl_local_physics_mononet_amp_phase.py`
- `src/models/cgl_local_physics_deeponet_amp_phase.py`
- `src/data/local_physics_single_case.py`

Configs preparees :

- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t1.yaml`
- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t5.yaml`
- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta0_mu1_t1.yaml`
- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta0_mu1_t5.yaml`
- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta05_mu0_t1.yaml`
- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta05_mu0_t5.yaml`
- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta05_mu1_t1.yaml`
- `configs/cgl_single_case_local_physics_mononet_amp_phase_alpha075_beta05_mu1_t5.yaml`

Launchers prepares :

- `launch/jz_submit_CGL_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t1_20h.slurm`
- `launch/jz_submit_CGL_local_physics_mononet_amp_phase_alpha075_beta0_mu0_t5_20h.slurm`
- `launch/jz_submit_CGL_local_physics_mononet_amp_phase_resume_20h.slurm`

Protocole retenu :

1. entrainer un seul reseau local partage sur une fenetre courte `Delta t`
2. utiliser uniquement :
   - residu PDE local
   - contraintes de bord
   - contrainte locale a `tau=0` contre l'etat d'entree du reseau
3. construire un banc d'etats par auto-rollout du modele
4. rejouer les etats precedents pour limiter l'oubli
5. benchmarker contre le solveur uniquement apres entrainement

Reference detaillee :

- `docs/protocol/local_monoreseau_single_case_physics_only_2026-06-16.md`

Condition de passage :

- commencer par `alpha=0.75, beta=0.0, mu=0.0`
- valider `t=1`
- ensuite seulement tenter `t=5`

### 4. `multireseau global`

Statut :

- protocole `physics-only` prepare
- script de relance disponible
- pas encore valide experimentalement

Constat :

- l'ancienne branche `global multistage` etait supervisee
- la nouvelle branche conserve le decoupage en stages mais retire toute cible solveur de la loss

Code prepare :

- `scripts/train_cgl_global_multinet_physics_only_amp_phase.py`
- `scripts/postprocess_cgl_global_multinet_physics_only_amp_phase.py`

Launchers prepares :

- `launch/jz_submit_CGL_global_multinet_physics_only_amp_phase_case_20h.slurm`

Configs preparees :

- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t1.yaml`
- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu0_t5.yaml`
- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta0_mu1_t5.yaml`
- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta05_mu0_t5.yaml`
- `configs/cgl_single_case_global_multinet_physics_only_amp_phase_alpha075_beta05_mu1_t5.yaml`

Protocole retenu :

1. un reseau par bloc temporel
2. warm-start du stage `k` depuis le meilleur modele du stage `k-1`
3. loss par stage :
   - residu PDE sur le bloc courant
   - BC
   - balance de masse
   - continuity inter-stage contre le modele precedent
4. benchmark solveur uniquement apres entrainement
5. validation d'abord sur `alpha=0.75, beta=0.0, mu=0.0` en `t<=1`, puis en `t<=5`

Reference detaillee :

- `docs/protocol/global_multireseau_single_case_physics_only_2026-06-16.md`

Condition de passage :

- d'abord verifier le cas `alpha=0.75, beta=0.0, mu=0.0`
- si pas de pic d'interface, etendre ensuite aux 4 single cases fixes

### 5. `multireseau local`

Statut :

- protocole `physics-only` prepare
- script de relance disponible
- pas encore valide experimentalement

Constat :

- c'est la famille la plus ambitieuse
- elle depend fortement d'un bon bootstrap `local monoreseau`

Code prepare :

- `scripts/train_cgl_local_multinet_physics_only_amp_phase.py`
- `scripts/postprocess_cgl_local_multinet_physics_only_amp_phase.py`

Launchers prepares :

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

Protocole retenu :

1. plusieurs reseaux locaux, chacun specialise sur un bloc temporel absolu
2. recouvrement partiel entre blocs
3. banc d'etats construit par rollout des reseaux courants
4. loss par reseau :
   - residu PDE local
   - BC locale
   - IC locale contre l'etat d'entree
   - coherence overlap contre les reseaux voisins
5. bootstrap recommande depuis un checkpoint `local monoreseau physics-only`

Reference detaillee :

- `docs/protocol/local_multireseau_single_case_physics_only_2026-06-16.md`

## Ordre de travail impose

1. figer `monoreseau global direct`
2. figer `monoreseau global curriculum`
3. reconstruire `monoreseau local` sans supervision
4. reconstruire `multireseau global` sans supervision
5. reconstruire `multireseau local` sans supervision

## Rappel important

Tant que les deux familles globales monoreseau ne sont pas refigees proprement, aucune conclusion comparative single-case n'est validee.
