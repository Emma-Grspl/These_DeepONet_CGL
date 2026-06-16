# Suivi Des Etapes

## Etapes deja faites

### 1. Definition du cadre du probleme

Fait :
- definition des single cases de reference
- fixation du domaine spatial
- fixation de l'horizon temporel cible `t_max = 5`
- definition explicite des 4 single cases etudies :
  - cas 1 : `alpha = 0.75`, `beta = 0.0`, `mu = 0.0`
  - cas 2 : `alpha = 0.75`, `beta = 0.0`, `mu = 1.0`
  - cas 3 : `alpha = 0.75`, `beta = 0.5`, `mu = 0.0`
  - cas 4 : `alpha = 0.75`, `beta = 0.5`, `mu = 1.0`
- choix du cas principal de diagnostic local :
  - `alpha = 0.75`
  - `beta = 0.0`
  - `mu = 0.0`

### 2. Solveur de reference

Fait :
- solveur classique CGL disponible
- solveur utilise comme reference pour tous les audits
- premieres analyses classiques deja generees

Reste a formaliser :
- rediger proprement l'etude de convergence du solveur
- figer explicitement les discretisations finales de reference

### 3. Comparaison single-case des configurations

Fait :
- `monoreseau / global_direct`
  - code disponible
  - runs faits sur les 4 cas
  - analyse faite
  - conclusion : mauvais
  - statut global : termine, a refaire pour le temps
  - detail par cas :
    - cas 1 `alpha075_beta0_mu0` : termine, a refaire pour le temps
    - cas 2 `alpha075_beta0_mu1` : termine, a refaire pour le temps
    - cas 3 `alpha075_beta05_mu0` : termine, a refaire pour le temps
    - cas 4 `alpha075_beta05_mu1` : termine, a refaire pour le temps

- `monoreseau / global_curriculum`
  - code disponible
  - runs faits sur les 4 cas
  - analyses finales produites
  - conclusion : meilleure baseline monoreseau, mais reste derriere `multireseau / global`
  - statut global : termine
  - resultat agrege :
    - `mean L2` sur 4 runs : `4.03%`
    - temps moyen : `8.97 h`

- `monoreseau / local_one_step_rollout`
  - code disponible
  - runs faits sur les 4 cas
  - analyses faites
  - conclusion : one-step correct, rollout ferme mauvais
  - statut global : termine, a refaire pour le temps
  - detail par cas :
    - cas 1 `alpha075_beta0_mu0` : termine, a refaire pour le temps
    - cas 2 `alpha075_beta0_mu1` : termine, a refaire pour le temps
    - cas 3 `alpha075_beta05_mu0` : termine, a refaire pour le temps
    - cas 4 `alpha075_beta05_mu1` : termine, a refaire pour le temps

- `monoreseau / local_multi_step_curriculum`
  - code disponible
  - runs faits sur les 4 cas
  - stabilisation numerique obtenue
  - conclusion : meilleur que `local_one_step_rollout`, mais encore trop mauvais en rollout
  - statut global : termine, a refaire pour le temps si comparaison budget stricte
  - detail par cas :
    - cas 1 `alpha075_beta0_mu0` : termine, a refaire pour le temps si necessaire
    - cas 2 `alpha075_beta0_mu1` : termine, a refaire pour le temps si necessaire
    - cas 3 `alpha075_beta05_mu0` : termine, a refaire pour le temps si necessaire
    - cas 4 `alpha075_beta05_mu1` : termine, a refaire pour le temps si necessaire

- `multireseau / local`
  - code disponible
  - tests faits sur les 4 cas
  - conclusion : one-step bon, rollout ferme mauvais
  - tests complementaires faits sur `alpha075_beta0_mu0` :
    - correction explicite du rollout : echec
    - representation `Re/Im` : meilleure que `amp/phase`, mais encore insuffisante
  - protocole retenu :
    - d'abord smoke test local sur `alpha075_beta0_mu0`
    - si pipeline propre mais rollout toujours mauvais : correction avant extension
    - si correction validee : lancement Jean Zay sur les 4 cas
  - statut global : termine comme diagnostic, non retenu en l'etat
  - resultat agrege :
    - `mean L2` sur 4 runs : `168.26%`
    - temps moyen reconstitue : `0.236 h`

- `multireseau / global`
  - code disponible
  - runs faits sur les 4 cas
  - analyses faites
  - heatmaps et snapshots generes
  - conclusion : meilleur compromis actuel
  - smoke test local valide
  - protocole multireseau valide
  - statut global : termine
  - resultat agrege :
    - `mean L2` sur 4 runs : `0.98%`
    - temps moyen reconstitue : `0.448 h`

### 4. Analyses produites

Fait :
- courbes `L2(t)` pour plusieurs familles
- heatmaps d'erreur
- snapshots temporels
- diagnostics one-step vs rollout pour les locaux
- comparaison globale des 4 runs `global_multistage`
- tests complementaires `multireseau / local` sur `alpha075_beta0_mu0` :
  - correction explicite de rollout
  - variante `Re/Im`

### 5. Decision intermediaire actuelle

Fait :
- meilleure configuration single-case fixee :
  - `multireseau / global`
- conclusion locale intermediaire :
  - `multireseau / local` reste inferieur
  - `Re/Im` ameliore le local mais ne suffit pas a le rendre competitif
- extension `t = 20` de `multireseau / global` analysee :
  - 3 cas sur 4 restent excellents jusqu'a `t = 20`
  - 1 cas presente un stage `[9,10]` isole mal appris
  - ce point est interprete comme un accident d'entrainement, pas comme une limite structurelle


## Etapes en cours

### 1. Finaliser la comparaison single-case

Fait :
- vue finale `single_case` reconstruite
- comparaison complete de toutes les familles disponible
- structure finale `all_runs` generee pour mono et multi
- tableaux et figures agreges disponibles a la racine de `analyses/single_case`

### 2. Tester la variante locale stabilisee

En cours :
- consolider la lecture quantitative de :
  - `local_direct_residual_multistep`

Objectif :
- figer proprement que la stabilisation numerique a ete obtenue
- mais que le rollout reste insuffisant pour une famille finale

### 3. Cadrer le protocole multireseau

Fait :
- protocole unique defini pour `multireseau / local` et `multireseau / global`
- memes sorties d'analyse imposees que pour le monoreseau
- mesure du temps total d'entrainement imposee
- smoke tests locaux valides sur `alpha075_beta0_mu0`
- extension Jean Zay realisee sur les 4 cas


## Etapes a venir

### 1. Tableau final de comparaison single-case

Fait :
- tableau unique disponible dans `analyses/single_case/all_config_summary.csv`
- figures agreges disponibles :
  - `analyses/single_case/mean_l2_vs_time_all_configs.png`
  - `analyses/single_case/mean_l2_vs_training_time_all_configs.png`

### 2. Choix final de la meilleure configuration

Fait :
- `multireseau / global` est la meilleure configuration single-case retenue
- `monoreseau / global_curriculum` est la meilleure baseline monoreseau
- les familles locales restent diagnostiques, non candidates finales

### 3. Passage au cas parametrique

A faire :
- prendre la meilleure configuration single-case
- l'etendre au probleme parametrique
- verifier la robustesse de la decomposition temporelle quand les parametres varient

Ordre retenu :
- 1. `sigma` seul avec les 4 familles fixes `alpha/beta/mu`
- 2. une seule variable libre a la fois :
  - `beta` variable, autres fixes
  - `alpha` variable, autres fixes
  - `mu` variable, autres fixes
- 3. `sigma + alpha`
- 4. `sigma + alpha + beta`
- 5. `sigma + alpha + beta + mu`

Structure cible :
- configs :
  - `CGL_amp_phase/configs/parameters/<campagne>/...`
- resultats :
  - `results/<NomDeCampagne>/run_<timestamp>_<jobid>/...`
- analyses :
  - `analyses/parameters/<campagne>/...`

Sorties minimales attendues pour chaque campagne :
- `timing_summary.txt`
- `timing_stages.csv`
- `rollout_metrics.csv`
- `summary.txt`
- `l2_vs_time.png`
- `error_heatmap.png`
- `snapshots.png`
- `comparison_animation.gif`
- `inference_timing.txt`
- `inference_timing.png`

### 4. Optimisation d'architecture

A faire :
- lancer Optuna sur la meilleure famille retenue
- optimiser :
  - largeur
  - profondeur
  - embeddings / fourier features
  - regularisation
  - learning rate
  - eventuellement le decoupage temporel

### 5. Documentation finale

A faire :
- rediger la justification du protocole
- figer les metriques officielles
- documenter les choix de solveur, d'architecture et de comparaison


## Priorite immediate

Ordre de travail recommande :
- 1. finir les runs `global_curriculum`
- 2. sortir les derniers plots `global_curriculum`
- 3. consolider le tableau comparatif single-case avec les temps manquants
- 4. lancer le premier palier parametrique en `multireseau / global`
