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
  - runs lances sur les 4 cas
  - analyses partielles faites
  - conclusion provisoire : nettement meilleur que global_direct
  - statut global : en cours de run
  - detail par cas :
    - cas 1 `alpha075_beta0_mu0` : en cours de run, reprise lancee
    - cas 2 `alpha075_beta0_mu1` : en cours de run, reprise lancee
    - cas 3 `alpha075_beta05_mu0` : en cours de run, reprise lancee
    - cas 4 `alpha075_beta05_mu1` : termine, a refaire pour le temps

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
  - premiere version codee et testee
  - premier run harmonized fait sur `alpha075_beta0_mu0`
  - conclusion : divergence numerique en NaN
  - nouvelle version residuelle stabilisee codee
  - statut global : a faire run
  - detail par cas :
    - cas 1 `alpha075_beta0_mu0` : a faire run
    - cas 2 `alpha075_beta0_mu1` : a faire run
    - cas 3 `alpha075_beta05_mu0` : a faire run
    - cas 4 `alpha075_beta05_mu1` : a faire run

- `multireseau / local`
  - code disponible
  - test fait sur `alpha075_beta0_mu0`
  - conclusion : mauvais en rollout
  - protocole retenu :
    - d'abord smoke test local sur `alpha075_beta0_mu0`
    - si pipeline propre mais rollout toujours mauvais : correction avant extension
    - si correction validee : lancement Jean Zay sur les 4 cas
  - statut global : en phase de cadrage local
  - detail par cas :
    - cas 1 `alpha075_beta0_mu0` : smoke test local a faire / reprendre proprement pour le temps
    - cas 2 `alpha075_beta0_mu1` : a faire apres validation locale
    - cas 3 `alpha075_beta05_mu0` : a faire apres validation locale
    - cas 4 `alpha075_beta05_mu1` : a faire apres validation locale

- `multireseau / global`
  - code disponible
  - runs faits sur les 4 cas
  - analyses faites
  - heatmaps et snapshots generes
  - conclusion : meilleur compromis actuel
  - protocole retenu :
    - d'abord smoke test local sur `alpha075_beta0_mu0`
    - verification des timings et du pipeline de post-traitement
    - puis rerun Jean Zay sur les 4 cas pour avoir une base propre avec le temps d'entrainement
  - statut global : a refaire pour le temps, avec validation locale prealable
  - detail par cas :
    - cas 1 `alpha075_beta0_mu0` : smoke test local a faire, puis a refaire pour le temps
    - cas 2 `alpha075_beta0_mu1` : a refaire pour le temps apres validation locale
    - cas 3 `alpha075_beta05_mu0` : a refaire pour le temps apres validation locale
    - cas 4 `alpha075_beta05_mu1` : a refaire pour le temps apres validation locale

### 4. Analyses produites

Fait :
- courbes `L2(t)` pour plusieurs familles
- heatmaps d'erreur
- snapshots temporels
- diagnostics one-step vs rollout pour les locaux
- comparaison globale des 4 runs `global_multistage`

### 5. Decision intermediaire actuelle

Fait :
- meilleure configuration single-case actuelle fixee provisoirement :
  - `multireseau / global`


## Etapes en cours

### 1. Finaliser la comparaison single-case

En cours :
- finir toutes les reprises `global_curriculum` jusqu'a `t = 5`
- consolider la comparaison finale avec :
  - precision
  - seuil `5%`
  - temps de calcul

### 2. Tester la variante locale stabilisee

En cours :
- lancer et analyser :
  - `local_direct_residual_multistep`

Objectif :
- verifier si l'ansatz local residuel avec facteur `dt`
- la borne douce
- et la loss multi-step progressive

permettent d'eviter les `NaN` et d'ameliorer le rollout ferme.

### 3. Cadrer le protocole multireseau

En cours :
- definir un protocole unique pour `multireseau / local` et `multireseau / global`
- imposer les memes sorties d'analyse que pour le monoreseau
- imposer la mesure du temps total d'entrainement

Ordre retenu :
- smoke test local sur un seul cas :
  - `alpha075_beta0_mu0`
- puis extension Jean Zay sur les 4 cas si le pipeline est valide


## Etapes a venir

### 1. Tableau final de comparaison single-case

A faire :
- construire un tableau unique avec les 6 configurations
- inclure au minimum :
  - `final_rel_l2`
  - `max_rel_l2`
  - `mean_rel_l2`
  - `first_t_gt_5pct`
  - temps total d'entrainement
  - nombre de relances / reprises

### 2. Choix final de la meilleure configuration

A faire :
- confirmer ou invalider `multireseau / global` comme meilleure famille finale
- conclure apres comparaison complete avec `global_curriculum`
- conclure apres test de la variante locale stabilisee
- conclure apres reruns multireseau avec temps d'entrainement homogenes

### 3. Passage au cas parametrique

A faire :
- prendre la meilleure configuration single-case
- l'etendre au probleme parametrique
- verifier la robustesse de la decomposition temporelle quand les parametres varient

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
- 2. tester `local_direct_residual_multistep`
- 3. valider localement le protocole `multireseau / local` et `multireseau / global`
- 4. figer le tableau comparatif single-case
- 5. confirmer la meilleure famille
- 6. passer au cas parametrique
