# Protocole Experimental

## Etape 1. Definir le probleme

Objectif :
- fixer les parametres de l'equation
- fixer le type de condition initiale
- fixer les intervalles de temps et d'espace

Elements a figer :
- equation CGL et convention de representation choisie
- parametres : `alpha`, `beta`, `mu`, `V`
- condition initiale : type d'IC et parametres geometriques associes
- domaine spatial : `x in [x_min, x_max]`
- horizon temporel : `t in [0, t_max]`
- discretisations de reference : `Nx`, `Nt`

Etat actuel :
- single-case principal utilise pour les diagnostics :
  - `alpha = 0.75`
  - `beta = 0.0`
  - `mu = 0.0`
  - IC de type `0`
  - domaine spatial `[-40, 40]`
  - horizon cible `t_max = 5`
- trois autres single cases ont aussi ete testes :
  - `alpha075_beta0_mu1`
  - `alpha075_beta05_mu0`
  - `alpha075_beta05_mu1`


## Etape 2. Definir le solveur de reference

Objectif :
- coder un solveur classique de reference
- verifier sa convergence
- le figer comme base de comparaison

Exigences :
- le solveur doit etre considere comme la verite de reference pour tous les audits
- toute metrique d'erreur doit etre mesuree contre ce solveur
- les normalisations doivent rester coherentes avec la sortie du solveur pour eviter des audits artificiellement faux

Etat actuel :
- un solveur classique CGL est en place et utilise comme reference
- des sorties de reference ont deja ete generees pour les 4 single cases
- des analyses classiques ont deja ete produites dans `analyses/classical_solveur`

Point a documenter proprement ensuite :
- etude de convergence explicite du solveur de reference
- choix final de `Nx`, `Nt` et justification


## Etape 3. Comparer les configurations single-case

Objectif :
- comparer plusieurs familles de modeles sur des single cases
- choisir la meilleure configuration en compromis precision / temps de calcul

Critere minimal :
- la precision doit rester sous le seuil `L2 = 5%` sur l'horizon cible

Configurations a comparer :
- `monoreseau / global_direct`
- `monoreseau / global_curriculum`
- `monoreseau / local_one_step_rollout`
- `monoreseau / local_multi_step_curriculum`
- `multireseau / local`
- `multireseau / global`

### Metriques d'analyse a fixer

Metriques principales :
- `final_rel_l2` a `t = 5`
- `max_rel_l2` sur tout l'horizon
- `mean_rel_l2` sur tout l'horizon
- `first_t_gt_5pct` : premier temps ou `L2 > 5%`
- temps total d'entrainement
- nombre de jobs / reprises necessaires

Supports visuels a produire pour chaque famille quand possible :
- `L2(t)`
- heatmap erreur absolue `|u_pred - u_ref|`
- snapshots a temps fixes `t = 0, 1, 2, 3, 4, 5`

Diagnostic local supplementaire :
- teacher forcing one-step
- rollout ferme
- audits courts `h1, h2, h4, h8`
- detection immediate des NaN / divergence

### Etat actuel par configuration

#### 1. Monoreseau / global_direct

Etat :
- code disponible
- runs effectues sur les 4 single cases
- analyses produites

Resultat :
- configuration terminee
- mauvaise globalement
- le seuil `5%` est franchi tres tot sur tous les cas

Conclusion actuelle :
- a conserver comme baseline negative
- pas candidat final

#### 2. Monoreseau / global_curriculum

Correspondance pratique :
- c'est la famille `global_causal / tchar_t5`

Etat :
- code disponible
- runs effectues sur les 4 single cases
- les 4 cas n'ont pas tous ete completes en meme temps initialement
- au moins un cas termine a `t = 5`
- des reprises ont ete lancees pour les autres

Resultat actuel :
- meilleur que `global_direct`
- nettement plus stable
- encore inferieur au `multireseau / global` sur les cas deja compares

Conclusion actuelle :
- baseline serieuse
- comparaison finale a terminer une fois tous les runs completes a `t = 5`

#### 3. Monoreseau / local_one_step_rollout

Correspondance pratique :
- `local_direct_one_step`

Etat :
- code disponible
- runs effectues sur les 4 single cases
- analyses produites

Resultat :
- one-step teacher forced raisonnable
- rollout ferme mauvais
- sur `alpha075_beta0_mu0`, le rollout explose tres vite alors que l'operateur de base est bien mieux appris que ce que le rollout laisse penser

Conclusion actuelle :
- important comme diagnostic
- pas candidat final en l'etat

#### 4. Monoreseau / local_multi_step_curriculum

Correspondance pratique :
- `local_direct_multistep`
- puis variante residuelle stabilisee en cours

Etat :
- code disponible
- premier run harmonized effectue sur `alpha075_beta0_mu0`
- premier essai : divergence numerique avec `NaN` des le debut
- nouvelle variante stabilisee codee :
  - ansatz local residuel avec facteur `dt`
  - borne douce via `tanh`
  - activation progressive de la loss multi-step
  - arret immediat en cas de `NaN`
- cette nouvelle variante est codee, a tester

Resultat actuel :
- pas encore valide
- pas encore compare sur les 4 cas

Conclusion actuelle :
- piste de recherche active
- non statuee

#### 5. Multireseau / local

Correspondance pratique :
- `local_multistage`

Etat :
- code disponible
- un run effectue sur `alpha075_beta0_mu0`
- pas encore etendu aux 4 cas

Resultat :
- echec en rollout
- diagnostic : l'echec vient surtout du rollout ferme, avec un mauvais depart tres precoce

Conclusion actuelle :
- non retenu en l'etat
- eventuellement utile comme objet de diagnostic, pas comme meilleur candidat actuel

#### 6. Multireseau / global

Correspondance pratique :
- `global_multistage`

Etat :
- code disponible
- runs effectues sur les 4 single cases
- analyses produites

Resultat :
- les 4 cas atteignent `t = 5`
- erreurs finales de l'ordre de `0.8%` a `1.3%`
- aucun cas ne depasse `5%`

Conclusion actuelle :
- meilleur compromis actuel
- configuration actuellement retenue comme meilleure candidate single-case

### Decision intermediaire actuelle

Configuration actuellement favorite :
- `multireseau / global`

Raison :
- meilleure precision
- stabilite sur les 4 single cases
- respecte le critere `L2 < 5%` sur tout l'horizon observe


## Etape 4. Passer du single-case au cas parametrique

Objectif :
- prendre la meilleure configuration single-case
- l'etendre au probleme parametrique

Strategie actuelle :
- prendre `multireseau / global` comme point de depart
- definir le schema de generalisation en parametres
- verifier que la decomposition temporelle reste pertinente quand les parametres varient

Questions a trancher :
- un reseau par bloc temporel mais commun a tous les parametres
- ou un autre schema si la dependance parametrique destabilise la decomposition


## Etape 5. Optimiser l'architecture avec Optuna

Objectif :
- une fois la meilleure famille fixee, lancer une optimisation d'architecture

Principe :
- ne pas lancer Optuna avant d'avoir fige la famille de modele
- optimiser ensuite :
  - largeur
  - profondeur
  - fourier features
  - regularisation
  - learning rate
  - eventuellement le decoupage temporel

Priorite :
- Optuna doit porter sur la configuration retenue apres l'etape 3
- a ce stade, la candidate naturelle est `multireseau / global`


## Evaluation de la coherence globale

Le plan est coherent.

Pourquoi :
- il separe proprement solveur de reference, comparaison single-case, generalisation parametrique, puis optimisation d'architecture
- il evite de lancer Optuna trop tot
- il impose un critere de precision explicite
- il permet de justifier scientifiquement le choix de la famille de modele avant de generaliser

Point a renforcer ensuite :
- ajouter une mesure explicite du cout de calcul dans les tableaux de comparaison
- figer un tableau final unique de synthese pour les 6 configurations
