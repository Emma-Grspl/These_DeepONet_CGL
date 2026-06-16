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
- des analyses classiques ont deja ete produites dans `analyses/single_case/reference_solver/classical_solver`

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

Resultat final :
- les 4 runs sont completes et figes pour l'analyse single-case
- meilleure baseline monoreseau
- resultat agrege :
  - `mean L2` sur 4 runs : `4.03%`
  - `final_rel_l2` moyen : `6.22%`
  - temps d'entrainement moyen : `8.97 h`

Conclusion finale :
- baseline serieuse
- utile pour la comparaison finale monoreseau
- reste inferieur au `multireseau / global`

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

Conclusion finale :
- important comme diagnostic
- pas candidat final en l'etat
- resultat agrege :
  - `mean L2` sur 4 runs : `428.02%`
  - temps d'entrainement moyen : `0.123 h`

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

Resultat final :
- les 4 cas ont ete testes
- stabilisation numerique obtenue
- meilleur que `local_one_step_rollout`
- mais rollout encore trop instable pour etre candidat final

Conclusion finale :
- utile comme diagnostic
- pas retenu comme meilleure configuration
- resultat agrege :
  - `mean L2` sur 4 runs : `18.35%`
  - temps d'entrainement moyen : `10.80 h`

#### 5. Multireseau / local

Correspondance pratique :
- `local_multistage`

Etat :
- code disponible
- runs effectues sur les 4 single cases
- analyses et diagnostics produits

Resultat :
- echec en rollout
- diagnostic : l'echec vient surtout du rollout ferme, pas du `one-step`

Conclusion finale :
- non retenu en l'etat
- eventuellement utile comme objet de diagnostic, pas comme meilleur candidat actuel
- resultat agrege :
  - `mean L2` sur 4 runs : `168.26%`
  - temps d'entrainement moyen : `0.236 h` reconstitue a partir des timestamps de fichiers

Tests complementaires effectues sur `alpha075_beta0_mu0` :
- diagnostic `one-step vs rollout` sur les 4 cas :
  - le `one-step` est bon partout
  - l'echec vient bien du rollout autoregressif
- test de correction explicite du rollout :
  - clamp amplitude
  - blend de phase en faible amplitude
  - lissage local
  - resultat : echec, plus instable que la baseline
- test de changement de representation vers `Re/Im` :
  - resultat : amelioration nette par rapport a `amp/phase`
  - mais rollout encore tres mauvais en absolu

Conclusion technique actualisee :
- le local multireseau n'est pas mal appris en `one-step`
- le verrou est la stabilite de la boucle fermee
- `Re/Im` est plus robuste que `amp/phase` pour le local
- en l'etat, la famille `multireseau / local` reste inferieure a `multireseau / global`

Protocole de validation a appliquer ensuite :
- phase 1 : smoke test local sur `alpha075_beta0_mu0`
- objectif phase 1 :
  - verifier l'absence de `NaN`
  - verifier l'ecriture de `timing_summary.txt`
  - verifier la reconstruction complete des sorties d'analyse
- sorties minimales obligatoires :
  - `rollout_metrics.csv`
  - `summary.txt`
  - `timing_summary.txt`
  - `error_heatmap.png`
  - `snapshots.png`
  - `comparison_animation.gif`
  - `inference_timing.txt`
  - `inference_timing.png`
- critere de passage a Jean Zay :
  - pas de divergence numerique
  - pipeline d'analyse complet
  - comportement du rollout interpretable
- phase 2 : lancement Jean Zay sur les 4 cas
- phase 3 : comparaison finale avec les autres familles sur :
  - precision
  - stabilite
  - temps total d'entrainement
  - cout en reprises si necessaire

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

Conclusion finale :
- meilleur compromis actuel
- configuration retenue comme meilleure configuration single-case
- resultat agrege :
  - `mean L2` sur 4 runs : `0.98%`
  - `final_rel_l2` moyen : `1.09%`
  - temps d'entrainement moyen : `0.448 h` reconstitue a partir des timestamps de fichiers

Resultat etendu :
- a `t = 20`, `multireseau / global` reste excellent sur `3` cas sur `4`
- le cas `alpha075_beta05_mu1` presente un bloc anormal sur `[9,10]`, interprete comme un stage isole mal appris et non comme une limite structurelle de la methode
- une relance de ce seul run est suffisante pour nettoyer cette anomalie

### Etat quantitatif final des single cases

Remarque :
- les valeurs ci-dessous sont les erreurs finales `final_rel_l2` a `t = 5` quand applicable
- pour certains runs multireseaux, le temps d'entrainement a ete reconstitue a partir des timestamps locaux faute de `timing_summary.txt` dans les resultats ramenes

#### Monoreseau / global_direct

- `alpha075_beta0_mu0` : `final_rel_l2 = 77.26%` | temps = `1.91 h`
- `alpha075_beta0_mu1` : `final_rel_l2 = 96.42%` | temps = `1.92 h`
- `alpha075_beta05_mu0` : `final_rel_l2 = 75.36%` | temps = `1.91 h`
- `alpha075_beta05_mu1` : `final_rel_l2 = 99.02%` | temps = `1.95 h`

#### Monoreseau / global_curriculum

- `alpha075_beta0_mu0` : `final_rel_l2 = 6.98%` | temps = `4.33 h`
- `alpha075_beta0_mu1` : `final_rel_l2 = 7.85%` | temps = `14.83 h`
- `alpha075_beta05_mu0` : `final_rel_l2 = 6.65%` | temps = `16.27 h`
- `alpha075_beta05_mu1` : `final_rel_l2 = 3.39%` | temps = `0.44 h`

#### Monoreseau / local_one_step_rollout

- `alpha075_beta0_mu0` : `final_rel_l2 = 2003.03%` | temps = `0.123 h`
- `alpha075_beta0_mu1` : `final_rel_l2 = 419.07%` | temps = `0.123 h`
- `alpha075_beta05_mu0` : `final_rel_l2 = 139.11%` | temps = `0.123 h`
- `alpha075_beta05_mu1` : `final_rel_l2 = 909.50%` | temps = `0.123 h`

#### Monoreseau / local_multi_step_curriculum

- `alpha075_beta0_mu0` : `final_rel_l2 = 29.96%` | temps = `10.63 h`
- `alpha075_beta0_mu1` : `final_rel_l2 = 62.59%` | temps = `10.94 h`
- `alpha075_beta05_mu0` : `final_rel_l2 = 21.96%` | temps = `10.71 h`
- `alpha075_beta05_mu1` : `final_rel_l2 = 53.38%` | temps = `10.92 h`

#### Multireseau / local

- `alpha075_beta0_mu0` : `final_rel_l2 = 257.33%` | temps = `0.245 h`
- `alpha075_beta0_mu1` : `final_rel_l2 = 107.22%` | temps = `0.241 h`
- `alpha075_beta05_mu0` : `final_rel_l2 = 236.61%` | temps = `0.241 h`
- `alpha075_beta05_mu1` : `final_rel_l2 = 599.92%` | temps = `0.220 h`

#### Multireseau / global

- `alpha075_beta0_mu0` : `final_rel_l2 = 0.81%` | temps = `0.451 h`
- `alpha075_beta0_mu1` : `final_rel_l2 = 1.29%` | temps = `0.446 h`
- `alpha075_beta05_mu0` : `final_rel_l2 = 0.99%` | temps = `0.445 h`
- `alpha075_beta05_mu1` : `final_rel_l2 = 1.28%` | temps = `0.449 h`

### Sorties finales figees

Les comparatifs single-case sont maintenant figes dans :
- `analyses/single_case/all_config_summary.csv`
- `analyses/single_case/mean_l2_vs_time_all_configs.png`
- `analyses/single_case/mean_l2_vs_training_time_all_configs.png`
- `analyses/single_case/monoreseau/all_runs/...`
- `analyses/single_case/multireseau/all_runs/...`

Protocole de validation a appliquer ensuite :
- phase 1 : smoke test local sur `alpha075_beta0_mu0`
- objectif phase 1 :
  - verifier l'absence de `NaN`
  - verifier l'ecriture de `timing_summary.txt` et `timing_stages.csv`
  - verifier la reconstruction complete des sorties d'analyse
- sorties minimales obligatoires :
  - `rollout_metrics.csv`
  - `summary.txt`
  - `timing_summary.txt`
  - `timing_stages.csv`
  - `error_heatmap.png`
  - `snapshots.png`
  - `comparison_animation.gif`
  - `inference_timing.txt`
  - `inference_timing.png`
- critere de passage a Jean Zay :
  - pas de divergence numerique
  - pipeline d'analyse complet
  - qualite du rollout confirmee sur le cas local
- phase 2 : lancement Jean Zay sur les 4 cas
- phase 3 : consolidation finale de la famille sur :
  - precision
  - stabilite
  - temps total d'entrainement
  - temps par stage si pertinent

### Decision intermediaire actuelle

Configuration retenue :
- `multireseau / global`

Raison :
- meilleure precision
- stabilite sur les 4 single cases
- respecte le critere `L2 < 5%` sur tout l'horizon observe
- reste tres bon sur l'extension `t = 20`, hors un stage isole a nettoyer sur un seul cas

### Protocole de reprise multireseau

Ordre retenu :
- 1. verifier localement `multireseau / local` sur `alpha075_beta0_mu0`
- 2. verifier localement `multireseau / global` sur `alpha075_beta0_mu0`
- 3. si les deux pipelines sont propres, lancer Jean Zay sur les 4 cas

Regles communes pour `multireseau / local` et `multireseau / global` :
- utiliser les memes 4 single cases que pour le monoreseau
- conserver les memes temps de snapshot : `t = 0, 1, 2, 3, 4, 5`
- conserver les memes metriques :
  - `final_rel_l2`
  - `max_rel_l2`
  - `mean_rel_l2`
  - `first_t_gt_5pct`
  - temps total d'entrainement
  - temps d'inference
- produire les memes artefacts visuels :
  - `L2(t)`
  - heatmap erreur absolue
  - snapshots
  - GIF

Decision de validation avant passage Jean Zay :
- `multireseau / local` :
  - si le smoke test local reste structurellement mauvais en rollout, ne pas etendre aux 4 cas sans correction
- `multireseau / global` :
  - si le smoke test local confirme la qualite du pipeline et des timings, etendre directement aux 4 cas

Decision actuelle :
- `multireseau / global` est retenu comme meilleure configuration pratique
- `multireseau / local` est conserve comme piste de diagnostic, mais pas comme candidat final en l'etat


## Etape 4. Passage au probleme parametrique

Objectif :
- partir de la meilleure configuration single-case
- l'etendre progressivement au probleme parametrique
- mesurer jusqu'ou la famille retenue reste stable et precise quand l'espace des parametres s'elargit

Configuration retenue comme base parametrique :
- `multireseau / global`

Parametres autorises a varier :
- `sigma` : ecart-type de la gaussienne
- `alpha`
- `beta`
- `mu`

Parametres fixes dans cette campagne parametrique :
- `V = 0`
- `x0 = 0`
- `type = gaussian`
- `A = 0.1`

Remarque de parametrisation :
- dans le code actuel, la largeur de gaussienne est historiquement notee `w0`
- dans le protocole experimental, on la designe ici par `sigma`

### Strategie generale

Principe :
- elargir l'espace parametrique par paliers
- commencer par la variation la plus simple
- verifier a chaque palier :
  - la stabilite de l'entrainement
  - la qualite de la prediction
  - le cout en temps d'entrainement

Metriques a conserver :
- `final_rel_l2`
- `max_rel_l2`
- `mean_rel_l2`
- `first_t_gt_5pct`
- temps total d'entrainement
- temps d'inference
- nombre de reprises si necessaire

Sorties d'analyse a conserver :
- `L2(t)`
- heatmaps d'erreur
- snapshots
- GIF
- timing d'inference

### Structure attendue de la campagne parametrique

Objectif de structure :
- avoir une arborescence lisible
- separer clairement les campagnes parametriques par palier
- garantir que les sorties soient comparables entre campagnes

#### Configs attendues

Convention recommandee :
- `CGL_amp_phase/configs/parameters/<campagne>/...`

Exemples :
- `CGL_amp_phase/configs/parameters/sigma_only/...`
- `CGL_amp_phase/configs/parameters/sigma_alpha/...`
- `CGL_amp_phase/configs/parameters/sigma_alpha_beta/...`
- `CGL_amp_phase/configs/parameters/full_parametric/...`
- `CGL_amp_phase/configs/parameters/one_variable_free/...`

#### Resultats attendus

Convention recommandee :
- `results/<NomDeCampagneParametrique>/run_<timestamp>_<jobid>/`

Sorties minimales dans chaque run :
- `timing_summary.txt`
- `timing_stages.csv`
- `rollout/rollout_metrics.csv`
- `rollout/summary.txt`
- `stage_*/checkpoints/model_best.pth`

#### Analyses attendues

Convention recommandee :
- `analyses/parameters/<campagne>/`

Organisation interne recommandee :
- `analyses/parameters/<campagne>/family_summary.csv`
- `analyses/parameters/<campagne>/all_cases_l2_vs_time.png`
- `analyses/parameters/<campagne>/all_cases_error_heatmaps.png`
- `analyses/parameters/<campagne>/run_<nom_du_cas>/`

Sorties minimales par cas :
- `l2_vs_time.png`
- `error_heatmap.png`
- `snapshots.png`
- `comparison_animation.gif`
- `summary.txt`
- `inference_timing.txt`
- `inference_timing.png`

#### Niveaux de comparaison a garder

Pour chaque campagne parametrique, conserver trois niveaux de lecture :
- niveau 1 : comparaison globale de la campagne
- niveau 2 : comparaison par sous-famille fixe
- niveau 3 : comparaison par cas individuel

Exemple au palier `sigma_only` :
- niveau campagne :
  - `all_cases_l2_vs_time.png`
  - `family_summary.csv`
- niveau sous-famille :
  - famille A : `alpha=0.75, beta=0, mu=0`
  - famille B : `alpha=0.75, beta=0, mu=1`
  - famille C : `alpha=0.75, beta=0.5, mu=0`
  - famille D : `alpha=0.75, beta=0.5, mu=1`
- niveau cas :
  - sorties detailles par run

### Palier 1. Variation de sigma seule autour des 4 cas single-case

But :
- tester si la meilleure famille generalise deja a une variation simple de la largeur initiale

Parametres :
- `sigma in [0.4, 0.8]`
- `alpha`, `beta`, `mu` fixes comme dans les 4 cas single-case

Concretement, cela definit 4 familles parametriques de base :
- famille A : `alpha = 0.75`, `beta = 0.0`, `mu = 0.0`, `sigma in [0.4, 0.8]`
- famille B : `alpha = 0.75`, `beta = 0.0`, `mu = 1.0`, `sigma in [0.4, 0.8]`
- famille C : `alpha = 0.75`, `beta = 0.5`, `mu = 0.0`, `sigma in [0.4, 0.8]`
- famille D : `alpha = 0.75`, `beta = 0.5`, `mu = 1.0`, `sigma in [0.4, 0.8]`

Lecture attendue :
- si ce palier echoue, la generalisation parametrique est deja trop difficile
- si ce palier reussit, il valide le passage du single-case a une premiere vraie famille parametrique

Retour des premiers runs effectues :
- un premier screening a ete lance avec :
  - `train_cases = 24`
  - `valid_cases = 8`
  - `stage_num_epochs = 15000`
  - `sigma in [0.4, 0.8]`
- conclusion : cette base est trop faible pour conclure proprement
- difficulte principale observee :
  - `sigma = 0.4` degrade fortement la generalisation
  - plusieurs familles franchissent `5%` tres tot
- interpretation actuelle :
  - le probleme n'est pas une instabilite du schema `global_multistage`
  - le probleme vient surtout d'une couverture parametrique insuffisante et d'un budget d'apprentissage trop faible

Nouvelle base retenue pour relance :
- `sigma in [0.6, 0.8]`
- evaluation sur `sigma = 0.6`, `0.7`, `0.8`
- `train_cases = 96`
- `valid_cases = 24`
- `stage_num_epochs = 30000`

Raison du resserrement initial sur `sigma` :
- retirer provisoirement la zone `sigma = 0.4` qui apparait comme le regime le plus difficile
- verifier d'abord si la famille apprend correctement une variation parametrique moderee avant d'ouvrir a nouveau la plage complete

### Palier 2. Variation de sigma et alpha, beta et mu fixes

But :
- tester l'effet d'une variation simultanee de la largeur initiale et du coefficient `alpha`

Parametres :
- `sigma in [0.4, 0.8]`
- `alpha in [0.5, 1.5]`
- `beta` fixe
- `mu` fixe

Comme au palier 1, cela se decline en 4 familles selon les couples fixes `(beta, mu)` des cas single-case.

### Palier 3. Variation de sigma, alpha et beta, mu fixe

But :
- tester l'extension a trois parametres libres, tout en gardant `mu` fixe

Parametres :
- `sigma in [0.4, 0.8]`
- `alpha in [0.5, 1.5]`
- `beta in [-0.5, 0.5]`
- `mu` fixe

Lecture attendue :
- cela permet de separer la difficulte venant de `mu` de celle venant du triplet `sigma, alpha, beta`

### Palier 4. Variation complete

But :
- tester le premier cas pleinement parametrique de base

Parametres :
- `sigma in [0.4, 0.8]`
- `alpha in [0.5, 1.5]`
- `beta in [-0.5, 0.5]`
- `mu in [0.0, 1.0]`

Lecture attendue :
- ce palier sert de reference finale pour la campagne parametrique de base

### Tests complementaires : une seule variable libre

Objectif :
- isoler l'effet de chaque parametre quand les autres sont fixes
- identifier quel parametre degrade le plus la generalisation

Tests proposes :
- test A : `beta` varie, `sigma`, `alpha`, `mu` fixes
- test B : `alpha` varie, `sigma`, `beta`, `mu` fixes
- test C : `mu` varie, `sigma`, `alpha`, `beta` fixes

Utilite :
- ces tests sont plus interpretables que les campagnes fully-coupled
- ils permettent d'identifier la source principale de la difficulte parametrique

Retour des premiers runs effectues :
- des screenings `alpha_only`, `beta_only` et `mu_only` ont deja ete lances avec la base initiale `24/8` et `15000` epochs/stage
- lecture qualitative actuelle :
  - `beta_only` est la direction la plus neutre
  - `mu_only` est intermediaire
  - `alpha_only` est plus difficile
  - `sigma_only` est de loin la direction la plus dure dans la base initiale
- conclusion :
  - la hierarchie de difficulte entre parametres est deja visible
  - mais la base initiale est trop faible pour tirer une conclusion quantitative definitive

Nouvelle base commune retenue pour toutes les campagnes `1 variable` :
- `train_cases = 96`
- `valid_cases = 24`
- `stage_num_epochs = 30000`
- memes 5 blocs temporels `[0,1]`, `[1,2]`, `[2,3]`, `[3,4]`, `[4,5]`

Hypothese de travail actuelle :
- avant d'augmenter le nombre de fenetres temporelles, il faut d'abord tester si plus de couverture parametrique et plus d'iterations suffisent
- l'augmentation du nombre de stages n'est pas retenue comme premier levier, car elle complexifie fortement la campagne sans isoler clairement la cause du defaut de generalisation

### Ordre recommande

Ordre pragmatique :
- 1. palier 1 : `sigma` seul
- 2. tests complementaires a une variable libre
- 3. palier 2 : `sigma + alpha`
- 4. palier 3 : `sigma + alpha + beta`
- 5. palier 4 : variation complete

Raison :
- ton protocole de base est coherent
- mais il est scientifiquement plus lisible de faire les tests "une variable libre" plus tot
- cela permet d'interpreter plus proprement un echec ulterieur sur les paliers plus larges

Conclusion actuelle sur le protocole parametrique :
- le plan est coherent
- il est progressif
- il permet de separer les difficultes de generalisation
- il est adapte a la configuration retenue `multireseau / global`
- les premiers runs montrent deja que la base d'entrainement single-case recyclee telle quelle n'est pas suffisante pour le parametrique
- la priorite immediate est donc :
  - augmenter la taille du pool parametrique
  - augmenter le budget d'entrainement par stage
  - relancer les campagnes `1 variable`
  - ne tester un raffinement du decoupage temporel qu'en second rideau


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
