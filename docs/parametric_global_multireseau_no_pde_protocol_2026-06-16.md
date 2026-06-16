# Protocole Parametrique Global Multireseau Sans Terme PDE - 2026-06-16

## Point de depart

Contrainte dure a conserver :

- aucune supervision solveur directe dans la loss
- solveur autorise uniquement pour audit et benchmark

Interpretation retenue ici de ta demande "enlever la physique de la loss" :

- on retire le residu PDE
- on retire la balance de masse
- on garde uniquement des contraintes structurelles non supervisees
- on remplace le signal physique supprime par de la distillation depuis des teachers `single-case physics-only` deja figes

Si on retire a la fois :

- le solveur dans la loss
- le residu PDE
- et tout teacher non supervise

alors le probleme devient mal pose. Le reseau peut satisfaire la continuite inter-stage et les bords tout en apprenant une dynamique vide ou quasi-stationnaire. Ce protocole evite cela en ajoutant des teachers `single-case` non supervises comme points d'ancrage.

## Idee generale

On garde la structure `global multireseau` :

- un reseau par bloc temporel
- warm-start interstage
- prediction finale piecewise par stage

Mais la loss devient :

- `L_anchor` : distillation sur des cas d'ancrage couverts par des teachers `single-case physics-only`
- `L_cont` : continuite a l'interface avec le stage precedent
- `L_overlap` : coherence sur une courte fenetre de recouvrement autour des interfaces
- `L_hist` : replay sur d'anciennes interfaces deja apprises pour limiter l'oubli
- `L_bc` : periodicite / coherence aux bords
- `L_param_smooth` : regularisation de l'interpolation en parametre entre cas voisins

Loss totale proposee :

```text
L = 1.00 L_anchor
  + 0.50 L_cont
  + 0.25 L_overlap
  + 0.10 L_hist
  + 0.10 L_bc
  + 0.05 L_param_smooth
```

Ce n'est pas une branche `physics-only`. C'est une branche `teacher-distilled`, sans solveur dans la loss, qui suppose que les teachers `single-case` eux-memes ont ete appris sans supervision solveur.

## Prerequis obligatoires

Avant toute campagne parametrique, il faut disposer de teachers `single-case physics-only` propres sur les points d'ancrage.

Sans ces teachers :

- `beta_only`, `alpha_only`, `mu_only`, `sigma_only` sans residu PDE ne sont pas scientifiquement exploitables

Format requis pour chaque teacher :

- un run fige
- checkpoint final stable
- audits `L2(t)` disponibles
- frontieres temporelles verifiees sur son cas

## Forme de la distillation

Pour un cas d'ancrage `p_anchor` :

- on choisit un teacher `T_k` pour chaque stage `k`
- le student parametrique `S_k` voit ce meme cas comme un simple echantillon du batch
- on penalise l'ecart entre `S_k(x,t,p_anchor)` et `T_k(x,t,p_anchor)` sur le bloc du stage

Pour les cas hors ancrage :

- pas de solveur dans la loss
- pas de residu PDE
- seulement `L_cont`, `L_overlap`, `L_hist`, `L_bc`, `L_param_smooth`

La distillation sert donc d'ancrage local, et les contraintes structurelles propagent une interpolation raisonnable entre ces points.

## Ce qu'on reprend des anciennes campagnes

### `alpha_only`

Constat historique utile :

- le run etait globalement bon
- le vrai point dur etait `alpha_max`
- le pic apparaissait en fin d'horizon, surtout sur `[4.5, 5.0]`

Consequence protocolaire :

- on garde un decoupage de queue fin
- on sur-echantillonne `alpha` proche de `1.5`
- on force plus d'epochs sur le dernier stage

### `beta_only`

Constat historique utile :

- c'etait la direction la plus neutre
- 5 gros blocs suffisaient

Consequence protocolaire :

- campagne de calibration
- si `beta_only` echoue, inutile d'ouvrir `alpha_only` et `sigma_only`

### `mu_only`

Constat historique utile :

- le run etait globalement propre
- la derive remontait surtout vers la fin
- le regime le plus fragile etait proche de `mu_min`

Consequence protocolaire :

- on garde un raffinement mi-fin de trajectoire
- on sur-echantillonne `mu` proche de `0.0`

### `sigma_only`

Constat historique utile :

- c'etait la direction la plus dure
- pics precoces autour de `t=0.1` et `t=0.2`
- le regime `sigma_min` etait le plus fragile
- la densification temporelle precoce ameliorait nettement la situation

Consequence protocolaire :

- on decoupe tres finement le debut
- on traite `sigma_only` comme 4 sous-campagnes separees suivant `(beta, mu)`
- on lance d'abord une plage nettoyee `w0 in [0.6, 0.8]`
- on ne re-ouvre `w0 < 0.6` qu'apres stabilisation

## Protocole commun

### 1. Teachers d'ancrage

Il faut figer les teachers suivants.

#### `alpha_only`

Campagne unique :

- `beta = 0.0`
- `mu = 0.0`
- `A = 0.1`
- `w0 = 0.6`

Teachers requis :

- `alpha = 0.5`
- `alpha = 1.0`
- `alpha = 1.5`

#### `beta_only`

Campagne unique :

- `alpha = 0.75`
- `mu = 0.0`
- `A = 0.1`
- `w0 = 0.6`

Teachers requis :

- `beta = -0.5`
- `beta = 0.0`
- `beta = 0.5`

#### `mu_only`

Campagne unique :

- `alpha = 0.75`
- `beta = 0.0`
- `A = 0.1`
- `w0 = 0.6`

Teachers requis :

- `mu = 0.0`
- `mu = 0.5`
- `mu = 1.0`

#### `sigma_only`

Quatre sous-campagnes distinctes :

- famille A : `alpha = 0.75`, `beta = 0.0`, `mu = 0.0`
- famille B : `alpha = 0.75`, `beta = 0.0`, `mu = 1.0`
- famille C : `alpha = 0.75`, `beta = 0.5`, `mu = 0.0`
- famille D : `alpha = 0.75`, `beta = 0.5`, `mu = 1.0`

Teachers requis par famille :

- `w0 = 0.6`
- `w0 = 0.7`
- `w0 = 0.8`

### 2. Distribution des batches

Repartition recommandee des cas dans un batch :

- `35%` cas d'ancrage teacher
- `40%` cas aleatoires uniformes dans la plage parametrique
- `25%` cas hard-focus

Le hard-focus depend de la famille :

- `alpha_only` : `alpha in [1.3, 1.5]`
- `beta_only` : aucun focus dur au premier passage
- `mu_only` : `mu in [0.0, 0.2]`
- `sigma_only` : `w0 in [0.6, 0.625]`

### 3. Overlap temporel

Comme il n'y a plus de residu PDE pour recoller les blocs, il faut renforcer les transitions :

- recouvrement d'entrainement de `10%` a `20%` entre stages voisins
- `L_overlap` evalue non seulement a `t_start`, mais sur une petite bande temporelle
- `L_hist` rejoue systematiquement les interfaces deja stabilisees

### 4. Audit

Le solveur ne revient qu'en audit.

Metriques minimales :

- `final_rel_l2`
- `max_rel_l2`
- `mean_rel_l2`
- `first_t_gt_5pct`

Points d'audit obligatoires :

- `min / mid / max` de la variable libre
- pour `sigma_only` : audit dense en `t = 0.1`, `0.2`, `0.5`, `0.8`, `1.0`
- pour `alpha_only` : audit dense sur `t = 4.0`, `4.5`, `5.0`

## Protocole par famille

### 1. `beta_only`

But :

- campagne la plus simple
- verifier que la branche `sans terme PDE` reste entrainable

Plage :

- `beta in [-0.5, 0.5]`

Decoupage temporel :

- `[0.0, 1.0]`
- `[1.0, 2.0]`
- `[2.0, 3.0]`
- `[3.0, 4.0]`
- `[4.0, 5.0]`

Budget recommande :

- `train_cases = 96`
- `valid_cases = 24`
- `stage_epochs = 25000`

Teachers d'ancrage :

- `beta = -0.5`
- `beta = 0.0`
- `beta = 0.5`

Critere de passage :

- si `beta_mid` ou `beta_max` franchit `5%` trop tot, on stoppe la branche

### 2. `mu_only`

But :

- exploiter la bonne stabilite historique
- surveiller uniquement la derive de fin

Plage :

- `mu in [0.0, 1.0]`

Decoupage temporel :

- `[0.0, 1.0]`
- `[1.0, 2.0]`
- `[2.0, 2.5]`
- `[2.5, 3.0]`
- `[3.0, 3.5]`
- `[3.5, 4.0]`
- `[4.0, 4.5]`
- `[4.5, 5.0]`

Budget recommande :

- `train_cases = 160`
- `valid_cases = 40`
- `stage_epochs = 25000`
- derniers deux stages : `35000`

Teachers d'ancrage :

- `mu = 0.0`
- `mu = 0.5`
- `mu = 1.0`

Focus dur :

- `mu in [0.0, 0.2]`

Critere de passage :

- toutes les evals `mu_min / mu_mid / mu_max` restent sous `5%`
- la derive de fin ne transforme pas `[4.0, 5.0]` en zone critique

### 3. `alpha_only`

But :

- traiter explicitement le pic sur `alpha_max` en fin d'horizon

Plage :

- `alpha in [0.5, 1.5]`

Decoupage temporel :

- `[0.0, 1.0]`
- `[1.0, 1.5]`
- `[1.5, 2.0]`
- `[2.0, 3.0]`
- `[3.0, 3.5]`
- `[3.5, 4.0]`
- `[4.0, 4.5]`
- `[4.5, 5.0]`

Budget recommande :

- `train_cases = 128`
- `valid_cases = 32`
- `stage_epochs = 30000`
- dernier stage : `50000`

Teachers d'ancrage :

- `alpha = 0.5`
- `alpha = 1.0`
- `alpha = 1.5`

Focus dur :

- `alpha in [1.3, 1.5]`

Renforcement specifique :

- `L_overlap` augmente de `25%` sur les deux derniers stages
- replay obligatoire du stage final a chaque tour de raffinement

Critere de passage :

- `alpha_max` ne doit plus presenter d'explosion nette sur `[4.5, 5.0]`
- un depassement ponctuel proche de `5%` est acceptable
- une remontee vers `8-10%` ne l'est pas

### 4. `sigma_only`

But :

- traiter la direction la plus dure
- priorite a la stabilite precoce

Plage de depart :

- `w0 in [0.6, 0.8]`

Re-ouverture optionnelle ensuite :

- `w0 in [0.4, 0.8]`

Decoupage `t <= 1` pour debug :

- `[0.0, 0.2]`
- `[0.2, 0.3]`
- `[0.3, 0.4]`
- `[0.4, 0.5]`
- `[0.5, 0.6]`
- `[0.6, 0.7]`
- `[0.7, 0.8]`
- `[0.8, 1.0]`

Decoupage `t <= 5` :

- `[0.0, 0.2]`
- `[0.2, 0.5]`
- `[0.5, 0.6]`
- `[0.6, 0.7]`
- `[0.7, 0.8]`
- `[0.8, 1.0]`
- `[1.0, 1.5]`
- `[1.5, 2.0]`
- `[2.0, 3.0]`
- `[3.0, 3.5]`
- `[3.5, 4.0]`
- `[4.0, 4.5]`
- `[4.5, 5.0]`

Budget recommande par famille `(beta, mu)` :

- `train_cases = 192`
- `valid_cases = 48`
- `t=1` debug : `18000` a `22000` epochs/stage
- `t=5` : `30000` epochs/stage
- stages `0` a `5` : `40000+`

Teachers d'ancrage par famille :

- `w0 = 0.6`
- `w0 = 0.7`
- `w0 = 0.8`

Focus dur :

- `w0 in [0.6, 0.625]`

Critere de passage :

- le pic autour de `t=0.1` et `t=0.2` doit descendre en dessous de `5%`
- si seule la famille `sigma_min` reste critique, on la traite a part
- si `sigma_mid` et `sigma_max` restent mauvais, on n'ouvre pas `w0 < 0.6`

## Ordre de lancement recommande

1. `beta_only`
2. `mu_only`
3. `alpha_only`
4. `sigma_only` famille A `beta0_mu0`
5. `sigma_only` famille B `beta0_mu1`
6. `sigma_only` famille C `beta05_mu0`
7. `sigma_only` famille D `beta05_mu1`

Raison :

- `beta_only` sert de campagne test
- `mu_only` est historiquement stable
- `alpha_only` est interpretable mais plus dur
- `sigma_only` doit rester la derniere branche a ouvrir

## Ce qu'il ne faut pas faire

- ne pas lancer `alpha_only`, `mu_only` ou `sigma_only` sans teachers d'ancrage
- ne pas ouvrir `sigma in [0.4, 0.8]` avant d'avoir stabilise `[0.6, 0.8]`
- ne pas rouvrir `alpha_beta_mu_sigma` tant que ces quatre campagnes ne sont pas propres

## Conclusion pratique

Si tu veux vraiment une branche parametrique `global multireseau` sans terme PDE, la seule version defendable est :

- no solver supervision in loss
- no PDE residual in loss
- teachers `single-case physics-only` comme ancrage
- continuity / overlap / replay / BC comme colle structurelle

Sans cette distillation d'ancrage, la campagne est trop mal posee pour etre interpretable.
