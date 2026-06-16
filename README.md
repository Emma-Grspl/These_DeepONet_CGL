# These_DeepONet_CGL

Projet de recherche sur l'approximation d'opérateurs pour l'équation de Ginzburg-Landau complexe 1D avec un `PI-DeepONet` (Physics-Informed Deep Operator Network).

L'objectif du dépôt est d'apprendre l'opérateur qui associe :
- un jeu de paramètres physiques,
- une condition initiale paramétrée,
- une coordonnée spatio-temporelle `(x, t)`,

à la solution complexe `u(x, t)` de la PDE, en utilisant principalement la physique comme signal d'entraînement.

Le dépôt contient aussi un solveur numérique CGL utilisé pour l'audit et le diagnostic, mais pas comme supervision directe dans la boucle d'entraînement principale.

## 1. Problème physique

Le code entraîne un modèle sur une forme 1D de l'équation de Ginzburg-Landau complexe :

```math
u_t = (1 + i\alpha) u_{xx} + \mu u - (1 + i\beta)|u|^2u - V u_x
```

où :
- `alpha` contrôle la dispersion/diffusion complexe,
- `beta` contrôle la non-linéarité complexe,
- `mu` est le terme linéaire de gain/perte,
- `V` est le terme d'advection,
- `u(x,t)` est un champ complexe.

Dans la configuration actuelle :
- le domaine spatial est `x in [-40, 40]`,
- le temps final par défaut est `t_max = 5`,
- les conditions initiales utilisées sont gaussiennes,
- `x0` est actuellement fixé à `0` dans la config fournie,
- `k` varie dans `[-2, 2]`.

## 2. Motivation scientifique

Le but n'est pas seulement d'approximer une trajectoire unique, mais d'apprendre un opérateur paramétrique sur une famille d'équations et de conditions initiales.

Cette approche est intéressante quand :
- on veut balayer un espace de paramètres rapidement après entraînement,
- le solveur numérique devient trop coûteux,
- on vise à terme des PDE plus complexes pour lesquelles il n'existe pas de vérité terrain dense facilement disponible.

Le choix assumé de ce dépôt est donc :
- entraînement principalement par résidu PDE et contraintes physiques,
- solveur numérique réservé à l'audit, au débogage et à l'évaluation.

## 3. Implications physiques importantes

La CGL générale n'est pas, en général, un système conservatif au sens NLS. On ne dispose donc pas d'une énergie universellement conservée simple.

En revanche, pour l'équation implémentée ici, on peut exploiter une balance intégrale de la norme `L2` :

```math
\frac{d}{dt}\int |u|^2 dx
=
-2\int |u_x|^2 dx
+2\mu \int |u|^2 dx
-2\int |u|^4 dx
```

quand les termes de bord ne contribuent pas.

Cette identité est utile car elle fournit une contrainte physique globale sans recourir à une solution de référence. Le code l'utilise maintenant comme perte auxiliaire de stabilisation.

## 4. Qu'est-ce qu'un PI-DeepONet ici ?

Le modèle apprend un opérateur

```text
(paramètres physiques, paramètres de CI, x, t) -> u(x, t)
```

avec deux sous-réseaux :
- un `branch net` pour encoder les paramètres,
- un `trunk net` pour encoder les coordonnées.

Le produit des représentations latentes `branch` et `trunk` reconstruit la sortie complexe.

Dans ce dépôt :
- la sortie est décomposée en partie réelle et imaginaire,
- la condition initiale gaussienne est imposée de manière analytique,
- le réseau apprend surtout la correction dynamique au-delà de `t=0`.

## 5. Description du modèle

Le modèle principal est défini dans [src/models/cgl_deeponet.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/models/cgl_deeponet.py).

### 5.1 Entrées

Le `branch net` reçoit un vecteur de 9 paramètres :
- `alpha`
- `beta`
- `mu`
- `V`
- `A`
- `w0`
- `x0`
- `k`
- `type`

Le `trunk net` reçoit les coordonnées `(x, t)`.

### 5.2 Normalisation

Le modèle normalise :
- linéairement les paramètres bornés,
- logarithmiquement `w0`,
- temporellement `t` sur `[0, t_max]`.

### 5.3 Encodage de Fourier

Le `trunk` utilise un encodage de Fourier multi-échelle pour améliorer la représentation des structures oscillantes et multirésolution.

### 5.4 Coordonnée géométrique adaptative

Le modèle ne travaille pas directement sur `x` brut. Il construit une coordonnée comobile :

```text
xi = (x - x0) / W(t)
```

avec une largeur dynamique `W(t)` dépendant de `w0` et `t`.

L'idée est de suivre l'étalement de l'enveloppe pour faciliter l'approximation.

### 5.5 Hard Constraint à t=0

La condition initiale est imposée analytiquement via un ansatz :
- le réseau reproduit exactement la gaussienne initiale à `t=0`,
- la correction apprise est activée progressivement via un facteur de transition temporelle.

Cela évite d'avoir à entraîner explicitement une loss de condition initiale.

## 6. Description de l'entraînement

La logique principale se trouve dans [src/training/trainer_CGL.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/training/trainer_CGL.py).

### 6.1 Philosophie

L'entraînement est principalement `physics-informed` :
- on ne supervise pas le modèle avec des profils solveur dans la loss principale,
- on optimise un mélange de contraintes locales et globales issues de la physique.

### 6.2 Termes de loss actuels

La loss d'entraînement combine maintenant :
- un résidu PDE fort absolu,
- un résidu PDE relatif,
- une perte faible intégrale sur quelques fonctions tests,
- une contrainte de bord de type Neumann sur `x_min` et `x_max`,
- une perte de balance de masse `L2`,
- une perte de continuité causale entre deux fenêtres temporelles.

Cette combinaison vise à éviter le cas classique :
- `loss` locale très faible,
- mais erreur solution globale encore élevée à l'audit.

### 6.3 Time marching causal

L'entraînement est organisé par fenêtres temporelles successives :
- on avance de `t_prev` vers `t_curr`,
- on valide localement le nouveau cap,
- on surveille l'oubli historique,
- on adapte `dt` si nécessaire.

Le sampler PDE causal mélange :
- une partie de points dans le passé,
- une partie de points près du front actif.

### 6.4 Continuité causale

Avant chaque nouvelle fenêtre, le modèle courant est figé comme `teacher`.

Pendant l'optimisation de la fenêtre suivante, le modèle entraîné est pénalisé s'il s'écarte trop de ce `teacher` sur la tranche `t = t_prev`.

Ce n'est pas une supervision par vérité terrain. C'est une contrainte de cohérence interne entre deux étapes de propagation.

### 6.5 RAR

Le dépôt utilise un mécanisme de `RAR` (Residual-based Adaptive Refinement) :
- génération d'un grand ensemble de points candidats,
- calcul du résidu PDE,
- sélection des points les plus difficiles.

La version actuelle renforce aussi les régions :
- à forte énergie locale,
- à fort gradient d'amplitude.

### 6.6 Finisher L-BFGS

Si Adam s'approche de la cible sans l'atteindre complètement, un passage `L-BFGS` est lancé comme étape de finition locale.

## 7. Audit et rôle du solveur

L'audit est géré dans `run_audit` dans [src/training/trainer_CGL.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/training/trainer_CGL.py).

Le solveur numérique CGL est implémenté dans [src/utils/solver_cgl.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/utils/solver_cgl.py).

Son rôle est :
- évaluer la progression réelle,
- détecter les plateaux,
- estimer l'erreur relative `L2`,
- vérifier l'oubli historique.

Son rôle n'est pas :
- fournir une loss supervisée dans la boucle d'entraînement standard.

L'audit échantillonne désormais les paramètres sur la même famille de problèmes, y compris `k`.

## 8. Architecture du code

Arborescence logique :

```text
.
├── configs/
│   └── cgl_config.yaml
├── launch/
│   ├── jz_submit_CGL.slurm
│   └── jz_submit_CGL_reprise.slurm
├── scripts/
│   ├── train_cgl.py
│   ├── train_cgl_resume.py
│   ├── test_debug.py
│   ├── test_visu.py
│   └── test_animation.py
├── src/
│   ├── data/
│   │   └── generators.py
│   ├── models/
│   │   └── cgl_deeponet.py
│   ├── physics/
│   │   └── pde_cgl.py
│   ├── plot/
│   │   ├── plot_animation.py
│   │   └── plot_snapshot.py
│   ├── training/
│   │   └── trainer_CGL.py
│   └── utils/
│       └── solver_cgl.py
├── analyse_cgl.py
└── requirements.txt
```

### 8.1 Modules principaux

- [configs/cgl_config.yaml](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/configs/cgl_config.yaml)
  configuration physique, modèle et entraînement.

- [src/models/cgl_deeponet.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/models/cgl_deeponet.py)
  architecture du PI-DeepONet.

- [src/physics/pde_cgl.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/physics/pde_cgl.py)
  calcul du résidu PDE et de ses composantes.

- [src/data/generators.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/data/generators.py)
  génération des batches PDE, interface causale, balance de masse, RAR.

- [src/training/trainer_CGL.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/training/trainer_CGL.py)
  boucle d'entraînement, navigation temporelle, audit, checkpointing.

- [src/utils/solver_cgl.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/src/utils/solver_cgl.py)
  solveur numérique CGL pour audit.

- [scripts/train_cgl.py](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/scripts/train_cgl.py)
  point d'entrée principal pour lancer un run.

## 9. Fichier de configuration

Le fichier principal est [configs/cgl_config.yaml](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/configs/cgl_config.yaml).

Il contient :
- la définition du problème physique,
- les bornes de paramètres,
- les hyperparamètres du modèle,
- les tailles de batch,
- les poids des pertes physiques auxiliaires,
- les zones temporelles de la navigation.

Les nouveaux poids utiles sont dans :

```yaml
training:
  physics_losses:
    pde_relative_weight: 0.5
    weak_weight: 0.05
    mass_weight: 0.05
    continuity_weight: 0.1
    mass_n_cases: 16
    mass_n_x: 128
    continuity_batch_size: 2048
```

Interprétation pratique :
- `pde_relative_weight` augmente l'importance du résidu relatif,
- `weak_weight` renforce la contrainte variationnelle,
- `mass_weight` impose davantage la balance globale,
- `continuity_weight` stabilise le passage d'une fenêtre temporelle à la suivante.

## 10. Boucle d'entraînement utilisée

La stratégie effective est la suivante :

1. initialiser le modèle et l'optimiseur,
2. démarrer à `t_prev = 0`,
3. proposer un cap `t_curr = t_prev + dt`,
4. faire un audit local rapide,
5. si nécessaire, lancer une optimisation adaptative sur la fenêtre,
6. valider aussi l'historique,
7. éventuellement lancer une `Rescue Loop` globale,
8. sauvegarder un checkpoint,
9. avancer au cap suivant,
10. finir par un polissage global.

Cette boucle joue le rôle de `navigator`.

## 11. Installation

Créer un environnement Python puis installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dépendances listées dans [requirements.txt](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/requirements.txt) :
- `matplotlib`
- `numpy`
- `scipy`
- `torch`
- `tqdm`
- `PyYAML`

## 12. Comment lancer le code

### 12.1 Nouveau run

```bash
python scripts/train_cgl.py --config configs/cgl_config.yaml
```

Le script :
- crée un dossier `results/CGL_Navigator_Run_<timestamp>/`,
- crée un sous-dossier `checkpoints/`,
- entraîne le modèle,
- sauvegarde les checkpoints et le modèle final.

### 12.2 Reprendre le dernier run

```bash
python scripts/train_cgl.py --resume latest
```

### 12.3 Reprendre un run spécifique

```bash
python scripts/train_cgl.py --resume results/CGL_Navigator_Run_YYYYMMDD-HHMMSS
```

ou :

```bash
python scripts/train_cgl_resume.py --resume latest
```

## 13. Lancer sur Jean Zay

Le dépôt contient des scripts SLURM dans [launch/](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/launch).

Fichiers disponibles :
- [launch/jz_submit_CGL.slurm](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/launch/jz_submit_CGL.slurm)
- [launch/jz_submit_CGL_reprise.slurm](/Users/emma.grospellier/Thèse/These_DeepOnet_CGL/launch/jz_submit_CGL_reprise.slurm)

Le flux typique est :
- adapter les ressources SLURM,
- charger l'environnement Python/CUDA adapté,
- soumettre avec `sbatch`.

## 14. Sorties et checkpoints

Les checkpoints sont stockés dans le dossier du run, typiquement :

```text
results/CGL_Navigator_Run_<timestamp>/checkpoints/
```

On y trouve :
- des checkpoints par cap temporel `ckpt_tXXXX.pth`,
- `model_latest.pth` pour la reprise automatique,
- éventuellement des diagnostics CSV selon les runs.

Le modèle final est aussi sauvegardé à la racine du run.

## 15. Conseils pratiques

Si la `loss` baisse mais que l'audit stagne :
- regarder séparément `PDE(abs/rel/weak)`,
- surveiller `Mass`,
- surveiller `Cont`,
- augmenter `pde_relative_weight` si le résidu absolu devient trop trompeur,
- réduire `continuity_weight` si l'apprentissage devient trop rigide,
- réduire `mass_weight` si la contrainte intégrale domine trop tôt.

Réglages raisonnables de départ :
- `pde_relative_weight: 0.5`
- `weak_weight: 0.05`
- `mass_weight: 0.02` à `0.05`
- `continuity_weight: 0.03` à `0.1`

## 16. Limites actuelles

- l'audit repose encore sur un solveur numérique, donc il reste coûteux,
- le problème actuellement configuré est centré sur des CI gaussiennes,
- la performance peut rester sensible au sampling en temps et en espace,
- le coût CPU de l'audit est nettement plus élevé que celui des tests purement syntaxiques.

## 17. Résumé

Ce dépôt implémente un PI-DeepONet pour la CGL 1D avec :
- condition initiale imposée analytiquement,
- entraînement principalement guidé par la physique,
- navigation temporelle causale,
- audit solveur séparé,
- contraintes physiques auxiliaires pour limiter les faux plateaux de résidu.

Le dépôt est conçu comme une base de travail vers des PDE plus complexes où la vérité terrain ne sera plus accessible pendant l'entraînement.
