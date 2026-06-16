# Organisation Du Projet

## Objectif

Le depot est maintenant recentre sur un seul axe :

- `single-case`
- `physics-only`
- `sans supervision classique`

Le parametrique a ete retire du chemin actif et devra etre reconstruit plus tard a partir d'une base single-case propre.

## Arborescence active

```text
These_DeepOnet_CGL/
├── src/
├── scripts/
├── configs/
├── launch/
├── results/
├── analyses/
│   └── single_case/
├── outputs/
├── run_assets/
├── run_registry/
├── docs/
│   ├── protocol/
│   ├── conclusions/
│   ├── project_constraints.md
│   ├── physics_only_reboot_plan_2026-06-16.md
│   └── supervision_audit_2026-06-16.md
└── archive/
```

## Regles

### 1. `src/`, `scripts/`, `configs/`, `launch/`

- ce sont les seuls repertoires de code actif
- tout nouveau developpement doit respecter la contrainte `physics-only`

### 2. `results/`

- ne contient que des runs bruts `single-case` non supervises
- pour l'instant, il est volontairement vide
- les deux premieres familles a y refixer sont :
  - `monoreseau global direct`
  - `monoreseau global curriculum`

### 3. `analyses/single_case/`

- contient uniquement :
  - les analyses single-case non supervisees
  - les vues du solveur de reference
- les futures analyses gelees devront etre rangees par famille :
  - `global_direct`
  - `global_curriculum`
  - `local_physics_only`
  - `global_multinet_physics_only`
  - `local_multinet_physics_only`

### 4. `run_assets/` et `run_registry/`

- servent a figer les campagnes retenues
- ils sont actuellement vides, a l'exception des fichiers de structure a venir
- toute campagne validee devra y laisser une trace minimale

### 5. `docs/`

- `docs/project_constraints.md` fixe la contrainte dure :
  - aucune supervision classique
- `docs/supervision_audit_2026-06-16.md` explique ce qui a ete purge
- `docs/protocol/` doit contenir uniquement les protocoles `single-case physics-only`

## Discipline pour la suite

- ne pas relancer de branche supervisee
- ne pas reintroduire de loss basee sur des cibles solveur
- ne figer que des campagnes single-case non supervisees
- ne rouvrir le parametrique qu'apres validation claire des familles single-case
