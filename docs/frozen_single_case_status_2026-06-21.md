# Etat single-case CGL - 2026-06-21

## Regle dure

- pas de supervision classique dans la loss
- evaluation solveur autorisee uniquement pour mesurer les erreurs apres entrainement
- `global_direct` et `local_mononet` sont figes tels quels et ne doivent plus etre relances

## Runs en cours sur Jean Zay

Les job names visibles dans `squeue` correspondent aux familles suivantes :

| Job name | Famille | Statut |
|---|---|---|
| `CGLA_CAS` | monoreseau global curriculum | relance autorisee pour completer jusqu'a `t=5` |
| `CGL_GMH_` | multireseau global historical | diagnostic en cours |
| `CGL_LM_C` | monoreseau local | ne plus relancer apres les jobs deja soumis |

Mapping probable si les jobs ont ete soumis dans l'ordre des boucles :

| Jobs | Configurations |
|---|---|
| `727483`-`727486` | `configs/cgl_case_alpha075_beta0_mu0_tchar_t5.yaml`, `configs/cgl_case_alpha075_beta0_mu1_tchar_t5.yaml`, `configs/cgl_case_alpha075_beta05_mu0_tchar_t5.yaml`, `configs/cgl_case_alpha075_beta05_mu1_tchar_t5.yaml` |
| `726999`-`727002` | `configs/cgl_single_case_global_multinet_physics_only_historical_amp_phase_alpha075_beta0_mu0_t5.yaml`, `configs/cgl_single_case_global_multinet_physics_only_historical_amp_phase_alpha075_beta0_mu1_t5.yaml`, `configs/cgl_single_case_global_multinet_physics_only_historical_amp_phase_alpha075_beta05_mu0_t5.yaml`, `configs/cgl_single_case_global_multinet_physics_only_historical_amp_phase_alpha075_beta05_mu1_t5.yaml` |
| `726991`-`726997` | `local_mononet` single-case physics-only, exact mapping a confirmer dans les logs Jean Zay |

Commande Jean Zay pour confirmer le mapping exact :

```bash
cd $WORK/These_DeepOnet_CGL || exit 1

for id in 727483 727484 727485 727486 726999 727000 727001 727002 726991 726992 726993 726994 726995 726996 726997
do
  echo "===== JOB $id ====="
  grep -h -E 'Config :|Run dir :|Resume :' slurm/log/*_${id}.out slurm/log/*${id}.out 2>/dev/null || true
  echo
done
```

## Familles figees

### `global_direct`

Statut : `frozen_no_relaunch`.

Runs conserves :

- `results/CGL_AmpPhase_alpha075_beta0_mu0_global_direct_t5/run_20260616-131117_492824`
- `results/CGL_AmpPhase_alpha075_beta0_mu1_global_direct_t5/run_20260616-131117_492825`
- `results/CGL_AmpPhase_alpha075_beta05_mu0_global_direct_t5/run_20260616-131124_492826`
- `results/CGL_AmpPhase_alpha075_beta05_mu1_global_direct_t5/run_20260616-131124_492828`

Artefacts conserves :

- `run_assets/single_case_physics_only/global_direct/`
- `analyses/single_case/global_direct/`
- `run_registry/single_case_physics_only_runs.csv`

Decision : ne plus relancer. Cette famille reste une baseline constatee.

### `local_mononet`

Statut : `frozen_no_relaunch`.

Runs conserves :

- `alpha075_beta0_mu0_t1`: `run_20260617-103731_534204`
- `alpha075_beta0_mu0_t5`: `run_20260617-103731_534209`
- `alpha075_beta0_mu1_t1`: `run_20260617-103731_534205`
- `alpha075_beta0_mu1_t5`: `run_20260617-103731_534210`
- `alpha075_beta05_mu0_t1`: `run_20260617-103730_534206`
- `alpha075_beta05_mu0_t5`: `run_20260617-103834_534211`
- `alpha075_beta05_mu1_t1`: `run_20260617-103731_534207`
- `alpha075_beta05_mu1_t5`: `run_20260617-103832_534212`

Artefacts conserves :

- checkpoint final utile : `model_final_local_physics_mononet_amp_phase.pth`
- evaluations : `evaluation/`
- timing : `timing_summary.txt`, `timing_stages.csv`
- configs, slurm generique, scripts et doc protocole
- `run_assets/single_case_physics_only/local_mononet/`
- `analyses/single_case/local_physics_only/`
- `run_registry/single_case_physics_only_runs.csv`

Decision : ne plus relancer. Les performances sont conservees comme resultat final de cette famille, meme si elles sont mauvaises.
