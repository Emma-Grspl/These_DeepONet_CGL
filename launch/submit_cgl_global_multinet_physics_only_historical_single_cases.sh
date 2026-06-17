#!/bin/bash
set -euo pipefail

cd "${WORK}/These_DeepOnet_CGL" || exit 1

launcher="launch/jz_submit_CGL_global_multinet_physics_only_historical_case_20h.slurm"

configs=(
  "configs/cgl_single_case_global_multinet_physics_only_historical_amp_phase_alpha075_beta0_mu0_t5.yaml"
  "configs/cgl_single_case_global_multinet_physics_only_historical_amp_phase_alpha075_beta0_mu1_t5.yaml"
  "configs/cgl_single_case_global_multinet_physics_only_historical_amp_phase_alpha075_beta05_mu0_t5.yaml"
  "configs/cgl_single_case_global_multinet_physics_only_historical_amp_phase_alpha075_beta05_mu1_t5.yaml"
)

for cfg in "${configs[@]}"; do
  echo "===== SUBMIT $cfg ====="
  sbatch --export=ALL,CONFIG_PATH="$cfg" "$launcher"
done
