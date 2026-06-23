#!/bin/bash
set -euo pipefail

cd "${WORK}/These_DeepOnet_CGL" || exit 1

launcher="launch/jz_submit_CGL_local_multinet_physics_only_amp_phase_case_20h.slurm"
cfg="configs/cgl_single_case_local_multinet_physics_only_v5_stageaudit_amp_phase_alpha075_beta0_mu0_t1.yaml"

sbatch --export=ALL,CONFIG_PATH="$cfg" "$launcher"
