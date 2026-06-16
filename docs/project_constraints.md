# Project Constraints

## Hard constraint - 2026-06-16

For CGL work in this repository:

- no classical supervised training
- no loss based on direct regression to solver targets
- no future campaign should be presented as valid if it uses `target_re`, `target_im`, `target_u_*`, or equivalent direct solver supervision

Accepted direction:

- physics-only training
- PDE residual
- boundary / continuity / conservation constraints
- benchmark evaluation against the solver only at evaluation time

Operational rule:

- if a new script introduces direct solver targets inside the training loss, it is out of scope for this project state

Reference:

- `docs/supervision_audit_2026-06-16.md`
