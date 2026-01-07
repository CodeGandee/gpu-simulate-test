# Developer

This section is for maintainers and contributors working on the repo’s implemented workflows:

- `001-compare-vidur-real-timing` (workload spec + real vs Vidur sim comparison)
- `002-reproduce-vidur-paper-fidelity` (paper-aligned fidelity metrics via `paper-fidelity`)

## Where things live

- Specs, tasks, and contracts:
  - `specs/001-compare-vidur-real-timing/`
  - `specs/002-reproduce-vidur-paper-fidelity/`
- Hydra config trees:
  - `configs/compare_vidur_real/`
  - `configs/paper_fidelity/`
- Implementation code: `src/gpu_simulate_test/`
- Validation:
  - unit tests: `tests/unit/`
  - manual scripts: `tests/manual/`
