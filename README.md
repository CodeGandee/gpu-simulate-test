# gpu-simulate-test

Testbed for evaluating GPU simulators with an emphasis on LLM/inference workloads.

## Goals

- Provide a repeatable way to run and compare simulators on the same workloads/configs.
- Track setup notes, pitfalls, and reproducible scripts for each simulator.
- Keep results and configs versioned.

## Quickstart

Initialize submodules:

```bash
git submodule update --init --recursive
```

Create the Pixi environment (Python + PyPI deps):

```bash
pixi install
```

Run commands inside the environment:

```bash
pixi run python -c "import torch; print(torch.__version__)"
```

Optional: configure `CODEX_HOME` and an HTTP proxy for tooling:

```bash
./setup-envs.sh --proxy auto
```

## External Assets (models/datasets)

Large machine-local assets are managed under `models/` and `datasets/` using an “external
reference” pattern (docs + bootstrap scripts are committed; the actual data is not).

- Bootstrap everything:
  - `bash models/bootstrap.sh`
  - `bash datasets/bootstrap.sh`
- Per-reference bootstraps:
  - Qwen3 model: `bash models/qwen3-0.6b/bootstrap.sh`
  - COCO 2017: `bash datasets/coco2017/bootstrap.sh`

Both bootstrap scripts support `EXTERNAL_REF_ROOT` to point at your local storage root.

## Simulators

First simulator to try:

- Vidur (Microsoft): https://github.com/microsoft/vidur

Other candidates:

- RealLM (Bespoke Silicon Group): https://github.com/bespoke-silicon-group/reallm
- LLMCompass (Princeton University): https://github.com/PrincetonUniversity/LLMCompass
- Accel-Sim Framework: https://github.com/accel-sim/accel-sim-framework

## Repo Layout

- `src/gpu_simulate_test/`: Python package code (src layout).
- `extern/`: third-party code (git submodules, e.g. `extern/vidur/`).
- `models/`: external model references (symlink-based, not committed).
- `datasets/`: external dataset references (symlink-based, not committed).
- `context/`: project notes, runbooks, and experiment/repro docs.
- `scripts/`: small helper scripts/entrypoints (repo-owned).
- `tests/`: test skeleton (`unit/`, `integration/`, `manual/`).
- `tmp/`: local scratch space (ignored by git).
