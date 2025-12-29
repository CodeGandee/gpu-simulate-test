# Repository Guidelines

## Project Structure & Module Organization

- `src/gpu_simulate_test/`: Python package code (installed editable via Pixi).
- `context/`: design notes, runbooks, and reproducibility docs for simulator experiments.
- `extern/vidur/`: Git submodule for the Vidur simulator.
- `magic-context/`: Git submodule used for repo “context”/workflow tooling.
- `pyproject.toml` + `pixi.lock` + `.pixi/`: Pixi-managed dev environment (Python + PyPI deps).
- `tmp/`: local scratch output (ignored by git). `/.codex/` is also ignored.

Initialize submodules when cloning:
`git submodule update --init --recursive`

## Build, Test, and Development Commands

- `pixi install`: resolve deps, update `pixi.lock`, and create `.pixi/envs/default`.
- `pixi run python -c "import torch; print(torch.__version__)"`: run Python inside the Pixi env.
- `pixi run python -m pip list`: inspect installed PyPI packages in the env.
- `./setup-envs.sh --proxy auto|none|http://host:port`: set `CODEX_HOME` and optional proxy vars.

GPU note: PyTorch is pinned to a CUDA 12.8 build (`+cu128`). A working NVIDIA driver/runtime is required for `torch.cuda.is_available()` to be `True`.

## Coding Style & Naming Conventions

- Python: follow PEP 8 with 4-space indentation; prefer explicit names and type hints where useful.
- Modules: keep simulator/workload integrations isolated (prefer new top-level packages under `src/`).
- Files/dirs: use `snake_case` for Python; avoid committing large generated artifacts (use `tmp/`).

## Testing Guidelines

No test suite is committed yet. If you add tests:
- Place them under `tests/` and name files `test_*.py`.
- Prefer `pytest`; run with `pixi run pytest`.

## Commit & Pull Request Guidelines

- Commits: short, imperative, sentence-case subjects (e.g., “Ignore tmp and .codex”).
- PRs: include a brief description, reproduction steps/commands run (e.g., `pixi install`), and link any related issues or experiment notes in `context/`.
