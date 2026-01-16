# Repository Guidelines

## Project Structure & Module Organization

- `src/gpu_simulate_test/`: Python package code (installed editable via Pixi).
- `context/`: design notes, runbooks, and reproducibility docs for simulator experiments.
- `extern/tracked/vidur/`: Git submodule for the Vidur simulator.
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

## Documentation & Diagrams

- If you add Mermaid `sequenceDiagram` figures, follow `magic-context/instructions/mermaid-seq-styling.md`.

## Testing Guidelines

No test suite is committed yet. If you add tests:
- Place them under `tests/` and name files `test_*.py`.
- Prefer `pytest`; run with `pixi run pytest`.

## Commit & Pull Request Guidelines

- Commits: short, imperative, sentence-case subjects (e.g., “Ignore tmp and .codex”).
- PRs: include a brief description, reproduction steps/commands run (e.g., `pixi install`), and link any related issues or experiment notes in `context/`.

## Active Technologies
- Python 3.13 (Pixi) + `pixi`, `torch==2.9.1+cu128`, `transformers`, `pandas`, `matplotlib`, Vidur (`extern/tracked/vidur` editable), Sarathi-Serve (`extern/tracked/sarathi-serve` editable) (001-compare-vidur-real-timing)
- Files under `/data1/huangzhe/code/gpu-simulate-test/tmp/` (CSV + JSON + Markdown reports) (001-compare-vidur-real-timing)
- Python 3.13 (Pixi) + `hydra-core` (configs + provenance), `torch==2.9.1+cu128` (real timing harness), `transformers` (optional real backend), `vidur` (editable submodule dependency), `pandas`/`pyarrow` (metrics I/O), `matplotlib`/`seaborn`/`plotly` (plots) (001-compare-vidur-real-timing)
- Files under `/data1/huangzhe/code/gpu-simulate-test/tmp/` (CSV + JSON + Markdown + plots); no DB (001-compare-vidur-real-timing)
- Python 3.13 (Pixi env; repo declares `requires-python >= 3.11`) + Pixi, Hydra (`hydra-core`), Vidur (`/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`), Sarathi-Serve (`/data1/huangzhe/code/gpu-simulate-test/extern/tracked/sarathi-serve`), PyTorch (`torch==2.9.1+cu128`), pandas/pyarrow, matplotlib/seaborn/plotly (002-reproduce-vidur-paper-fidelity)
- Filesystem (CSV/JSON/Markdown artifacts under `/data1/huangzhe/code/gpu-simulate-test/tmp/` and `/data1/huangzhe/code/gpu-simulate-test/results/`) (002-reproduce-vidur-paper-fidelity)
- Python 3.13 (Pixi) + Hydra (`hydra-core`), PyTorch (`torch==2.9.1+cu128`), Vidur (editable submodule), Sarathi-Serve (editable submodule), pandas/pyarrow, matplotlib/seaborn/plotly (003-paper-fidelity-more-models)
- Python 3.13 (Pixi), repo supports `>=3.11` + `hydra-core`/OmegaConf, Vidur (submodule), Sarathi-Serve (submodule), pandas/pyarrow, torch/transformers (optional real backend) (004-vidur-cli)
- Filesystem (CSV/JSON/YAML/Markdown) under the resolved workspace root (004-vidur-cli)
- Python 3.13 (Pixi env; repo supports `>=3.11`) + Pixi, PyTorch (CUDA build), Hydra/OmegaConf, pandas, Vidur (`/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`) (005-vidur-mlp-cuda-driver)
- Filesystem artifacts (CSV/JSON/Markdown) under `/data1/huangzhe/code/gpu-simulate-test/tmp/` and `/data1/huangzhe/code/gpu-simulate-test/results/` (005-vidur-mlp-cuda-driver)

## Recent Changes
- 001-compare-vidur-real-timing: Added Python 3.13 (Pixi) + `pixi`, `torch==2.9.1+cu128`, `transformers`, `pandas`, `matplotlib`, Vidur (`extern/tracked/vidur` editable), Sarathi-Serve (`extern/tracked/sarathi-serve` editable)
