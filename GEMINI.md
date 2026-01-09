# gpu-simulate-test Context

## Project Overview

`gpu-simulate-test` is a testbed for evaluating GPU simulators, with a specific focus on **Vidur** (MLSys'24) and Large Language Model (LLM) inference workloads. The goal is to provide a reproducible environment to compare simulated results against real-world GPU inference runs, track fidelity gaps, and manage configurations.

## Architecture

The project is a Python package (`gpu_simulate_test`) managed by `pixi`.

*   **`src/gpu_simulate_test/`**: Core package.
    *   `cli/`: Entry points for various tools (exposed via `pixi`).
    *   `vidur_ext/`: Wrappers and extensions for the Vidur simulator (`extern/tracked/vidur`).
    *   `real_bench/`: Harness for running real GPU inference benchmarks.
    *   `paper_fidelity/`: Tools specifically for reproducing/verifying Vidur paper results.
    *   `config/`: Configuration management (using `hydra-core`).
*   **`extern/`**: Contains third-party dependencies as git submodules (e.g., `vidur`, `sarathi-serve`).
*   **`configs/`**: Hydra configuration files (`.yaml`).
*   **`models/` & `datasets/`**: "External reference" pattern for large assets. Bootstrap scripts download data to a local scratch path while keeping the repo light.

## Environment & Setup

The project uses **Pixi** for dependency management (Python + PyPI + System libs).

### Prerequisite
```bash
git submodule update --init --recursive
```

### Installation
```bash
pixi install
```

### External Assets
Models and datasets are not stored in the repo. Use bootstrap scripts:
```bash
bash models/bootstrap.sh          # Set up model directory structure
bash datasets/bootstrap.sh        # Set up dataset directory structure
bash models/qwen3-0.6b/bootstrap.sh # Example: Download/link specific model
```

## Key Commands (Pixi Tasks)

All commands should be run via `pixi run <task>` or inside the `pixi shell`.

| Task | Command | Description |
| :--- | :--- | :--- |
| `workload-spec` | `python -m gpu_simulate_test.cli.workload_spec` | Generate/validate workload specifications. |
| `real-bench` | `python -m gpu_simulate_test.cli.real_bench` | Run real GPU inference benchmarks. |
| `vidur-profile` | `python -m gpu_simulate_test.cli.vidur_profile` | Generate profiling data for Vidur. |
| `vidur-sim` | `python -m gpu_simulate_test.cli.vidur_sim` | Run the Vidur simulator. |
| `compare-runs` | `python -m gpu_simulate_test.cli.compare_runs` | Compare simulation vs. real results. |
| `paper-fidelity`| `python -m gpu_simulate_test.cli.paper_fidelity`| Run fidelity checks against Vidur paper results. |
| `docs-serve` | `mkdocs serve` | Serve local documentation at `http://127.0.0.1:8000`. |
| `docs-build` | `mkdocs build --strict` | Build static documentation. |

## Development Conventions

*   **Python Version:** 3.13 (managed by Pixi).
*   **Style:** PEP 8, 4-space indentation.
*   **Testing:**
    *   Located in `tests/` (`unit/`, `integration/`, `manual/`).
    *   Run with `pixi run pytest`.
*   **Configuration:** Uses `hydra-core`. Configs are in `configs/`. Do not hardcode paths or parameters in code; use the config system.
*   **Artifacts:** Large outputs should go to `tmp/` or `results/` (ignored by git).
