# Data Model: Vidur MLP profiling validation and provenance

**Branch**: `005-vidur-mlp-cuda-driver`  
**Date**: 2026-01-16  
**Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/005-vidur-mlp-cuda-driver/spec.md`

## Entities

### Profiling Run

Represents one execution of a profiling workflow for a specific model + hardware + run configuration.

Key attributes:

- `run_id`: unique identifier for the run
- `started_at`, `ended_at`: timestamps
- `git_commit`: code revision identifier
- `git_dirty`: whether local changes were present
- `env`: environment snapshot (key environment variables + versions)
- `params`: fully resolved run configuration used for the run

### Profiling Root

Represents the staged artifact directory used by simulations and reports.

Key attributes:

- `profiling_root`: absolute path to the root directory
- `hardware_id` / `device`, `network_device`
- applicability constraints:
  - tensor-parallel and pipeline-parallel settings (if required for compatibility checks)
  - CPU overhead inclusion settings (profiled vs expected by consumer)

### MLP Timing Dataset

Represents the staged MLP compute profiling CSV (e.g., `data/profiling/compute/<device>/<model_id>/mlp.csv`).

Key attributes:

- `csv_path`: absolute path
- `row_count`, `column_count`
- `input_size_column`: canonical independent variable (token count)
- `time_stats_columns`: list of all columns matching `time_stats.*` (validated set)

### MLP Validation Result

Represents the outcome of validating the MLP timing dataset.

Key attributes:

- `csv_path`: absolute path to the dataset validated
- `mode`: validation strictness mode (`strict` or `non_strict`)
- `row_count`, `column_count`, `time_column_count`
- `missing_cells_total`: number of missing cells across all validated `time_stats.*` columns
- `missing_columns`: list of required columns that were not present (if applicable)
- `zero_heavy_columns`: list of columns that exceed the configured zero-heavy threshold
- `thresholds`:
  - `small_input_threshold`: token count below which zeros are tolerated
  - `zero_heavy_limit`: fraction above which zeros are flagged
- `warnings`: list of human-readable warnings emitted in non-strict mode

### Profiling Provenance Record

Represents the metadata record written alongside profiling outputs (e.g., `profiling_meta.json`).

Key attributes:

- `schema_version`
- `run_type`, `run_id`
- `git_commit`, `git_dirty`, `env`, `params`
- `profiling_commands`: commands used for MLP/attention/CPU overhead profiling
- `profiling_outputs`: paths to staged outputs
- `mlp_validation`: embedded `MLP Validation Result` (new)

