# Research: Vidur MLP profiling missing driver-launched kernels

**Branch**: `005-vidur-mlp-cuda-driver`  
**Date**: 2026-01-16  
**Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/005-vidur-mlp-cuda-driver/spec.md`  
**Primary Issue**: `/data1/huangzhe/code/gpu-simulate-test/context/issues/known/issue-vidur-mlp-profiling-misses-cuda-driver-kernels.md`

## Findings

### 1) Current failure mechanism

- Upstream tracer: `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/vidur/profiling/utils/record_function_tracer.py`
  - Only counts `cat == "cuda_runtime"` events inside each `cat == "user_annotation"` region.
  - Correlates to a matching event via `event["args"]["correlation"]` and sums the correlated event’s `"dur"`.
  - Drops an operation entirely when the summed `cuda_time == 0`, which leads to missing keys in `time_stats`.
- Upstream MLP CSV generation: `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/vidur/profiling/mlp/main.py`
  - Expands `time_stats` dicts into columns via `pandas.json_normalize(...)`, which yields NaNs when keys are missing.
- Our staging currently masks missing measurements:
  - `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/profile_runner.py` fills all `time_stats.*` NaNs with `0.0`, turning “missing measurement” into “0 ms”.

### 2) Trace correlation fields and event categories

The existing tracer already relies on a stable correlation mechanism:

- Launch-ish events can appear as either:
  - `cat == "cuda_runtime"`, or
  - `cat == "cuda_driver"` (e.g., `cuLaunchKernel`)
- GPU execution events appear as:
  - `cat == "kernel"` (and potentially GPU memcpy/memset categories depending on trace)
- Both launch-ish and execution events can contain `event["args"]["correlation"]`, which can be used to connect “launch” to “GPU work”.

Conclusion: the tracer’s attribution gap is categorical (it ignores `cuda_driver`) rather than structural (correlation is unavailable).

### 3) Profiling-method control

Vidur’s MLP profiler defaults to `record_function`:

- `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/vidur/profiling/mlp/main.py` uses `--profile_method` with default `record_function`.

Decision:

- **Decision**: Profiling method selection is explicit via run configuration; wrappers pass `--profile_method` explicitly to the Vidur entrypoint (no hidden defaults in wrapper code).
- **Rationale**: Makes runs reproducible and avoids “accidental record_function” usage.
- **Alternatives considered**:
  - Keep relying on Vidur defaults → rejected (not explicit; easy to regress).

### 4) Improving record-function attribution without committing Vidur submodule changes

We will avoid committing changes inside the Vidur submodule.

- **Decision**: Implement a local `RecordFunctionTracerV2` (in `src/gpu_simulate_test/vidur_ext/`) and switch Vidur’s MLP wrapper to use it when `profile_method=record_function`.
- **Rationale**: Keeps upstream submodule pinned while allowing robust attribution.
- **Alternatives considered**:
  - Patch `extern/tracked/vidur/.../record_function_tracer.py` directly → rejected (requires committing/updating submodule pointer).

Implementation note: Vidur’s MLP profiling uses Ray actors (`ray.remote(MlpWrapper)`); the wrapper imports `RecordFunctionTracer` as a module global in `vidur.profiling.mlp.mlp_wrapper`. The plan will patch the symbol used by `MlpWrapper` (not just the utils module) so the updated tracer is what gets used during profiling.

### 5) Validation and fallback behavior

Decisions (from the clarified spec):

- Strict by default; explicit non-strict allowed.
- Missing values always fail; “zero-heavy” triggers fail in strict and warn in non-strict.
- Validation enforced both when creating the profiling root and when consuming it.
- Fail fast by default; opt-in automatic fallback retries using the alternate profiling method.
- Validate all `time_stats.*` summary stats for all ops that appear in the dataset (columns present).

Provenance:

- Existing profiling meta already captures `git_commit`, `git_dirty`, an environment snapshot, and full resolved run configuration:
  - `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/profiling_bundle.py`
  - `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/profiling.py`
- **Decision**: Embed MLP validation results into existing meta records (no new artifact location required).

