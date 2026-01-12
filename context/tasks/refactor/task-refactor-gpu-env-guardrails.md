# Refactor Plan: GPU env guardrails (GSIM_CUDA_VISIBLE_DEVICES)

## What to Refactor

- Add a single, reusable guard that:
  - Loads repo-local `.env` (when present).
  - Requires `GSIM_CUDA_VISIBLE_DEVICES` for GPU-using code paths.
  - Applies `CUDA_VISIBLE_DEVICES=GSIM_CUDA_VISIBLE_DEVICES` before any CUDA work.
- Replace ad-hoc / implicit GPU selection spread across:
  - `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
  - `src/gpu_simulate_test/real_bench/backends/sarathi_backend.py`
  - `src/gpu_simulate_test/real_bench/backends/transformers_backend.py`
  - `src/gpu_simulate_test/vidur_ext/profile_runner.py`
  - `src/gpu_simulate_test/vidur_ext/vidur_profiling_*_main.py`

## Why Refactor

- Safety: avoid accidentally exposing “all GPUs” to PyTorch/Sarathi/Ray (known to crash on some hosts with MIG/odd GPUs).
- Reproducibility: ensure profiling, real runs, and any GPU tooling use the exact same pinned GPU set.
- Debuggability: fail fast with a clear error when GPU pinning is missing, instead of silently selecting a default GPU.

## How to Refactor

1. Introduce `gpu_simulate_test/env_guard.py`:
   - `find_repo_root()` (search upward for `pyproject.toml`).
   - `load_dotenv_if_present(repo_root)` (minimal `.env` parser; does not override existing env vars).
   - `require_gsim_cuda_visible_devices()` (raises if missing/empty).
   - `apply_cuda_visible_devices_from_gsim()` (sets `CUDA_VISIBLE_DEVICES` accordingly).
2. Call the guard from every GPU entrypoint **before** importing/using CUDA libs:
   - Sarathi backends: before importing `sarathi`.
   - Transformers backend: before importing `torch` when `device.startswith("cuda")`.
   - Vidur profiling: before importing `torch` and before spawning profiling subprocesses.
   - Vidur profiling wrapper mains: at the top of `main()` (so direct runs are guarded too).
3. Remove/avoid any “default GPU auto-selection” code in favor of explicit `GSIM_CUDA_VISIBLE_DEVICES`.
4. Add unit tests for:
   - `.env` parsing behavior.
   - “missing `GSIM_CUDA_VISIBLE_DEVICES`” error path.
   - “sets `CUDA_VISIBLE_DEVICES` from `GSIM_CUDA_VISIBLE_DEVICES`” behavior.
5. (Optional but recommended) ignore `.env` in git to prevent accidental commits.

## Impact Analysis

- **Behavior change**: GPU workflows now hard-require `GSIM_CUDA_VISIBLE_DEVICES`.
  - Risk: existing scripts that relied on `CUDA_VISIBLE_DEVICES` (or defaults) will fail.
  - Mitigation: clear error message and `.env` support so local workflows can set it once.
- **Compatibility**: CPU-only commands should continue to work without `GSIM_CUDA_VISIBLE_DEVICES` as long as they don’t touch CUDA.
- **Ray/Sarathi**: pinning must happen before any Ray/Sarathi initialization so workers inherit the correct GPU visibility.

## Expected Outcome

- Any GPU-using workflow either:
  - Runs with a deterministic, explicitly pinned GPU set (`CUDA_VISIBLE_DEVICES` derived from `GSIM_CUDA_VISIBLE_DEVICES`), or
  - Refuses to run with an actionable error.
- Fewer “works on my machine” failures caused by hidden GPU visibility differences.

## Before/After (essential snippets)

### Before: implicit CUDA_VISIBLE_DEVICES selection in Sarathi paper-fidelity runner

```py
# src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py
existing = os.environ.get("CUDA_VISIBLE_DEVICES")
if existing:
    desired_cuda_visible_devices = existing
else:
    desired_cuda_visible_devices = _default_cuda_visible_devices() or "0"
os.environ["CUDA_VISIBLE_DEVICES"] = desired_cuda_visible_devices
```

### After: explicit guardrail via GSIM_CUDA_VISIBLE_DEVICES

```py
from gpu_simulate_test.env_guard import apply_cuda_visible_devices_from_gsim

desired_cuda_visible_devices = apply_cuda_visible_devices_from_gsim()
```

### Before: Vidur profiling imports torch without a repo-wide GPU pinning convention

```py
# src/gpu_simulate_test/vidur_ext/profile_runner.py
import torch
if not torch.cuda.is_available():
    raise RuntimeError(...)
```

### After: enforce pinning before any CUDA probing

```py
from gpu_simulate_test.env_guard import apply_cuda_visible_devices_from_gsim

apply_cuda_visible_devices_from_gsim()
import torch
```

## References

- `magic-context/instructions/planning/make-refactor-plan.md`
- `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real.md`
- `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
- `src/gpu_simulate_test/real_bench/backends/sarathi_backend.py`
- `src/gpu_simulate_test/real_bench/backends/transformers_backend.py`
- `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- `src/gpu_simulate_test/vidur_ext/vidur_profiling_cpu_overhead_main.py`
