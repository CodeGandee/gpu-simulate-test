# Troubleshooting

## Common issues

### `torch.cuda.is_available()` is `False`

- Check driver and runtime: `nvidia-smi`
- Confirm Pixi env uses the CUDA build: `pixi run python -c "import torch; print(torch.__version__)"` (should include `+cu128` in this repo)

### Model reference missing (`models/*/source-data/...` not found)

- Recreate the symlink: `bash models/<model>/bootstrap.sh` (or `bash models/bootstrap.sh`)
- If your local model storage root differs, set `GSIM_MODELS_ROOT` before running the script.

### `backend=sarathi` fails to load Qwen3

- Ensure the Sarathi submodule is initialized: `git submodule update --init --recursive`
- Ensure you are running inside Pixi: `pixi run real-bench ...`
- Qwen3 requires Sarathi support for the `Qwen3ForCausalLM` architecture (provided by the `extern/tracked/sarathi-serve` submodule on branch `hz-dev`).

### `paper-fidelity repro` fails early with “CUDA is required”

- `paper-fidelity repro` uses Sarathi-Serve for the real run; it requires `torch.cuda.is_available() == True`.
- If you only want to validate the trace path on CPU, run:
  - `pixi run paper-fidelity trace --scenario llama2_7b_arxiv --workload static`

### `paper-fidelity repro` fails with “sarathi is required”

- Initialize submodules: `git submodule update --init --recursive`
- Ensure Sarathi is available in the Pixi env (this repo expects the editable `extern/tracked/sarathi-serve` integration).

### `paper-fidelity profile --include-cpu-overhead` fails or is unexpectedly slow

- Ensure the model weights are available locally (this repo typically uses `models/*/source-data` via `bash models/<model>/bootstrap.sh`).
- Run long profiling jobs in `tmux` and log progress: `pixi run paper-fidelity profile ... 2>&1 | tee run.log` (Hydra `chdir` puts `run.log` next to the other run outputs).
- If you are doing sim-vs-real parity work, explicitly set `scenario.vidur.skip_cpu_overhead_modeling=true|false` and keep profiling, simulation, and real runs consistent.

### Trace subset errors (`trace_subset.*`)

Common causes:

- Out-of-bounds: `trace_subset.begin/end` exceed the trace length.
- Empty range: `begin >= end`.
- `trace_subset.kind=indices` used with a timed trace source (`trace_source.kind=trace_csv` or `legacy_workload_dir`).
  - Fix: use `trace_subset.kind=range` for timed sources.

### Ray “metrics exporter agent” errors

Ray may log errors about failing to connect to a metrics exporter agent. For this workflow it is usually benign as long as the run produces the output CSVs.

## More runbook notes

- `context/runbooks/001-compare-vidur-real-timing-troubleshooting.md`
- Paper fidelity quickstart: `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
