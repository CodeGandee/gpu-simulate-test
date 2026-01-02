# Troubleshooting

## Common issues

### `torch.cuda.is_available()` is `False`

- Check driver and runtime: `nvidia-smi`
- Confirm Pixi env uses the CUDA build: `pixi run python -c "import torch; print(torch.__version__)"` (should include `+cu128` in this repo)

### Model reference missing (`models/qwen3-0.6b/source-data/...` not found)

- Recreate the symlink: `bash models/qwen3-0.6b/bootstrap.sh`
- If your local model storage root differs, set `EXTERNAL_REF_ROOT` before running the script.

### `backend=sarathi` fails to load Qwen3

- Ensure the Sarathi submodule is initialized: `git submodule update --init --recursive`
- Ensure you are running inside Pixi: `pixi run real-bench ...`
- Qwen3 requires Sarathi support for the `Qwen3ForCausalLM` architecture (provided by the `extern/tracked/sarathi-serve` submodule on branch `hz-dev`).

### Ray “metrics exporter agent” errors

Ray may log errors about failing to connect to a metrics exporter agent. For this workflow it is usually benign as long as the run produces the output CSVs.

## More runbook notes

- `context/runbooks/001-compare-vidur-real-timing-troubleshooting.md`

