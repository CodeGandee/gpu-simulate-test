# Runbook: 001-compare-vidur-real-timing troubleshooting

This runbook covers common failure modes when running the end-to-end workflow:
`workload-spec` → `real-bench` → `vidur-profile` → `vidur-sim` → `compare-runs`.

## Missing GPU (A100-required stages)

**Symptoms**
- `real-bench` (transformers backend) errors with `torch.cuda.is_available() is False`.
- `vidur-profile` errors with `Vidur profiling requires a CUDA-capable GPU`.

**Checks**
- `nvidia-smi` shows the GPU and driver.
- `pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`

**Fix**
- Ensure the NVIDIA driver/runtime is installed and compatible with the PyTorch CUDA build pinned by this repo (`+cu128`).

## Missing model reference (tokenizer/model files)

**Symptoms**
- `FileNotFoundError` for `models/qwen3-0.6b/source-data` or `.../config.json`.

**Fix**
- Repair the external reference symlink:
  - `bash models/qwen3-0.6b/bootstrap.sh`
  - Or `bash models/bootstrap.sh` to (re)bootstrap all model references.

## Missing profiling data (Vidur simulation prereqs)

**Symptoms**
- `vidur-sim` fails fast with `Missing profiling inputs:` and a list of required CSV paths.

**What’s required (TP=1 / PP=1 baseline)**
- `<profiling_root>/data/profiling/compute/<device>/<model_id>/mlp.csv`
- `<profiling_root>/data/profiling/compute/<device>/<model_id>/attention.csv`

**Fix**
- Generate a profiling bundle:
  - `pixi run vidur-profile vidur.profiling.root=<profiling_root>`
- Or copy profiling artifacts into the expected layout under `<profiling_root>/data/profiling/...`.

## Tokenization mismatch (workload vs backend)

**Symptoms**
- Real and sim distributions look wildly different even for tiny workloads.
- `num_prefill_tokens` in `trace_lengths.csv` does not correspond to what the backend uses.

**Fix**
- Ensure workload generation and real benchmark use the same tokenizer/model reference (`models/qwen3-0.6b/source-data`).
- Prefer consistent tokenizer settings (e.g., avoid adding special tokens when counting prompt tokens).

## Early stopping alignment (real < requested decode)

**Behavior**
- `compare-runs` truncates simulator token rows per request using real `num_decode_tokens_actual`.

**Checks**
- Inspect `tmp/comparisons/<comparison_id>/summary.md` and `tmp/comparisons/<comparison_id>/tables/*.csv`.

