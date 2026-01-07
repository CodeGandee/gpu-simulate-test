# Raw results (ignored by default)

All subdirectories under `results/raw/` are ignored by default (see `results/raw/.gitignore`).

If you need to intentionally keep a specific raw directory in git, add a negation rule for it in
`results/raw/.gitignore`.

## Contents

- `results/raw/vendor-results/`: Vidur **vendor (submodule) simulation outputs** produced by running `python -m vidur.main`
  as described in `extern/tracked/vidur/README.md`, using Vidur’s shipped profiling + `data/processed_traces/*.csv`.
  - `results/raw/vendor-results/sarathi-serve/dynamic/`: Poisson arrivals + Sarathi scheduler (vendor-style dynamic runs).
  - `results/raw/vendor-results/vllm/static/`: Static arrivals + vLLM scheduler (vendor-style static runs).
  - Layout (both trees): `.../<gpu>-<model>-<trace>/<timestamp>/...`
  - Each `<timestamp>/` contains Vidur outputs like `request_metrics.csv`, `config.json`, `chrome_trace.json`, and `plots/*.csv`.
  - Each subtree has its own `manifest.json` and cache directory (`cache/` or `vidur-cache/`) for predictor reuse.

- `results/raw/vidur-profiling/`: Host-generated **Vidur profiling bundles** (compute microbenchmarks) produced on this
  machine.
  - Produced by `pixi run vidur-profiling` (`scripts/run_vidur_profiling_llama2_7b.sh`).
  - Layout: `results/raw/vidur-profiling/<model-slug>/<scheduler-name>/<run_id>/`
  - Convenience: `results/raw/vidur-profiling/<model-slug>/<scheduler-name>/latest` is a local symlink that can point to
    the most recent run on this host (note: it is ignored by git via `results/raw/.gitignore`).
  - Each run directory is a Vidur-compatible profiling root containing:
    - Curated outputs (useful): `data/profiling/compute/<device>/<org>/<model>/{mlp.csv,attention.csv}`
    - Provenance: `profiling_meta.json` (commands, params, environment snapshot)
    - Debug-only intermediates: `cache/` (raw profiler outputs; safe to delete if you only need the curated CSVs)
  - Caveat: `attention.csv` row counts are not stable across runs and are not directly comparable to Vidur’s vendor CSVs.
    The attention profiler’s grid size is mainly controlled by `--num_tensor_parallel_workers` (often multiple values in
    vendor bundles) and also depends on `--max_seq_len`, `--min_batch_size/--max_batch_size`, and
    `--profile_only_decode/--profile_only_prefill` (plus Vidur/Sarathi version/backends). Use `profiling_meta.json` to
    compare the actual knobs used.
  - Example (latest run): `results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest/`
  - Example (timestamped run): `results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_15-15-15-281697026/`
