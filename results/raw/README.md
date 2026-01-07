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
  - Each run directory is a Vidur-compatible profiling root containing:
    - Curated outputs (useful): `data/profiling/compute/<device>/<org>/<model>/{mlp.csv,attention.csv}`
    - Provenance: `profiling_meta.json` (commands, params, environment snapshot)
    - Debug-only intermediates: `cache/` (raw profiler outputs; safe to delete if you only need the curated CSVs)
  - Example (complete run): `results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_10-43-39-975600338/`
