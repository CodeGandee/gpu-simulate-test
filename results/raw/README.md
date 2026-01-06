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
