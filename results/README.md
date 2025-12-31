# Results

This directory is intended to hold experiment results and reports.

## Layout

- `results/raw/`
  - Raw materials produced while running experiments (logs, dumps, large intermediate files, etc).
  - Subdirs should be organized as: `raw/<YYYY-MM-DD>/<timestamp>-<what>/` (e.g., `raw/2025-12-31/20251231-073000-vidur-qwen3-sim/`).
  - By default, `results/raw/<subdir>/` contents are ignored by git via `results/raw/.gitignore`.
  - If you want to keep specific raw files in git, explicitly de-ignore them (add `!path` rules in
    `results/raw/.gitignore`, or add a narrower `.gitignore` inside the subdir).
- `results/reports/`
  - Git-tracked experiment reports: summary markdowns, small JSONs, experiment configs (YAML), plots (small), etc.
  - Subdirs should be organized as: `reports/<YYYY-MM-DD>/<timestamp>-<what>/` (e.g., `reports/2025-12-31/20251231-073000-vidur-qwen3-sim/`).
  - These are used to describe the setup, what was run, the observed results, and any conclusions.
  - Add ignore rules under `results/reports/` only when needed (on-demand, based on what you generate).
