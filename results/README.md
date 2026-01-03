# Results

This directory collects experiment artifacts and human-readable reports.

## Layout

- `results/raw/`
  - Raw materials (run logs, dumps, intermediate artifacts, etc).
  - These can be large, so **everything under `results/raw/` is ignored by default** via
    `results/raw/.gitignore`.
  - If you intentionally want to keep some raw materials in git, you must explicitly *de-ignore*
    them in `results/raw/.gitignore`.
- `results/reports/`
  - Git-tracked experiment reports: summary markdowns, small JSONs, configs (YAML), small tables,
    and other lightweight artifacts that describe how the experiment was run and what it showed.
  - Add ignore rules only on demand (if a report grows too large or contains machine-local data).

## Naming convention

Subdirectories are organized as:

- `results/raw/<YYYY-MM-DD>/<timestamp>-<what>/`
- `results/reports/<YYYY-MM-DD>/<timestamp>-<what>/`

Example:

- `results/reports/2026-01-02/20260102-120000Z-qwen3-0.6b-vidur-vs-sarathi/`

