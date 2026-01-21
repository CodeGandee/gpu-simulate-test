# MLP validation and fallback (Vidur profiling roots)

This repo stages Vidur MLP profiling into a Vidur-compatible `mlp.csv` and then consumes it in simulation (`vidur-sim`, `vidur-cli svr sim`, and paper-fidelity sim). Because some profilers can produce incomplete timing grids, the repo provides explicit controls for:

- **Validation strictness** (`strict` vs `non_strict`)
- **NaN handling** (`nan_policy`)
- **Profiling fallback** (optional retry with another profiler when validation fails)

---

## Missing required columns vs missing cells

Two kinds of “missing data” exist, and they behave differently:

- **Missing required columns**: a required `time_stats.<op>.<stat>` column is absent (e.g., `time_stats.op.mean` does not exist at all).
  - Always a hard error.
  - Rationale: consumers cannot train a model for a target that is not present in the dataframe schema.
- **Missing cells**: required columns exist, but some rows contain `NaN` for a required target.
  - Policy-driven via `nan_policy` (details below).

`nan_policy` only applies to **missing cells** (NaNs). It does not “recreate” missing columns.

---

## Where validation happens

The same core validator (`validate_mlp_csv`) is applied in two places, but with different config namespaces:

- **Profiling / staging** (producing `mlp.csv`):
  - `profiling.mlp.validation.mode`
  - `profiling.mlp.validation.nan_policy`
  - `profiling.mlp.fallback.*` (profiling-only)
- **Consumption** (loading a profiling root for sim/report):
  - `vidur.validation.mlp.mode`
  - `vidur.validation.mlp.nan_policy`

Important: the profiling-stage `nan_policy` controls whether staging fails. The consumption-stage `nan_policy` controls how the simulator trains predictors (drop rows vs fill values).

---

## NaN policy (`nan_policy`)

Supported values:

- `auto`: resolved from `mode` (strict → `reject`, non_strict → `drop`)
- `reject`: fail on any missing cell in core timing targets
- `drop`: allow missing cells; consumers drop missing samples per target before sklearn training
- `zero`: allow missing cells; consumers fill missing targets with `0.0` per target before sklearn training

### Effective policy resolution

`mode` only affects NaN handling when `nan_policy=auto`.

| nan_policy | mode=strict | mode=non_strict |
|-----------:|-------------|-----------------|
| `auto`     | `reject`    | `drop`          |
| `reject`   | `reject`    | `reject`        |
| `drop`     | `drop`      | `drop`          |
| `zero`     | `zero`      | `zero`          |

---

## Profiling fallback (`profiling.mlp.fallback.*`)

Fallback is a **profiling-only** feature that can retry MLP profiling with another method when validation fails.

- `profiling.mlp.fallback.enabled=true`
- `profiling.mlp.fallback.method=cuda_event` (example)

Key implication: if missing cells are *allowed* (effective policy `drop` or `zero`), then missing-cell issues will not fail validation and fallback will not run. Fallback only triggers on validation failures (e.g., missing cells under `reject`, zero-heavy signals in `strict`, missing required columns).

---

## What happens in practice (profiling vs consumption)

### Profiling stage (staging `mlp.csv`)

- Missing required columns: **fail** (always)
- Missing cells (NaNs):
  - effective `reject`: **fail**
  - effective `drop` / `zero`: **warn** and stage the CSV
- Zero-heavy signals:
  - `strict`: **fail**
  - `non_strict`: **warn**
- If validation fails:
  - fallback enabled: rerun MLP profiling using `profiling.mlp.fallback.method`
  - fallback disabled: stop with an error

### Consumption stage (training predictors from `mlp.csv`)

Missing required columns still fail before any sklearn training begins.

For missing cells:

| Effective policy | Consumer action | Provenance in `run_meta.json` |
|------------------|-----------------|-------------------------------|
| `reject` | fail fast | `mlp_validation` |
| `drop` | per-target dropna before `.fit()` | `mlp_validation`, `mlp_nan_drop` |
| `zero` | per-target fillna(0.0) before `.fit()` | `mlp_validation`, `mlp_nan_fill_zero` |

Notes:

- `drop` can still fail if an individual target ends up with **0 usable rows** after dropping NaNs (that model cannot be trained).
- `zero` will not drop rows due to missing targets, but it can bias predictions downward (a filled `0.0` is a synthetic value, not a measurement).

---

## Recommendations

- Highest fidelity: prefer `profiling.mlp.profile_method=cuda_event` (or enable fallback to `cuda_event`) and keep `nan_policy=reject`.
- Best-effort consumption of imperfect roots:
  - use `nan_policy=drop` when you want to avoid introducing synthetic targets
  - use `nan_policy=zero` when you need the pipeline to complete and you accept the bias risk
- Tutorial defaults (`docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli`): `record_function`, `mode=strict`, `nan_policy=zero`, fallback disabled (chosen to keep the tutorial runnable while making missing-data handling explicit).
