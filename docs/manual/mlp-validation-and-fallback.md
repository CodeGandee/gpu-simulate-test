# MLP validation and fallback: implications and recommended combinations

This repo’s Vidur workflows have two *separate* places where MLP profiling quality is enforced:

1. **Profiling (staging)**: `svr profile` (or `vidur-profile`, `paper-fidelity profile`) writes `mlp.csv` and validates it.
2. **Consumption (simulation/reporting)**: `svr sim` (or `vidur-sim`, `paper-fidelity repro`) validates the profiling root again and may patch Vidur’s sklearn trainer.

The knobs below control what happens when `mlp.csv` contains missing or suspicious values.

## Terms

### Missing required columns vs missing cells (NaNs)

- **Missing required columns**: a required header (for example `time_stats.mlp_up_proj.median`) is absent from the CSV.
  - This is always a hard error (there is no way to “drop NaNs” if the column does not exist).
- **Missing cells (NaNs)**: the required column exists, but some rows have missing values.
  - This can be either fatal (`reject`) or allowed (`drop`), depending on policy.

The validator reports:

- `missing_columns`: list of required columns that are missing.
- `missing_cells_total`: total number of missing cells across all core timing target columns.

## Profiling-stage knobs (staging `mlp.csv`)

These knobs apply to `svr profile` (and other profiling entrypoints that stage `mlp.csv`).

### Validation mode

- `profiling.mlp.validation.mode=strict|non_strict`
  - Affects **zero-heavy** checks.
  - Also affects NaN behavior when `nan_policy=auto`.

### NaN policy (profiling)

- `profiling.mlp.validation.nan_policy=auto|reject|drop|zero`

Effective policy resolution:

- `auto + strict` ⇒ **reject**
- `auto + non_strict` ⇒ **drop**
- explicit `reject|drop` overrides the mode for NaN handling
- `zero` ⇒ **allow NaNs** during validation; consumers fill missing targets with 0 during training

### Fallback rerun (profiling)

- `profiling.mlp.fallback.enabled=true|false`
- `profiling.mlp.fallback.method=<profile_method>` (commonly `cuda_event`)

Important semantics:

- Fallback is only attempted **after a validation failure**.
- Fallback is a **full MLP profiling rerun** using the fallback method.
  - It does **not** selectively “re-profile only missing ops/kernels”.
  - The staged `mlp.csv` is replaced by the fallback method’s output.

## Consumption-stage knobs (sim/report)

These knobs apply to `svr sim` (and other consumers of a profiling root).

- `vidur.validation.mlp.mode=strict|non_strict`
- `vidur.validation.mlp.nan_policy=auto|reject|drop|zero`

If the effective policy is `drop`, consumers apply a local patch so Vidur’s sklearn training drops NaNs **per target**
column before fitting. If the filtered dataset becomes empty for any required target, the run still fails with a
remediation hint.

## Combination matrix (what happens)

This table focuses on *NaN/missing-cell* behavior. Missing required columns always fail.

### Profiling (staging) outcomes

| Profiling knobs | If `mlp.csv` has NaNs | Outcome |
|---|---|---|
| `mode=strict` + `nan_policy=auto` + `fallback.enabled=false` | NaNs present | **FAIL** in `svr profile` (effective policy is `reject`) |
| `mode=strict` + `nan_policy=auto` + `fallback.enabled=true` | NaNs present | **RERUN** full MLP profiling with fallback method; if fallback output validates, **PASS** |
| `mode=non_strict` + `nan_policy=auto` + `fallback.enabled=false` | NaNs present | **PASS** in `svr profile` with warnings (effective policy is `drop`) |
| `mode=non_strict` + `nan_policy=auto` + `fallback.enabled=true` | NaNs present | Same as above: **PASS** with warnings; fallback is **not** triggered because validation does not fail |
| any mode + `nan_policy=reject` + `fallback.enabled=false` | NaNs present | **FAIL** in `svr profile` |
| any mode + `nan_policy=drop` + `fallback.enabled=false` | NaNs present | **PASS** in `svr profile` with warnings |
| any mode + `nan_policy=zero` + `fallback.enabled=false` | NaNs present | **PASS** in `svr profile` with warnings |

### Consumption (sim) outcomes

| Sim knobs | If staged `mlp.csv` has NaNs | Outcome |
|---|---|---|
| `mode=strict` + `nan_policy=auto` | NaNs present | **FAIL** in `svr sim` (effective policy is `reject`) |
| `mode=non_strict` + `nan_policy=auto` | NaNs present | **PASS** (effective policy is `drop`; per-target dropna patch enabled) |
| any mode + `nan_policy=drop` | NaNs present | **PASS** (per-target dropna patch enabled) |
| any mode + `nan_policy=zero` | NaNs present | **PASS** (per-target fillna(0) patch enabled) |
| any mode + `nan_policy=reject` | NaNs present | **FAIL** |

## Why these behaviors exist (and why they matter)

1. **`record_function` can produce missing per-op samples** on some hosts/workloads because it attributes GPU time by
   parsing profiler traces and correlating launches to kernels. When attribution misses some ops for some token sizes,
   those ops become NaN in `mlp.csv`.
2. **`cuda_event` is usually more robust** (direct CUDA timing), but it changes what is being measured (and can change
   sim-vs-real agreement).
3. **Fallback is intentionally coarse-grained** (rerun MLP profiling entirely) because Vidur’s MLP profiler emits a
   single `mlp.csv` per run, not a per-op cache that can be partially repaired.
4. **Profiling-stage “drop” is not sufficient by itself**: even if profiling allows NaNs, consumers must also allow
   NaNs (or explicitly drop them) or simulation will still fail.

## Recommended defaults (practical)

- High-fidelity runs: prefer producing a complete `mlp.csv` without NaNs.
  - Use `profiling.mlp.profile_method=cuda_event`, or fix the root cause if `record_function` is desired.
- “Keep going” runs:
  - Profiling: `profiling.mlp.validation.mode=non_strict profiling.mlp.validation.nan_policy=auto`
  - Sim: `vidur.validation.mlp.mode=non_strict vidur.validation.mlp.nan_policy=auto`
  - Understand that dropping NaNs reduces training data per target and can worsen the sim-vs-real gap.
- Avoid `nan_policy=zero` unless you explicitly want “fill missing timings with 0” semantics for debugging; it is
  generally less faithful than dropping NaNs per target.
