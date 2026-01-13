# Requirements: Vidur sim-vs-real (paper-fidelity)

## Header
- **Purpose**: Define requirements for running *comparable* Vidur simulation vs real inference experiments so the gap is interpretable.
- **Status**: Working draft
- **Date**: 2026-01-13
- **Primary reference (authoritative)**: `configs/paper_fidelity/` + `src/gpu_simulate_test/cli/paper_fidelity.py`
- **Related implementations**:
  - Orchestration: `scripts/paper_fidelity_sweep.sh`
  - Vidur runner: `src/gpu_simulate_test/vidur_ext/sim_runner.py`
  - Real runner (Sarathi): `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
  - Report writer: `src/gpu_simulate_test/paper_fidelity/report.py`

---

## 1) Scope and definitions

### What “comparable” means

Sim-vs-real results are only interpretable as “simulator fidelity” when **the two runs represent the same workload and the same timing boundary**:

1. **Same request shapes**: `num_prefill_tokens` and `num_decode_tokens` per request match.
2. **Same arrivals**: the request arrival schedule is identical.
3. **Same topology/parallelism**: TP/PP and topology assumptions match (or are explicitly recorded as different).
4. **Same timing boundary**: compared metrics measure the same phases.
5. **Same decode semantics**: early termination (EOS) does not silently change decode lengths.

### Static vs dynamic workloads

Paper-fidelity operates over a canonical trace file with columns:

`arrived_at,num_prefill_tokens,num_decode_tokens,request_id`

- **Static**: all requests arrive at time 0 (`arrived_at=0`). No capacity search; no target QPS.
- **Dynamic**: requests arrive over time (non-zero `arrived_at`) via a Poisson process. The workflow performs capacity discovery on the real backend and runs at an operating QPS (recorded in `inputs/capacity.json`).

The final report (`summary.md`) MUST clearly state whether the run is static or dynamic and explain the difference.

---

## 2) Hard requirements (must-haves)

### R1. One canonical trace (single source of truth)

The experiment MUST derive both sim and real inputs from one trace (request shapes + arrivals):

- `tmp/paper_fidelity/traces/<scenario.name>/trace.csv`

Never generate two independent Poisson schedules for sim vs real.

### R2. Determinism and provenance

Every run MUST record enough metadata to reproduce the result:

- Run inside Pixi (`pixi run ...`)
- Record `git commit` + `git dirty` + environment snapshot
- Persist metadata (`run_meta.json`, `trace_meta.json`, and `capacity.json` for dynamic runs)

### R3. Decode-length semantics are enforced

Real replay MUST prevent early termination and validate decoded token counts:

- Paper-fidelity uses `ignore_eos: true` and validates actual decode lengths.
- If decoded token counts do not match the trace, treat the run as invalid for fidelity conclusions.

### R4. TP/PP consistency across profiling + sim + real

TP/PP (and any topology assumptions) MUST be set explicitly and kept consistent:

- Real: `scenario.real.parallel.tensor_parallel_size`, `scenario.real.parallel.pipeline_parallel_size`
- Sim: `scenario.vidur.tensor_parallel_size`, `scenario.vidur.num_pipeline_stages`, `scenario.vidur.network_device`, `scenario.vidur.device`
- Profiling root selection MUST match these settings (profiling bundles are not universally interchangeable).

### R5. Parity-critical scheduler knobs are explicit (do not trust defaults)

For sim-vs-real fidelity work, do not rely on defaults in either system. Explicitly set and record:

- Scheduler type and config (`scenario.vidur.scheduler.*`, `scenario.real.scheduler.*`)
- Chunk size / iteration limits (if applicable)
- Batch caps / concurrency limits
- `max_tokens` and trace generation constraints

Rationale: different defaults can yield “plausible” metrics while producing large prefill/decode split errors.

### R6. CPU overhead parity (default: counted)

For sim-vs-real runs, CPU overhead is counted by default:

- Simulator: `scenario.vidur.skip_cpu_overhead_modeling=false`
- Profiling: `profiling.include_cpu_overhead=true` so the profiling root contains CPU overhead measurements

If CPU overhead modeling is disabled (`scenario.vidur.skip_cpu_overhead_modeling=true`), the final report (`summary.md`) MUST include a clear warning that this is non-default and that sim-vs-real gaps must be interpreted accordingly.

### R7. Required artifacts (report directory contract)

Each run MUST write:

- `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<run_tag>/run_meta.json`
- `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<run_tag>/scores.json`
- `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<run_tag>/summary.md`
- `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<run_tag>/figs/*.svg`

The report MUST include workload mode + CPU overhead modeling status (+ warnings).

---

## 3) How to run (current workflow)

### Recommended: sweep script

Use `scripts/paper_fidelity_sweep.sh` for repeatable runs. It supports global TP/PP for all cases and writes an append-only log (`cases.jsonl`).

Example (single scenario, TP=4, run both static and dynamic at small scale):

```bash
bash scripts/paper_fidelity_sweep.sh \
  --scenarios <scenario_key> \
  --workloads static,dynamic \
  --scale small \
  --tp 4 \
  --pp 1 \
  --run-id my_run_001
```

Notes:
- CPU overhead microbenchmarks are included in profiling by default; pass `--no-include-cpu-overhead` only for debugging.
- Report directories are written under `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/...`.

### Manual: single case (debugging)

```bash
# 1) Create a host-matched profiling root (prints the path on the last line).
profiling_root="$(pixi run paper-fidelity profile --scenario <scenario_key> | tail -n 1)"

# 2) Run static or dynamic repro using that profiling root.
pixi run paper-fidelity repro --scenario <scenario_key> --workload static --scale small \
  "scenario.vidur.profiling_root=${profiling_root}" \
  "scenario.vidur.tensor_parallel_size=<tp>" \
  "scenario.real.parallel.tensor_parallel_size=<tp>"
```

---

## 4) Post-run checklist (acceptance criteria)

### A) Workload parity

- Same number of requests consumed by sim and real.
- Decode token counts match the trace (paper-fidelity enforces this).
- Arrivals monotonic and non-negative.

### B) Boundary parity

- Metrics compared are the paper-fidelity normalized request-level metrics from both sim and real.
- Scheduler/parallelism knobs are recorded and aligned.

### C) CPU overhead visibility (report UX)

- `summary.md` states whether the run is `static` or `dynamic` and explains the difference.
- `summary.md` states whether CPU overhead modeling is enabled or disabled.
- If disabled, `summary.md` includes a warning that this is non-default for sim-vs-real parity.

---

## 5) Implementation notes (dynamic runs)

- Dynamic arrivals are generated from exponential inter-arrivals with mean `1/qps`, using `workload.seed` for determinism (`src/gpu_simulate_test/paper_fidelity/traces.py:add_poisson_arrivals()`).
- Dynamic runs perform capacity discovery on the real backend, then run at `qps_85 = 0.85 * capacity_qps` (recorded in `inputs/capacity.json`).
