# Contract: run directory layout and artifacts

**Feature**: `specs/004-vidur-cli/spec.md`  
**Date**: 2026-01-14  

This document is a compact contract for what files exist in a `vidur-cli` run directory and where to find their schemas.

## Required files

Within `<run_dir>`:

- `run_state.json`
  - Schema: `specs/004-vidur-cli/contracts/run_state.schema.json`
- `resources.json`
  - Schema: `specs/004-vidur-cli/contracts/resources.schema.json`

## Optional provenance files

- `resolved_config.yaml`
  - Written by `svr init-run` as a best-effort snapshot of presets/overrides/resources used for the run.

## Trace files (after `svr trace`)

- `trace/trace.csv`
  - Required columns: `request_id`, `arrival_time_ns`, `num_prefill_tokens`, `num_decode_tokens`
- `trace/trace_meta.json`
  - Schema: `specs/004-vidur-cli/contracts/trace_meta.schema.json`
- `trace/trace_lengths.csv` *(compatibility)*
  - Used by legacy Vidur runners; includes `request_id`, `prompt_id`, `num_prefill_tokens`, `num_decode_tokens`
- `trace/trace_intervals.csv` *(compatibility)*
  - Used by legacy Vidur runners; includes `request_id`, `inter_arrival_ns`, `arrival_time_ns`

## Failure file (on stage failure)

- `failure.json`
  - Schema: `specs/004-vidur-cli/contracts/failure.schema.json`

## Report files (after `svr report`)

- `report/summary.md`
- `report/figs/*` (optional)
- `report/tables/*` (optional)
