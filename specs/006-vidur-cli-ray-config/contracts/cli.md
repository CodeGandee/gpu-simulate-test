# Contract: `vidur-cli` Ray runtime configuration

**Feature**: `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`  
**Date**: 2026-01-20  

This document defines the externally visible CLI + config contract additions for Ray runtime configuration.

## Supported Ray settings (initial scope)

`vidur-cli` supports only the following Ray settings via config:

- `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES`
- `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION`
- `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE`

If the configuration includes any other key under `ray.env`, `vidur-cli` must fail fast and list the supported keys.

## Precedence (per setting)

For each supported key, precedence is:

1. Environment (`RAY_*`)
2. Configuration (`cfg.ray.env`)
3. Ray default (no injection)

`vidur-cli` must never override a user-set `RAY_*` env var.

## Stages affected

At minimum:
- `vidur-cli svr profile`: applies Ray settings before profiling starts.
- `vidur-cli svr real`: applies Ray settings before starting a Ray-using backend (e.g., Sarathi); for non-Ray backends, it may be a no-op.

## No-Ray compute profiling option

Config key (in the `vidur_profile` workflow config):

- `profiling.compute.use_ray` (bool, default `true`)

When `false`, compute profiling must not start Ray and must still produce downstream-compatible profiling outputs; unsupported no-Ray configurations must fail fast with an actionable error.

## Output observability

For every Ray-using stage:

- `vidur-cli` must emit an effective settings report that includes each supported key, its effective value (or “default/unset”), and the source (`environment` | `configuration` | `default`).
- The report should not break the stdout “primary output path” convention; printing the report to stderr and/or writing a JSON artifact is acceptable.

