# Contract: Ray settings artifacts

**Feature**: `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`  
**Date**: 2026-01-20  

This document defines the on-disk artifact contract for Ray settings observability.

## `ray_settings.json` (v1)

**Purpose**: Persist the effective Ray runtime settings (value + provenance) used for a stage.

**Proposed locations**:
- `<run_dir>/profile/ray_settings.json`
- `<run_dir>/real/ray_settings.json` (only when the selected backend uses Ray)

**Schema**: `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/contracts/ray_settings.schema.json`

