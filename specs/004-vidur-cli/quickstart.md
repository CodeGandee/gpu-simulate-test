# Quickstart: `vidur-cli` (Phase 1)

**Repo root (example)**: `/data1/huangzhe/code/gpu-simulate-test`  
**Feature**: `specs/004-vidur-cli/spec.md`  

This quickstart shows the intended end-user workflow once `vidur-cli` is implemented.

## 0) Setup

From the repo root:

```bash
pixi install
```

## 1) Optional: project-local config TOML

Create a project-local config file in your current working directory:

```bash
mkdir -p .vidur-config
cat > .vidur-config/default.toml << 'EOF'
[resources]
repo_root = "/data1/huangzhe/code/gpu-simulate-test"
workspace_dir = "default"
# models_root = "/abs/path/to/models"      # optional
# datasets_root = "/abs/path/to/datasets"  # optional
EOF
```

## 2) Initialize a run

```bash
# Option A (intended final UX):
pixi run vidur-cli svr init-run model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default

# Option B (dev invocation, equivalent):
pixi run python -m gpu_simulate_test.cli.vidur_cli svr init-run model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default
```

The command prints a `<run_dir>` path. Use it in subsequent steps.

## 3) Run stages (step-by-step)

```bash
pixi run vidur-cli svr trace   --run-dir <run_dir>
pixi run vidur-cli svr profile --run-dir <run_dir>
pixi run vidur-cli svr sim     --run-dir <run_dir>
pixi run vidur-cli svr real    --run-dir <run_dir>
pixi run vidur-cli svr report  --run-dir <run_dir>
```

## 4) Inspect outputs

Key artifacts live under `<run_dir>`:

- `run_state.json` (stage state + artifact pointers)
- `resources.json` (resolved resources + provenance)
- `trace/trace.csv` and `trace/trace_meta.json`
- `report/summary.md` (final comparison summary)

## 5) Common diagnostics

- Print resolved resources/config search path: `vidur-cli --print-resolved ...`
- List available presets: `vidur-cli configs list --group model`
- Inspect resolved resources: `vidur-cli resources show`
