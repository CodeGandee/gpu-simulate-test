# Quickstart: `vidur-cli` (Phase 1)

**Repo root (example)**: `/data1/huangzhe/code/gpu-simulate-test`  
**Feature**: `specs/004-vidur-cli/spec.md`  

This quickstart shows the intended end-user workflow once `vidur-cli` is implemented.

## 0) Setup

From the repo root:

```bash
pixi install
```

## 1) Resource resolution (env + project-local TOML)

`vidur-cli` resolves filesystem resources with this precedence:

- env vars
- project-local config TOML
- repo fallback / `<pwd>` fallback

### Environment variables

- `GSIM_VIDUR_CLI_USER_CONFIG`: config TOML path (relative paths resolve relative to `<pwd>`)
- `GSIM_REPO_ROOT`: repo root (must contain `pyproject.toml` + `configs/compare_vidur_real/`)
- `GSIM_MODELS_ROOT`: models root (optional; defaults to `<repo_root>/models`)
- `GSIM_DATASETS_ROOT`: datasets root (optional; defaults to `<repo_root>/datasets`)
- `GSIM_VIDUR_WORKSPACE_DIR`: workspace dir selector (relative ⇒ `<pwd>/.vidur-output/<value>`, absolute ⇒ used as-is)
- `GSIM_VIDUR_CLI_HYDRA_CONFIG_DIRS`: extra Hydra config roots (split by `:`; higher precedence first)

### Optional: project-local config TOML

Create a project-local config file in your current working directory:

```bash
mkdir -p .vidur-config
cat > .vidur-config/default.toml << 'EOF'
[resources]
repo_root = "/data1/huangzhe/code/gpu-simulate-test"
workspace_dir = "default"
# models_root = "/abs/path/to/models"      # optional
# datasets_root = "/abs/path/to/datasets"  # optional

[hydra]
# config_dirs = ["/abs/path/to/my/configs"] # optional
EOF
```

## 2) Initialize a run

```bash
# Option A (repo-root UX):
pixi run vidur-cli svr init-run model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default

# Option B (run-from-anywhere via manifest path):
GSIM_REPO_ROOT=/data1/huangzhe/code/gpu-simulate-test \
pixi run -m /data1/huangzhe/code/gpu-simulate-test vidur-cli svr init-run \
  model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default

# Option C (dev invocation, equivalent):
pixi run python -m gpu_simulate_test.cli.vidur_cli svr init-run model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default
```

The command prints a `<run_dir>` path. Use it in subsequent steps.

## 3) Run stages (step-by-step)

```bash
# Trace generation options:
#
# - from a lengths-only CSV (minimal example):
cat > lengths.csv <<'EOF'
num_prefill_tokens,num_decode_tokens
8,16
12,16
EOF
pixi run vidur-cli svr trace --run-dir <run_dir> --from-lengths ./lengths.csv
#
# - import an existing canonical trace:
# pixi run vidur-cli svr trace --run-dir <run_dir> --import-trace /abs/path/to/trace.csv
#
# - default generation from workload config (prompts + tokenizer + schedule):
# pixi run vidur-cli svr trace --run-dir <run_dir>

pixi run vidur-cli svr profile --run-dir <run_dir>
# (optional) disable CPU overhead measurement:
# pixi run vidur-cli svr profile --run-dir <run_dir> --no-include-cpu-overhead
pixi run vidur-cli svr sim     --run-dir <run_dir>
pixi run vidur-cli svr real    --run-dir <run_dir>
pixi run vidur-cli svr report  --run-dir <run_dir>
```

## 4) Inspect outputs

Key artifacts live under `<run_dir>`:

- `run_state.json` (stage state + artifact pointers)
- `resources.json` (resolved resources + provenance)
- `trace/trace.csv` and `trace/trace_meta.json`
- `report/summary.md` (final comparison summary; includes arrival kind + CPU overhead status)

## 5) Common diagnostics

- Print resolved resources/config search path: `vidur-cli --print-resolved ...`
- List available presets: `vidur-cli configs list --group model`
- Inspect resolved resources: `vidur-cli resources show`

Examples (from any directory `<pwd>`):

```bash
GSIM_REPO_ROOT=/data1/huangzhe/code/gpu-simulate-test \
pixi run -m /data1/huangzhe/code/gpu-simulate-test vidur-cli resources show

GSIM_REPO_ROOT=/data1/huangzhe/code/gpu-simulate-test \
pixi run -m /data1/huangzhe/code/gpu-simulate-test vidur-cli configs list --group model
```

If a higher-precedence config root shadows a lower-precedence preset key, `configs list` prints a `WARNING:` line to stderr with both paths.
