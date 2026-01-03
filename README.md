# gpu-simulate-test

Testbed for evaluating GPU simulators with an emphasis on LLM/inference workloads.

## Goals

- Provide a repeatable way to run and compare simulators on the same workloads/configs.
- Track setup notes, pitfalls, and reproducible scripts for each simulator.
- Keep results and configs versioned.

## Quickstart

Initialize submodules:

```bash
git submodule update --init --recursive
```

Create the Pixi environment (Python + PyPI deps):

```bash
pixi install
```

Run commands inside the environment:

```bash
pixi run python -c "import torch; print(torch.__version__)"
```

Optional: configure `CODEX_HOME` and an HTTP proxy for tooling:

```bash
./setup-envs.sh --proxy auto
```

## External Assets (models/datasets)

Large machine-local assets are managed under `models/` and `datasets/` using an “external
reference” pattern (docs + bootstrap scripts are committed; the actual data is not).

- Bootstrap everything:
  - `bash models/bootstrap.sh`
  - `bash datasets/bootstrap.sh`
- Per-reference bootstraps:
  - Qwen3 model: `bash models/qwen3-0.6b/bootstrap.sh`
  - COCO 2017: `bash datasets/coco2017/bootstrap.sh`

Both bootstrap scripts support `EXTERNAL_REF_ROOT` to point at your local storage root.

### Vidur paper LLMs (reference + stats)

The Vidur (MLSys'24) paper submodule (`extern/tracked/vidur`) evaluates across these LLMs, and you’ll see these names/IDs show up in configs/notes in this repo:

| Paper name | Vidur/HF model id | Params | Layers | Embedding | Attn heads | Attention type | ModelScope |
|---|---|---:|---:|---:|---:|---|---|
| LLaMA2-7B | `meta-llama/Llama-2-7b-hf` | 7B | 32 | 4096 | 32 | Multi-Head Attention | https://modelscope.cn/models/meta-llama/Llama-2-7b-hf |
| LLaMA2-70B | `meta-llama/Llama-2-70b-hf` | 70B | 80 | 8192 | 64 | Group-Head Attention | https://modelscope.cn/models/meta-llama/Llama-2-70b-hf |
| InternLM-20B | `internlm/internlm-20b` | 20B | 60 | 5120 | 40 | Multi-Head Attention | https://modelscope.cn/models/internlm/internlm-20b |
| Qwen-72B | `Qwen/Qwen-72B` | 72B | 80 | 8192 | 64 | Multi-Head Attention | https://modelscope.cn/models/Qwen/Qwen-72B |

Stats source: Vidur’s config-explorer demo page (`extern/tracked/vidur/vidur/config_optimizer/analyzer/dashboard/intro_page.py`).

### Downloading with ModelScope

ModelScope SDK supports downloading a full model repo snapshot via `snapshot_download` (official docs: https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E4%B8%8B%E8%BD%BD).

1) Install the SDK:

```bash
pixi run python -m pip install -U modelscope
```

2) (Optional) set an access token (needed for gated/private models):

```bash
export MODELSCOPE_API_TOKEN="..."
```

Token page: https://modelscope.cn/my/myaccesstoken

3) Download a model into your external storage (pick a `local_dir` under your `EXTERNAL_REF_ROOT` / scratch disk):

```bash
pixi run python -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('Qwen/Qwen-72B', local_dir='PATH/TO/Qwen-72B'))"
```

Examples (one per model):

```bash
pixi run python -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('meta-llama/Llama-2-7b-hf', local_dir='PATH/TO/Llama-2-7b-hf'))"
pixi run python -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('meta-llama/Llama-2-70b-hf', local_dir='PATH/TO/Llama-2-70b-hf'))"
pixi run python -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('internlm/internlm-20b', local_dir='PATH/TO/internlm-20b'))"
pixi run python -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('Qwen/Qwen-72B', local_dir='PATH/TO/Qwen-72B'))"
```

Notes:
- `snapshot_download(..., local_dir=...)` downloads the repo into that directory; `cache_dir=...` uses the ModelScope cache layout.
- You can filter downloads with `allow_patterns=` / `ignore_patterns=` (e.g., only `*.safetensors` + configs).
- LLaMA2 checkpoints are typically license-gated; you may need to accept the upstream license terms before downloads succeed.
- 70B/72B checkpoints are very large; plan disk space and download time accordingly.

## Simulators

First simulator to try:

- Vidur (Microsoft): https://github.com/microsoft/vidur

Other candidates:

- RealLM (Bespoke Silicon Group): https://github.com/bespoke-silicon-group/reallm
- LLMCompass (Princeton University): https://github.com/PrincetonUniversity/LLMCompass
- Accel-Sim Framework: https://github.com/accel-sim/accel-sim-framework

## Q&A (Vidur simulation vs real GPU timing)

### Q: Does `vidur-sim` run on CPU or GPU?

A: `vidur-sim` runs on **CPU**. It uses a GPU-generated profiling bundle (e.g. A100 kernel timing + comm profiles) and a performance model to **simulate/predict GPU execution time** for a workload.

### Q: Is it meaningful to compare `vidur-sim` results with real GPU inference timing?

A: Yes — you should interpret the comparison as **simulated GPU latency prediction vs measured GPU latency**. It is meaningful, but it is not “two measurements of the same runtime”, so expect gaps:

- The simulator’s accuracy depends on how well the profiling/model matches your real inference stack (kernels, precision, scheduler/batching, KV-cache behavior, etc.).
- In this repo’s current `vidur-sim` wrapper, CPU overhead modeling is disabled (`skip_cpu_overhead_modeling=True` in `src/gpu_simulate_test/vidur_ext/sim_runner.py`), so CPU-side costs in a real server stack won’t be reflected.
- Token-level latencies from `vidur-sim` are derived from request-level metrics (not measured token-by-token); see `src/gpu_simulate_test/vidur_ext/sim_runner.py`.
- `real-bench` replays requests sequentially; if your workload has `arrival_time_ns=0` for many requests, later requests’ `ttft_ns` will include queueing behind earlier ones, which can distort direct TTFT comparisons.
- For a “no queueing” baseline with `real-bench`, set `workload.arrival.inter_arrival_ns` large enough that each request completes before the next arrives (check `completion_time_ns[i] <= arrival_time_ns[i+1]` in the real run’s `request_metrics.csv`).

## Repo Layout

- `src/gpu_simulate_test/`: Python package code (src layout).
- `extern/`: third-party code (git submodules under `extern/tracked/`).
- `models/`: external model references (symlink-based, not committed).
- `datasets/`: external dataset references (symlink-based, not committed).
- `context/`: project notes, runbooks, and experiment/repro docs.
- `scripts/`: small helper scripts/entrypoints (repo-owned).
- `tests/`: test skeleton (`unit/`, `integration/`, `manual/`).
- `tmp/`: local scratch space (ignored by git).
