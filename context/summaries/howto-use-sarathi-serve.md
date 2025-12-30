# How to use Sarathi-Serve (`sarathi`) in this repo

## HEADER
- **Purpose**: Practical notes for running Sarathi-Serve (the `sarathi` inference engine) inside this repo’s Pixi environment, for offline inference and for its OpenAI-compatible HTTP server.
- **Status**: Active
- **Date**: 2025-12-30
- **Dependencies**:
  - `pyproject.toml` (Pixi env; `sarathi` installed editable)
  - `extern/tracked/sarathi-serve/` (Sarathi-Serve source tree)
  - GPU runtime (for actual inference; Sarathi builds/uses CUDA extensions)
- **Target**: Contributors who need a real inference engine baseline for simulator comparisons.

## 1) Setup (Pixi + submodules)

1. Initialize submodules (if needed):
   - `git submodule update --init --recursive`
2. Install/update the Pixi environment:
   - `pixi install`
3. Sanity-check import:
   - `pixi run python -c "import sarathi; print(sarathi.__version__, sarathi.__file__)"`

## 2) Offline inference (Python API)

Sarathi’s “offline” mode is just Python: construct a `SystemConfig`, create an `LLMEngine`, enqueue requests, then call `step()` until all requests finish.

Minimal example (dummy weights; does not download model weights but still instantiates the model on GPU):

```python
from sarathi import LLMEngine, SamplingParams
from sarathi.config import (
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
    ReplicaConfig,
    SarathiSchedulerConfig,
    SystemConfig,
)

system_config = SystemConfig(
    replica_config=ReplicaConfig(output_dir="tmp/sarathi_runs"),
    model_config=ModelConfig(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        trust_remote_code=False,
        load_format="dummy",
        dtype="float16",
        max_model_len=2048,
    ),
    parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
    scheduler_config=SarathiSchedulerConfig(chunk_size=64, max_num_seqs=4),
    metrics_config=MetricsConfig(write_metrics=True, enable_chrome_trace=False),
)

engine = LLMEngine.from_system_config(system_config)
params = SamplingParams(temperature=0.0, max_tokens=32)

engine.add_request("Hello, my name is", params)
while engine.has_unfinished_requests():
    for out in engine.step():
        if out.finished:
            print(out.text)
```

Notes:

- Sarathi writes metrics under `replica_config.output_dir/replica_0/`.
- `metrics_store.plot()` (via `engine.plot_metrics()`) may require Chrome (Plotly/Kaleido image export). If you don’t have Chrome installed, skip plotting and just collect raw CSV/JSON outputs.

## 3) OpenAI-compatible server

Sarathi provides an OpenAI-compatible API server via:

- Module: `sarathi.entrypoints.openai.api_server`
- Config dataclass: `sarathi.entrypoints.openai.config.OpenAIServerConfig`

Example (single GPU; dummy weights for a quick “does the server run?” smoke test):

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python -m sarathi.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --output_dir tmp/sarathi_openai_server \
  --model_config_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model_config_trust_remote_code False \
  --model_config_load_format dummy \
  --model_config_dtype float16 \
  --parallel_config_pipeline_parallel_size 1 \
  --parallel_config_tensor_parallel_size 1 \
  --scheduler_config_type SARATHI \
  --sarathi_scheduler_config_chunk_size 64 \
  --sarathi_scheduler_config_max_num_seqs 64
```

Quick client checks:

- `curl http://localhost:8000/v1/models`
- `curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"Hello","max_tokens":16}'`

## 4) Model support note (Qwen3)

As of the currently tracked Sarathi-Serve version, the model registry does **not** include Qwen3 (`architectures: ["Qwen3ForCausalLM"]`). You can confirm what Sarathi supports in:

- `extern/tracked/sarathi-serve/sarathi/model_executor/model_loader.py`

For Qwen3 ground-truth timing, you’ll either need to:

- Add Qwen3 support to Sarathi (implement the model class + register it), or
- Use a different real inference engine that already supports Qwen3 (e.g., vLLM) and treat it as a separate comparison slice.
