# Q&A: vidur-usage

## Introduction

This Q&A captures practical usage notes for Vidur (the LLM inference simulator), aimed at developers (including future maintainers) working in this repo.

**Related docs**
- `context/summaries/howto-use-vidur.md`
- `context/summaries/note-of-vidur-paper.md`
- `extern/tracked/vidur/README.md`
- `extern/tracked/vidur/paper/tex/main.tex`

**Key entrypoints and modules**
- `extern/tracked/vidur/vidur/main.py`
- `extern/tracked/vidur/vidur/simulator.py`
- `extern/tracked/vidur/vidur/config/config.py`
- `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py`
- `extern/tracked/vidur/vidur/scheduler/utils/memory_planner.py`
- `extern/tracked/vidur/vidur/profiling/mlp/mlp_wrapper.py`
- `extern/tracked/vidur/vidur/profiling/attention/attention_wrapper.py`

## Do I need to load model weights into RAM/VRAM to run a Vidur simulation, and can I simulate a 600B model on limited hardware?
> Last revised at: `2026-01-06T08:28:02Z` | Last revised base commit: `5dd6037d2dcb151e46639d07e0fde6a3765ab894`

- **Simulation does not load a real checkpoint**: Vidur’s simulator consumes a model spec (`model_config`) plus profiled/predicted operator runtimes (CSV inputs) and schedules events; it does not instantiate transformer layers or load full weights during simulation (`extern/tracked/vidur/vidur/simulator.py`, `extern/tracked/vidur/vidur/entities/replica.py`, `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py`).
- **Simulation is event-driven and does not compute token outputs**: The simulator advances a virtual clock over request/batch events and uses predicted runtimes to emit latency/throughput/utilization metrics; it does not run a forward pass or generate correct inference results (`extern/tracked/vidur/vidur/simulator.py`, `extern/tracked/vidur/vidur/execution_time_predictor/base_execution_time_predictor.py`).
- **Memory feasibility is modeled analytically**: Per-device parameter bytes and KV-cache bytes are computed from the model spec; the replica scheduler derives max batch/request slots from device memory and asserts if even one request cannot fit (`extern/tracked/vidur/vidur/utils/param_counter.py`, `extern/tracked/vidur/vidur/scheduler/utils/memory_planner.py`).
- **So “limited local VRAM” is usually not a blocker for simulation**: You can run the simulator on a modest machine while *simulating* a large deployment (many GPUs, large memory) as long as the *simulated* deployment config is feasible and the required profiling inputs exist (`extern/tracked/vidur/vidur/config/config.py`).
- **But profiling/onboarding can require significant GPU memory**: Generating the profiling CSVs uses GPU kernels and allocates dummy/sharded weight tensors (MLP) and KV-cache blocks (attention); for very large hidden sizes this may require high TP sharding and/or large GPUs (`extern/tracked/vidur/vidur/profiling/mlp/mlp_wrapper.py`, `extern/tracked/vidur/vidur/profiling/attention/attention_wrapper.py`).
- **For a 600B model specifically**: Vidur would need a corresponding `BaseModelConfig` entry and profiling data keyed by that model name/device; otherwise the execution-time predictor won’t find matching rows and the run will fail or be meaningless (`extern/tracked/vidur/vidur/config/model_config.py`, `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py`).

## Different inference engines (HF Transformers / vLLM / Sarathi-Serve / SGLang) behave differently; how does Vidur capture those differences?
> Last revised at: `2026-01-06T08:40:11Z` | Last revised base commit: `5dd6037d2dcb151e46639d07e0fde6a3765ab894`

- **Vidur factors “engine behavior” into (a) scheduling policy + (b) operator runtime model**: batches/events come from the scheduler; per-batch time comes from the execution-time predictor (`extern/tracked/vidur/vidur/scheduler/global_scheduler/base_global_scheduler.py`, `extern/tracked/vidur/vidur/execution_time_predictor/base_execution_time_predictor.py`).
- **Scheduler differences are explicit, pluggable modules**: Vidur ships replica schedulers for `vllm`, `orca`, `sarathi`, `faster_transformer`, and `lightllm` via `ReplicaSchedulerType`/configs and the corresponding implementations under `extern/tracked/vidur/vidur/scheduler/replica_scheduler/` (e.g. `vllm_replica_scheduler.py`, `sarathi_replica_scheduler.py`, `faster_transformer_replica_scheduler.py`, `lightllm_replica_scheduler.py`).
- **KV-cache / preemption semantics are modeled at the scheduler level**: block allocation, watermarking, preemption, and request restart are part of the scheduler logic and request state machine (`extern/tracked/vidur/vidur/scheduler/replica_scheduler/vllm_replica_scheduler.py`, `extern/tracked/vidur/vidur/entities/request.py`, `extern/tracked/vidur/vidur/scheduler/utils/memory_planner.py`).
- **Kernel/runtime differences are captured only if profiling matches the engine’s kernel stack**: the predictor reads profiling CSVs keyed by `{DEVICE}/{MODEL}` and uses them to predict per-operator times (`extern/tracked/vidur/vidur/config/config.py`, `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py`). If another engine uses different kernels/backends, you need new profiling inputs for that stack.
- **CPU-side overheads can be modeled separately (when enabled)**: Vidur includes CPU overhead terms (schedule/prepare inputs/sampler/process outputs/etc.) and provides a profiler that measures them using Sarathi’s `LLMEngine` (`extern/tracked/vidur/vidur/entities/execution_time.py`, `extern/tracked/vidur/vidur/profiling/cpu_overhead/benchmark_runner.py`).
- **“Captured behavior” is validated against a specific baseline, not every engine by default**: the paper’s fidelity evaluation compares against an optimized vLLM-based serving stack extended with multiple scheduling policies (`extern/tracked/vidur/paper/tex/5-eval.tex`).
- **Engines not implemented as schedulers are not faithfully simulated out-of-the-box**: there is no replica scheduler for HF Transformers or SGLang in `ReplicaSchedulerType` today (`extern/tracked/vidur/vidur/config/config.py`), so supporting them means implementing their batching/memory policy and collecting matching profiling data.

## Using vLLM as an example, what “events” are supported by Vidur?
> Last revised at: `2026-01-06T08:58:53Z` | Last revised base commit: `5dd6037d2dcb151e46639d07e0fde6a3765ab894`

- **Vidur events are simulator control-flow events** (request routing/scheduling/batch lifecycle), not per-kernel launch events; the event enum is `EventType` (`extern/tracked/vidur/vidur/types/event_type.py`) and concrete classes live in `extern/tracked/vidur/vidur/events/`.
- **Request arrival → global scheduling**: `RequestArrivalEvent` enqueues the request and triggers `GlobalScheduleEvent` (`extern/tracked/vidur/vidur/events/request_arrival_event.py`, `extern/tracked/vidur/vidur/events/global_schedule_event.py`).
- **Global scheduling → per-replica scheduling**: `GlobalScheduleEvent` maps queued requests to replicas and emits `ReplicaScheduleEvent` for the affected replicas (`extern/tracked/vidur/vidur/events/global_schedule_event.py`, `extern/tracked/vidur/vidur/scheduler/global_scheduler/base_global_scheduler.py`).
- **Replica scheduling (vLLM policy) → stage-0 arrival**: `ReplicaScheduleEvent` calls the replica scheduler (`VLLMReplicaScheduler`) to form batches/microbatches and emits `BatchStageArrivalEvent` for pipeline stage 0 (`extern/tracked/vidur/vidur/events/replica_schedule_event.py`, `extern/tracked/vidur/vidur/scheduler/replica_scheduler/vllm_replica_scheduler.py`, `extern/tracked/vidur/vidur/events/batch_stage_arrival_event.py`).
- **Stage scheduling → stage completion**: `ReplicaStageScheduleEvent` starts a batch stage (using predicted runtime from the execution-time predictor) and emits a `BatchStageEndEvent` at `t + execution_time` (`extern/tracked/vidur/vidur/events/replica_stage_schedule_event.py`, `extern/tracked/vidur/vidur/scheduler/replica_stage_scheduler/replica_stage_schduler.py`, `extern/tracked/vidur/vidur/entities/execution_time.py`).
- **Stage completion → next stage or batch end**: `BatchStageEndEvent` either forwards the batch to the next stage (`BatchStageArrivalEvent`) or ends it (`BatchEndEvent`) if this was the last stage (`extern/tracked/vidur/vidur/events/batch_stage_end_event.py`, `extern/tracked/vidur/vidur/events/batch_end_event.py`).
- **Batch end → request progress + reschedule**: `BatchEndEvent` advances request token counters, frees memory for completed requests, and re-triggers `ReplicaScheduleEvent` (`extern/tracked/vidur/vidur/events/batch_end_event.py`, `extern/tracked/vidur/vidur/entities/request.py`).
- **vLLM-specific behaviors are modeled inside the vLLM replica scheduler, not as separate event types**: KV-block allocation, watermarking, preemption, and restart live in `VLLMReplicaScheduler` + the request state machine (`extern/tracked/vidur/vidur/scheduler/replica_scheduler/vllm_replica_scheduler.py`, `extern/tracked/vidur/vidur/entities/request.py`).
