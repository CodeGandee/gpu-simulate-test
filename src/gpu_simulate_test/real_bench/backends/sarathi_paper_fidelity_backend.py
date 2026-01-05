"""
Sarathi-Serve backend for paper-fidelity trace replay.

This runner replays a canonical `trace.csv` (token-length-only) against Sarathi-Serve and writes
paper-aligned request metrics under `tmp/paper_fidelity/`.

Key design choice: rely on Sarathi's in-engine metric definitions (via its metrics store) rather than
client-side timestamping, to keep metric boundaries aligned with Vidur.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import write_csv, write_json
from gpu_simulate_test.paper_fidelity.traces import TraceSpec, read_trace_csv


PAPER_FIDELITY_REQUIRED_SARATHI_COLUMNS = [
    "request_scheduling_delay",
    "request_execution_plus_preemption_time_normalized",
    "request_e2e_time_normalized",
    "request_num_decode_tokens",
]


@dataclass(frozen=True)
class SarathiPaperFidelityInputs:
    scenario_name: str
    trace_csv: Path
    model_id: str
    model_ref: Path
    seed: int = 42
    max_tokens: int = 4096
    chunk_size: int = 16
    max_num_seqs: int = 16
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    cuda_visible_devices: str | None = None


def convert_sequence_metrics_to_request_metrics(sequence_metrics_csv: Path) -> pd.DataFrame:
    """Map Sarathi `sequence_metrics.csv` to the paper-fidelity request metrics schema."""
    df = pd.read_csv(sequence_metrics_csv)
    if "Request Id" not in df.columns:
        raise ValueError(f"{sequence_metrics_csv}: missing required column 'Request Id'")

    out = df.rename(columns={"Request Id": "request_id"})
    missing = [c for c in PAPER_FIDELITY_REQUIRED_SARATHI_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"{sequence_metrics_csv}: missing required columns: {missing}")
    return out


def _default_cuda_visible_devices() -> str | None:
    """Best-effort selection of usable GPUs for single-process workflows.

    Some environments may have GPUs in MIG-enabled mode without instances (or other unusable states)
    that can cause PyTorch CUDA initialization to fail when all devices are visible.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,mig.mode.current", "--format=csv,noheader"],
            text=True,
        )
        candidates: list[str] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            idx, mig_mode = [p.strip() for p in line.split(",", 1)]
            if mig_mode.lower() == "disabled":
                candidates.append(idx)
        if candidates:
            return ",".join(candidates)
    except Exception:
        pass
    return None


def run_sarathi_paper_fidelity(
    inputs: SarathiPaperFidelityInputs,
    *,
    out_dir: Path,
) -> Path:
    """Replay a trace on Sarathi and return the written request_metrics.csv path."""
    if inputs.cuda_visible_devices is not None:
        desired_cuda_visible_devices = inputs.cuda_visible_devices
    else:
        existing = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing:
            desired_cuda_visible_devices = existing
        else:
            desired_cuda_visible_devices = _default_cuda_visible_devices() or "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = desired_cuda_visible_devices

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for the Sarathi paper-fidelity runner; run inside the Pixi env.") from e

    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError(
            "CUDA is required for the Sarathi paper-fidelity runner (torch.cuda.is_available() is False)."
        )

    if not inputs.model_ref.exists():
        raise FileNotFoundError(
            "Model assets are required for Sarathi replay; missing model_ref: "
            f"{inputs.model_ref} (see models/*/bootstrap.sh)"
        )

    try:
        # Patch Sarathi's Ray worker initialization to preserve CUDA_VISIBLE_DEVICES.
        # Sarathi unsets CUDA_VISIBLE_DEVICES by default, which can expose unusable GPUs on some hosts.
        from sarathi.engine import base_llm_engine as _base_llm_engine  # type: ignore

        def _noop() -> None:
            return None

        class _PinnedRayWorker:  # type: ignore
            def __init__(self, init_cached_hf_modules: bool = False) -> None:
                if init_cached_hf_modules:
                    from transformers.dynamic_module_utils import init_hf_modules  # type: ignore

                    init_hf_modules()
                # Ray may set CUDA_VISIBLE_DEVICES to an empty value for actors without GPU requests.
                # Sarathi intentionally unsets it; instead, set it explicitly to a safe subset.
                os.environ["CUDA_VISIBLE_DEVICES"] = desired_cuda_visible_devices
                self.worker = None

            def init_worker(self, worker_init_fn):
                self.worker = worker_init_fn()

            def __getattr__(self, name):
                return getattr(self.worker, name)

            def execute_method(self, method, *args, **kwargs):
                executor = getattr(self, method)
                return executor(*args, **kwargs)

        _base_llm_engine.unset_cuda_visible_devices = _noop  # type: ignore[attr-defined]
        _base_llm_engine.RayWorker = _PinnedRayWorker  # type: ignore[attr-defined]

        from sarathi import LLMEngine, SamplingParams  # type: ignore
        from sarathi.config import (  # type: ignore
            MetricsConfig,
            ModelConfig,
            ParallelConfig,
            ReplicaConfig,
            SarathiSchedulerConfig,
            SystemConfig,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "sarathi is required for the Sarathi paper-fidelity runner; run inside the Pixi env "
            "and ensure `extern/tracked/sarathi-serve` is initialized."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)

    trace = read_trace_csv(inputs.trace_csv, spec=TraceSpec(max_tokens=inputs.max_tokens))
    trace = trace.reset_index(drop=True)

    replica_config = ReplicaConfig(output_dir=str(out_dir / "sarathi"))
    model_config = ModelConfig(model=str(inputs.model_ref), seed=int(inputs.seed))
    parallel_config = ParallelConfig(
        tensor_parallel_size=int(inputs.tensor_parallel_size),
        pipeline_parallel_size=int(inputs.pipeline_parallel_size),
    )
    scheduler_config = SarathiSchedulerConfig(chunk_size=int(inputs.chunk_size), max_num_seqs=int(inputs.max_num_seqs))
    metrics_config = MetricsConfig(
        write_metrics=True,
        enable_chrome_trace=False,
        enable_op_level_metrics=False,
        enable_cpu_op_level_metrics=False,
        keep_individual_batch_metrics=False,
        enable_request_outputs=False,
    )
    system_config = SystemConfig(
        replica_config=replica_config,
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        metrics_config=metrics_config,
    )

    engine = LLMEngine.from_system_config(system_config)

    prompt_cache: dict[int, list[int]] = {}

    def _prompt_token_ids(n: int) -> list[int]:
        cached = prompt_cache.get(n)
        if cached is None:
            cached = [0] * n
            prompt_cache[n] = cached
        return cached

    start = time.monotonic()
    next_idx = 0
    while next_idx < len(trace) or engine.has_unfinished_requests():
        now = time.monotonic()

        # Submit any requests whose arrival time has passed.
        while next_idx < len(trace) and now >= start + float(trace.loc[next_idx, "arrived_at"]):
            prefill = int(trace.loc[next_idx, "num_prefill_tokens"])
            decode = int(trace.loc[next_idx, "num_decode_tokens"])
            seq_id = str(int(trace.loc[next_idx, "request_id"]) if "request_id" in trace.columns else next_idx)
            sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=decode)
            engine.add_request(
                prompt=None,
                prompt_token_ids=_prompt_token_ids(prefill),
                sampling_params=sampling_params,
                arrival_time=start + float(trace.loc[next_idx, "arrived_at"]),
                seq_id=seq_id,
            )
            next_idx += 1

        if next_idx < len(trace):
            # Avoid a busy spin if the next arrival is in the future and the engine is idle.
            next_arrival_s = start + float(trace.loc[next_idx, "arrived_at"])
            if not engine.has_unfinished_requests() and now < next_arrival_s:
                time.sleep(min(0.01, max(0.0, next_arrival_s - now)))
                continue

        engine.step()

    # Sarathi's MetricsStore.plot() writes PNGs via Plotly/Kaleido (requires Chrome).
    # For paper-fidelity we only need `sequence_metrics.csv`, so write the CSV directly.
    metrics_store = engine.metrics_store
    all_seq_metrics = list(metrics_store.seq_metrics_time_distributions.values()) + list(
        metrics_store.seq_metrics_histogram.values()
    )
    metrics_store._save_as_csv(  # type: ignore[attr-defined]
        dataseries_list=all_seq_metrics,
        key_to_join="Request Id",
        base_path=metrics_store.output_dir,
        file_name="sequence_metrics",
    )

    seq_csv = out_dir / "sarathi" / "replica_0" / "sequence_metrics.csv"
    if not seq_csv.exists():
        raise FileNotFoundError(f"Sarathi did not produce sequence_metrics.csv at {seq_csv}")

    request_df = convert_sequence_metrics_to_request_metrics(seq_csv)
    req_csv = out_dir / "request_metrics.csv"
    write_csv(req_csv, request_df, required_columns=["request_id", *PAPER_FIDELITY_REQUIRED_SARATHI_COLUMNS])

    run_meta = {
        "schema_version": "v1",
        "run_type": "real",
        "scenario_name": inputs.scenario_name,
        "started_at": None,
        "ended_at": None,
        "trace_csv": str(inputs.trace_csv.resolve()),
        "sarathi_sequence_metrics_csv": str(seq_csv.resolve()),
        "request_metrics_csv": str(req_csv.resolve()),
    }
    write_json(out_dir / "run_meta.json", run_meta)
    return req_csv
