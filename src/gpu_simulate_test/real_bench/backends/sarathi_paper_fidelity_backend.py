"""
Sarathi-Serve backend for paper-fidelity trace replay.

This runner replays a canonical `trace.csv` (token-length-only) against Sarathi-Serve and writes
paper-aligned request metrics under `tmp/paper_fidelity/`.

Key design choice: rely on Sarathi's in-engine metric definitions (via its metrics store) rather than
client-side timestamping, to keep metric boundaries aligned with Vidur.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gpu_simulate_test.env_guard import (
    apply_cuda_visible_devices_from_gsim,
    patch_sarathi_preserve_cuda_visible_devices,
)
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
    ignore_eos: bool = True


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


def _parse_sarathi_request_id(value: object) -> int:
    """Convert Sarathi 'Request Id' values like '0_12' into trace request_id ints."""
    s = str(value)
    if "_" in s:
        _, s = s.split("_", 1)
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(f"Unexpected Sarathi Request Id value: {value!r}") from e


def _validate_decode_token_counts(
    *,
    trace: pd.DataFrame,
    request_df: pd.DataFrame,
    ignore_eos: bool,
) -> None:
    """Ensure Sarathi produced exactly `num_decode_tokens` per request.

    For paper-fidelity, Vidur assumes fixed-length decode per request (as in the trace). Sarathi
    can stop early if it encounters EOS (or stop strings) unless `ignore_eos=True`.
    """
    if "request_id" in trace.columns:
        trace_request_ids = pd.to_numeric(trace["request_id"], errors="raise").astype(int)
    else:
        trace_request_ids = pd.Series(range(len(trace)), dtype=int)
    expected_decode = pd.to_numeric(trace["num_decode_tokens"], errors="raise").astype(int)
    expected = pd.DataFrame(
        {
            "trace_request_id": trace_request_ids,
            "expected_num_decode_tokens": expected_decode,
        }
    )

    if expected["trace_request_id"].duplicated().any():
        dup = expected.loc[expected["trace_request_id"].duplicated(), "trace_request_id"].unique()[:5].tolist()
        raise ValueError(f"trace.csv has duplicate request_id values (e.g. {dup}); request_id must be unique.")

    actual = request_df[["request_id", "request_num_decode_tokens"]].copy()
    actual["trace_request_id"] = actual["request_id"].map(_parse_sarathi_request_id)
    actual["actual_num_decode_tokens"] = pd.to_numeric(actual["request_num_decode_tokens"], errors="raise").astype(int)
    actual = actual[["trace_request_id", "actual_num_decode_tokens"]]

    if actual["trace_request_id"].duplicated().any():
        dup = actual.loc[actual["trace_request_id"].duplicated(), "trace_request_id"].unique()[:5].tolist()
        raise ValueError(
            f"Sarathi produced duplicate request ids after parsing (e.g. {dup}); "
            "this runner assumes a single replica."
        )

    expected_ids = set(expected["trace_request_id"].tolist())
    actual_ids = set(actual["trace_request_id"].tolist())
    missing = sorted(expected_ids - actual_ids)[:5]
    extra = sorted(actual_ids - expected_ids)[:5]
    if missing or extra:
        raise ValueError(
            "Sarathi replay request ids do not match trace request ids. "
            f"missing_in_sarathi={missing}, extra_in_sarathi={extra}"
        )

    merged = actual.merge(expected, on="trace_request_id", how="inner")
    mismatches = merged.loc[merged["actual_num_decode_tokens"] != merged["expected_num_decode_tokens"]]
    if len(mismatches) == 0:
        return

    sample = mismatches.head(10)
    details = "; ".join(
        f"id={int(rid)} expected={int(exp)} got={int(got)}"
        for rid, exp, got in zip(
            sample["trace_request_id"].tolist(),
            sample["expected_num_decode_tokens"].tolist(),
            sample["actual_num_decode_tokens"].tolist(),
        )
    )
    remedy = (
        "Set `scenario.real.sampling.ignore_eos=true` (and ensure no stop strings) "
        "to force fixed-length decode."
    )
    if ignore_eos:
        remedy = f"Even with ignore_eos=true, token counts mismatched. {remedy}"

    raise ValueError(
        "Sarathi replay produced request_num_decode_tokens that do not match trace num_decode_tokens "
        f"(ignore_eos={ignore_eos}). First mismatches: {details}. {remedy}"
    )


def run_sarathi_paper_fidelity(
    inputs: SarathiPaperFidelityInputs,
    *,
    out_dir: Path,
) -> Path:
    """Replay a trace on Sarathi and return the written request_metrics.csv path."""
    apply_cuda_visible_devices_from_gsim()
    patch_sarathi_preserve_cuda_visible_devices()

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

    warmup_prefill = max(1, min(int(inputs.chunk_size), int(inputs.max_tokens) - 1))
    warmup_decode = 1
    warmup_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=bool(inputs.ignore_eos),
        max_tokens=warmup_decode,
    )
    engine.add_request(
        prompt=None,
        prompt_token_ids=_prompt_token_ids(warmup_prefill),
        sampling_params=warmup_sampling_params,
        arrival_time=time.monotonic(),
        seq_id="warmup",
    )
    while engine.has_unfinished_requests():
        engine.step()
    engine.reset_metrics()

    start = time.monotonic()
    next_idx = 0
    while next_idx < len(trace) or engine.has_unfinished_requests():
        now = time.monotonic()

        # Submit any requests whose arrival time has passed.
        while next_idx < len(trace) and now >= start + float(trace.loc[next_idx, "arrived_at"]):
            prefill = int(trace.loc[next_idx, "num_prefill_tokens"])
            decode = int(trace.loc[next_idx, "num_decode_tokens"])
            seq_id = str(int(trace.loc[next_idx, "request_id"]) if "request_id" in trace.columns else next_idx)
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                ignore_eos=bool(inputs.ignore_eos),
                max_tokens=decode,
            )
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
    _validate_decode_token_counts(trace=trace, request_df=request_df, ignore_eos=bool(inputs.ignore_eos))
    req_csv = out_dir / "request_metrics.csv"
    write_csv(req_csv, request_df, required_columns=["request_id", *PAPER_FIDELITY_REQUIRED_SARATHI_COLUMNS])

    run_meta = {
        "schema_version": "v1",
        "run_type": "real",
        "scenario_name": inputs.scenario_name,
        "started_at": None,
        "ended_at": None,
        "warmup": {
            "enabled": True,
            "prefill_tokens": warmup_prefill,
            "decode_tokens": warmup_decode,
            "ignore_eos": bool(inputs.ignore_eos),
        },
        "trace_csv": str(inputs.trace_csv.resolve()),
        "sarathi_sequence_metrics_csv": str(seq_csv.resolve()),
        "request_metrics_csv": str(req_csv.resolve()),
    }
    write_json(out_dir / "run_meta.json", run_meta)
    return req_csv
