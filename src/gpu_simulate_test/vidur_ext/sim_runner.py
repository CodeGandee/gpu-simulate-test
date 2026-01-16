"""
Vidur simulation wrappers for paper-fidelity workflows.

This module runs the Vidur simulator and converts its emitted metrics into the schemas used by
this repository's paper-fidelity scoring pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import read_csv, utcnow_iso, write_csv, write_json
from gpu_simulate_test.paper_fidelity.traces import TraceSpec, read_trace_csv
from gpu_simulate_test.vidur_ext.profiling_root import ProfilingRootLayout, validate_profiling_root


def _default_vidur_cache_dir(*, out_dir: Path) -> Path:
    """Return the default Vidur cache directory under a run output directory.

    Parameters
    ----------
    out_dir
        Simulation output directory.

    Returns
    -------
    pathlib.Path
        Cache directory path (`<out_dir>/vidur-cache`), created if missing.
    """
    cache_dir = out_dir / "vidur-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass(frozen=True)
class VidurSimInputs:
    """Legacy inputs for `run_vidur_sim` (split-trace layout)."""

    workload_dir: Path
    profiling_root: Path
    model_id: str
    device: str = "a100"
    network_device: str = "a100_pairwise_nvlink"
    tensor_parallel_size: int = 1
    num_pipeline_stages: int = 1
    mlp_validation_mode: str = "strict"
    mlp_small_input_threshold: int = 128
    mlp_zero_heavy_limit: float = 0.01
    seed: int = 42
    max_tokens: int = 4096
    # Vidur's interface uses the negative form.
    skip_cpu_overhead_modeling: bool = True
    # Guardrails: `strict` rejects placeholder-like dummy CPU overhead CSVs.
    cpu_overhead_validation: str = "strict"
    # Optional parity-critical scheduler knobs. If unset, Vidur defaults apply.
    scheduler_chunk_size: int | None = None
    scheduler_batch_size_cap: int | None = None
    scheduler_block_size: int | None = None
    scheduler_watermark_blocks_fraction: float | None = None


def _build_vidur_trace_csv(inputs: VidurSimInputs, *, out_dir: Path) -> Path:
    lengths = read_csv(
        inputs.workload_dir / "trace_lengths.csv",
        required_columns=["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"],
        context="trace_lengths",
    )
    intervals = read_csv(
        inputs.workload_dir / "trace_intervals.csv",
        required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"],
        context="trace_intervals",
    )
    merged = pd.merge(lengths, intervals, on=["request_id"], how="inner").sort_values("request_id")
    merged = merged.reset_index(drop=True)

    trace_df = pd.DataFrame(
        {
            "arrived_at": merged["arrival_time_ns"].astype(float) / 1e9,
            "num_prefill_tokens": merged["num_prefill_tokens"].astype(int),
            "num_decode_tokens": merged["num_decode_tokens"].astype(int),
        }
    )

    trace_path = out_dir / "vidur_trace.csv"
    trace_df.to_csv(trace_path, index=False)
    return trace_path


def _standardize_vidur_outputs(
    *,
    workload_dir: Path,
    vidur_request_metrics_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workload_lengths = read_csv(
        workload_dir / "trace_lengths.csv",
        required_columns=["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"],
        context="trace_lengths",
    )
    workload_intervals = read_csv(
        workload_dir / "trace_intervals.csv",
        required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"],
        context="trace_intervals",
    )
    workload = pd.merge(workload_lengths, workload_intervals, on=["request_id"], how="inner").sort_values(
        "request_id"
    )

    raw = pd.read_csv(vidur_request_metrics_csv)
    if "Request Id" not in raw.columns:
        raise ValueError(f"Unexpected Vidur request_metrics.csv schema: missing 'Request Id' ({vidur_request_metrics_csv})")

    raw = raw.rename(columns={"Request Id": "request_id"})
    for col in ["request_e2e_time", "prefill_e2e_time"]:
        if col not in raw.columns:
            raise ValueError(f"Unexpected Vidur request_metrics.csv schema: missing '{col}' ({vidur_request_metrics_csv})")

    merged = pd.merge(workload, raw[["request_id", "request_e2e_time", "prefill_e2e_time"]], on=["request_id"], how="inner")
    merged = merged.sort_values("request_id").reset_index(drop=True)

    arrival_ns = merged["arrival_time_ns"].astype(int)
    ttft_ns = (merged["prefill_e2e_time"].astype(float) * 1e9).round().astype(int)
    first_token_time_ns = arrival_ns + ttft_ns
    completion_time_ns = arrival_ns + (merged["request_e2e_time"].astype(float) * 1e9).round().astype(int)

    num_prefill_tokens = merged["num_prefill_tokens"].astype(int)
    num_decode_tokens = merged["num_decode_tokens"].astype(int)
    num_decode_tokens_actual = num_decode_tokens.copy()

    request_df = pd.DataFrame(
        {
            "request_id": merged["request_id"].astype(int),
            "arrival_time_ns": arrival_ns,
            "first_token_time_ns": first_token_time_ns,
            "ttft_ns": ttft_ns,
            "completion_time_ns": completion_time_ns,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "num_decode_tokens_actual": num_decode_tokens_actual,
            "status": ["ok"] * len(merged),
        }
    )

    token_rows: list[dict] = []
    for row in request_df.to_dict(orient="records"):
        rid = int(row["request_id"])
        n = int(row["num_decode_tokens_actual"])
        t0 = int(row["first_token_time_ns"])
        tN = int(row["completion_time_ns"])

        if n <= 0:
            continue
        if n == 1:
            times = [t0]
        else:
            step = (tN - t0) / float(n - 1)
            times = [int(round(t0 + step * i)) for i in range(n)]

        prev = None
        for i, t in enumerate(times):
            token_rows.append(
                {
                    "request_id": rid,
                    "token_index": int(i),
                    "token_time_ns": int(t),
                    "token_latency_ns": 0 if prev is None else int(t - prev),
                }
            )
            prev = t

    token_df = pd.DataFrame(token_rows)
    return request_df, token_df


def run_vidur_sim(inputs: VidurSimInputs, *, out_dir: Path, run_meta: dict) -> None:
    layout = ProfilingRootLayout(
        profiling_root=inputs.profiling_root,
        device=inputs.device,
        model_id=inputs.model_id,
        network_device=inputs.network_device,
        tensor_parallel_size=inputs.tensor_parallel_size,
        num_pipeline_stages=inputs.num_pipeline_stages,
        mlp_validation_mode=str(inputs.mlp_validation_mode),
        mlp_small_input_threshold=int(inputs.mlp_small_input_threshold),
        mlp_zero_heavy_limit=float(inputs.mlp_zero_heavy_limit),
        skip_cpu_overhead_modeling=bool(inputs.skip_cpu_overhead_modeling),
        cpu_overhead_validation=str(inputs.cpu_overhead_validation),
    )
    validate_profiling_root(layout)

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = _build_vidur_trace_csv(inputs, out_dir=out_dir)

    profiling_base = inputs.profiling_root / "data" / "profiling"

    from vidur.config import (
        ClusterConfig,
        MetricsConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        SarathiSchedulerConfig,
        SimulationConfig,
        TraceRequestGeneratorConfig,
    )
    from vidur.simulator import Simulator

    replica_config = ReplicaConfig(
        model_name=inputs.model_id,
        num_pipeline_stages=int(inputs.num_pipeline_stages),
        tensor_parallel_size=int(inputs.tensor_parallel_size),
        device=str(inputs.device),
        network_device=str(inputs.network_device),
    )
    replica_scheduler_config = SarathiSchedulerConfig()
    if inputs.scheduler_chunk_size is not None:
        replica_scheduler_config.chunk_size = int(inputs.scheduler_chunk_size)
    if inputs.scheduler_batch_size_cap is not None:
        replica_scheduler_config.batch_size_cap = int(inputs.scheduler_batch_size_cap)
    if inputs.scheduler_block_size is not None:
        replica_scheduler_config.block_size = int(inputs.scheduler_block_size)
    if inputs.scheduler_watermark_blocks_fraction is not None:
        replica_scheduler_config.watermark_blocks_fraction = float(inputs.scheduler_watermark_blocks_fraction)

    cluster_config = ClusterConfig(
        num_replicas=1,
        replica_config=replica_config,
        replica_scheduler_config=replica_scheduler_config,
    )
    request_generator_config = TraceRequestGeneratorConfig(trace_file=str(trace_csv), max_tokens=int(inputs.max_tokens))

    exec_cfg = RandomForrestExecutionTimePredictorConfig(
        compute_input_file=str(profiling_base / "compute/{DEVICE}/{MODEL}/mlp.csv"),
        attention_input_file=str(profiling_base / "compute/{DEVICE}/{MODEL}/attention.csv"),
        all_reduce_input_file=str(profiling_base / "network/{NETWORK_DEVICE}/all_reduce.csv"),
        send_recv_input_file=str(profiling_base / "network/{NETWORK_DEVICE}/send_recv.csv"),
        cpu_overhead_input_file=str(profiling_base / "cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv"),
        skip_cpu_overhead_modeling=bool(inputs.skip_cpu_overhead_modeling),
    )

    metrics_cfg = MetricsConfig(
        write_metrics=True,
        enable_chrome_trace=False,
        store_plots=False,
        store_operation_metrics=False,
        store_token_completion_metrics=False,
        store_request_metrics=True,
        store_batch_metrics=False,
        store_utilization_metrics=False,
        output_dir=str(out_dir / "vidur_raw"),
        cache_dir=str(_default_vidur_cache_dir(out_dir=out_dir).resolve()),
    )

    sim_cfg = SimulationConfig(
        seed=int(inputs.seed),
        cluster_config=cluster_config,
        request_generator_config=request_generator_config,
        execution_time_predictor_config=exec_cfg,
        metrics_config=metrics_cfg,
    )

    simulator = Simulator(sim_cfg)
    simulator.run()
    simulator.metric_store.plot()

    raw_dir = Path(sim_cfg.metrics_config.output_dir)
    raw_request_metrics = raw_dir / "request_metrics.csv"
    if not raw_request_metrics.exists():
        raise FileNotFoundError(f"Vidur did not produce request_metrics.csv under {raw_dir}")

    request_df, token_df = _standardize_vidur_outputs(
        workload_dir=inputs.workload_dir,
        vidur_request_metrics_csv=raw_request_metrics,
    )

    write_csv(
        out_dir / "request_metrics.csv",
        request_df,
        required_columns=[
            "request_id",
            "arrival_time_ns",
            "first_token_time_ns",
            "ttft_ns",
            "num_prefill_tokens",
            "num_decode_tokens",
            "num_decode_tokens_actual",
            "status",
        ],
    )
    write_csv(
        out_dir / "token_metrics.csv",
        token_df,
        required_columns=["request_id", "token_index", "token_time_ns", "token_latency_ns"],
    )

    run_meta = dict(run_meta)
    run_meta.setdefault("ended_at", utcnow_iso())
    run_meta.setdefault("vidur_raw_dir", str(raw_dir))
    run_meta.setdefault("vidur_trace_csv", str(trace_csv))
    write_json(out_dir / "run_meta.json", run_meta)


PAPER_FIDELITY_REQUIRED_VIDUR_COLUMNS = [
    "request_scheduling_delay",
    "request_execution_plus_preemption_time_normalized",
    "request_e2e_time_normalized",
    "request_num_decode_tokens",
]


def _validate_vidur_decode_token_counts(*, trace_csv: Path, request_df: pd.DataFrame, max_tokens: int) -> None:
    trace = read_trace_csv(trace_csv, spec=TraceSpec(max_tokens=int(max_tokens)))
    if "request_id" in trace.columns:
        trace_request_ids = pd.to_numeric(trace["request_id"], errors="raise").astype(int)
    else:
        trace_request_ids = pd.Series(range(len(trace)), dtype=int)

    expected = pd.DataFrame(
        {
            "request_id": trace_request_ids,
            "expected_num_decode_tokens": pd.to_numeric(trace["num_decode_tokens"], errors="raise").astype(int),
        }
    )
    if expected["request_id"].duplicated().any():
        dup = expected.loc[expected["request_id"].duplicated(), "request_id"].unique()[:5].tolist()
        raise ValueError(f"trace.csv has duplicate request_id values (e.g. {dup}); request_id must be unique.")

    actual = request_df[["request_id", "request_num_decode_tokens"]].copy()
    actual["request_id"] = pd.to_numeric(actual["request_id"], errors="raise").astype(int)
    actual["actual_num_decode_tokens"] = pd.to_numeric(actual["request_num_decode_tokens"], errors="raise").astype(int)
    actual = actual[["request_id", "actual_num_decode_tokens"]]

    if actual["request_id"].duplicated().any():
        dup = actual.loc[actual["request_id"].duplicated(), "request_id"].unique()[:5].tolist()
        raise ValueError(f"Vidur produced duplicate request ids (e.g. {dup}); request_id must be unique.")

    expected_ids = set(expected["request_id"].tolist())
    actual_ids = set(actual["request_id"].tolist())
    missing = sorted(expected_ids - actual_ids)[:5]
    extra = sorted(actual_ids - expected_ids)[:5]
    if missing or extra:
        raise ValueError(
            "Vidur request ids do not match trace request ids (runs are not comparable). "
            f"missing_in_vidur={missing}, extra_in_vidur={extra}"
        )

    merged = actual.merge(expected, on="request_id", how="inner")
    mismatches = merged.loc[merged["actual_num_decode_tokens"] != merged["expected_num_decode_tokens"]]
    if len(mismatches) == 0:
        return

    sample = mismatches.head(10)
    details = "; ".join(
        f"id={int(rid)} expected={int(exp)} got={int(got)}"
        for rid, exp, got in zip(
            sample["request_id"].tolist(),
            sample["expected_num_decode_tokens"].tolist(),
            sample["actual_num_decode_tokens"].tolist(),
        )
    )
    raise ValueError(
        "Vidur request_num_decode_tokens does not match trace num_decode_tokens (runs are not comparable). "
        f"First mismatches: {details}"
    )


def convert_vidur_request_metrics_to_paper_fidelity(raw_request_metrics_csv: Path) -> pd.DataFrame:
    """Convert Vidur's raw `request_metrics.csv` to the paper-fidelity schema.

    The primary transformation is renaming `Request Id` → `request_id`, while preserving
    Vidur's normalized metric columns without recomputation.

    Parameters
    ----------
    raw_request_metrics_csv
        Path to Vidur's raw `request_metrics.csv`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing at least `request_id` and the required normalized metric columns.

    Raises
    ------
    ValueError
        If the input CSV does not match the expected Vidur schema.
    """
    raw = pd.read_csv(raw_request_metrics_csv)
    if "Request Id" not in raw.columns:
        raise ValueError(
            "Unexpected Vidur request_metrics.csv schema: missing 'Request Id' "
            f"({raw_request_metrics_csv})"
        )

    df = raw.rename(columns={"Request Id": "request_id"})
    missing = [c for c in PAPER_FIDELITY_REQUIRED_VIDUR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Unexpected Vidur request_metrics.csv schema: missing required columns "
            f"{missing} ({raw_request_metrics_csv})"
        )
    return df


@dataclass(frozen=True)
class VidurPaperFidelitySimInputs:
    """Inputs for running Vidur in the paper-fidelity pipeline."""

    scenario_name: str
    trace_csv: Path
    profiling_root: Path
    model_id: str
    device: str = "a100"
    network_device: str = "a100_pairwise_nvlink"
    tensor_parallel_size: int = 1
    num_pipeline_stages: int = 1
    mlp_validation_mode: str = "strict"
    mlp_small_input_threshold: int = 128
    mlp_zero_heavy_limit: float = 0.01
    seed: int = 42
    max_tokens: int = 4096
    # Vidur's interface uses the negative form.
    skip_cpu_overhead_modeling: bool = True
    # Guardrails: `strict` rejects placeholder-like dummy CPU overhead CSVs.
    cpu_overhead_validation: str = "strict"
    # Parity-critical scheduler knobs. Vidur has its own defaults; for sim-vs-real
    # experiments these should be set explicitly to match the real runner.
    scheduler_type: str = "sarathi"
    scheduler_chunk_size: int | None = None
    scheduler_batch_size_cap: int | None = None
    scheduler_block_size: int | None = None
    scheduler_watermark_blocks_fraction: float | None = None


def run_vidur_paper_fidelity_sim(
    inputs: VidurPaperFidelitySimInputs,
    *,
    out_dir: Path,
    run_meta: dict,
) -> Path:
    """Run Vidur and write a paper-fidelity `request_metrics.csv`.

    Parameters
    ----------
    inputs
        Simulation inputs (scenario name, trace path, profiling root, and model/hardware settings).
    out_dir
        Output directory under which Vidur's raw metrics and the standardized paper-fidelity CSVs
        are written.
    run_meta
        Metadata dict to persist as `run_meta.json` alongside the outputs.

    Returns
    -------
    pathlib.Path
        Path to the written `request_metrics.csv` in paper-fidelity schema.
    """
    layout = ProfilingRootLayout(
        profiling_root=inputs.profiling_root,
        device=inputs.device,
        model_id=inputs.model_id,
        network_device=inputs.network_device,
        tensor_parallel_size=inputs.tensor_parallel_size,
        num_pipeline_stages=inputs.num_pipeline_stages,
        mlp_validation_mode=str(inputs.mlp_validation_mode),
        mlp_small_input_threshold=int(inputs.mlp_small_input_threshold),
        mlp_zero_heavy_limit=float(inputs.mlp_zero_heavy_limit),
        skip_cpu_overhead_modeling=bool(inputs.skip_cpu_overhead_modeling),
        cpu_overhead_validation=str(inputs.cpu_overhead_validation),
    )
    validate_profiling_root(layout)

    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = utcnow_iso()

    profiling_base = inputs.profiling_root / "data" / "profiling"

    from vidur.config import (
        ClusterConfig,
        MetricsConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        SarathiSchedulerConfig,
        SimulationConfig,
        TraceRequestGeneratorConfig,
    )
    from vidur.simulator import Simulator

    replica_config = ReplicaConfig(
        model_name=inputs.model_id,
        num_pipeline_stages=int(inputs.num_pipeline_stages),
        tensor_parallel_size=int(inputs.tensor_parallel_size),
        device=str(inputs.device),
        network_device=str(inputs.network_device),
    )
    scheduler_type = str(inputs.scheduler_type or "sarathi").lower()
    if scheduler_type != "sarathi":
        raise ValueError(
            f"Unsupported Vidur replica scheduler type for paper-fidelity sim: {scheduler_type!r}. "
            "Set scenario.vidur.scheduler.type=sarathi (or extend sim_runner.py to support more)."
        )

    replica_scheduler_config = SarathiSchedulerConfig()
    if inputs.scheduler_chunk_size is not None:
        replica_scheduler_config.chunk_size = int(inputs.scheduler_chunk_size)
    if inputs.scheduler_batch_size_cap is not None:
        replica_scheduler_config.batch_size_cap = int(inputs.scheduler_batch_size_cap)
    if inputs.scheduler_block_size is not None:
        replica_scheduler_config.block_size = int(inputs.scheduler_block_size)
    if inputs.scheduler_watermark_blocks_fraction is not None:
        replica_scheduler_config.watermark_blocks_fraction = float(inputs.scheduler_watermark_blocks_fraction)

    cluster_config = ClusterConfig(
        num_replicas=1,
        replica_config=replica_config,
        replica_scheduler_config=replica_scheduler_config,
    )
    request_generator_config = TraceRequestGeneratorConfig(
        trace_file=str(inputs.trace_csv),
        max_tokens=int(inputs.max_tokens),
    )

    exec_cfg = RandomForrestExecutionTimePredictorConfig(
        compute_input_file=str(profiling_base / "compute/{DEVICE}/{MODEL}/mlp.csv"),
        attention_input_file=str(profiling_base / "compute/{DEVICE}/{MODEL}/attention.csv"),
        all_reduce_input_file=str(profiling_base / "network/{NETWORK_DEVICE}/all_reduce.csv"),
        send_recv_input_file=str(profiling_base / "network/{NETWORK_DEVICE}/send_recv.csv"),
        cpu_overhead_input_file=str(profiling_base / "cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv"),
        skip_cpu_overhead_modeling=bool(inputs.skip_cpu_overhead_modeling),
    )

    metrics_cfg = MetricsConfig(
        write_metrics=True,
        enable_chrome_trace=False,
        store_plots=False,
        store_operation_metrics=False,
        store_token_completion_metrics=False,
        store_request_metrics=True,
        store_batch_metrics=False,
        store_utilization_metrics=False,
        output_dir=str(out_dir / "vidur_raw"),
        cache_dir=str(_default_vidur_cache_dir(out_dir=out_dir).resolve()),
    )

    sim_cfg = SimulationConfig(
        seed=int(inputs.seed),
        cluster_config=cluster_config,
        request_generator_config=request_generator_config,
        execution_time_predictor_config=exec_cfg,
        metrics_config=metrics_cfg,
    )

    simulator = Simulator(sim_cfg)
    simulator.run()
    simulator.metric_store.plot()

    raw_dir = Path(sim_cfg.metrics_config.output_dir)
    raw_request_metrics = raw_dir / "request_metrics.csv"
    if not raw_request_metrics.exists():
        raise FileNotFoundError(f"Vidur did not produce request_metrics.csv under {raw_dir}")

    request_df = convert_vidur_request_metrics_to_paper_fidelity(raw_request_metrics)
    _validate_vidur_decode_token_counts(
        trace_csv=inputs.trace_csv,
        request_df=request_df,
        max_tokens=int(inputs.max_tokens),
    )
    write_csv(
        out_dir / "request_metrics.csv",
        request_df,
        required_columns=["request_id", *PAPER_FIDELITY_REQUIRED_VIDUR_COLUMNS],
    )

    meta = dict(run_meta)
    meta.setdefault("schema_version", "v1")
    meta.setdefault("run_type", "sim")
    meta.setdefault("scenario_name", inputs.scenario_name)
    meta.setdefault("started_at", started_at)
    meta.setdefault("ended_at", utcnow_iso())
    meta.setdefault("vidur_raw_dir", str(raw_dir.resolve()))
    meta.setdefault("trace_csv", str(inputs.trace_csv.resolve()))
    write_json(out_dir / "run_meta.json", meta)

    return out_dir / "request_metrics.csv"
