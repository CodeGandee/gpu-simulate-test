from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import read_csv, utcnow_iso, write_csv, write_json
from gpu_simulate_test.vidur_ext.profiling_root import ProfilingRootLayout, validate_profiling_root


@dataclass(frozen=True)
class VidurSimInputs:
    workload_dir: Path
    profiling_root: Path
    model_id: str
    device: str = "a100"
    network_device: str = "a100_pairwise_nvlink"
    tensor_parallel_size: int = 1
    num_pipeline_stages: int = 1
    seed: int = 42


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
        skip_cpu_overhead_modeling=True,
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
    cluster_config = ClusterConfig(num_replicas=1, replica_config=replica_config)
    request_generator_config = TraceRequestGeneratorConfig(trace_file=str(trace_csv), max_tokens=4096)

    exec_cfg = RandomForrestExecutionTimePredictorConfig(
        compute_input_file=str(profiling_base / "compute/{DEVICE}/{MODEL}/mlp.csv"),
        attention_input_file=str(profiling_base / "compute/{DEVICE}/{MODEL}/attention.csv"),
        all_reduce_input_file=str(profiling_base / "network/{NETWORK_DEVICE}/all_reduce.csv"),
        send_recv_input_file=str(profiling_base / "network/{NETWORK_DEVICE}/send_recv.csv"),
        cpu_overhead_input_file=str(profiling_base / "cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv"),
        skip_cpu_overhead_modeling=True,
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
