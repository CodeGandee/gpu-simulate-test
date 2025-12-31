from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import write_csv
from gpu_simulate_test.workloads.arrival_schedule import ArrivalScheduleConfig, build_trace_intervals


def _csv_bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_trace_intervals_deterministic_bytes(tmp_path: Path) -> None:
    out_csv = tmp_path / "trace_intervals.csv"

    cfg = ArrivalScheduleConfig(kind="poisson", seed=123, poisson_rate_per_s=2.0)
    df1 = build_trace_intervals(8, config=cfg)
    write_csv(
        out_csv,
        df1,
        required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"],
    )
    b1 = _csv_bytes(out_csv)

    df2 = build_trace_intervals(8, config=cfg)
    write_csv(
        out_csv,
        df2,
        required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"],
    )
    b2 = _csv_bytes(out_csv)

    assert b1 == b2


def test_trace_intervals_required_columns_and_invariants() -> None:
    cfg = ArrivalScheduleConfig(kind="fixed_interval", seed=0, inter_arrival_ns=10)
    df = build_trace_intervals(5, config=cfg)

    assert set(["request_id", "inter_arrival_ns", "arrival_time_ns"]).issubset(df.columns)
    assert df["request_id"].tolist() == [0, 1, 2, 3, 4]

    inter = df["inter_arrival_ns"].astype(int).tolist()
    arrival = df["arrival_time_ns"].astype(int).tolist()

    # arrival_time_ns == cumsum(inter_arrival_ns)
    cumsum = []
    s = 0
    for x in inter:
        s += x
        cumsum.append(s)
    assert arrival == cumsum

