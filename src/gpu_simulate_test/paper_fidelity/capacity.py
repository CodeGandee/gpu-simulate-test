"""
Capacity discovery utilities for paper-fidelity dynamic workloads.

Capacity is defined as the maximum QPS that is not overloaded under an overload criterion
(default: overloaded if P99(request_scheduling_delay) > 5s). The dynamic operating point is
`qps_85 = 0.85 * capacity_qps` by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from gpu_simulate_test.io import write_json


@dataclass(frozen=True)
class CapacityCriterion:
    metric: str = "request_scheduling_delay"
    quantile: float = 0.99
    threshold_s: float = 5.0


@dataclass(frozen=True)
class CapacityResult:
    capacity_qps: float
    qps_85: float
    criterion: CapacityCriterion

    def to_dict(self) -> dict:
        return {"capacity_qps": self.capacity_qps, "qps_85": self.qps_85, "criterion": asdict(self.criterion)}


def is_overloaded(df: pd.DataFrame, *, criterion: CapacityCriterion) -> bool:
    """Return True if df violates the overload criterion."""
    if criterion.metric not in df.columns:
        raise ValueError(f"missing required metric column: {criterion.metric}")

    value = float(df[criterion.metric].astype(float).quantile(float(criterion.quantile)))
    return value > float(criterion.threshold_s)


def discover_capacity(
    *,
    run_at_qps: Callable[[float], pd.DataFrame],
    min_qps: float,
    max_qps: float,
    max_iters: int,
    criterion: CapacityCriterion,
    operating_point_fraction: float = 0.85,
) -> CapacityResult:
    """Binary search the max QPS that is not overloaded."""
    if min_qps < 0 or max_qps < 0:
        raise ValueError("min_qps and max_qps must be >= 0")
    if max_qps < min_qps:
        raise ValueError("max_qps must be >= min_qps")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    if not (0.0 < operating_point_fraction <= 1.0):
        raise ValueError("operating_point_fraction must be in (0, 1]")

    lo = float(min_qps)
    hi = float(max_qps)

    for _ in range(int(max_iters)):
        mid = (lo + hi) / 2.0
        df = run_at_qps(mid)
        if is_overloaded(df, criterion=criterion):
            hi = mid
        else:
            lo = mid

    capacity_qps = lo
    return CapacityResult(
        capacity_qps=capacity_qps,
        qps_85=float(operating_point_fraction) * capacity_qps,
        criterion=criterion,
    )


def write_capacity_json(path: Path, *, result: CapacityResult) -> None:
    """Write `capacity.json` containing capacity QPS and derived 85% operating point."""
    write_json(path, result.to_dict())

