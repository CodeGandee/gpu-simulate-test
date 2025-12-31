from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArrivalScheduleConfig:
    kind: Literal["fixed_interval", "poisson"]
    seed: int
    inter_arrival_ns: int = 0
    poisson_rate_per_s: float = 1.0


def build_trace_intervals(num_requests: int, *, config: ArrivalScheduleConfig) -> pd.DataFrame:
    if num_requests < 0:
        raise ValueError(f"num_requests must be >= 0, got {num_requests}")

    request_id = np.arange(num_requests, dtype=np.int64)
    inter_arrival_ns = np.zeros(num_requests, dtype=np.int64)

    if num_requests > 1:
        if config.kind == "fixed_interval":
            if config.inter_arrival_ns < 0:
                raise ValueError(
                    f"inter_arrival_ns must be >= 0, got {config.inter_arrival_ns}"
                )
            inter_arrival_ns[1:] = int(config.inter_arrival_ns)
        elif config.kind == "poisson":
            if config.poisson_rate_per_s <= 0:
                raise ValueError(
                    f"poisson_rate_per_s must be > 0, got {config.poisson_rate_per_s}"
                )
            rng = np.random.default_rng(int(config.seed))
            samples_s = rng.exponential(scale=1.0 / float(config.poisson_rate_per_s), size=num_requests - 1)
            samples_ns = np.maximum(0, np.round(samples_s * 1e9).astype(np.int64))
            inter_arrival_ns[1:] = samples_ns
        else:
            raise ValueError(f"Unknown schedule kind: {config.kind}")

    arrival_time_ns = np.cumsum(inter_arrival_ns, dtype=np.int64)

    return pd.DataFrame(
        {
            "request_id": request_id,
            "inter_arrival_ns": inter_arrival_ns,
            "arrival_time_ns": arrival_time_ns,
        }
    )

