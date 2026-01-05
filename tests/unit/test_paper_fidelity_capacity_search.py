from __future__ import annotations

import pandas as pd

from gpu_simulate_test.paper_fidelity.capacity import CapacityCriterion, discover_capacity


def test_capacity_search_binary_search_finds_threshold_crossing() -> None:
    criterion = CapacityCriterion(metric="request_scheduling_delay", quantile=0.99, threshold_s=5.0)

    def run_at_qps(qps: float) -> pd.DataFrame:
        # Synthetic: overload once qps exceeds 10 (P99 scheduling delay jumps above threshold).
        delay = 1.0 if qps <= 10.0 else 10.0
        return pd.DataFrame({"request_scheduling_delay": [delay] * 200})

    result = discover_capacity(
        run_at_qps=run_at_qps,
        min_qps=0.0,
        max_qps=20.0,
        max_iters=6,
        criterion=criterion,
        operating_point_fraction=0.85,
    )

    assert 9.9 <= result.capacity_qps <= 10.1
    assert result.qps_85 == 0.85 * result.capacity_qps

