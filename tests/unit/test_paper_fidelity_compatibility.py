from __future__ import annotations

import pandas as pd
import pytest

from gpu_simulate_test.paper_fidelity.scoring import validate_sim_vs_real_compatibility


def _metrics_df(*, request_ids: list[object], decode_tokens: list[int]) -> pd.DataFrame:
    n = len(request_ids)
    return pd.DataFrame(
        {
            "request_id": request_ids,
            "request_scheduling_delay": [0.0] * n,
            "request_execution_plus_preemption_time_normalized": [1.0] * n,
            "request_e2e_time_normalized": [1.0] * n,
            "request_num_decode_tokens": decode_tokens,
        }
    )


def test_validate_sim_vs_real_compatibility_accepts_sarathi_prefixed_ids() -> None:
    sim = _metrics_df(request_ids=[0, 1, 2], decode_tokens=[4, 4, 4])
    real = _metrics_df(request_ids=["0_0", "0_1", "0_2"], decode_tokens=[4, 4, 4])
    validate_sim_vs_real_compatibility(sim_df=sim, real_df=real)


def test_validate_sim_vs_real_compatibility_rejects_id_mismatch() -> None:
    sim = _metrics_df(request_ids=[0, 1, 2], decode_tokens=[4, 4, 4])
    real = _metrics_df(request_ids=[0, 1], decode_tokens=[4, 4])
    with pytest.raises(ValueError, match="request id sets differ"):
        validate_sim_vs_real_compatibility(sim_df=sim, real_df=real)


def test_validate_sim_vs_real_compatibility_rejects_decode_token_mismatch() -> None:
    sim = _metrics_df(request_ids=[0, 1, 2], decode_tokens=[4, 4, 4])
    real = _metrics_df(request_ids=[0, 1, 2], decode_tokens=[4, 4, 5])
    with pytest.raises(ValueError, match="request_num_decode_tokens mismatch"):
        validate_sim_vs_real_compatibility(sim_df=sim, real_df=real)

