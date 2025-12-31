from __future__ import annotations

import pandas as pd

from gpu_simulate_test.analysis.compare import align_tokens_by_actual_decode
from gpu_simulate_test.analysis.load_metrics import RunMetrics


def test_align_tokens_truncates_sim_by_real_actual() -> None:
    real = RunMetrics(
        run_dir=None,  # type: ignore[arg-type]
        request_metrics=pd.DataFrame(
            {
                "request_id": [0, 1],
                "num_decode_tokens_actual": [2, 1],
            }
        ),
        token_metrics=pd.DataFrame(
            {
                "request_id": [0, 0, 1],
                "token_index": [0, 1, 0],
                "token_time_ns": [10, 20, 15],
                "token_latency_ns": [0, 10, 0],
            }
        ),
        run_meta={},
    )
    sim = RunMetrics(
        run_dir=None,  # type: ignore[arg-type]
        request_metrics=pd.DataFrame(
            {
                "request_id": [0, 1],
                "num_decode_tokens_actual": [5, 5],
            }
        ),
        token_metrics=pd.DataFrame(
            {
                "request_id": [0, 0, 0, 1, 1],
                "token_index": [0, 1, 2, 0, 1],
                "token_time_ns": [11, 21, 31, 16, 26],
                "token_latency_ns": [0, 10, 10, 0, 10],
            }
        ),
        run_meta={},
    )

    real_tok, sim_tok = align_tokens_by_actual_decode(real=real, sim=sim)
    assert real_tok["request_id"].tolist() == [0, 0, 1]
    assert sim_tok["request_id"].tolist() == [0, 0, 1]
    assert sim_tok["token_index"].tolist() == [0, 1, 0]

