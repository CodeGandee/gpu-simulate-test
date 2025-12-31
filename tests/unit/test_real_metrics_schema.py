from __future__ import annotations

import pandas as pd

from gpu_simulate_test.real_bench.backends.base import TokenEvent
from gpu_simulate_test.real_bench.metrics import build_metrics_frames


def test_build_metrics_frames_schema_and_invariants() -> None:
    token_events = [
        TokenEvent(token_index=0, token_time_ns=110),
        TokenEvent(token_index=1, token_time_ns=210),
        TokenEvent(token_index=2, token_time_ns=260),
    ]

    request_df, token_df = build_metrics_frames(
        request_id=7,
        arrival_time_ns=100,
        token_events=token_events,
        num_prefill_tokens=12,
        num_decode_tokens=3,
        backend="transformers",
    )

    assert isinstance(request_df, pd.DataFrame)
    assert isinstance(token_df, pd.DataFrame)

    for c in [
        "request_id",
        "arrival_time_ns",
        "first_token_time_ns",
        "ttft_ns",
        "num_prefill_tokens",
        "num_decode_tokens",
        "num_decode_tokens_actual",
        "status",
    ]:
        assert c in request_df.columns

    for c in ["request_id", "token_index", "token_time_ns", "token_latency_ns"]:
        assert c in token_df.columns

    row = request_df.iloc[0].to_dict()
    assert row["request_id"] == 7
    assert row["arrival_time_ns"] == 100
    assert row["first_token_time_ns"] == 110
    assert row["ttft_ns"] == 10
    assert row["num_decode_tokens"] == 3
    assert row["num_decode_tokens_actual"] == 3
    assert row["status"] == "ok"

    assert token_df["token_index"].tolist() == [0, 1, 2]
    assert token_df["token_time_ns"].tolist() == [110, 210, 260]
    assert token_df["token_latency_ns"].tolist() == [0, 100, 50]

