from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from gpu_simulate_test.io import read_csv
from gpu_simulate_test.workloads.prompts import read_prompts_jsonl


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt_id: str
    prompt_text: str
    arrival_time_ns: int
    num_prefill_tokens: int
    num_decode_tokens: int


def load_workload_requests(workload_dir: Path) -> list[RequestSpec]:
    prompts_path = workload_dir / "prompts.jsonl"
    lengths_path = workload_dir / "trace_lengths.csv"
    intervals_path = workload_dir / "trace_intervals.csv"

    prompts = read_prompts_jsonl(prompts_path)
    prompt_text_by_id = {p.prompt_id: p.text for p in prompts}

    lengths = read_csv(
        lengths_path,
        required_columns=["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"],
        context="trace_lengths",
    )
    intervals = read_csv(
        intervals_path,
        required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"],
        context="trace_intervals",
    )

    merged = pd.merge(lengths, intervals, on=["request_id"], how="inner")
    merged = merged.sort_values("request_id").reset_index(drop=True)

    specs: list[RequestSpec] = []
    for row in merged.to_dict(orient="records"):
        prompt_id = str(row["prompt_id"])
        if prompt_id not in prompt_text_by_id:
            raise ValueError(f"prompt_id {prompt_id!r} missing from prompts.jsonl")
        specs.append(
            RequestSpec(
                request_id=int(row["request_id"]),
                prompt_id=prompt_id,
                prompt_text=prompt_text_by_id[prompt_id],
                arrival_time_ns=int(row["arrival_time_ns"]),
                num_prefill_tokens=int(row["num_prefill_tokens"]),
                num_decode_tokens=int(row["num_decode_tokens"]),
            )
        )
    return specs


def replay_schedule(requests: Iterable[RequestSpec]) -> Iterator[RequestSpec]:
    start_ns = time.monotonic_ns()
    for req in sorted(requests, key=lambda r: r.arrival_time_ns):
        target_ns = start_ns + int(req.arrival_time_ns)
        while True:
            now_ns = time.monotonic_ns()
            remaining = target_ns - now_ns
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining / 1e9))
        yield req

