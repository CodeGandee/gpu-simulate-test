from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

from gpu_simulate_test.workloads.prompts import PromptRecord


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class TraceLengthsConfig:
    num_decode_tokens: int


def load_hf_tokenizer(tokenizer_ref: Path):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for HuggingFace tokenizer loading; "
            "run inside the Pixi env (`pixi install`)."
        ) from e

    if not tokenizer_ref.exists():
        raise FileNotFoundError(f"tokenizer_ref does not exist: {tokenizer_ref}")

    return AutoTokenizer.from_pretrained(
        str(tokenizer_ref),
        use_fast=True,
        trust_remote_code=True,
    )


def compute_trace_lengths(
    prompts: Sequence[PromptRecord],
    *,
    tokenizer: Tokenizer,
    config: TraceLengthsConfig,
) -> pd.DataFrame:
    if config.num_decode_tokens < 0:
        raise ValueError(f"num_decode_tokens must be >= 0, got {config.num_decode_tokens}")

    rows = []
    for request_id, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt.text)
        rows.append(
            {
                "request_id": int(request_id),
                "prompt_id": prompt.prompt_id,
                "num_prefill_tokens": int(max(0, len(token_ids))),
                "num_decode_tokens": int(config.num_decode_tokens),
            }
        )
    return pd.DataFrame(rows)

