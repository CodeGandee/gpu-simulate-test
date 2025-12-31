from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TokenEvent:
    token_index: int
    token_time_ns: int
    token_id: int | None = None


class RealBackend(Protocol):
    def warmup(self) -> None: ...

    def run_request(self, *, prompt: str, max_new_tokens: int) -> list[TokenEvent]: ...

