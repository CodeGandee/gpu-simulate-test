from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str


def read_prompts_jsonl(path: Path) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{idx + 1}: invalid JSON: {e}") from e

            if "prompt_id" not in obj or "text" not in obj:
                raise ValueError(f"{path}:{idx + 1}: requires 'prompt_id' and 'text'")
            prompts.append(PromptRecord(prompt_id=str(obj["prompt_id"]), text=str(obj["text"])))
    return prompts


def write_prompts_jsonl(path: Path, prompts: Iterable[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt_id": p.prompt_id, "text": p.text}, ensure_ascii=False, sort_keys=True))
            f.write("\n")

