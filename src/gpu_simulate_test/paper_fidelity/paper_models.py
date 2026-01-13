from __future__ import annotations

PAPER_MODEL_SCENARIOS: list[str] = [
    "internlm_20b_arxiv",
    "llama2_70b_arxiv",
    "qwen_72b_arxiv",
]

EXCLUDED_SCENARIOS: set[str] = {
    "qwen3_0.6b_arxiv",
    "qwen3_0_6b_arxiv",
    "qwen3_0.6b",
}


def validate_paper_model_scenarios(scenarios: list[str]) -> None:
    bad = sorted(set(scenarios) & EXCLUDED_SCENARIOS)
    if bad:
        raise ValueError(f"Excluded from paper-model matrix: {bad}")

