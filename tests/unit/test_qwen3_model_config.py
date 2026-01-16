from __future__ import annotations

from pathlib import Path

import pytest

from gpu_simulate_test.vidur_ext.qwen3_model_config import QWEN3_0_6B_MODEL_ID, maybe_register_qwen3_0_6b


def test_maybe_register_qwen3_0_6b_noop_for_other_models(tmp_path: Path) -> None:
    assert maybe_register_qwen3_0_6b(model_id="meta-llama/Llama-2-7b-hf", tokenizer_ref=tmp_path) is False


def test_maybe_register_qwen3_0_6b_requires_config_when_selected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing Qwen3 config.json"):
        maybe_register_qwen3_0_6b(model_id=QWEN3_0_6B_MODEL_ID, tokenizer_ref=tmp_path)

