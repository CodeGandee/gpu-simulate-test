from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from vidur.config.model_config import BaseModelConfig
from vidur.types import ActivationType, NormType


QWEN3_0_6B_MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class Qwen3ModelRef:
    config_json: Path


@dataclass
class Qwen3_0_6BModelConfig(BaseModelConfig):
    num_layers: int = 28
    num_q_heads: int = 16
    num_kv_heads: int = 8
    embedding_dim: int = 1024
    mlp_hidden_dim: int = 3072
    max_position_embeddings: int = 40960
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 151936
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 1000000
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return QWEN3_0_6B_MODEL_ID


def register_qwen3_0_6b(*, model_ref: Qwen3ModelRef) -> None:
    if not model_ref.config_json.exists():
        raise FileNotFoundError(f"Missing Qwen3 config.json: {model_ref.config_json}")

    cfg = json.loads(model_ref.config_json.read_text(encoding="utf-8"))
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": Qwen3_0_6BModelConfig.num_layers,
        "num_attention_heads": Qwen3_0_6BModelConfig.num_q_heads,
        "num_key_value_heads": Qwen3_0_6BModelConfig.num_kv_heads,
        "hidden_size": Qwen3_0_6BModelConfig.embedding_dim,
        "intermediate_size": Qwen3_0_6BModelConfig.mlp_hidden_dim,
        "max_position_embeddings": Qwen3_0_6BModelConfig.max_position_embeddings,
        "vocab_size": Qwen3_0_6BModelConfig.vocab_size,
    }
    mismatches = {
        k: {"expected": v, "actual": cfg.get(k)}
        for k, v in expected.items()
        if cfg.get(k) != v
    }
    if mismatches:
        raise ValueError(f"Qwen3 config.json mismatch: {mismatches}")

    # Import side-effect is sufficient: BaseModelConfig.create_from_name uses subclass discovery.
    _ = Qwen3_0_6BModelConfig


def maybe_register_qwen3_0_6b(*, model_id: str, tokenizer_ref: Path) -> bool:
    """Register the Qwen3-0.6B model config if the requested model matches.

    Parameters
    ----------
    model_id
        HuggingFace model id (e.g., `meta-llama/Llama-2-7b-hf`).
    tokenizer_ref
        Path to the model's tokenizer/config directory (must contain `config.json` for Qwen3).

    Returns
    -------
    bool
        True if the Qwen3 config was registered, otherwise False.
    """
    if str(model_id) != QWEN3_0_6B_MODEL_ID:
        return False
    register_qwen3_0_6b(model_ref=Qwen3ModelRef(config_json=tokenizer_ref / "config.json"))
    return True
