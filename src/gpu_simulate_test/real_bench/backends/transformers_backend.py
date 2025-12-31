from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from gpu_simulate_test.real_bench.backends.base import TokenEvent


@dataclass
class _TimingStreamer:
    token_ids: list[int]
    token_times_ns: list[int]

    def put(self, value) -> None:  # transformers calls this with token ids
        try:
            import torch  # type: ignore

            if isinstance(value, torch.Tensor):
                ids = value.detach().cpu().flatten().tolist()
            else:
                ids = [int(value)]
        except Exception:
            # Best-effort fallback
            ids = [int(x) for x in value] if isinstance(value, (list, tuple)) else [int(value)]

        now_ns = time.monotonic_ns()
        for token_id in ids:
            self.token_ids.append(int(token_id))
            self.token_times_ns.append(int(now_ns))

    def end(self) -> None:
        return


class TransformersBackend:
    def __init__(self, *, model_ref: Path, device: str) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "transformers + torch are required; run inside the Pixi env (`pixi install`)."
            ) from e

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"device={device} requested but torch.cuda.is_available() is False; "
                "run on an NVIDIA machine with a working driver/runtime."
            )

        if not model_ref.exists():
            raise FileNotFoundError(f"model_ref does not exist: {model_ref}")

        self._torch = torch
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_ref),
            use_fast=True,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_ref),
            torch_dtype="auto",
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

        if getattr(self._tokenizer, "pad_token_id", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def warmup(self) -> None:
        _ = self.run_request(prompt="warmup", max_new_tokens=1)

    def run_request(self, *, prompt: str, max_new_tokens: int) -> list[TokenEvent]:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)

        streamer = _TimingStreamer(token_ids=[], token_times_ns=[])

        if self._device.type == "cuda":
            self._torch.cuda.synchronize()

        with self._torch.no_grad():
            _ = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                streamer=streamer,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        if self._device.type == "cuda":
            self._torch.cuda.synchronize()

        events: list[TokenEvent] = []
        for idx, (token_id, t_ns) in enumerate(zip(streamer.token_ids, streamer.token_times_ns)):
            events.append(TokenEvent(token_index=int(idx), token_time_ns=int(t_ns), token_id=int(token_id)))
        return events

