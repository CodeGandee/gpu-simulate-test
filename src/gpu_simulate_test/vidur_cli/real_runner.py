"""
Token-length replay runner for `vidur-cli svr real`.

This module replays a canonical token-length trace (`trace/trace.csv`) against a
real backend without requiring the original prompt dataset.

Backends
--------
- `transformers`: HuggingFace model.generate with synthetic `input_ids`
- `sarathi`: Sarathi-Serve engine with `prompt_token_ids`
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import assert_columns, utcnow_iso
from gpu_simulate_test.real_bench.backends.base import TokenEvent
from gpu_simulate_test.real_bench.metrics import build_metrics_frames, write_run_outputs


TRACE_REQUIRED_COLUMNS = ["request_id", "arrival_time_ns", "num_prefill_tokens", "num_decode_tokens"]


@dataclass(frozen=True)
class RealReplayResult:
    out_dir: Path


def run_token_length_replay(
    *,
    trace_csv: Path,
    backend: str,
    model_id: str,
    model_ref: Path | None,
    device: str,
    out_dir: Path,
    sarathi_chunk_size: int | None = None,
    sarathi_max_num_seqs: int | None = None,
    sarathi_max_tokens: int | None = None,
    sarathi_ignore_eos: bool | None = None,
) -> RealReplayResult:
    """Replay a token-length trace and write metrics under `out_dir`."""
    trace_csv = trace_csv.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    trace = pd.read_csv(trace_csv)
    assert_columns(trace, TRACE_REQUIRED_COLUMNS, context=str(trace_csv))
    trace = trace.copy()
    trace["request_id"] = pd.to_numeric(trace["request_id"], errors="raise").astype("int64")
    trace["arrival_time_ns"] = pd.to_numeric(trace["arrival_time_ns"], errors="raise").astype("int64")
    trace["num_prefill_tokens"] = pd.to_numeric(trace["num_prefill_tokens"], errors="raise").astype("int64")
    trace["num_decode_tokens"] = pd.to_numeric(trace["num_decode_tokens"], errors="raise").astype("int64")

    trace = trace.sort_values(["arrival_time_ns", "request_id"]).reset_index(drop=True)

    if backend == "transformers":
        impl = _TransformersTokenLengthBackend(model_ref=_require_model_ref(model_ref), device=device)
        impl.warmup()
        return _run_sequential_token_length_replay(
            trace=trace,
            impl=impl,
            out_dir=out_dir,
            trace_csv=trace_csv,
            backend=backend,
            model_id=model_id,
        )

    if backend == "sarathi":
        return _run_sarathi_token_length_replay(
            trace=trace,
            trace_csv=trace_csv,
            model_id=model_id,
            model_ref=_require_model_ref(model_ref),
            out_dir=out_dir,
            chunk_size=int(sarathi_chunk_size or 16),
            max_num_seqs=int(sarathi_max_num_seqs or 16),
            max_tokens=int(sarathi_max_tokens or 4096),
            ignore_eos=True if sarathi_ignore_eos is None else bool(sarathi_ignore_eos),
        )

    raise ValueError(f"Unknown backend: {backend}")


def _require_model_ref(model_ref: Path | None) -> Path:
    if model_ref is None:
        raise ValueError("model_ref is required for backend=transformers")
    p = model_ref.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"model_ref does not exist: {p}")
    return p


class _TokenLengthBackend:
    def warmup(self) -> None: ...

    def run_request(self, *, prefill_tokens: int, max_new_tokens: int) -> list[TokenEvent]: ...


@dataclass
class _TimingStreamer:
    token_ids: list[int]
    token_times_ns: list[int]
    skip_prompt_tokens: int
    skipped_prompt_tokens: int = 0

    def put(self, value) -> None:  # transformers calls this with token ids
        try:
            import torch  # type: ignore

            if isinstance(value, torch.Tensor):
                ids = value.detach().cpu().flatten().tolist()
            else:
                ids = [int(value)]
        except Exception:
            ids = [int(x) for x in value] if isinstance(value, (list, tuple)) else [int(value)]

        remaining = int(self.skip_prompt_tokens) - int(self.skipped_prompt_tokens)
        if remaining > 0:
            drop = min(len(ids), remaining)
            self.skipped_prompt_tokens += int(drop)
            ids = ids[drop:]

        if not ids:
            return

        now_ns = time.monotonic_ns()
        for token_id in ids:
            self.token_ids.append(int(token_id))
            self.token_times_ns.append(int(now_ns))

    def end(self) -> None:
        return


class _TransformersTokenLengthBackend:
    def __init__(self, *, model_ref: Path, device: str) -> None:
        if device.startswith("cuda"):
            from gpu_simulate_test.env_guard import apply_cuda_visible_devices_from_gsim

            apply_cuda_visible_devices_from_gsim()

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

        self._fill_token_id = int(getattr(self._tokenizer, "eos_token_id", 0) or 0)

    def warmup(self) -> None:
        _ = self.run_request(prefill_tokens=1, max_new_tokens=1)

    def run_request(self, *, prefill_tokens: int, max_new_tokens: int) -> list[TokenEvent]:
        if prefill_tokens < 1:
            raise ValueError(f"prefill_tokens must be >= 1, got {prefill_tokens}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        input_ids = self._torch.full(
            (1, int(prefill_tokens)),
            fill_value=int(self._fill_token_id),
            dtype=self._torch.long,
            device=self._device,
        )

        streamer = _TimingStreamer(
            token_ids=[],
            token_times_ns=[],
            skip_prompt_tokens=int(prefill_tokens),
        )

        if self._device.type == "cuda":
            self._torch.cuda.synchronize()

        with self._torch.no_grad():
            eos_id = int(getattr(self._tokenizer, "eos_token_id", None) or self._fill_token_id or 0)
            _ = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                min_new_tokens=int(max_new_tokens),
                do_sample=False,
                streamer=streamer,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
            )

        if self._device.type == "cuda":
            self._torch.cuda.synchronize()

        events: list[TokenEvent] = []
        for idx, (token_id, t_ns) in enumerate(zip(streamer.token_ids, streamer.token_times_ns)):
            events.append(TokenEvent(token_index=int(idx), token_time_ns=int(t_ns), token_id=int(token_id)))
        return events


def _run_sequential_token_length_replay(
    *,
    trace: pd.DataFrame,
    impl: _TokenLengthBackend,
    out_dir: Path,
    trace_csv: Path,
    backend: str,
    model_id: str,
) -> RealReplayResult:
    run_start_ns = time.monotonic_ns()
    started_at = utcnow_iso()

    request_frames: list[pd.DataFrame] = []
    token_frames: list[pd.DataFrame] = []

    for row in trace.to_dict(orient="records"):
        request_id = int(row["request_id"])
        arrival_time_ns = int(row["arrival_time_ns"])
        prefill = int(row["num_prefill_tokens"])
        decode = int(row["num_decode_tokens"])

        target_ns = run_start_ns + arrival_time_ns
        while True:
            now_ns = time.monotonic_ns()
            remaining = target_ns - now_ns
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining / 1e9))

        token_events_abs = impl.run_request(prefill_tokens=prefill, max_new_tokens=decode)
        token_events_rel = [
            TokenEvent(
                token_index=int(ev.token_index),
                token_time_ns=int(ev.token_time_ns) - run_start_ns,
                token_id=ev.token_id,
            )
            for ev in token_events_abs
        ]

        req_df, tok_df = build_metrics_frames(
            request_id=request_id,
            arrival_time_ns=arrival_time_ns,
            token_events=token_events_rel,
            num_prefill_tokens=prefill,
            num_decode_tokens=decode,
            backend=str(backend),
        )
        request_frames.append(req_df)
        token_frames.append(tok_df)

    request_df = pd.concat(request_frames, ignore_index=True) if request_frames else pd.DataFrame()
    token_df = pd.concat(token_frames, ignore_index=True) if token_frames else pd.DataFrame()

    run_meta = {
        "schema_version": "v1",
        "run_type": "real",
        "backend": str(backend),
        "model": str(model_id),
        "trace_csv": str(trace_csv),
        "started_at": started_at,
        "ended_at": utcnow_iso(),
    }
    write_run_outputs(out_dir, request_df=request_df, token_df=token_df, run_meta=run_meta)
    return RealReplayResult(out_dir=out_dir.resolve())


def _run_sarathi_token_length_replay(
    *,
    trace: pd.DataFrame,
    trace_csv: Path,
    model_id: str,
    model_ref: Path,
    out_dir: Path,
    chunk_size: int,
    max_num_seqs: int,
    max_tokens: int,
    ignore_eos: bool,
) -> RealReplayResult:
    from gpu_simulate_test.env_guard import apply_cuda_visible_devices_from_gsim, patch_sarathi_preserve_cuda_visible_devices

    apply_cuda_visible_devices_from_gsim()
    patch_sarathi_preserve_cuda_visible_devices()

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for the Sarathi token-length runner; run inside the Pixi env.") from e
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("CUDA is required for the Sarathi token-length runner (torch.cuda.is_available() is False).")

    try:
        from sarathi import LLMEngine, SamplingParams  # type: ignore
        from sarathi.config import (  # type: ignore
            MetricsConfig,
            ModelConfig,
            ParallelConfig,
            ReplicaConfig,
            SarathiSchedulerConfig,
            SystemConfig,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "sarathi is required for the Sarathi backend; run inside the Pixi env "
            "and ensure `extern/tracked/sarathi-serve` is initialized."
        ) from e

    if not model_ref.exists():
        raise FileNotFoundError(f"model_ref does not exist: {model_ref}")

    out_dir.mkdir(parents=True, exist_ok=True)

    engine = LLMEngine.from_system_config(
        SystemConfig(
            replica_config=ReplicaConfig(output_dir=str(out_dir / "sarathi")),
            model_config=ModelConfig(model=str(model_ref)),
            parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
            scheduler_config=SarathiSchedulerConfig(chunk_size=int(chunk_size), max_num_seqs=int(max_num_seqs)),
            metrics_config=MetricsConfig(
                write_metrics=True,
                enable_chrome_trace=False,
                enable_op_level_metrics=False,
                enable_cpu_op_level_metrics=False,
                keep_individual_batch_metrics=False,
                enable_request_outputs=False,
            ),
        )
    )

    prompt_cache: dict[int, list[int]] = {}

    def _prompt_token_ids(n: int) -> list[int]:
        cached = prompt_cache.get(n)
        if cached is None:
            cached = [0] * n
            prompt_cache[n] = cached
        return cached

    warmup_prefill = max(1, min(int(chunk_size), int(max_tokens) - 1))
    warmup_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=bool(ignore_eos),
    )
    engine.add_request(
        prompt=None,
        prompt_token_ids=_prompt_token_ids(warmup_prefill),
        sampling_params=warmup_sampling_params,
        arrival_time=time.monotonic(),
        seq_id="warmup",
    )
    while engine.has_unfinished_requests():
        engine.step()

    engine.reset_metrics()

    run_start_s = time.monotonic()
    run_start_ns = time.monotonic_ns()
    started_at = utcnow_iso()

    @dataclass
    class _SeqState:
        request_id: int
        arrival_time_ns: int
        num_prefill_tokens: int
        num_decode_tokens: int
        prev_len: int
        token_events_abs: list[TokenEvent]

    next_idx = 0
    seq_states: dict[str, _SeqState] = {}

    trace_records = trace.to_dict(orient="records")
    while next_idx < len(trace_records) or engine.has_unfinished_requests():
        now_ns = time.monotonic_ns()

        while next_idx < len(trace_records):
            row = trace_records[next_idx]
            arrival_time_ns = int(row["arrival_time_ns"])
            if now_ns < run_start_ns + arrival_time_ns:
                break

            request_id = int(row["request_id"])
            prefill = int(row["num_prefill_tokens"])
            decode = int(row["num_decode_tokens"])

            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=int(decode),
                ignore_eos=bool(ignore_eos),
            )
            seq_id = str(request_id)
            engine.add_request(
                prompt=None,
                prompt_token_ids=_prompt_token_ids(int(prefill)),
                sampling_params=sampling_params,
                seq_id=seq_id,
                arrival_time=run_start_s + (arrival_time_ns / 1e9),
            )
            seq_states[seq_id] = _SeqState(
                request_id=request_id,
                arrival_time_ns=arrival_time_ns,
                num_prefill_tokens=prefill,
                num_decode_tokens=decode,
                prev_len=0,
                token_events_abs=[],
            )
            next_idx += 1

        if not engine.has_unfinished_requests():
            if next_idx < len(trace_records):
                next_arrival_ns = int(trace_records[next_idx]["arrival_time_ns"])
                target_ns = run_start_ns + next_arrival_ns
                remaining = target_ns - now_ns
                if remaining > 0:
                    time.sleep(min(0.05, remaining / 1e9))
            continue

        step_outputs = engine.step()
        token_time_ns = time.monotonic_ns()

        for out in step_outputs:
            seq_id = getattr(out, "seq_id", None)
            if seq_id is None or seq_id not in seq_states:
                continue
            st = seq_states[seq_id]
            token_ids = list(getattr(out, "token_ids", []))
            if len(token_ids) <= st.prev_len:
                continue
            new_ids = token_ids[st.prev_len :]
            for token_id in new_ids:
                st.token_events_abs.append(
                    TokenEvent(
                        token_index=int(st.prev_len),
                        token_time_ns=int(token_time_ns),
                        token_id=int(token_id),
                    )
                )
                st.prev_len += 1

    request_frames: list[pd.DataFrame] = []
    token_frames: list[pd.DataFrame] = []
    for row in trace_records:
        request_id = int(row["request_id"])
        seq_id = str(request_id)
        st = seq_states.get(seq_id)
        if st is None:
            raise RuntimeError(f"Sarathi replay missing request_id={request_id} (seq_id={seq_id}).")

        token_events_rel = [
            TokenEvent(
                token_index=int(ev.token_index),
                token_time_ns=int(ev.token_time_ns) - run_start_ns,
                token_id=ev.token_id,
            )
            for ev in st.token_events_abs
        ]
        if len(token_events_rel) != int(st.num_decode_tokens):
            raise ValueError(
                "Sarathi replay produced num_decode_tokens_actual that does not match trace num_decode_tokens "
                f"(request_id={request_id} expected={st.num_decode_tokens} got={len(token_events_rel)})."
            )

        req_df, tok_df = build_metrics_frames(
            request_id=int(st.request_id),
            arrival_time_ns=int(st.arrival_time_ns),
            token_events=token_events_rel,
            num_prefill_tokens=int(st.num_prefill_tokens),
            num_decode_tokens=int(st.num_decode_tokens),
            backend="sarathi",
        )
        request_frames.append(req_df)
        token_frames.append(tok_df)

    request_df = pd.concat(request_frames, ignore_index=True) if request_frames else pd.DataFrame()
    token_df = pd.concat(token_frames, ignore_index=True) if token_frames else pd.DataFrame()

    from gpu_simulate_test.io import write_csv, write_json
    from gpu_simulate_test.paper_fidelity.scoring import PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS
    from gpu_simulate_test.real_bench.backends.sarathi_paper_fidelity_backend import (
        convert_sequence_metrics_to_request_metrics,
    )

    paper_fidelity_dir = out_dir / "paper_fidelity"
    paper_fidelity_csv = paper_fidelity_dir / "request_metrics.csv"

    metrics_store = engine.metrics_store
    all_seq_metrics = list(metrics_store.seq_metrics_time_distributions.values()) + list(
        metrics_store.seq_metrics_histogram.values()
    )
    metrics_store._save_as_csv(  # type: ignore[attr-defined]
        dataseries_list=all_seq_metrics,
        key_to_join="Request Id",
        base_path=metrics_store.output_dir,
        file_name="sequence_metrics",
    )

    sarathi_sequence_metrics_csv = out_dir / "sarathi" / "replica_0" / "sequence_metrics.csv"
    if not sarathi_sequence_metrics_csv.exists():
        raise FileNotFoundError(f"Sarathi did not produce sequence_metrics.csv at {sarathi_sequence_metrics_csv}")

    pf_df = convert_sequence_metrics_to_request_metrics(sarathi_sequence_metrics_csv)

    expected_decode_by_id = {int(r["request_id"]): int(r["num_decode_tokens"]) for r in trace_records}
    pf_ids = pf_df["request_id"].map(
        lambda x: int(str(x).split("_", 1)[1]) if "_" in str(x) else int(str(x))
    )
    pf_decode = pd.to_numeric(pf_df["request_num_decode_tokens"], errors="raise").astype(int)
    for rid, got in zip(pf_ids.tolist(), pf_decode.tolist()):
        expected = expected_decode_by_id.get(int(rid))
        if expected is None:
            raise RuntimeError(f"Sarathi metrics produced unexpected request_id={rid}.")
        if int(got) != int(expected):
            raise RuntimeError(
                f"Sarathi metrics decode token mismatch for request_id={rid}: expected={expected} got={got}."
            )

    write_csv(paper_fidelity_csv, pf_df, required_columns=PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS)
    write_json(
        paper_fidelity_dir / "run_meta.json",
        {
            "schema_version": "v1",
            "run_type": "real",
            "backend": "sarathi",
            "generated_at": utcnow_iso(),
            "trace_csv": str(trace_csv.resolve()),
            "sequence_metrics_csv": str(sarathi_sequence_metrics_csv.resolve()),
            "request_metrics_csv": str(paper_fidelity_csv.resolve()),
            "scheduler": {"chunk_size": int(chunk_size), "max_num_seqs": int(max_num_seqs)},
            "parallel": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
            "ignore_eos": bool(ignore_eos),
            "max_tokens": int(max_tokens),
        },
    )

    run_meta = {
        "schema_version": "v1",
        "run_type": "real",
        "backend": "sarathi",
        "model": str(model_id),
        "model_ref": str(model_ref),
        "trace_csv": str(trace_csv),
        "scheduler": {"chunk_size": int(chunk_size), "max_num_seqs": int(max_num_seqs)},
        "parallel": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
        "ignore_eos": bool(ignore_eos),
        "max_tokens": int(max_tokens),
        "paper_fidelity": {
            "request_metrics_csv": str(paper_fidelity_csv.resolve()),
            "sequence_metrics_csv": str(sarathi_sequence_metrics_csv.resolve()),
        },
        "started_at": started_at,
        "ended_at": utcnow_iso(),
    }
    write_run_outputs(out_dir, request_df=request_df, token_df=token_df, run_meta=run_meta)
    return RealReplayResult(out_dir=out_dir.resolve())
