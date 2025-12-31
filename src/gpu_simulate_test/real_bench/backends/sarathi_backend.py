from __future__ import annotations

import time
from pathlib import Path

from gpu_simulate_test.real_bench.backends.base import TokenEvent


class SarathiBackend:
    def __init__(self, *, model: str, out_dir: Path) -> None:
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

        self._SamplingParams = SamplingParams

        replica_config = ReplicaConfig(output_dir=str(out_dir / "sarathi"))
        model_config = ModelConfig(model=model)
        parallel_config = ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1)
        scheduler_config = SarathiSchedulerConfig(chunk_size=16, max_num_seqs=1)
        metrics_config = MetricsConfig(write_metrics=False, enable_chrome_trace=False)
        system_config = SystemConfig(
            replica_config=replica_config,
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            metrics_config=metrics_config,
        )

        self._engine = LLMEngine.from_system_config(system_config)

    def warmup(self) -> None:
        _ = self.run_request(prompt="warmup", max_new_tokens=1)

    def run_request(self, *, prompt: str, max_new_tokens: int) -> list[TokenEvent]:
        seq_id = str(time.time_ns())
        sampling_params = self._SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_new_tokens))
        self._engine.add_request(prompt, sampling_params, seq_id=seq_id, arrival_time=time.monotonic())

        events: list[TokenEvent] = []
        prev_len = 0
        while self._engine.has_unfinished_requests():
            step_outputs = self._engine.step()
            now_ns = time.monotonic_ns()

            for out in step_outputs:
                if out.seq_id != seq_id:
                    continue

                token_ids = list(out.token_ids)
                if len(token_ids) > prev_len:
                    new_ids = token_ids[prev_len:]
                    for token_id in new_ids:
                        events.append(
                            TokenEvent(
                                token_index=int(prev_len),
                                token_time_ns=int(now_ns),
                                token_id=int(token_id),
                            )
                        )
                        prev_len += 1
        return events

