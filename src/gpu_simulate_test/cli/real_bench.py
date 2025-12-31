from __future__ import annotations

import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso
from gpu_simulate_test.real_bench.metrics import build_metrics_frames, write_run_outputs
from gpu_simulate_test.real_bench.replay import load_workload_requests


register_omegaconf_resolvers()


@hydra.main(
    config_path="../../../configs/compare_vidur_real",
    config_name="real_bench",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    out_dir = Path.cwd()

    workload_dir = Path(cfg.workload.workload_dir).expanduser()
    requests = load_workload_requests(workload_dir)

    backend_name = str(cfg.backend.name)
    if backend_name == "transformers":
        from gpu_simulate_test.real_bench.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend(
            model_ref=Path(cfg.model.tokenizer_ref).expanduser(),
            device=str(cfg.hardware.device),
        )
    elif backend_name == "sarathi":
        from gpu_simulate_test.real_bench.backends.sarathi_backend import SarathiBackend

        backend = SarathiBackend(model=str(cfg.model.model_id), out_dir=out_dir)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    backend.warmup()

    run_start_ns = time.monotonic_ns()
    started_at = utcnow_iso()

    request_frames: list[pd.DataFrame] = []
    token_frames: list[pd.DataFrame] = []

    for req in sorted(requests, key=lambda r: r.arrival_time_ns):
        target_ns = run_start_ns + int(req.arrival_time_ns)
        while True:
            now_ns = time.monotonic_ns()
            remaining = target_ns - now_ns
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining / 1e9))

        token_events_abs = backend.run_request(prompt=req.prompt_text, max_new_tokens=req.num_decode_tokens)
        token_events_rel = [
            type(ev)(
                token_index=ev.token_index,
                token_time_ns=int(ev.token_time_ns) - run_start_ns,
                token_id=ev.token_id,
            )
            for ev in token_events_abs
        ]

        req_df, tok_df = build_metrics_frames(
            request_id=req.request_id,
            arrival_time_ns=req.arrival_time_ns,
            token_events=token_events_rel,
            num_prefill_tokens=req.num_prefill_tokens,
            num_decode_tokens=req.num_decode_tokens,
            backend=backend_name,
        )
        request_frames.append(req_df)
        token_frames.append(tok_df)

    request_df = pd.concat(request_frames, ignore_index=True) if request_frames else pd.DataFrame()
    token_df = pd.concat(token_frames, ignore_index=True) if token_frames else pd.DataFrame()

    ended_at = utcnow_iso()
    repo_root = Path(cfg.paths.repo_root)
    git = get_git_info(repo_root=repo_root)

    run_meta = {
        "schema_version": "v1",
        "run_type": "real",
        "run_id": str(cfg.output.run_id),
        "backend": backend_name,
        "model": str(cfg.model.model_id),
        "workload_dir": str(workload_dir.resolve()),
        "hardware": OmegaConf.to_container(cfg.hardware, resolve=True),
        "started_at": started_at,
        "ended_at": ended_at,
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "params": OmegaConf.to_container(cfg, resolve=True),
        "hydra": {"config_path": "configs/compare_vidur_real", "config_name": "real_bench"},
    }

    write_run_outputs(out_dir, request_df=request_df, token_df=token_df, run_meta=run_meta)
    print(str(out_dir))


if __name__ == "__main__":
    main()

