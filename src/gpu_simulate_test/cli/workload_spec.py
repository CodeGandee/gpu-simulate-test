from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.io import get_git_info, utcnow_iso, write_csv, write_json
from gpu_simulate_test.workloads.arrival_schedule import ArrivalScheduleConfig, build_trace_intervals
from gpu_simulate_test.workloads.prompts import read_prompts_jsonl, write_prompts_jsonl
from gpu_simulate_test.workloads.token_lengths import TraceLengthsConfig, compute_trace_lengths, load_hf_tokenizer


register_omegaconf_resolvers()


def generate_workload_spec(cfg: DictConfig) -> Path:
    out_dir = Path.cwd()

    prompts_in = Path(cfg.workload.prompts).expanduser()
    prompts = read_prompts_jsonl(prompts_in)

    prompts_out = out_dir / "prompts.jsonl"
    write_prompts_jsonl(prompts_out, prompts)

    tokenizer_ref = Path(cfg.model.tokenizer_ref).expanduser()
    tokenizer = load_hf_tokenizer(tokenizer_ref)

    lengths = compute_trace_lengths(
        prompts,
        tokenizer=tokenizer,
        config=TraceLengthsConfig(num_decode_tokens=int(cfg.workload.num_decode_tokens)),
    )
    trace_lengths = out_dir / "trace_lengths.csv"
    write_csv(
        trace_lengths,
        lengths,
        required_columns=["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"],
    )

    arrival_cfg = ArrivalScheduleConfig(
        kind=str(cfg.workload.arrival.kind),
        seed=int(cfg.workload.arrival.seed),
        inter_arrival_ns=int(cfg.workload.arrival.inter_arrival_ns),
        poisson_rate_per_s=float(cfg.workload.arrival.poisson_rate_per_s),
    )
    intervals = build_trace_intervals(len(prompts), config=arrival_cfg)
    trace_intervals = out_dir / "trace_intervals.csv"
    write_csv(
        trace_intervals,
        intervals,
        required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"],
    )

    repo_root = Path(cfg.paths.repo_root)
    git = get_git_info(repo_root=repo_root)

    workload_id = str(cfg.output.workload_id)
    meta = {
        "schema_version": "v1",
        "workload_id": workload_id,
        "created_at": utcnow_iso(),
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "model": str(cfg.model.model_id),
        "seed": int(cfg.workload.seed),
        "tokenizer_ref": str(tokenizer_ref.resolve()),
        "paths": {
            "prompts": str(prompts_out.resolve()),
            "trace_lengths": str(trace_lengths.resolve()),
            "trace_intervals": str(trace_intervals.resolve()),
        },
        "hydra": {"config_path": "configs/compare_vidur_real", "config_name": "workload_spec"},
        "params": OmegaConf.to_container(cfg, resolve=True),
    }
    write_json(out_dir / "workload_meta.json", meta)

    return out_dir


@hydra.main(
    config_path="../../../configs/compare_vidur_real",
    config_name="workload_spec",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    out_dir = generate_workload_spec(cfg)
    print(str(out_dir))


if __name__ == "__main__":
    main()
