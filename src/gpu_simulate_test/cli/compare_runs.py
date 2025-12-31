from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from gpu_simulate_test.analysis.compare import align_tokens_by_actual_decode
from gpu_simulate_test.analysis.load_metrics import load_run_metrics
from gpu_simulate_test.analysis.report import ReportPaths, write_report
from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso, write_json


register_omegaconf_resolvers()


@hydra.main(
    config_path="../../../configs/compare_vidur_real",
    config_name="compare_runs",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    out_dir = Path.cwd()

    real_run_dir = Path(cfg.real_run_dir).expanduser()
    sim_run_dir = Path(cfg.sim_run_dir).expanduser()

    real = load_run_metrics(real_run_dir)
    sim = load_run_metrics(sim_run_dir)

    real_tok, sim_tok = align_tokens_by_actual_decode(real=real, sim=sim)

    percentiles = [0.5, 0.9, 0.99]
    paths = ReportPaths(
        out_dir=out_dir,
        summary_md=out_dir / "summary.md",
        tables_dir=out_dir / "tables",
        figs_dir=out_dir / "figs",
    )

    ttft_table, token_table = write_report(
        paths,
        ttft_real=real.request_metrics["ttft_ns"].astype(float),
        ttft_sim=sim.request_metrics["ttft_ns"].astype(float),
        token_lat_real=real_tok["token_latency_ns"].astype(float),
        token_lat_sim=sim_tok["token_latency_ns"].astype(float),
        percentiles=percentiles,
        real_run_dir=real.run_dir,
        sim_run_dir=sim.run_dir,
    )

    repo_root = Path(cfg.paths.repo_root)
    git = get_git_info(repo_root=repo_root)
    run_meta = {
        "schema_version": "v1",
        "run_type": "compare",
        "run_id": str(cfg.output.comparison_id),
        "real_run_dir": str(real.run_dir),
        "sim_run_dir": str(sim.run_dir),
        "started_at": utcnow_iso(),
        "ended_at": utcnow_iso(),
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "params": OmegaConf.to_container(cfg, resolve=True),
        "hydra": {"config_path": "configs/compare_vidur_real", "config_name": "compare_runs"},
        "tables": {
            "ttft_percentiles_csv": str((paths.tables_dir / "ttft_percentiles.csv").resolve()),
            "token_latency_percentiles_csv": str((paths.tables_dir / "token_latency_percentiles.csv").resolve()),
        },
    }
    write_json(out_dir / "run_meta.json", run_meta)
    print(str(out_dir))


if __name__ == "__main__":
    main()

