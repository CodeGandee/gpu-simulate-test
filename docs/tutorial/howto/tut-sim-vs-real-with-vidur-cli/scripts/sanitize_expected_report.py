#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanitize a vidur-cli report snapshot for git tracking.")
    p.add_argument("--expected-dir", required=True, help="Path to the copied <run_dir>/report/ directory.")
    p.add_argument("--run-dir", default="", help="Absolute <run_dir> to replace with <RUN_DIR>.")
    p.add_argument("--repo-root", default="", help="Absolute repo root to replace with <REPO_ROOT>.")
    p.add_argument("--model-ref", default="", help="Absolute model_ref path to replace with <MODEL_REF>.")
    p.add_argument(
        "--run-tag-placeholder",
        default="<RUN_TAG>",
        help="Placeholder string to use for the report title run tag.",
    )
    return p.parse_args()


def _replace_str(s: str, *, run_dir: str, repo_root: str, model_ref: str) -> str:
    if run_dir and s.startswith(run_dir):
        return "<RUN_DIR>" + s[len(run_dir) :]
    if repo_root and s.startswith(repo_root):
        return "<REPO_ROOT>" + s[len(repo_root) :]
    if model_ref and s.startswith(model_ref):
        return "<MODEL_REF>" + s[len(model_ref) :]
    return s


def _walk(x, *, run_dir: str, repo_root: str, model_ref: str):
    if isinstance(x, str):
        return _replace_str(x, run_dir=run_dir, repo_root=repo_root, model_ref=model_ref)
    if isinstance(x, list):
        return [_walk(v, run_dir=run_dir, repo_root=repo_root, model_ref=model_ref) for v in x]
    if isinstance(x, dict):
        return {k: _walk(v, run_dir=run_dir, repo_root=repo_root, model_ref=model_ref) for k, v in x.items()}
    return x


def main() -> None:
    args = _parse_args()
    expected_dir = Path(args.expected_dir).expanduser().resolve()
    run_dir = str(Path(args.run_dir).expanduser().resolve()) if args.run_dir else ""
    repo_root = str(Path(args.repo_root).expanduser().resolve()) if args.repo_root else ""
    model_ref = str(Path(args.model_ref).expanduser().resolve()) if args.model_ref else ""

    summary_md = expected_dir / "summary.md"
    if summary_md.exists():
        text = summary_md.read_text(encoding="utf-8")
        for src, dst in [(run_dir, "<RUN_DIR>"), (repo_root, "<REPO_ROOT>"), (model_ref, "<MODEL_REF>")]:
            if src:
                text = text.replace(src, dst)

        lines = text.splitlines()
        if lines and lines[0].startswith("# Sim-vs-Real Report:"):
            lines[0] = f"# Sim-vs-Real Report: {args.run_tag_placeholder}"
        summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    run_meta = expected_dir / "run_meta.json"
    if run_meta.exists():
        meta = json.loads(run_meta.read_text(encoding="utf-8"))
        meta = _walk(meta, run_dir=run_dir, repo_root=repo_root, model_ref=model_ref)
        if isinstance(meta, dict) and "run_dir" in meta:
            meta["run_dir"] = "<RUN_DIR>"
        run_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
