# Requirements: Vidur tutorial (using the LLaMA2-7B paper-fidelity report as a worked example)

## HEADER
- **Purpose**: Define requirements for a “how to use Vidur properly” tutorial that teaches portable concepts (so readers can reproduce the approach in their own environments/projects), using this repo’s paper-fidelity report as the canonical worked example and using this repo’s scripts/env-vars/guardrails as concrete examples.
- **Status**: Working draft
- **Date**: 2026-01-12
- **Canonical worked example (must reference)**: `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/` (and its guide `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/intro-to-the-report.md`)
- **Primary references (authoritative)**:
  - Paper-fidelity workflow: `specs/002-reproduce-vidur-paper-fidelity/` and `configs/paper_fidelity/`
  - Environment setup: `context/instructions/prep-dev-env.md`
  - Troubleshooting: `docs/manual/troubleshooting.md`
  - Existing requirements doc to align with: `context/tasks/working/req-vidur-sim-vs-real.md`
- **Related implementations (tutorial should point to these)**:
  - CLI: `src/gpu_simulate_test/cli/paper_fidelity.py`
  - Host profiling orchestration: `src/gpu_simulate_test/paper_fidelity/profiling.py`
  - Vidur sim runner: `src/gpu_simulate_test/vidur_ext/sim_runner.py`
  - Real runner backend (Sarathi): `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
  - GPU guardrails: `src/gpu_simulate_test/env_guard.py`, `src/sitecustomize.py`
  - CPU overhead profiling/validation: `src/gpu_simulate_test/vidur_ext/profile_runner.py`, `src/gpu_simulate_test/vidur_ext/cpu_overhead_validation.py`

---

## 1) Scope and audience

### S1. Audience

The tutorial MUST be understandable by:
- A Python ML developer with no Vidur experience who understands core LLM inference concepts (prefill vs decode, batching, KV cache, throughput/latency tradeoffs, TP/PP) and can run shell commands.

### S2. Goals (“use Vidur properly” means)

The tutorial MUST teach the user to:
- Generate or select a **correct** Vidur profiling root for their hardware + topology.
- Run a **trace-driven** Vidur simulation in a way that is comparable to a real replay runner.
- Validate that the run is **credible** (profiling inputs present, token semantics aligned, and key knobs explicitly set).
- Interpret the produced metrics and identify the most common “false fidelity” failure modes (e.g., dummy CPU overheads, early EOS, mismatched scheduler knobs).
- Port the same workflow to a different repo/project by understanding the *reasons* behind each step (not just memorizing this repo’s commands).

### S3. Non-goals

The tutorial MUST NOT try to:
- Explain all of Vidur internals or reproduce the entire Vidur paper suite.
- Teach general GPU/Linux environment setup beyond what this repo already documents.
- Assume the reader knows Ray internals or Sarathi/Vidur implementation details.

---

## 2) Canonical worked example requirement

### E1. The tutorial MUST use this report as the running example

- `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/summary.md`
- `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/run_meta.json`
- `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/scores.json`
- `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/figs/`

The tutorial MUST explicitly point out, from `run_meta.json`, that:
- The run used a **host** profiling root (not the paper bundle).
- CPU overhead modeling was **enabled** and validated (`status: ok`).
- The environment snapshot includes CUDA availability and GPU name.

### E2. The tutorial MUST include the exact reproduction commands used by the example

At minimum, include (as code blocks) the two commands, with a note that paths/dates will differ:
- `pixi run paper-fidelity profile --scenario llama2_7b_arxiv --include-cpu-overhead`
- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static --scale small "scenario.vidur.profiling_root=..." "scenario.vidur.skip_cpu_overhead_modeling=false"`

The tutorial MUST show how to locate the resulting:
- profiling root under `tmp/paper_fidelity/profiling_roots/...`
- report directory under `results/reports/<YYYY-MM-DD>/paper_fidelity/...`

---

## 3) Tutorial content requirements (what must be included)

### R0. Quick mental model + glossary (ML-centric)

The tutorial MUST include a short “mental model” section that explains, in ML inference terms:
- What Vidur is modeling (GPU kernel time models + scheduler + queueing + optional CPU/runtime overhead models).
- What a “profiling root/bundle” is (calibration data: per-op latency models for a specific model + hardware + topology).
- What the “real runner” is in this repo (Sarathi replay) and what boundary it measures.
- The difference between **portable concepts** (e.g., “pin GPUs so worker processes see the same devices”) and **repo-specific mechanisms** (e.g., `GSIM_CUDA_VISIBLE_DEVICES`, `paper-fidelity` CLI).

The tutorial MUST include a small glossary mapping Vidur/Sarathi terms to common ML inference terminology, e.g.:
- prefill / prompt processing
- decode / token generation loop
- scheduler `chunk_size` (token granularity per iteration)
- `max_num_seqs` / `batch_size_cap` (inflight sequence cap)
- KV cache `block_size` (page size / memory planning unit)
- TP/PP (tensor/pipeline parallelism)

### R1. Prerequisites section

The tutorial MUST cover prerequisites at two levels:

1) **Portable prerequisites** (apply outside this repo):
- A working CUDA driver/runtime compatible with your PyTorch build.
- A single-GPU “hello world” that can allocate CUDA memory.
- Local model assets (or a clearly documented model loading strategy) consistent between profiling and real replay.

2) **This repo’s concrete setup** (example implementation):
- Pixi env setup (`pixi install`) and “always use `pixi run ...`”.
- Submodule initialization.
- GPU pinning requirement via `.env` + `GSIM_CUDA_VISIBLE_DEVICES`.
- Model assets requirement for the “real” runner (e.g., `models/llama2-7b-hf/source-data`).

It MUST include at least one “sanity check” command and sample output:
- `pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"`

### R2. Artifacts and directory layout section

The tutorial MUST explain where to find:
- traces: `tmp/paper_fidelity/traces/<scenario>/trace.csv`
- sim outputs: `tmp/paper_fidelity/runs/<scenario>/sim/request_metrics.csv`
- real outputs: `tmp/paper_fidelity/runs/<scenario>/real/request_metrics.csv`
- profiling roots: `tmp/paper_fidelity/profiling_roots/<scenario>/<run_id>/`
- reports: `results/reports/<date>/paper_fidelity/<scenario>/`

The tutorial MUST explain which artifacts are intended to be machine-local and not committed (especially `tmp/` and `results/`).

### R3. “Profiling root correctness” section (must be explicit)

The tutorial MUST teach how to verify a profiling root is usable:
- Required files exist (MLP, attention, and if enabled: CPU overhead `cpu_overheads.csv`).
- Profiling root matches the scenario’s `device`, `network_device`, and `model_id`.

It MUST include a checklist covering:
- model id directory: `<profiling_root>/data/profiling/.../<model_id>/...`
- CPU overhead status: “missing vs ok vs placeholder-like”

### R4. CPU overhead modeling section (must be prominent)

The tutorial MUST explain:
- What CPU overhead modeling is (what it tries to account for in sim-vs-real).
- Why dummy/placeholder CPU overhead inputs lead to misleading “fidelity”.
- How this repo enforces guardrails (validation modes `strict`/`warn`/`off`).

It MUST include:
- A short snippet showing how to `head` the `cpu_overheads.csv` and what columns look like.
- A “what to do if CPU profiling fails” subsection pointing to `docs/manual/troubleshooting.md` and Ray logs under `/tmp/ray/session_latest/logs/`.

### R5. Config + knob hygiene section (don’t rely on defaults)

The tutorial MUST demonstrate:
- Where scenario knobs live (`configs/paper_fidelity/scenario/*.yaml`).
- How to override config values from the CLI (Hydra overrides).
- Which knobs are “parity-critical” (must be explicitly set for sim-vs-real comparisons), including:
  - scheduler chunking and batch cap
  - TP/PP settings
  - `ignore_eos` / decode-length consistency
  - CPU overhead modeling enable/disable

It MUST frame these knobs in ML-inference terms (e.g., “this is your max concurrent sequences”, “this changes how many tokens are processed per scheduling step”), not only simulator jargon.

### R6. Metrics interpretation section

The tutorial MUST explain:
- The meaning of the “normalized” metrics (per-token normalization), with the key formula(s).
- How percent error is computed and how pass/warn/fail thresholds are applied.
- Why you should not compare runs with mismatched request ids or decode token counts.

It MUST include a “how to reason about discrepancies” subsection aimed at ML practitioners, including examples like:
- If prefill is off but decode matches: likely tokenization/prompt-length handling or prefill modeling mismatch.
- If both prefill+decode are uniformly low: often missing CPU/runtime overhead modeling or using a non-host-matched profiling root.

It MUST show how to inspect:
- `summary.md` (human-readable)
- `scores.json` (machine-readable; includes per-percentile values)
- `run_meta.json` (provenance + resolved config)

### R7. Figures section

The tutorial MUST explain what the plots in `figs/` represent (at least ECDF + percentile plots) and how to use them for sanity checking.

### R8. Troubleshooting / common pitfalls section

The tutorial MUST include a “pitfalls” list with remediation steps, including at least:
- GPU pinning / Ray worker CUDA issues
- missing model assets / wrong model path
- early EOS / token mismatch issues
- using a paper-provided profiling bundle for host-fidelity conclusions
- accidentally enabling CPU overhead modeling with missing/placeholder CPU overheads

It MUST also include a “caveats when using Vidur” subsection that is explicit about:
- What Vidur does *not* model by default (or only models when explicitly enabled via profiling inputs), and how that can bias conclusions.
- Which conclusions are safe to draw from a run (e.g., comparative trends) vs which require stricter parity/profiling (e.g., absolute wall-clock fidelity).

It MUST explicitly call out “easy-to-make mistakes we have already made in this repo” and why they cause problems, including at least:
- **Running with dummy/placeholder CPU overhead inputs** (or missing `cpu_overheads.csv`) while thinking CPU overhead modeling is “on” → leads to misleading underprediction; fix by profiling CPU overheads on-host and validating status `ok`.
- **Letting Sarathi/Ray unset or override GPU visibility** (e.g., `CUDA_VISIBLE_DEVICES` cleared inside Ray workers when `num_gpus=0`) → manifests as “No CUDA GPUs are available” inside workers; fix by enforcing GPU pinning and preserving `CUDA_VISIBLE_DEVICES` in workers.
- **Relying on Vidur/Sarathi defaults for parity-critical scheduler knobs** (chunk size, max inflight sequences) → can make prefill vs decode split wrong even if headline metrics look plausible; fix by explicitly setting and recording these knobs.
- **Token-count mismatches due to early stopping** (end-of-sequence behavior) → makes sim vs real incomparable; fix by disabling early EOS in the real runner and validating token counts match.
- **Using a non-host-matched profiling root** (wrong device/topology/model) → yields “reasonable-looking” but invalid numbers; fix by ensuring the profiling root keys match the scenario and are generated on the target host for fidelity claims.

For each caveat/mistake, the tutorial MUST provide:
- “How to recognize it” (symptoms in logs / `summary.md` / `run_meta.json`)
- “How to fix it” (specific commands or config overrides)
- “How to prevent it next time” (guardrails/checklist items)

---

## 4) Reproducibility and examples requirements

### X1. Step-by-step reproduction with “example outputs”

For each major step, the tutorial MUST include:
- The command to run.
- A short snippet of expected output (CLI output or “jupyter-cell style” output).
- A short “Why this step exists” paragraph written at the portable-concept level (so the reader can replicate it without this repo).

Minimum required steps:
1) Setup / sanity check
2) Profiling root generation (with CPU overhead)
3) Repro run (sim + real + score + report)
4) Inspect summary + JSON provenance

### X2. Clear “what changes between machines”

The tutorial MUST explicitly state what will vary:
- report date path (`results/reports/<YYYY-MM-DD>/...`)
- profiling root run id directory
- GPU name / environment snapshot
- Environment management tooling (Pixi/conda/venv) and how to translate the commands accordingly.

---

## 5) Delivery requirements (where the final tutorial lives)

The requirements doc MUST specify, in the final tutorial PR, at least one of:
- A new tutorial page under `docs/manual/` (and linked from `docs/manual/index.md`), or
- A developer-oriented guide under `docs/developer/` (and linked from `docs/developer/index.md`).

(This requirements doc does not pick the final location; the tutorial author must pick one and link it.)

---

## 6) Writing/style requirements

### W1. Use repo conventions for commands and paths

- All commands MUST be `pixi run ...` unless explicitly about Pixi itself.
- All file paths and commands MUST be in code blocks / monospace formatting.

### W1.5 Acronyms must be expanded on first use

To avoid misunderstandings for readers new to Vidur, the tutorial MUST expand any acronym the first time it appears, e.g.:

- **tensor parallelism (TP)**
- **pipeline parallelism (PP)**
- **key/value cache (KV cache)**
- **cumulative distribution function (CDF)** / **empirical CDF (ECDF)**
- **requests per second (RPS)** / **queries per second (QPS)** (choose one and be consistent)

After first use, the tutorial MAY use the acronym alone.

### W1.6 Teach concepts first; use this repo as an example implementation

The tutorial MUST be written so that readers can reproduce the workflow in their own project:

- Each section MUST first explain the *concept* and *reasoning* (portable).
- Then it MUST show how this repo implements the concept (commands, config paths, env vars).
- Where this repo uses a repo-specific trick (e.g., `.env` loading, `GSIM_CUDA_VISIBLE_DEVICES`, `paper-fidelity` CLI), the tutorial MUST explicitly label it as “this repo’s implementation detail” and briefly describe the underlying generic requirement (e.g., consistent GPU visibility across worker processes).

### W2. Keep the tutorial actionable

- Prefer short, copy-pastable blocks.
- Prefer “checklists” over long prose where possible.

### W3. Diagrams (optional, but if present must follow styling rules)

If the tutorial includes Mermaid `sequenceDiagram`, it MUST follow:
- `magic-context/instructions/mermaid-seq-styling.md`

---

## 7) Acceptance criteria (definition of done)

The tutorial is acceptable when:
- A fresh developer can follow it end-to-end and produce a report directory similar to the worked example.
- The tutorial teaches how to confirm the run is **comparable** (same trace, consistent token semantics, parity-critical knobs aligned, profiling root validated).
- The tutorial clearly explains how to interpret `summary.md` + `scores.json` + `run_meta.json` and spot non-credible runs.
