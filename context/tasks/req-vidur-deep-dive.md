# Requirements: Vidur deep-dive tutorial (paper-led, code-backed, with LLaMA2-7B case study)

## HEADER
- **Purpose**: Define requirements for writing a deep-dive tutorial that explains *how Vidur works* by combining (1) Vidur’s research paper as the primary narrative, (2) Vidur’s internal implementation (Python classes/configs), and (3) this repo’s practical usage via the LLaMA2-7B sim-vs-real workflow and report.
- **Status**: Working draft
- **Date**: 2026-01-12
- **Intended audience**: A Python machine learning (ML) developer with strong LLM inference experience (prefill/decode, batching, key/value (KV) cache, tensor/pipeline parallelism) and some familiarity with discrete-event simulation concepts, but no prior Vidur codebase familiarity.
- **Primary lead reference (paper)**:
  - Source (TeX): `paper/tex/main.tex` (and section files under `paper/tex/`)
  - If a PDF is present in-tree, link it; otherwise, reference the TeX as canonical.
- **Case study (must use)**:
  - Report: `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/`
  - Report guide: `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/intro-to-the-report.md`
  - Tutorial (practical): `context/summaries/vidur-kb/tutorials/howto-use-vidur-for-llama2-7b.md`
- **Repo references (authoritative for “how we run it here”)**:
  - Paper-fidelity spec/configs: `specs/002-reproduce-vidur-paper-fidelity/`, `configs/paper_fidelity/`
  - CLI pipeline: `src/gpu_simulate_test/cli/paper_fidelity.py`
  - Host profiling: `src/gpu_simulate_test/paper_fidelity/profiling.py`, `src/gpu_simulate_test/vidur_ext/profile_runner.py`
  - Vidur simulation runner: `src/gpu_simulate_test/vidur_ext/sim_runner.py`
  - Real replay (Sarathi): `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
  - GPU guardrails: `src/gpu_simulate_test/env_guard.py`, `src/sitecustomize.py`
  - CPU overhead validation: `src/gpu_simulate_test/vidur_ext/cpu_overhead_validation.py`
- **Vidur implementation entrypoints (must cite by path)**:
  - `vidur/main.py`
  - `vidur/simulator.py`
  - `vidur/config/config.py`
  - `vidur/types/event_type.py`
  - `vidur/events/`
  - `vidur/execution_time_predictor/`
  - `vidur/scheduler/`
  - `vidur/profiling/`
- **Related knowledge base (optional citations)**:
  - `context/summaries/vidur-kb/qa-vidur-usage.md`
  - `context/tasks/req-vidur-tutorial.md` (style constraints: concepts-first, portable)
  - `context/tasks/working/req-vidur-sim-vs-real.md` (parity-critical requirements)

---

## 1) Deliverable and scope

### D1. Deliverable

The deep-dive tutorial MUST be a single primary Markdown document plus a local figures directory:

- Primary doc: (author chooses final location; must be linked somewhere discoverable)
- Figures: `<tutorial-dir>/figures/` (SVG/PNG; keep assets small)

This requirements doc does not mandate the final doc path, but it MUST be placed in a stable, discoverable location (see “Delivery requirements”).

### D2. What “paper-led, code-backed” means

The tutorial MUST:
- Use the paper’s structure and claims as the main narrative spine.
- For each major paper claim/section, map it to:
  - the relevant Python modules/classes/config keys in `vidur/`
  - and a concrete observation from the LLaMA2-7B case study (report artifacts or run logs).

### D3. Non-goals

The tutorial MUST NOT:
- Reproduce the entire paper evaluation suite.
- Explain generic LLM concepts in depth (assume ML reader knows them).
- Assume readers will use this repo forever; it should teach transferable concepts.

---

## 2) Required narrative structure (paper → code → case study)

### N1. Paper section mapping

The tutorial MUST include a “paper map” section that lists the relevant paper sections (by section name/number as found in TeX) and, for each:

- 1–2 sentence “what the paper is claiming/defining”
- The corresponding Vidur code modules/classes
- The corresponding knobs in this repo’s paper-fidelity configs (when applicable)

### N2. Repeated pattern per section

For each major section of the deep dive, the tutorial MUST follow this pattern:

1) **Concept (paper)**: restate the definition/model/assumption.
2) **Implementation (code)**: point to the precise classes/files and key parameters.
3) **Concrete evidence (case study)**: show a small snippet of real output (CLI log, JSON excerpt, or a figure) from the LLaMA2-7B run.
4) **Caveats**: where the model/implementation can mislead (and how to detect it).

---

## 3) Required technical content (must include)

### C0. Quick glossary (ML + simulation)

The tutorial MUST include a glossary that expands acronyms on first use and maps:

- ML inference terms (prefill/decode, KV cache, TP/PP, scheduler chunking)
- Simulation terms (event queue, virtual time, discrete-event simulation, state machine)
- Vidur terms (replica scheduler, stage scheduler, execution-time predictor, profiling root)

### C1. System boundary and what is being simulated

The tutorial MUST clearly define the system boundary for:

- Vidur simulation outputs (what time is being modeled)
- Real replay outputs (Sarathi metrics boundary)
- How this repo makes them comparable (trace + metric alignment)

It MUST explicitly call out that the simulator does not generate tokens; it advances a virtual clock over events.

### C2. Workload modeling (trace, arrivals, token semantics)

The tutorial MUST explain:

- The trace schema used in this repo (what columns matter conceptually)
- Static vs dynamic arrivals, and why arrivals must match across sim and real for comparability
- The “early EOS” hazard and how this repo avoids it (concept + config knob)

It MUST show how the LLaMA2-7B example derives a trace (including a snippet from `trace.csv`).

### C3. Discrete-event simulation loop (core mechanics)

The tutorial MUST explain the simulator’s core mechanics:

- The main event loop (event queue ordering, time advancement)
- Key event types and what state they mutate
- How requests progress across prefill/decode and across pipeline stages (if PP > 1)

It MUST cite:

- `vidur/simulator.py`
- `vidur/types/event_type.py`
- At least 3 representative event classes in `vidur/events/`

It MUST include at least one diagram:

- a `sequenceDiagram` or flow diagram that shows “request arrival → scheduling → batch execution → completion”.

If Mermaid is used, it MUST follow `magic-context/instructions/mermaid-seq-styling.md`.

### C4. Scheduling model (replica/global/stage schedulers)

The tutorial MUST explain:

- The difference between global scheduling vs replica scheduling vs stage scheduling in Vidur.
- How a particular “engine policy” is represented (e.g., Sarathi/vLLM replica scheduler types).
- Why parity-critical knobs (chunk size, inflight cap, block size, watermark) strongly affect outcomes.

It MUST map the LLaMA2-7B config knobs to scheduler behavior, including:

- `scenario.real.scheduler.chunk_size` and `scenario.real.scheduler.max_num_seqs`
- `scenario.vidur.scheduler.chunk_size`, `scenario.vidur.scheduler.batch_size_cap`, `scenario.vidur.scheduler.block_size`, `scenario.vidur.scheduler.watermark_blocks_fraction`

It MUST include a “defaults are dangerous” warning grounded in a real failure mode observed in this repo (with symptoms + prevention checklist).

### C5. Runtime model (execution-time prediction)

The tutorial MUST explain:

- What inputs the execution-time predictor uses (profiling CSVs)
- How the predictor is trained/fit and then used at runtime (high-level)
- Which parts correspond to prefill vs decode operators

It MUST cite:

- `vidur/execution_time_predictor/sklearn_execution_time_predictor.py`
- The profiling outputs under `data/profiling/...` (conceptual structure)

It MUST show at least one concrete log snippet from the LLaMA2-7B run showing model fitting/prediction phases (from `paper-fidelity repro` output).

### C6. Profiling (MLP, attention, and host/topology matching)

The tutorial MUST explain, conceptually:

- Why profiling must match model/hardware/topology/kernel stack for fidelity claims
- What a profiling root contains and how Vidur looks up rows for a given run

In this repo, it MUST show:

- How to generate a host profiling root (`paper-fidelity profile`)
- Where that root is stored

It MUST include a “host-matched vs paper-provided profiling” section explaining which conclusions are valid under each mode.

### C7. CPU overhead modeling (what it is, why it matters, guardrails)

The tutorial MUST explain:

- What CPU/runtime overhead terms represent (scheduling overhead, preparing inputs, sampler, etc.)
- Why enabling CPU overhead modeling without real CPU overhead measurements leads to systematic underprediction
- How to detect/avoid dummy/placeholder overhead inputs

It MUST cite this repo’s guardrails and validation behavior (by path), and it MUST reference the LLaMA2-7B report’s CPU overhead status block (`status: ok`).

### C8. Metrics, normalization, and scoring (what’s compared)

The tutorial MUST explain:

- The meaning of “normalized” metrics (per-token normalization) and why it’s used
- Percent error definition and scoring thresholds
- ECDF/percentile plots and what to look for (uniform shift vs tail skew)

It MUST connect the “paper fidelity” notion to the paper (where possible) and to this repo’s report fields:

- `summary.md`, `scores.json`, `run_meta.json`

It MUST include one “walkthrough of a score row” showing how to interpret p50 vs p95 and what “warn” implies.

---

## 4) Reproduction requirements (must include step-by-step with outputs)

The deep dive MUST include a reproduction section that mirrors `context/tasks/task-anything.md` style:

- prerequisites
- what artifacts are produced
- how to interpret results
- steps to reproduce, with representative outputs (CLI snippets or “jupyter-cell style” blocks)

Minimum steps (must be present):

1) Environment sanity check (PyTorch + CUDA)
2) Host profiling root generation (include CPU overhead)
3) Repro run (sim + real + report)
4) Inspecting the report (CPU overhead status, score table, figures)

Each step MUST include:
- “Why this step exists” (portable concept)
- “This repo’s implementation” (exact command)
- “Expected output shape” (what the user should see)

---

## 5) Caveats and “mistakes we already made” (must include)

The deep dive MUST include a dedicated “Caveats” section that enumerates easy-to-make mistakes that lead to misleading results, and it MUST include the ones we have observed in this repo:

- Dummy/missing CPU overhead inputs while thinking CPU overhead modeling is “on”.
- GPU visibility issues in worker processes (Ray/Sarathi clearing/overriding `CUDA_VISIBLE_DEVICES`).
- Relying on defaults for parity-critical knobs (chunk size, inflight cap, etc.).
- Token mismatch due to early EOS stopping.
- Using non-host-matched profiling roots for host-fidelity claims.

For each caveat, it MUST provide:
- symptoms (where it shows up: logs, `summary.md`, `run_meta.json`)
- how to fix
- how to prevent

---

## 6) Writing/style requirements

### W1. Acronyms expanded on first use

Any acronym MUST be expanded the first time it appears (e.g., tensor parallelism (TP), pipeline parallelism (PP), key/value cache (KV cache), empirical cumulative distribution function (ECDF), queries per second (QPS)).

### W2. Concepts-first, portable framing

The tutorial MUST be written so a reader could reproduce the approach in their own repo:

- Explain concepts and reasoning first.
- Then show how this repo implements it (commands/configs/guardrails).
- Explicitly label repo-specific mechanisms (e.g., `GSIM_CUDA_VISIBLE_DEVICES`, `paper-fidelity` CLI) and describe their generic requirement.

### W3. Code pointers over code dumps

Prefer:
- file paths + class/function names
- small excerpts (≤ ~15 lines) only when essential

Avoid:
- large pasted code blocks

---

## 7) Delivery requirements

The author MUST choose a discoverable location for the final deep-dive tutorial and link it from at least one index:

- `docs/manual/index.md` or `docs/developer/index.md`, or
- `context/summaries/vidur-kb/` with a pointer from `context/summaries/vidur-kb/qa-vidur-usage.md`

Figures MUST be stored under `<tutorial-dir>/figures/` and referenced via relative paths.

---

## 8) Acceptance criteria (definition of done)

The deep-dive tutorial is acceptable when:

- A reader can trace “paper concept → code component → case-study artifact” for the main ideas.
- A reader can reproduce a LLaMA2-7B sim-vs-real report and understand *why* each step exists.
- The tutorial clearly warns about common pitfalls and teaches how to validate credibility before interpreting fidelity.
- The tutorial provides enough code pointers that an experienced ML engineer can continue exploring Vidur internals without getting lost.
