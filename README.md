# gpu-simulate-test

Testbed for evaluating GPU simulators with an emphasis on LLM/inference workloads.

## Goals

- Provide a repeatable way to run and compare simulators on the same workloads/configs.
- Track setup notes, pitfalls, and reproducible scripts for each simulator.
- Keep results and configs versioned.

## Simulators

First simulator to try:

- Vidur (Microsoft): https://github.com/microsoft/vidur

Other candidates:

- RealLM (Bespoke Silicon Group): https://github.com/bespoke-silicon-group/reallm
- LLMCompass (Princeton University): https://github.com/PrincetonUniversity/LLMCompass
- Accel-Sim Framework: https://github.com/accel-sim/accel-sim-framework

## Repo layout (initial)

- `simulators/`: per-simulator setup + wrappers
- `workloads/`: models/traces/configs used across simulators
- `scripts/`: one-shot helpers (setup, run, collect, summarize)
- `results/`: raw outputs and summarized metrics (where appropriate)

