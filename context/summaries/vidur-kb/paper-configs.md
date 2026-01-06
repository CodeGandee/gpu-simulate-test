# Vidur Paper Fidelity Simulations Configurations

This file documents the simulation configurations derived from the Vidur paper to reproduce the "optimal deployment configuration" results.

## Source Data

- **Primary Source**: `graphs/parallel_coord.pdf` (Parallel Coordinates Plot / Sankey Diagram) from the Vidur paper.
- **TeX Source**: `extern/tracked/vidur/paper/tex/5-eval.tex` (Section 5.3 "What-if Analysis").
- **Figure Reference**: Figure 6 in the paper (`figures-tex/fig-best-configs.tex`).

## Interpretation Methodology

The configurations in `paper-configs.json` were manually extracted by tracing the flows in the parallel coordinates diagram. The diagram maps the following dimensions for each `(Model, Dataset)` pair:

1.  **SKU**: A100 vs H100
2.  **BS (Batch Size)**: 64, 128, 256
3.  **Scheduler**: Sarathi-Serve vs vLLM vs Orca (visualized as flows)
4.  **TP Dim (Tensor Parallelism)**: 1, 2, 4
5.  **PP Dim (Pipeline Parallelism)**: 1, 2

### Visual Mapping Keys (Inferred)

-   **SKU**: Top flow -> A100, Bottom flow -> H100
-   **Batch Size**: Top flow -> 64, Middle flow -> 128, Bottom flow -> 256
-   **Scheduler**: Top flow -> Sarathi-Serve, Bottom flow -> vLLM
-   **TP Dim**: Top -> 1, Middle -> 2, Bottom -> 4
-   **PP Dim**: Top -> 1, Bottom -> 2

## JSON Data

The structured JSON representation of these configurations is saved in [paper-configs.json](./paper-configs.json).
