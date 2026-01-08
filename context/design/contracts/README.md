# contracts/

Data entity contracts and invariants.

- Defines **schemas**, required fields, and semantic meaning for data artifacts (CSV/JSON/dir layouts).
- Captures **validation rules** and backwards-compatibility expectations.
- Stays **implementation-agnostic**; link to code/tests/specs where the contract is enforced.

Key docs:

- `context/design/contracts/def-perf-metrics.md`: Request-level latency metric definitions (normalized E2E vs execution+preemption).
