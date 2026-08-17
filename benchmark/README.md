# Graph Swarm Recurrence Benchmark

Keep benchmark construction separate from implementation code.

Each recurrence family should define:

- acquisition tasks that expose a failure/recovery pattern,
- later transfer tasks where reuse may be applicable,
- an objective evaluator,
- a manual rationale documenting why transfer is defensible,
- chronological ordering that prevents future-memory leakage.
