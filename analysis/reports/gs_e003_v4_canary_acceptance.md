# GS-E003 Sprint3B v4 canary acceptance criteria

This document is a predeclared engineering gate. It was created before any
live v4 result exists. No Kilo, model/provider, Hugging Face, or Neo4j call is
represented here.

## Protocol invariants

The canary is accepted for development expansion only if all of the following
are true:

- exactly 6 planned runs: GS-T006 B0/T, GS-T007 B0/T, and GS-T008 B0/T;
- B0 and T use the same provider and model;
- B0 and T receive byte-identical `R13B_SYSTEM_PROMPT` text and SHA256;
- `max_actions` is 28 and `max_requests` is 24 for both conditions;
- tools and environment rules are identical to the declared canary policy;
- recurrence metadata identifies `frozen_recurrence_matcher_v2_structured_pytest`;
- no evaluator-only benchmark metadata reaches the task prompt or model runtime.

## Trajectory goals

For each of the six runs, record and inspect:

- zero unsupported shell-composition attempts;
- zero attempts to use unavailable `rg` or `file` utilities due to agent assumptions;
- the first mutation action;
- the first test action;
- total tool calls;
- model requests;
- termination reason.

## Graph Swarm safety

- GS-T007 receives no inapplicable advice;
- no non-expected `RecoveryPattern` is selected;
- false-advice rate must not increase through broad trigger firing.

## Graph Swarm regression

- GS-T006 remains capable of legitimate expected-pattern retrieval when its
  trigger opportunity occurs;
- GS-T008 remains capable of legitimate expected-pattern retrieval when its
  trigger opportunity occurs;
- GS-T008 treatment-task success is an explicit engineering non-regression
  check.

## Research interpretation

- The six-run canary is not required to show a statistically superior T result
  over B0.
- Methodology must not be tuned after individual outcomes merely to force a
  positive treatment result.
- The canary determines whether the runtime correction is safe enough to
  expand to the remaining development tasks.

## Planned entrypoint

The historical v3 launcher is not the v4 entrypoint because it hard-codes the
ten-task v3 protocol and freeze contract. The dedicated v4 module is:

```text
python -m experiments.run_sprint3b_kilo_v4_canary --project-root . --config configs/experiments/sprint3b_kilo_v4_canary.yaml --freeze research/evidence/results/GS-E003/sprint3b_kilo_v4_canary/freeze.json
```

The offline package-freeze command is:

```text
python -m experiments.run_sprint3b_kilo_v4_canary --project-root . --config configs/experiments/sprint3b_kilo_v4_canary.yaml --freeze research/evidence/results/GS-E003/sprint3b_kilo_v4_canary/freeze.json --create-freeze
```

The dedicated loader validates the config path, six-entry execution plan,
artifact root, freeze artifact, R13B prompt ID/hash, and corrected recurrence
metadata before execution. It reuses the existing Sprint 3 environment,
objective, advisory, and recurrence primitives without changing their logic.

## V4 freeze dependency surface

The freeze covers the complete directly behavior-critical path, without
hashing unrelated repository code:

- orchestration and protocol: `experiments/run_sprint3b_kilo_v4_canary.py`,
  `experiments/run_sprint3b_kilo_v3.py`,
  `experiments/run_sprint3b_reduced_b0_t.py`, `experiments/sprint3.py`, and
  `experiments/sprint3a.py`;
- agent construction/runtime: prompts, coding agent, advisory/dependencies,
  hooks, model-output/pacing/stagnation modules, and all five registered tool
  implementations;
- treatment advisory/retrieval: advisory formatting/service, Neo4j repository,
  graph queries/read models/repository validation, integration runtime,
  retrieval applicability/candidates/query/scorer/service, recovery embeddings,
  and `memory/recovery_evidence.py`;
- recurrence/objective/environment/artifacts: research runner/contracts,
  artifact writer, benchmark environment/runtime-smoke helpers, domain action,
  actions, advice, behavior, events, failures, environment, recovery-pattern,
  and task contracts;
- declared inputs: the v4 YAML, Kilo model YAML, benchmark manifest and
  recurrence annotations, and benchmark-environment policy.

In particular, `src/graph_swarm/memory/recovery_evidence.py` is frozen because
the structured-pytest recurrence decision uses its trusted test-execution and
argument normalization semantics. A focused test simulates a changed hash for
that file and confirms `load_canary_context()` rejects the existing freeze.

## Required freeze/live sequence

1. Commit and push the reviewed source, config, and tests.
2. Confirm a clean Git working tree.
3. Create the freeze with `--create-freeze`.
4. Inspect and retain the resulting `freeze.json`.
5. Run `--preflight-only` and require `READY`.
6. Execute the six-run command exactly once; the live path rejects an existing
   primary artifact root before provider or task execution.

The preflight path covers frozen package/hash integrity, required credentials
and Kilo provider readiness, treatment embedding/Hugging Face readiness,
read-only Neo4j connectivity and canonical-pattern reads, and the selected
T006–T008 benchmark environments. It creates no primary result artifact and
does not execute a benchmark task. Neo4j write operations remain forbidden.
