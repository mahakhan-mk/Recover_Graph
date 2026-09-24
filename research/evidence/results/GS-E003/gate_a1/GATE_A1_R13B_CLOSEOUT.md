# Gate A1 R13b Closeout

## Verdict

| Field | Value |
|---|---|
| experiment | GS-E003 |
| gate | Gate A1 |
| revision | R13b |
| status | PASSED |
| logical tasks | 5 |
| completed tasks | 5 |
| eligible acquisition tasks | 5 |
| acquisition corpus ready | true |
| manifest errors | [] |

The canonical acquisition root is:

`research/evidence/results/GS-E003/gate_a1/acquisition-r13b-20260923T115128Z`

The freeze was verified against commit `857bd073de420d0129435e3bb47e96db3741ad4f`.

Gate A1 passes because every selected logical task has complete trusted recovery lineage, successful objective-anchored recovery, and a persisted recovery pattern. This conclusion was cross-checked against the selected task artifact and, where applicable, its selected retry artifact; it is not inferred from counts alone.

## Final configuration identity

- `run_revision = R13b`
- `config_version = gate-a1-r13b-v2`
- `prompt_version = v2-runtime-guidance`
- coding model: `nex-agi/nex-n2.5-pro:free`
- recovery abstraction model: `cohere/north-mini-code:free`
- command policy: `structured_argv_canonicalization_v2`
- objective anchoring: `objective_anchored_v1`
- recovery-event semantics: `objective_anchor_trusted_mutation_objective_success_v1`
- agent timeout environment variable: `GRAPH_SWARM_AGENT_TIMEOUT_SECONDS`
- effective agent ceilings: T001 900 seconds; T002/T003 3600 seconds; T004/T005 7200 seconds
- objective timeout: 900 seconds
- pre-mutation stagnation guard: `pre_mutation_stagnation_guard_v1`, nudge 1200 seconds, abort 1800 seconds

## Selected logical results

| Task | Selected mechanism | Selected run | Pattern ID | Source → applicability | Verification |
|---|---|---|---|---|---|
| GS-T001 | initial | `GS-E003-A1-R13b-GS-T001-d2418a3ebc4a4ea9963cb5207bd73f3f` | `recovery-pattern-ad07a6a45718b848a30ad377` | `run_command` → `edit_file` | observed_successful |
| GS-T002 | controlled retry, attempt 2 | `GS-E003-A1-R13b-GS-T002-bf6b0cd4d66949a7b1c2234c031ee770` | `recovery-pattern-162e3999c4a2c66a1ff647ed` | `run_command` → `edit_file` | observed_successful |
| GS-T003 | pattern-only retry, attempt 1 | `GS-E003-A1-R13b-GS-T003-385aa250afca478c93f272ca58df68db` | `recovery-pattern-f490f62ab931191c6eac6db1` | `edit_file` → `edit_file` | observed_successful |
| GS-T004 | controlled retry, attempt 2 | `GS-E003-A1-R13b-GS-T004-ac49dfcc4dd64f6c9acaaab4a5ed885e` | `recovery-pattern-99a54266f940e1d4648f4698` | `run_command` → `edit_file` | observed_successful |
| GS-T005 | initial | `GS-E003-A1-R13b-GS-T005-ba0d94867f6944cd8341a28a44c4013b` | `recovery-pattern-fd7b65022b22dc5f2a42816f` | `run_command` → `edit_file` | observed_successful |

All five selected logical results have `task_success=true`, `acquisition_success=true`, `complete_trusted_lineage=true`, `pattern_created=true`, `pattern_embedded=true`, and `pattern_persisted=true`. Their objective-success trigger IDs and complete field-level verification are recorded in `GATE_A1_R13B_CLOSEOUT.json`.

## Retry and anomaly history

- GS-T002 initial run `GS-E003-A1-R13b-GS-T002-690b2420ef934192ac19c4bd3c112492` was unsuccessful with a documented agent wall-clock timeout. The successful controlled retry `GS-E003-A1-R13b-GS-T002-bf6b0cd4d66949a7b1c2234c031ee770` is the selected logical result. The failed initial artifact remains preserved.
- GS-T003 original run `GS-E003-A1-R13b-GS-T003-385aa250afca478c93f272ca58df68db` reached trusted recovery lineage, but downstream structured abstraction/pattern creation failed. The explicit `single_explicit_pattern_retry_v1` reused that trusted evidence and completed abstraction, embedding, and persistence. The original artifact remains preserved.
- GS-T004 initial run `GS-E003-A1-R13b-GS-T004-645478c46c064554a16408334b007222` reached its documented 3600-second wall-clock timeout without complete trusted recovery lineage. Controlled retry attempt 2, `GS-E003-A1-R13b-GS-T004-ac49dfcc4dd64f6c9acaaab4a5ed885e`, is selected. The timeout artifact remains preserved.
- GS-T005 succeeded initially and required no retry.

The following are development-history facts, not experimental treatment effects: structured argv canonicalization, fail-soft command handling, objective anchoring, source/applicability separation, runtime guidance, environment-configurable timeout, controlled failed-task retry, pattern-only retry, pre-mutation stagnation guard, and selected-attempt manifest aggregation.

## Preservation and execution boundary

The canonical R13b root and all original, retry, pattern-retry, task, workspace, configuration, objective, graph-persistence, and provenance evidence were preserved. This closeout sprint did not run the coding agent, abstraction model, embedding model, provider calls, acquisition, objective evaluation against benchmark workspaces, or Neo4j acquisition writes. No T001–T005 task was rerun.

