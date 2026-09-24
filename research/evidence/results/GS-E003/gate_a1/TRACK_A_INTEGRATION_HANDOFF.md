# Track A → Integration Point 1 Handoff

## 1. What Track A accomplished

Gate A1 acquisition corpus is complete: 5/5 logical acquisition tasks are eligible, verified/observed recovery patterns are persisted, and the canonical acquisition root is frozen. These are observed repository and evidence states, not causal claims.

## 2. Canonical paths

- acquisition root: `research/evidence/results/GS-E003/gate_a1/acquisition-r13b-20260923T115128Z`
- machine-readable closeout: `research/evidence/results/GS-E003/gate_a1/GATE_A1_R13B_CLOSEOUT.json`
- human-readable closeout: `research/evidence/results/GS-E003/gate_a1/GATE_A1_R13B_CLOSEOUT.md`
- machine-readable handoff: `research/evidence/results/GS-E003/gate_a1/TRACK_A_INTEGRATION_HANDOFF.json`

## 3. Frozen models

- coding: `nex-agi/nex-n2.5-pro:free`
- recovery abstraction: `cohere/north-mini-code:free`

## 4. Frozen runtime/config identity

- `run_revision = R13b`
- `config_version = gate-a1-r13b-v2`
- `prompt_version = v2-runtime-guidance`
- command policy: `structured_argv_canonicalization_v2`
- configuration hash: `606d69aac4ed506d1a3b738052bfef34238f8463777eec5fb08188e15d237529`
- objective anchoring: `objective_anchored_v1`
- recovery-event semantics: `objective_anchor_trusted_mutation_objective_success_v1`
- source/applicability metadata is recorded separately in the selected artifacts.

## 5. Logical task table

| Task | Selected run | Selected mechanism | Acquisition success | Trusted lineage | Pattern ID | Pattern persisted | Notes |
|---|---|---|---|---|---|---|---|
| GS-T001 | `GS-E003-A1-R13b-GS-T001-d2418a3ebc4a4ea9963cb5207bd73f3f` | initial | true | true | `recovery-pattern-ad07a6a45718b848a30ad377` | true | original successful acquisition |
| GS-T002 | `GS-E003-A1-R13b-GS-T002-bf6b0cd4d66949a7b1c2234c031ee770` | controlled retry, attempt 2 | true | true | `recovery-pattern-162e3999c4a2c66a1ff647ed` | true | initial timeout retained |
| GS-T003 | `GS-E003-A1-R13b-GS-T003-385aa250afca478c93f272ca58df68db` | pattern-only retry, attempt 1 | true | true | `recovery-pattern-f490f62ab931191c6eac6db1` | true | trusted evidence reused; original artifact retained |
| GS-T004 | `GS-E003-A1-R13b-GS-T004-ac49dfcc4dd64f6c9acaaab4a5ed885e` | controlled retry, attempt 2 | true | true | `recovery-pattern-99a54266f940e1d4648f4698` | true | original 3600-second timeout retained |
| GS-T005 | `GS-E003-A1-R13b-GS-T005-ba0d94867f6944cd8341a28a44c4013b` | initial | true | true | `recovery-pattern-fd7b65022b22dc5f2a42816f` | true | no retry required |

## 6. Retry history

GS-T002’s initial failed run is `GS-E003-A1-R13b-GS-T002-690b2420ef934192ac19c4bd3c112492`; its successful controlled retry is selected logically. GS-T003’s original run reached trusted recovery lineage but its downstream structured abstraction/pattern creation failed; `single_explicit_pattern_retry_v1` selected the successful pattern-only retry while reusing the trusted evidence. GS-T004’s initial run `GS-E003-A1-R13b-GS-T004-645478c46c064554a16408334b007222` timed out at 3600 seconds; controlled retry attempt 2 is selected. All failed/original evidence remains preserved.

## 7. Runtime guard

- overall agent timeout is configurable through `GRAPH_SWARM_AGENT_TIMEOUT_SECONDS`.
- the final acquisition profile used 7200 seconds where explicitly overridden; the selected artifact records retain the effective timeout per task.
- pre-mutation nudge: 1200 seconds.
- pre-mutation abort: 1800 seconds.
- objective timeout: 900 seconds.

Later final controlled experiments must freeze identical resource limits across conditions. Timeout and guard changes must not be introduced selectively between experimental conditions.

## 8. What Integration Point 1 must consume

Integration Point 1 should consume the frozen Gate A1 recovery corpus, canonical graph recovery records, selected logical task results, pattern IDs, applicability metadata, source provenance, and environment metadata. Do not reconstruct recovery evidence from raw traces when canonical records already exist.

## 9. What must NOT happen

- Do not rerun T001–T005.
- Do not mutate historical Gate A1 artifacts.
- Do not silently replace selected retry attempts.
- Do not change coding or abstraction models while claiming the same configuration.
- Do not preload future evaluation outcomes.
- Do not allow later tasks to use future history.
- Do not change final baseline/treatment resource limits independently.
- Do not describe observed recovery as causal proof.

## 10. Integration readiness marker

TRACK_A_GATE_A1_COMPLETE_READY_FOR_INTEGRATION_POINT_1
