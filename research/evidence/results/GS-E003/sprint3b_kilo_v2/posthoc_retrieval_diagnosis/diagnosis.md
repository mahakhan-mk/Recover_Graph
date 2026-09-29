# Sprint 3B Kilo v2 post-hoc retrieval diagnosis

Read-only audit of GS-T006..GS-T015. No experiment, Graph Swarm retrieval behavior, benchmark artifact, or Neo4j data was modified. Kilo was not called.

## Summary

- T tasks inspected: 10
- Unique actual actions inspected: 229
- Actions with at least one eligible pattern: 0
- Actions where retrieval selected a pattern: 0
- Actions where AdvisoryService produced advice: 0

## Candidate rejection counts

| Reason | Count |
|---|---:|
| `operation_mismatch` | 1135 |
| `runtime_mismatch` | 1145 |
| `tool_mismatch` | 1135 |

## Expected-pattern status

| Task | Expected pattern | Source task | Status | Evidence |
|---|---|---|---|---|
| GS-T006 | `recovery-pattern-ad07a6a45718b848a30ad377` | GS-T001 | **B** | present but rejected: operation_mismatch=23, runtime_mismatch=24, tool_mismatch=23 |
| GS-T007 | `recovery-pattern-162e3999c4a2c66a1ff647ed` | GS-T002 | **F** | never applicable to any actual action |
| GS-T008 | `recovery-pattern-f490f62ab931191c6eac6db1` | GS-T003 | **B** | present but rejected: operation_mismatch=24, runtime_mismatch=25, tool_mismatch=24 |
| GS-T009 | `recovery-pattern-99a54266f940e1d4648f4698` | GS-T004 | **F** | never applicable to any actual action |
| GS-T010 | `recovery-pattern-fd7b65022b22dc5f2a42816f` | GS-T005 | **F** | never applicable to any actual action |
| GS-T011 | `recovery-pattern-ad07a6a45718b848a30ad377` | GS-T001 | **F** | never applicable to any actual action |
| GS-T012 | `recovery-pattern-162e3999c4a2c66a1ff647ed` | GS-T002 | **F** | never applicable to any actual action |
| GS-T013 | `recovery-pattern-f490f62ab931191c6eac6db1` | GS-T003 | **F** | never applicable to any actual action |
| GS-T014 | `recovery-pattern-99a54266f940e1d4648f4698` | GS-T004 | **F** | never applicable to any actual action |
| GS-T015 | `recovery-pattern-fd7b65022b22dc5f2a42816f` | GS-T005 | **F** | never applicable to any actual action |

## Acquisition-only marker audit

No: the Sprint 3B EnvironmentContext has no markers, so the existing service compares no marker keys; the acquisition-only markers do not cause rejection or eligibility.

Historical keys: `memory_write_only, retrieval_performed`; current keys: ``; compared keys: `(none)`.

## Evidence-supported root causes

- **vector candidate generation** — All 229 action replays returned the five canonical vector candidates; no vector-generation failure was observed.
- **chronology/verification** — No chronology, invalidation, stale, or verification rejection occurred.
- **structural applicability** — The frozen patterns all require edit_file/edit_file. Most generated actions were run_command or read_file, so they had no structural match.
- **environment applicability** — The two structurally matching edit_file actions were rejected because the runner supplied runtime=python while every canonical pattern requires runtime=docker.
- **ranking** — Ranking was never reached for an eligible candidate.
- **AdvisoryService conversion** — No selected pattern reached AdvisoryService conversion.
- **agent injection** — No advice event was injected; this is downstream of zero retrieval selections.

The coverage failure occurs before ranking: structural action-class mismatch explains the majority of actions, and the runner environment/runtime mismatch rejects the only two `edit_file` actions. Retrieval, AdvisoryService conversion, and injection therefore have no selected-pattern path to process.

Full candidate/action evidence is in `diagnosis.json`.
