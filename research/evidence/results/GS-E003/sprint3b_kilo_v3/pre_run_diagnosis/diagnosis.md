# Sprint 3B Kilo v3 corrected offline replay

Read-only audit of GS-T006..GS-T015. No experiment, Graph Swarm retrieval behavior, benchmark artifact, or Neo4j data was modified. Kilo was not called.

## Summary

- T tasks inspected: 10
- Unique actual actions inspected: 229
- Actions with at least one eligible pattern: 2
- Actions where retrieval selected a pattern: 2
- Actions where AdvisoryService produced advice: 2

## Candidate rejection counts

| Reason | Count |
|---|---:|
| `operation_mismatch` | 1135 |
| `runtime_mismatch` | 0 |
| `tool_mismatch` | 1135 |
| `version_mismatch:python` | 235 |

## Expected-pattern status

| Task | Expected pattern | Source task | Status | Evidence |
|---|---|---|---|---|
| GS-T006 | `recovery-pattern-ad07a6a45718b848a30ad377` | GS-T001 | **E** | selected and AdviceResult returned advice |
| GS-T007 | `recovery-pattern-162e3999c4a2c66a1ff647ed` | GS-T002 | **F** | never applicable to any actual action |
| GS-T008 | `recovery-pattern-f490f62ab931191c6eac6db1` | GS-T003 | **E** | selected and AdviceResult returned advice |
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

- **vector candidate generation** â€” All 229 action replays returned the five canonical vector candidates; no vector-generation failure was observed.
- **chronology/verification** â€” No chronology, invalidation, stale, or verification rejection occurred.
- **structural applicability** â€” The frozen patterns all require edit_file/edit_file. Most generated actions were run_command or read_file, so they had no structural match.
- **environment applicability** â€” The corrected replay supplies the prepared Docker runtime and Python version to T; structurally matching edit_file actions are no longer rejected for runtime mismatch.
- **ranking** â€” Corrected replay ranking is reached only where a candidate survives applicability.
- **AdvisoryService conversion** â€” Only corrected retrieval selections reach AdvisoryService conversion.
- **agent injection** â€” Observed artifact advice events remain historical v2 evidence; this replay does not mutate or rerun the agent.

This replay corrects only the T EnvironmentContext runtime boundary. The separate structural applicability question is reported below as a counterfactual and is not applied to production retrieval.

## CURRENT vs FAILURE-TRIGGER structural audit

The audit reuses the exact vector candidates produced before policy projection. `CURRENT` uses `applicability_tool`/`applicability_operation`; `FAILURE-TRIGGER` uses `source_tool`/`source_operation`.

Full candidate/action evidence is in `diagnosis.json`.

### GS-T006 — expected `recovery-pattern-ad07a6a45718b848a30ad377`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | 18 | 1 | 1 | True | 1 | 0 |
| failure_trigger | 1 | 19 | 19 | True | 19 | 0 |

### GS-T007 — expected `recovery-pattern-162e3999c4a2c66a1ff647ed`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | None | 0 | 0 | False | 0 | 0 |
| failure_trigger | 1 | 14 | 14 | False | 0 | 14 |

### GS-T008 — expected `recovery-pattern-f490f62ab931191c6eac6db1`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | 20 | 1 | 1 | True | 1 | 0 |
| failure_trigger | 1 | 20 | 20 | True | 20 | 0 |

### GS-T009 — expected `recovery-pattern-99a54266f940e1d4648f4698`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | None | 0 | 0 | False | 0 | 0 |
| failure_trigger | 1 | 16 | 16 | False | 0 | 16 |

### GS-T010 — expected `recovery-pattern-fd7b65022b22dc5f2a42816f`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | None | 0 | 0 | False | 0 | 0 |
| failure_trigger | 1 | 14 | 14 | False | 0 | 14 |

### GS-T011 — expected `recovery-pattern-ad07a6a45718b848a30ad377`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | None | 0 | 0 | False | 0 | 0 |
| failure_trigger | 1 | 16 | 16 | False | 0 | 16 |

### GS-T012 — expected `recovery-pattern-162e3999c4a2c66a1ff647ed`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | None | 0 | 0 | False | 0 | 0 |
| failure_trigger | 1 | 18 | 18 | True | 18 | 0 |

### GS-T013 — expected `recovery-pattern-f490f62ab931191c6eac6db1`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | None | 0 | 0 | False | 0 | 0 |
| failure_trigger | 1 | 0 | 0 | False | 0 | 0 |

### GS-T014 — expected `recovery-pattern-99a54266f940e1d4648f4698`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | None | 0 | 0 | False | 0 | 0 |
| failure_trigger | 1 | 0 | 0 | False | 0 | 0 |

### GS-T015 — expected `recovery-pattern-fd7b65022b22dc5f2a42816f`

| Policy | First match | Eligible actions | Selections | Expected selected | Same-pattern advice | Incorrect selections |
|---|---:|---:|---:|---|---:|---:|
| current | None | 0 | 0 | False | 0 | 0 |
| failure_trigger | 1 | 20 | 20 | False | 0 | 20 |
