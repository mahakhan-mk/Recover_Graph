# GS-E003 trigger-applicability development audit

Read-only replay of retained GS-T006..GS-T015 trajectories and v3 vector candidate scores. Kilo was not called; no benchmark or Neo4j data was modified.

## Result

- Actions inspected: 229
- Corrected-trigger advice opportunities: 2
- Corrected expected-pattern task coverage: 2
- Corrected non-expected selections: 0

## Policy comparison

A uses the current edit_file recovery-action key. B uses the source failure action key. C requires source trigger action/intent, exact text anchors from the current task and historical context, chronology/status, runtime, and exact versions unless the trigger explicitly opts out.

| Policy | Advice opportunities | Actions with no eligible candidate | Non-expected selections |
|---|---:|---:|---:|
| current | 2 | 227 | 0 |
| source_failure_trigger | 8 | 92 | 5 |
| corrected_trigger | 2 | 224 | 0 |

| Task | Expected pattern | A first eligible | B first eligible | C first eligible | C selected | C advice count | C non-expected |
|---|---|---:|---:|---:|---|---:|---:|
| GS-T006 | `recovery-pattern-ad07a6a45718b848a30ad377` | 18 | 1 | 19 | `recovery-pattern-ad07a6a45718b848a30ad377` | 1 | 0 |
| GS-T007 | `recovery-pattern-162e3999c4a2c66a1ff647ed` | none | 1 | none | `none` | 0 | 0 |
| GS-T008 | `recovery-pattern-f490f62ab931191c6eac6db1` | 20 | 1 | 15 | `recovery-pattern-f490f62ab931191c6eac6db1` | 1 | 0 |
| GS-T009 | `recovery-pattern-99a54266f940e1d4648f4698` | none | 1 | none | `none` | 0 | 0 |
| GS-T010 | `recovery-pattern-fd7b65022b22dc5f2a42816f` | none | 1 | none | `none` | 0 | 0 |
| GS-T011 | `recovery-pattern-ad07a6a45718b848a30ad377` | none | 1 | none | `none` | 0 | 0 |
| GS-T012 | `recovery-pattern-162e3999c4a2c66a1ff647ed` | none | 1 | none | `none` | 0 | 0 |
| GS-T013 | `recovery-pattern-f490f62ab931191c6eac6db1` | none | none | none | `none` | 0 | 0 |
| GS-T014 | `recovery-pattern-99a54266f940e1d4648f4698` | none | none | none | `none` | 0 | 0 |
| GS-T015 | `recovery-pattern-fd7b65022b22dc5f2a42816f` | none | 1 | none | `none` | 0 | 0 |

## Version applicability audit

Exact Python-version equality remains enabled for the corrected trigger path. The retained pilot shows acquisition/runtime version mismatches for GS-T013 and GS-T014, but does not establish that either recovery is safe across Python versions; therefore C does not remove this gate.

## Manual review of non-expected advice

No corrected-trigger non-expected selections were produced.
