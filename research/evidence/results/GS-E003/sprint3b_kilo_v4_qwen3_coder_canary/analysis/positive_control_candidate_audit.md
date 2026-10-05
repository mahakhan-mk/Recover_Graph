# GS-E003 positive-control candidate audit

This is a read-only, pre-outcome audit of GS-T009 through GS-T015. No previous treatment result was used to select a candidate. No task-specific rule was added, no trigger/context matching was loosened, and no Kilo, benchmark, or Neo4j write was performed.

## Result

No existing task from GS-T009 through GS-T015 is a scientifically defensible controlled development positive-control pair under the current RecoveryTrigger semantics.

The corrected offline replay inspected 229 actions. It found no eligible action for any of these seven tasks. The current-policy replay likewise found no eligible action. GS-T013 and GS-T014 additionally fail the exact Python-version applicability gate (`3.10.12` versus the frozen patterns' `3.12.1`). The other tasks are runtime-compatible but lack a defensible source-trigger context intersection in the observed action stream; generic terms such as “variable,” “error,” “wrong,” “operator,” or “test” are deliberately insufficient.

## Candidate-by-candidate assessment

| Task | Expected pattern | Source task | Environment | Defensible opportunity? | Main blocker | Suitable control? |
|---|---|---|---|---|---|---|
| GS-T009 | `recovery-pattern-99a54266f940e1d4648f4698` | GS-T004 | Docker, Python 3.12.1 | No | No exact operator-semantic trigger/context intersection; replay found no eligible action. | No |
| GS-T010 | `recovery-pattern-fd7b65022b22dc5f2a42816f` | GS-T005 | Docker, Python 3.12.1 | No | No exact missing-iteration trigger/context intersection; replay found no eligible action. | No |
| GS-T011 | `recovery-pattern-ad07a6a45718b848a30ad377` | GS-T001 | Docker, Python 3.12.1 | No | Rendering indentation is not the source locale/delta context; broad matching would invite false transfer. | No |
| GS-T012 | `recovery-pattern-162e3999c4a2c66a1ff647ed` | GS-T002 | Docker, Python 3.12.1 | No | No exact missing-initialization/dataflow context intersection; replay found no eligible action. | No |
| GS-T013 | `recovery-pattern-f490f62ab931191c6eac6db1` | GS-T003 | Docker, Python 3.10.12 | No | Exact version mismatch, plus no source-specific prerequisite-ordering context. | No |
| GS-T014 | `recovery-pattern-99a54266f940e1d4648f4698` | GS-T004 | Docker, Python 3.10.12 | No | Exact version mismatch, plus no source-specific operator-semantic context. | No |
| GS-T015 | `recovery-pattern-fd7b65022b22dc5f2a42816f` | GS-T005 | Docker, Python 3.12.1 | No | Empty-document output does not establish the source iteration anchors; replay found no eligible action. | No |

## Frozen relationships and trigger boundaries

The frozen transfer mapping is:

- GS-T009 → `99a542...` from GS-T004, operator semantics, occurrence 2.
- GS-T010 → `fd7b...` from GS-T005, missing iteration, occurrence 2.
- GS-T011 → `ad07...` from GS-T001, conditional polarity, occurrence 3.
- GS-T012 → `162e...` from GS-T002, missing initialization/assignment, occurrence 3.
- GS-T013 → `f490...` from GS-T003, prerequisite ordering, occurrence 3.
- GS-T014 → `99a542...` from GS-T004, operator semantics, occurrence 3.
- GS-T015 → `fd7b...` from GS-T005, missing iteration, occurrence 3.

For a candidate to be defensible, a legitimate runtime action must satisfy the frozen trigger semantics: compatible runtime/version facts, a trusted test-execution intent (`run_tests` or an unambiguous structured pytest command), and a non-generic context anchor shared by the current task and historical failure trigger. The existing tasks fail one or more of these requirements. The diagnostic’s corrected-trigger table records `none` for every one of GS-T009 through GS-T015.

## Best basis for one new development pair

If one new pair must be constructed, the best unused-by-this-canary basis is `recovery-pattern-162e3999c4a2c66a1ff647ed` from GS-T002, the missing-initialization/assignment pattern. This is a construction basis only, not a recommendation to run now.

The new task must be related but non-identical: a different repository/problem and a separately preregistered fail-to-pass/pass-to-pass set, with no reuse of GS-T007 or GS-T012 artifacts or gold solution. It should remain in Docker on exact Python 3.12.1, expose its first legitimate opportunity at a trusted test action, and share at least one genuine subject-matter anchor with the source failure trigger. Generic defect vocabulary cannot supply that anchor. Recovery applicability must remain `edit_file/edit_file`; no version-sensitive opt-out, threshold change, prompt change, or task-specific rule is warranted by this audit.

No task is created or run here.

Recommendation: C. No existing candidate is scientifically defensible; construct one new preregistered development recurrence pair before running.
