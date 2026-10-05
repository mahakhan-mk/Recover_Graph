# GS-E003 T008 regression and timing audit

This is a read-only audit. No Kilo call, model/provider call, benchmark execution, Docker benchmark execution, Neo4j write, or experiment-behavior change was performed.

## Determination

`recovery-pattern-f490f62ab931191c6eac6db1` remains selectable under the current implementation. The current Qwen T008 run selected it once at vector score `0.7200925350189209`; the retained v3 T008 run selected the same expected pattern twice. No non-expected pattern was selected in either trajectory or in the corrected offline replay.

The current Qwen source fix completed at `2026-09-30T18:09:22.094244Z`. Advice was issued at `2026-09-30T18:09:26.989074Z`, after that edit. The historical v3 trajectory issued advice at `17:25:36` and `17:26:14`, but both planned `edit_file` actions were intercepted; the actual source-changing `write_file` recovery completed at `17:27:42`. Thus historical advice preceded the actual recovery, although the model had already planned equivalent edits.

The difference is explained by model trajectory and evidence timing, not by a demonstrated trigger/applicability regression. The current run did not execute a failed test before advice: `run_tests` was intercepted at the pre-tool advisory boundary. The later explicit `python test_date_add.py` command passed both the three-argument and two-argument cases.

## Ordered current Qwen trajectory

| Model request | Planned tool + operation | Executed tool + operation | Result and evidence |
|---:|---|---|---|
| 1 | `run_command/run_command`, shell-composition search | `run_command/run_command` | Rejected; no edit. |
| 2 | `run_command/run_command`, file listing | `run_command/run_command` | Success; exploration. |
| 3 | `read_file/read_file`, `sqlglot/dialects/spark.py` | `read_file/read_file` | Success; broken `_build_dateadd` ordering observed. |
| 4 | `edit_file/edit_file`, assign `expression` before use | `edit_file/edit_file` | Success; relevant source edit completed. |
| 5 | `run_tests/run_tests` | None | Intercepted by advice; no failed test execution. |
| 5 | Recovery advice | Advice event | Expected pattern selected after the source edit. |
| 6 | `write_file/write_file`, create `test_date_add.py` | `write_file/write_file` | Success; behavior evidence marked structural change. |
| 7 | `run_command/run_command`, `python test_date_add.py` | `run_command/run_command` | Success for both DATE_ADD forms; objective result true. |
| 8 | Final response | None | No recurrence event observed. |

The persisted behavior evidence says `behavior_changed: true` because the next action (`write_file`) differed structurally from the advised `run_tests` action. That is not source-edit attribution: the source had already changed before advice was issued.

## Ordered retained v3 trajectory

| Model request | Planned tool + operation | Executed tool + operation | Result and evidence |
|---:|---|---|---|
| 1 | `run_command/run_command` status and shell-composition listing | Same tools | One success and one tool-shape failure. |
| 2 | `run_command/run_command` diff/grep; `read_file/read_file` source | Same tools | Source and tests located. |
| 3 | Diff/status and source/test reads | Same tools | Workspace state and surrounding implementations inspected. |
| 4 | Targeted pytest, direct reproduction, HEAD source | Same tools | Targeted pytest and reproduction failed before recovery. |
| 5 | `edit_file/edit_file`, correct ordering | None | Advice issued before execution; planned edit was not applied. |
| 6 | Reads, diffs, reproduction, targeted pytest | Same tools | Second targeted pytest failed; source still broken. |
| 7 | `edit_file/edit_file`, correct ordering | None | Advice issued before execution; planned edit was not applied. |
| 8 | `read_file/read_file`, source inspection | Same tool | Confirmed the edits had not applied. |
| 9 | `write_file/write_file`, corrected `spark.py` | `write_file/write_file` | Actual recovery/edit completed. |
| 10 | Reads and diff/status commands | Same tools | Repair inspected; no later failed test recorded. |
| 11 | Final diff diagnostics | None | Tool-call budget ended the run; objective result true. |

Historical failed test executions were recorded before the actual recovery at approximately `17:25:02` and `17:25:56`. Historical advice was pre-tool advice, but it preceded the actual source-changing `write_file` at `17:27:42`.

## Answers to the regression questions

1. Yes. The expected `recovery-pattern-f490f62ab931191c6eac6db1` remained eligible and selected.
2. No. No non-expected pattern was selected.
3. In the current Qwen run, the source fix occurred before Graph Swarm advice.
4. In the historical run, advice preceded the actual recovery/edit; the model had only planned equivalent edits before the advice boundaries.
5. No genuine trigger/applicability regression is demonstrated. The corrected replay reports T008 expected-pattern eligibility and zero non-expected selections.
6. The current difference is primarily model trajectory plus timing/instrumentation. It is not supported as an evaluator failure or implementation defect. The evaluator did not observe a current failed test before advice, while the historical run did; the advice evidence itself is not a causal source-change measure.
7. Yes. `RecoveryPatternApplicabilityService` has an explicit test-action bridge: a historical test-execution trigger can match a current `run_tests` action. The current Qwen advice is the concrete post-edit example.
8. This is expected but weak causal evidence, compounded by an instrumentation limitation. `record_post_advice_action` records structural next-action change, not whether the source repair occurred after advice. The evidence does not demonstrate a generic mechanism bug requiring correction.

## Current T008 classification

`GS-T008`: inconclusive.

The task passed and the expected pattern was selected, but the advice arrived after the source edit and before any executed failed test. That supports neither prevention evidence nor a regression claim. It is also insufficient as mechanism evidence because the successful verification cannot be causally attributed to the advice.

For completeness, the other current Qwen canary classifications are:

- `GS-T006`: mechanism evidence. The expected pattern was selected, one advice event was emitted, behavior evidence recorded a post-advice action change, and the objective passed. This does not establish causal prevention.
- `GS-T007`: safety evidence, bounded. No advice or non-expected pattern was emitted, but the request budget expired during exploration and the objective did not pass; therefore this is not mechanism or prevention evidence.

## Evidence boundary

The offline corrected-trigger replay inspected 229 actions across GS-T006 through GS-T015 and found two corrected advice opportunities, two expected-pattern task coverages, and zero corrected non-expected selections. For T008 it found the expected pattern as the first corrected eligible selection and no non-expected selection. This is diagnostic evidence, not a new run.

Recommendation: C. No existing candidate is scientifically defensible; construct one new preregistered development recurrence pair before running.
