# GS-E003 RFR evaluator correction

Date: 2026-09-30
Scope: evaluator-only offline replay of the retained GS-E003 Sprint3B v3 primary artifacts for GS-T006–GS-T011.

No model/Kilo call was made, no Neo4j write was made, and no retained artifact was modified.

## Correction

`experiments/sprint3.py::make_recurrence_matcher` now preserves the existing failed `run_tests` path and additionally evaluates a failed `run_command` only when its trusted structured planned argv is a supported pytest invocation:

- `pytest ...`
- `python -m pytest ...`
- `python3 -m pytest ...`
- the same forms with executable paths, matched by basename

Arbitrary commands, arbitrary Python commands, command failure text, and pytest text in arguments do not qualify. Recurrence still requires a substring match against the frozen `FAIL_TO_PASS` target or its existing pytest-target normalization.

The runner passes a typed `RecurrenceEventStream` containing the same `AgentEvent` sequence plus the trusted planned-action map. This preserves the public callback shape and persisted event schema while making structured argv available at the evaluator boundary. Frozen benchmark data remains evaluator-only; it is not passed to prompts, retrieval, `RecoveryPattern`, or treatment runtime.

## Offline replay result

| condition/task | artifact `known_failure_repeated` | old failed test executions | corrected failed test executions | frozen target matched | old result | corrected result |
|---|:---:|---|---|:---:|:---:|:---:|
| B0 / GS-T006 | false | #12 `run_tests`; no target | #12 `run_tests`; no target | no | false | false |
| B0 / GS-T007 | false | — | #20 `python -m pytest tests/test_idtracking.py -q` | yes: 2 IDs | false | true |
| B0 / GS-T008 | false | — | — | — | false | false |
| B0 / GS-T009 | false | — | #25 `python -m pytest -q patsy/mgcv_cubic_splines.py` | yes: `test_crs_compat` | false | true |
| B0 / GS-T010 | false | — | — | — | false | false |
| B0 / GS-T011 | false | — | #20 and #21 `python -m pytest` render commands | yes: render targets | false | true |
| T / GS-T006 | false | — | — | — | false | false |
| T / GS-T007 | false | — | — | — | false | false |
| T / GS-T008 | false | — | #13 and #18 `python -m pytest tests/dialects/test_spark.py::TestSpark -q` | yes: `test_spark` | false | true |
| T / GS-T009 | false | — | — | — | false | false |
| T / GS-T010 | false | — | — | — | false | false |
| T / GS-T011 | false | — | #11 `pytest tests/render/test_text.py -q` | yes: render targets | false | true |

The complete per-artifact machine-readable record, including every frozen target and every matched signature, is in [gs_e003_rfr_evaluator_correction.json](gs_e003_rfr_evaluator_correction.json).

### Numerators and changed artifacts

- Old recurrence numerator: **0/12** retained primary artifacts.
- Corrected recurrence numerator: **5/12** retained primary artifacts.
- Changed `false -> true`: **B0/GS-T007, B0/GS-T009, B0/GS-T011, T/GS-T008, T/GS-T011**.
- Seven failed pytest executions through `run_command` were recognized by the corrected evaluator; all seven matched at least one frozen target/signature.
- No observed failed pytest command remained non-recurrence because of a frozen-target mismatch. The failed B0/GS-T006 `run_tests` event remained false because its collection errors did not contain either frozen Tornado target.

The artifact boolean is intentionally not rewritten: every retained artifact still records its historical `known_failure_repeated: false` value.

## Focused behavior coverage

The added tests cover:

1. failed `run_tests` with a matching target remains true;
2. failed `run_tests` with an unrelated failure remains false;
3. failed `pytest ...` through `run_command` matches;
4. failed `python -m pytest ...` matches;
5. failed `python3 -m pytest ...` matches;
6. successful pytest remains false;
7. failed arbitrary `run_command` remains false;
8. failed arbitrary Python remains false;
9. failed pytest without a frozen target match remains false; and
10. frozen `FAIL_TO_PASS` data is used only by the evaluator and is absent from the task prompt.

## Validation

- Focused recurrence tests: `12 passed`.
- Relevant evaluator/runner/experiment tests: `16 passed, 97 deselected, 1 warning` (the warning is the existing Pydantic Evals event-loop deprecation warning).
- Full `tests/unit/research/test_sprint3.py`: `59 passed`.
- Ruff on all changed Python files: passed (`All checks passed!`).
- Pyright: passed (`0 errors, 0 warnings, 0 informations`).
- A targeted Pyright invocation including `experiments/sprint3.py` reports 11 pre-existing diagnostics at unrelated lines in that legacy module; none are at the evaluator correction lines.
- `git diff --check`: passed. Git emitted only the existing LF-to-CRLF normalization warnings.

No full repository test suite was run; no model or Neo4j access was used.

## Files changed

- `src/graph_swarm/memory/recovery_evidence.py`
- `src/graph_swarm/research/runner.py`
- `src/graph_swarm/research/__init__.py`
- `experiments/sprint3.py`
- `tests/unit/research/test_sprint3.py`
- this report and its JSON companion

No prompt, agent tool, budget, advisory, retrieval/applicability, model/provider, benchmark annotation, frozen evidence, or historical artifact file was changed.
