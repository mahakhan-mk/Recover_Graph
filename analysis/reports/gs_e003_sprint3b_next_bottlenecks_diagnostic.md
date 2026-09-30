# GS-E003 Sprint3B next-bottlenecks diagnostic

Date: 2026-09-30  
Scope: read-only diagnostic after the RecoveryTrigger applicability correction.

No source, test, benchmark, evidence, model/provider, or Neo4j state was changed. No benchmark was rerun.

## Executive conclusion

The next bottlenecks are:

1. **Confirmed prompt/runtime wiring defect.** The Sprint3B YAML records `system_prompt: ROLLOUT1_SYSTEM_PROMPT` and a source path, but those fields are not declared in `ExperimentConfiguration` and are not passed to `create_coding_agent`. Both B0 and T therefore run with the short `ROLLOUT1_SYSTEM_PROMPT`; neither receives the R13B runtime guidance about structured argv, portable commands, `read_file`, or `edit_file`.
2. **Confirmed recurrence-evaluator under-detection.** `experiments/sprint3.py::make_recurrence_matcher` considers only failed `run_tests` events. In the retained v3 primary artifacts, seven unambiguous failed pytest executions were sent through `run_command`; all were recorded with `known_failure_repeated: false`. This makes the observed zero recurrence-trigger count an evaluator floor, not evidence of zero recurrence.
3. **Likely trajectory/budget bottleneck.** The complete retained audit shows all 20 diagnostic runs exhausted the configured 20-call audit budget. Treatment runs accumulated 10 invalid shell-syntax attempts and 22 unavailable-`rg` failures, and eight of ten treatment runs stopped before an edit. Advice is delivered pre-tool through `ModelRetry`, which adds a reconsideration/model-request path and can consume budget around the intervention.

The right next implementation sprint is a small, evidence-preserving correction sprint: wire an explicit Sprint3B runtime prompt through the runner, broaden recurrence matching to unambiguous test-intent `run_command` events, and add focused tests/replay fixtures. Raising `max_actions` should not be the first intervention.

## Evidence boundary

The retained v3 primary run artifacts contain B0/T pairs for GS-T006 through GS-T011 only: 12 artifacts. There are no retained v3 primary artifacts for GS-T012 through GS-T015.

The v3 `pre_run_diagnosis/diagnosis.json` contains a 20-run B0/T budget audit, but `analysis/diagnose_sprint3b_kilo_v3_retrieval.py` explicitly sets its source run root to `research/evidence/results/GS-E003/sprint3b_kilo_v2`. Thus the 20-row budget table below is a retained v3 diagnostic artifact whose trajectory source is v2, not a complete set of v3 Kilo primary runs. Detailed model-request, tool-count, test, and advice claims below are limited to the 12 actual v3 primary artifacts unless marked otherwise.

Relevant retained v3 summary fields: 229 corrected-replay actions inspected; 10 T tasks inspected; `benchmark_tasks_rerun: false`; `kilo_called: false`; read-only diagnostic; 2 advisory opportunities and 2 expected-pattern selections.

## B0/T orchestration and prompt trace

### Actual prompt text

`src/graph_swarm/agent/prompts.py` defines the actual default prompt as:

```text
Work only inside the provided workspace. Inspect relevant files before editing them.
Use the registered tools for all filesystem and process operations; never claim a
command or test ran unless you actually used a tool. Make the smallest reasonable
change, then run tests to verify it. Continue only within the configured resource
limits and stop when tests pass or the allowed run budget is exhausted. Do not
access hidden benchmark or research metadata.
```

The R13B prompt appends this runtime guidance:

```text
Runtime guidance:
Use run_command only for a single executable with structured argv. Shell
pipelines, redirection, &&, ||, and shell composition are unsupported.
Do not assume optional shell utilities such as rg or file are installed.
Prefer the registered repository tools and portable commands already known to
be available.
Use read_file to inspect relevant source regions.
For a small exact source-code change after locating the relevant text, prefer
edit_file instead of reconstructing an entire file or trying to perform
shell-based text editing.
Once the relevant defect and expected behavior are understood, make the
smallest reasonable edit and validate it instead of continuing broad,
unrelated repository exploration.
```

`R13B_SYSTEM_PROMPT` is the concatenation of those two blocks. It is the prompt used by the Gate A1 R13b acquisition path (`gate_a1_acquisition.py` passes it into `gate_a1_r13.py`).

### Sprint3B path

- `configs/experiments/sprint3b_kilo_v3.yaml` contains `system_prompt: ROLLOUT1_SYSTEM_PROMPT`, `system_prompt_source: src/graph_swarm/agent/prompts.py`, `max_actions: 28`, and `max_requests: 24`.
- `ExperimentConfiguration` in `src/graph_swarm/research/runner.py` does not declare `system_prompt` or `system_prompt_source`. The validated configuration therefore does not carry those YAML values into agent construction.
- `ExperimentRunner.run_case` builds the same task prompt and calls `run_coding_agent` with the configured action/request limits. The T path deliberately uses the same pre-tool path as B0; there is no T-specific prompt.
- `_build_agent` calls `create_coding_agent(settings, model=self.model, capabilities=[step_persistence])` without a `system_prompt` argument.
- `create_coding_agent` consequently falls back to `system_prompt or ROLLOUT1_SYSTEM_PROMPT`.
- `run_sprint3b_kilo_v3.py::build_condition_runner` constructs both conditions with the same `ExperimentRunner` configuration. B0 has `advisory_service=None`; T adds the real treatment repository/advisory service and fail-closed controls. This is the material B0/T runtime difference, not a prompt difference.
- `run_coding_agent` maps `max_actions` directly to PydanticAI `tool_calls_limit`. Advice is produced in `prepare_tool_action` before controlled tool execution; when advice exists it raises `ModelRetry`, causing another model reconsideration path before the next actual tool call.

Therefore the configured v3 value of 28 is the actual primary-run ceiling, but the YAML prompt label is metadata-only. The complete retained 20-run audit has a separate configured audit budget of 20 and must not be conflated with the v3 primary ceiling.

## Complete retained 20-run budget audit

`artifact_tool_calls` is the number retained in the artifact; `budget_consumed` is the parsed `UsageLimitExceeded` count. `pre-edit stop` means the retained audit found no edit before termination. A blank first-edit field means no edit was observed in that audit.

| condition | task | artifact calls | budget consumed | first edit | pre-edit stop | git inspections | invalid shell | `rg` failures |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| B0 | T006 | 17 | 21 | — | yes | 4 | 1 | 1 |
| B0 | T007 | 14 | 21 | — | yes | 4 | 0 | 0 |
| B0 | T008 | 16 | 22 | — | yes | 6 | 1 | 2 |
| B0 | T009 | 16 | 22 | — | yes | 4 | 0 | 0 |
| B0 | T010 | 19 | 24 | — | yes | 5 | 1 | 2 |
| B0 | T011 | 19 | 23 | — | yes | 6 | 1 | 1 |
| B0 | T012 | 17 | 23 | — | yes | 7 | 0 | 0 |
| B0 | T013 | 20 | 23 | 20 | no | 10 | 0 | 0 |
| B0 | T014 | 17 | 21 | — | yes | 6 | 1 | 2 |
| B0 | T015 | 14 | 21 | — | yes | 4 | 0 | 1 |
| T | T006 | 19 | 24 | 18 | no | 9 | 0 | 1 |
| T | T007 | 17 | 21 | — | yes | 4 | 1 | 2 |
| T | T008 | 19 | 23 | 18 | no | 5 | 0 | 0 |
| T | T009 | 17 | 21 | — | yes | 8 | 1 | 1 |
| T | T010 | 16 | 21 | — | yes | 2 | 0 | 1 |
| T | T011 | 15 | 22 | — | yes | 5 | 2 | 2 |
| T | T012 | 18 | 24 | — | yes | 9 | 1 | 2 |
| T | T013 | 20 | 23 | — | yes | 7 | 0 | 0 |
| T | T014 | 19 | 24 | — | yes | 6 | 0 | 2 |
| T | T015 | 20 | 24 | — | yes | 7 | 0 | 2 |

All 20 records exhausted the 20-call audit limit. The treatment side stopped before an edit on 8/10 records; B0 stopped before an edit on 9/10. This is a budget/trajectory signal, not a causal treatment estimate, because the 20-row source is the v2-derived retained diagnostic.

## Actual v3 primary trajectories: GS-T006–GS-T011

The following values are from each retained v3 `steps.sqlite` and `artifact.json`. `first test` means the first clearly test-like action identified in the retained action sequence; a `run_command` test is still a `run_command` for evaluator purposes.

| condition/task | calls; model requests | completed tool counts (`read/edit/write/run_tests/run_command`) | first mutation | first test-like action | task success | advice |
|---|---|---|---:|---:|:---:|---:|
| B0/T006 | 26; 12 | 2/1/0/1/22 | 11 | 12 (`run_tests`) | yes | 0 |
| B0/T007 | 26; 8 | 8/1/0/0/17 | 26 | 17 (`run_command`) | yes | 0 |
| B0/T008 | 27; 8 | 10/1/0/0/16 | 24 | 26 (`run_command`) | yes | 0 |
| B0/T009 | 27; 16 | 8/1/0/0/18 | 26 | 25 (`run_command`) | yes | 0 |
| B0/T010 | 27; 7 | 3/0/0/0/24 | — | — | no | 0 |
| B0/T011 | 28; 12 | 12/0/0/0/16 | — | 20 (`run_command`) | no | 0 |
| T/T006 | 27; 8 | 11/0/0/0/16 | — | — | no | 1 |
| T/T007 | 24; 6 | 10/0/0/0/14 | — | — | no | 0 |
| T/T008 | 24; 11 | 7/0/1/0/16 | 20 (`write_file`) | 11 (`run_command`) | yes | 2 |
| T/T009 | 24; 5 | 9/0/0/0/15 | — | — | no | 0 |
| T/T010 | 26; 9 | 6/0/0/0/20 | — | — | no | 0 |
| T/T011 | 17; 19 started / 18 completed | 7/0/0/0/10 | — | 11 (`run_command`) | no | 3 |

The retained B0 artifacts and T/T006–T010 artifacts terminate at the v3 28-call tool budget, including successful tasks; T/T011 has an incomplete/blank termination field while its event stream ends after the same budget-pressure trajectory. The apparent `task_success` values are benchmark/run artifact outcomes; they do not mean an edit occurred. B0 succeeded without treatment on T006–T009, while the retained T artifacts succeeded only on T008; this partial set is too small and provenance-mixed for a treatment effect claim.

### Advice timing and budget interaction

In the actual v3 treatment artifacts, advice was delivered at `before_tool_execution` / `pre_tool`:

- T006: one selected historical recovery, not accepted by the artifact (`advice_accepted: false`).
- T008: two deliveries of the same selected recovery pattern, accepted (`advice_accepted: true`).
- T011: three deliveries of the same selected recovery pattern, accepted.

The retained SQLite event counts show the intervention path: T006 had 28 tool-call starts but 27 completions; T008 had 26 starts but 24 completions; T011 had 20 starts but 17 completions and one incomplete model request. Advice is not a separately registered filesystem/process tool call, but `ModelRetry` adds model reconsideration and attempted-call accounting around the pre-tool boundary. This is consistent with advice consuming budget pressure, but the retained data does not support a clean causal estimate of how many calls were attributable to advice.

The corrected offline retrieval replay found only two eligible/advisory boundaries: GS-T006 selected `recovery-pattern-ad07a6a45718b848a30ad377`, and GS-T008 selected `recovery-pattern-f490f62ab931191c6eac6db1`. The existing corrected applicability report records both as expected selections and zero non-expected selections. That result should not be conflated with the actual treatment artifact advice counts for T011, which came from the retained primary artifact and reflect a separate observed retrieval event.

## Recurrence evaluator audit

`experiments/sprint3.py::make_recurrence_matcher` currently:

1. ignores occurrence 1;
2. loops over events;
3. accepts only `event.result.tool_name == "run_tests"` with `success == false`;
4. searches output/error for the frozen `fail_to_pass` test ID or its pytest target form.

It does not inspect `run_command`, even when its structured argv is unambiguously `pytest` or `python -m pytest`. It also correctly avoids treating arbitrary failed shell commands as test failures; the missing piece is a narrow test-intent classifier, not a blanket `run_command` match.

### Failed test executions visible in retained v3 primary artifacts

The following seven failed test executions were observed through `run_command` and therefore are invisible to the current matcher. Each retained artifact still has `known_failure_repeated: false`.

- B0/T007: `python -m pytest tests/test_idtracking.py -q`; failed IDs `tests/test_idtracking.py::test_complex` and `tests/test_idtracking.py::test_if_branching_stores`.
- B0/T009: `python -m pytest -q patsy/mgcv_cubic_splines.py`; failed ID `patsy/mgcv_cubic_splines.py::test_crs_compat`. A later repeat of the same command passed.
- B0/T011: first `python -m pytest tests/render/test_text.py -q`; failed IDs `tests/render/test_text.py::test_render_text[True-False-True-expected_output0]`, `tests/render/test_text.py::test_render_text[True-True-True-expected_output1]`, `tests/render/test_text.py::test_render_text[False-False-True-expected_output2]`, `tests/render/test_text.py::test_render_text[False-True-True-expected_output3]`, `tests/render/test_text.py::test_render_text_given_depth[True-2-expected_output2]`, `tests/render/test_text.py::test_render_text_encoding[2-utf-8-expected_output1]`, and `tests/render/test_text.py::test_render_text_with_license_info_and_reversed_tree[utf-8-expected_output0]`. A second narrower command failed `tests/render/test_text.py::test_render_text_encoding[2-utf-8-expected_output1]` again.
- T/T008: two failed executions of `python -m pytest tests/dialects/test_spark.py::TestSpark -q`; both reported `tests/dialects/test_spark.py::TestSpark::test_spark`.
- T/T011: failed `pytest tests/render/test_text.py -q`; it reported the same seven render-test IDs listed for B0/T011.

B0/T006 also had a failed `run_tests` event, but its retained output showed unrelated collection errors in `maint/test/cython/cythonapp_test.py` and `maint/test/redbot/red_test.py`, not a frozen task target. The current matcher therefore correctly returns false for that event under its current policy. The seven `run_command` failures are the evaluator false negatives.

The bounded conclusion is **underdetection / inconclusive for the full v3 run set**, not genuine zero recurrence-trigger rate. For the six retained v3 task pairs, the evidence is strong that the current matcher misses recurrence-shaped test failures; it is not sufficient to compute a corrected experiment-level RFR because v3 primary artifacts for T012–T015 are absent and the frozen fail-to-pass lists are not embedded in each artifact.

## Findings by confidence

### Confirmed

- Sprint3B B0/T do not receive the R13B runtime guidance. The YAML prompt fields do not reach `create_coding_agent`.
- B0/T share the same task prompt and agent-construction path; T adds advisory runtime services but no alternate system prompt.
- The recurrence matcher has a hard `run_tests` tool-name gate.
- Seven unambiguous failed pytest executions in retained v3 primary evidence are consequently missed.
- All 20 retained diagnostic records exhaust the 20-call audit budget; 8/10 T records stop before edit.

### Likely

- Missing structured-argv/portable-tool guidance contributes to unsupported-shell attempts, unavailable `rg`, repeated repository-state inspection, and delayed/no mutation.
- Treatment advice adds pre-tool reconsideration and can consume budget, particularly when repeated (T/T008 and T/T011), but the retained sample cannot isolate its causal effect.
- The two corrected RecoveryTrigger applicability selections are not the next bottleneck: they are expected-pattern selections with zero non-expected selections in the corrected offline replay.

### Inconclusive / not estimable here

- Full v3 B0/T success or RFR for T012–T015: primary v3 artifacts are absent.
- Complete v3 model-request counts for the 20-run audit: not retained in the v3 diagnosis; only the 12 primary SQLite artifacts expose them.
- Exact corrected recurrence numerator/denominator for the full experiment: frozen fail-to-pass lists are not embedded in the primary artifacts, and the evaluator currently misses `run_command` test intent.
- A causal claim that treatment advice caused any specific failure or budget exhaustion.

## Smallest next implementation sprint

1. Make the Sprint3B runtime prompt explicit in the condition-runner/agent construction path, with B0 and T receiving the same frozen prompt. Use the R13B runtime guidance or an explicitly frozen Sprint3B equivalent; do not leave `system_prompt` as YAML-only metadata.
2. Extend `make_recurrence_matcher` with a conservative structured-argv classifier for `run_command` test intent (`pytest`, `python -m pytest`, and equivalent portable forms), retaining the existing frozen-signature check and excluding arbitrary failed commands.
3. Add focused tests/replay fixtures for prompt selection, B0/T prompt equality, test-intent classification, and the seven observed failure shapes.
4. Re-run the read-only evaluator/unit checks first. Do not increase the action budget until prompt/runtime and evaluator instrumentation are corrected, because a larger budget would mainly create more unclassified trajectory data.

## Source map

- Prompt definitions: `src/graph_swarm/agent/prompts.py:3-28`.
- Agent default prompt and usage limits: `src/graph_swarm/agent/coding_agent.py:158-188, 324-384, 486-498`.
- Advisory pre-tool boundary: `src/graph_swarm/agent/advisory.py:26-107`.
- Configuration loading, run path, and agent construction: `src/graph_swarm/research/runner.py:105-137, 207-232, 884-902, 1004-1022`.
- B0/T runner construction: `experiments/run_sprint3b_kilo_v3.py:488-529, 680-720`.
- R13B Gate A1 prompt injection: `src/graph_swarm/research/gate_a1_acquisition.py:2355-2392`; forwarding in `src/graph_swarm/research/gate_a1_r13.py:303-317, 783-787`.
- Recurrence matcher: `experiments/sprint3.py:519-548`.
- v3 diagnosis provenance: `analysis/diagnose_sprint3b_kilo_v3_retrieval.py:1050-1055`.
- Retained evidence: `research/evidence/results/GS-E003/sprint3b_kilo_v3/`.
