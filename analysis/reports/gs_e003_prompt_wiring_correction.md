# GS-E003 prompt wiring correction

This is an offline verification report for the development canary. No Kilo,
model, provider, retrieval service, or treatment execution was run.

- Configuration: `configs/experiments/sprint3b_kilo_v4_canary.yaml`
- Experiment identity: `GS-E003` development canary; this is not GS-E004
- System prompt identifier: `R13B_SYSTEM_PROMPT`
- Resolved prompt SHA256 (exact UTF-8 bytes): `f3bf4187cc9bfabcf2551aa3988e592a2d0f2d0e61ad2984d7d90ddf13d54502`
- B0 prompt SHA256: `f3bf4187cc9bfabcf2551aa3988e592a2d0f2d0e61ad2984d7d90ddf13d54502`
- T prompt SHA256: `f3bf4187cc9bfabcf2551aa3988e592a2d0f2d0e61ad2984d7d90ddf13d54502`
- B0/T prompt equality: verified by identical resolved prompt text and selector
- Max actions: `28`
- Max requests: `24`
- Provider/model: `kilo` / `nex-agi/nex-n2.5-pro`
- Tools: unchanged from the v3 policy (`read_file`, `write_file`, `edit_file`, `run_tests`, `run_command`)

`ExperimentRunner._build_agent` resolves the configured identifier and passes
the resulting prompt explicitly as `system_prompt` to `create_coding_agent`.
The B0 and T runner construction checks use the same loaded canary
configuration, so their prompt ID, prompt bytes, and prompt SHA256 are equal;
their only runtime difference remains the injected advisory/memory service.

No standard execution path for this canary silently relies on the coding
agent's default prompt. Custom injected agent factories remain an explicit
test/integration override and are not used by the canary runner construction.

`configs/experiments/sprint3b_kilo_v3.yaml` was not modified. Existing v2/v3
evidence was not modified. The correction does not change retrieval,
RecoveryTrigger, AdvisoryService, RFR evaluation, model/provider, tools, or
resource limits.

Validation summary:

- Focused prompt/runner/Sprint3B tests: `45 passed`
- Corrected recurrence evaluator tests: `12 passed`
- Ruff: passed
- Pyright: passed with `0 errors, 0 warnings, 0 informations`
- `git diff --check`: passed
- Full suite: `729 passed, 13 skipped, 12 failed`
- Full-suite failures were classified as missing frozen fixtures/baselines and
  frozen input-hash checks that correctly reject the changed prompt/runner
  source hashes; frozen evidence was not updated to suppress them.
