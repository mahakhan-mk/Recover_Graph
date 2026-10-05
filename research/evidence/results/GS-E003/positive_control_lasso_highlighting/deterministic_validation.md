# GS-T018 deterministic validation

Validation used the frozen Pygments SWE-smith workspace, the derived image
`graph-swarm/gs-t018-python312:prepared`, upstream image digest
`sha256:ba403c3f4f52fb6ed76c0809b541f2a78ff7147a978a665c962ae11313002dd9`,
Python 3.12.1, and `--network none`.

- Frozen mutation applied: exact FAIL_TO_PASS exited 1 with both target tests
  failing. Repeated run: same result.
- Known repaired state restored: exact FAIL_TO_PASS exited 0 with 2 passed.
  Repeated run: same result.
- Frozen PASS_TO_PASS exited 0 with 2 passed. Repeated run: same result.
- The exact commands are valid pytest node targets.
- Offline replay of current hardened applicability found exactly
  `recovery-pattern-fd7b65022b22dc5f2a42816f`; no unrelated pattern was
  eligible.
- The first advisory boundary is the pre-tool `prepare_tool_action` boundary
  while planning the first eligible pytest `run_command`, before controlled
  test execution and before any source edit. The failed test observation then
  precedes source inspection and repair.
- No B0/T run, model/provider call, agent run, or Neo4j write occurred.
