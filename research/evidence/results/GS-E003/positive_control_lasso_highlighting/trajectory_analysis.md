# GS-T018 offline trajectory analysis

Status: candidate passes the pre-registration gates; no B0/T run was performed.

## Candidate

GS-T018 is `pygments__pygments.27649ebb.func_pm_remove_cond__ai6vpsek` at base
revision `d53d06ab2c467be43d32f68881422698ee33adf6`. It is the Pygments
LassoLexer candidate, not the current Rich or Patsy development case. Its
prepared environment reuses the Gate-A1 validated Pygments image
`swebench/swesmith.x86_64.pygments_1776_pygments.27649ebb` and asserts Python
`3.12.1` with execution network disabled.

## Frozen trajectory before any treatment edit

- Initially the model knows only the task statement: built-in and member names
  receive the wrong Pygments token class, and the two Lasso example fixtures
  are the failing boundary. It does not know which lexer method, state stack,
  or conditional was removed.
- The repair is not identifiable from the statement alone. Several mechanisms
  could explain the symptom: Lasso regex states, delimiter handling, token
  post-processing, the built-in/member tables, or fixture expectations.
- A competent agent has a natural first action: run the two FAIL_TO_PASS
  fixture targets. The expected first trusted failure boundary is
  `python -m pytest tests/examplefiles/lasso/json.lasso9:: tests/examplefiles/lasso/json.lasso:: -q`.
  The expected result is a deterministic fixture/golden mismatch showing that
  Lasso names are emitted as `Name.Other` rather than `Name.Builtin`.
- The actual first legitimate Graph Swarm opportunity is the pre-tool boundary
  when the model plans the first eligible `run_command` pytest action. The
  runtime records that action, evaluates applicability, and can raise
  `ModelRetry` before `controlled_run_command` executes. The completed failing
  pytest result is therefore the next trusted evidence boundary, not the first
  advisory boundary.
- This is still before any edit that can restore the lexer path. After the
  advisory, the model can run the frozen fixtures, observe the golden-token
  mismatch, and then inspect `LassoLexer.get_tokens_unprocessed()` before its
  first `edit_file`. The source repair is not disclosed before that test-first
  trajectory; a model that edits immediately remains a possible recurrence
  path, but the architecture does not postpone advice until after the edit.

## Expected recovery and semantic transfer

The expected frozen pattern is
`recovery-pattern-fd7b65022b22dc5f2a42816f`, the stackprinter
syntax-highlighting/token-iteration recovery. The genuine shared anchors are
the syntax-highlighting pipeline, lexer-produced token streams, and preserving
token classification while iterating emitted tokens. They are semantic/dataflow
anchors, not the generic words “test”, “failure”, “variable”, or “edit”.

The historical recovery can change the next action: after observing the
fixture failure, it directs the model to inspect the lexer’s token iteration
and downstream token conversion before changing regex definitions or fixture
data. In this concrete implementation, that inspection exposes the missing
Lasso delimiter-state setup and the missing built-in/member reclassification
branch. The guidance is therefore useful after failure observation and does
not disclose the exact source edit in the task statement.

Offline hardened-applicability replay predicts exactly the expected pattern at
the failing test boundary. The other four canonical patterns have no shared
non-generic trigger-context anchor and are rejected; generic execution terms
cannot establish applicability. No B0 outcome was used for this decision.

## Recurrence opportunity in B0

If B0 edits before testing, it may discover the missing `get_tokens_unprocessed`
path through source inspection and repair without advice. If it follows the
normal test-first path, the golden fixture failure creates a real pre-repair
advisory opportunity. The candidate is intentionally harder than a direct
undefined-variable or empty-iterator defect because the failure is a token
classification regression with two deleted behaviors in one lexer hook.

The candidate is rejected if later execution shows the source repair is obvious
before the first trusted test boundary or if the frozen FAIL_TO_PASS/PASS_TO_PASS
contract is not reproduced. This document is an offline selection record, not
an execution result.
