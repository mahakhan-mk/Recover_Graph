# GS-E003 deterministic validation

Status: `UNSUITABLE_POSITIVE_CONTROL`

## A–D: deterministic behavior and environment

The provisional Patsy mutation removed the `token_strings` accumulator initialization from `_read_python_expr`.

- Broken workspace: `patsy/parse_formula.py::test__tokenize_formula` failed with `NameError: name 'token_strings' is not defined`.
- Repaired workspace: the same test function passed.
- PASS_TO_PASS checks for token spacing and `Origin` behavior passed in both workspaces.
- Docker reported Python `3.12.1`.

The image’s cached environment lacks pytest/numpy, so these exact test bodies were executed inside the Docker Python process with minimal import stubs. This candidate was stopped before benchmark registration; no treatment run was attempted.

## E–F: frozen applicability stop gate

The unchanged applicability service considered `recovery-pattern-162e3999c4a2c66a1ff647ed` eligible at the preregistered structured `run_command`/pytest boundary with Docker/Python 3.12.1.

However, it also considered three unrelated run-command patterns eligible:

- `recovery-pattern-ad07a6a45718b848a30ad377`
- `recovery-pattern-99a54266f940e1d4648f4698`
- `recovery-pattern-fd7b65022b22dc5f2a42816f`

The common context match is the generic `command` token in frozen test-failure signatures and structured command arguments. That does not satisfy the manually defensible shared-anchor requirement. Therefore F fails, and the candidate is unsuitable. No retrieval/applicability relaxation was attempted.

No benchmark manifest or recurrence annotation was updated, and no B0/T primary artifact was created.

## Repository validation

- Both JSON reports parse successfully.
- Ruff passed for `src` and `tests`.
- Pyright passed with 0 errors and 0 warnings.
- The focused applicability suite passed: 8 tests.
