# GS-E003 development positive-control preregistration

Status: `UNSUITABLE_POSITIVE_CONTROL`

This is a provisional, pre-treatment record for `GS-T016`; it was not added to the benchmark manifest or recurrence annotation because the frozen applicability stop gate failed.

## Candidate

- Repository: `swesmith/pydata__patsy.a5d16484`
- Base revision: `aac7539dbb86b0cc177a8c74e9a7a28bda354cda`
- Family: `missing_initialization_or_assignment`
- Source historical task: `GS-T002`
- Expected pattern: `recovery-pattern-162e3999c4a2c66a1ff647ed`
- Environment: Docker, Python `3.12.1`
- Expected operation: `edit_file/edit_file`

The candidate is a formula-tokenizer failure caused by an omitted accumulator initialization in `patsy/parse_formula.py`. Its independent shared anchor is lexical/text analysis with token accumulation before downstream parsing. It uses different code and test IDs from GS-T002, GS-T007, and GS-T012.

## Preregistered objective

FAIL_TO_PASS: `patsy/parse_formula.py::test__tokenize_formula`

PASS_TO_PASS:

- `patsy/tokens.py::test_pretty_untokenize_and_normalize_token_spacing`
- `patsy/origin.py::test_Origin`

Expected first boundary: structured `run_command` invoking `python -m pytest patsy/parse_formula.py::test__tokenize_formula -q`.

## Stop-gate result

The broken workspace failed with the expected `NameError`; the repaired workspace passed the same validation function, and the unrelated token-spacing/origin checks passed. Docker reported Python `3.12.1`.

The unchanged applicability service considered the expected pattern eligible, but also considered three unrelated run-command patterns eligible at the same boundary: `ad07a6a45718b848a30ad377`, `99a54266f940e1d4648f4698`, and `fd7b65022b22dc5f2a42816f`. The common match is the generic `command` token in frozen test-failure signatures, not the intended lexical-analysis anchor. This fails requirement F and is not a scientifically clean positive control.

Per instruction, work stopped. No retrieval/applicability code, thresholds, prompt, tools, budgets, manifest, annotation, or agent experiment was changed. No Kilo, model, provider, benchmark-agent, or Neo4j call was made.
