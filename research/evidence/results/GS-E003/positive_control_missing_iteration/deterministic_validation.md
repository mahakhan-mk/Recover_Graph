# GS-T017 deterministic validation

All offline gates passed in the Docker image built from `python:3.12.1-slim` with Python `3.12.1`, pytest `8.3.5`, Pygments `2.18.0`, and markdown-it-py `3.0.0`. Test containers used `--network none`.

- A: the broken workspace fails `tests/test_syntax.py::test_python_render_line_range_indent_guides` with exit code 1 and an empty rendered result.
- B: the repaired workspace passes that boundary with exit code 0 (`1 passed`).
- C: both pass-to-pass tests pass with exit code 0 (`2 passed`) in both workspaces.
- D: Docker and Python 3.12.1 are confirmed.
- E: the unchanged hardened applicability service selects `recovery-pattern-fd7b65022b22dc5f2a42816f`.
- F: the other four canonical patterns are rejected with `trigger_context_mismatch`; none is eligible or preferred at this boundary.
- G: the anchor is semantic/dataflow evidence in `rich/syntax.py:508-540`: `line_tokenize()` iterates `lexer.get_tokens(code)`, partitions token text by newline, and `tokens_to_spans()` consumes those fragments for line-range highlighting. Generic execution words were not used.

No B0/T treatment run, model/provider call, or Neo4j write was performed.
