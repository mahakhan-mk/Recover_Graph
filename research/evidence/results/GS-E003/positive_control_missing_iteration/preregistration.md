# GS-T017 preregistration

Status: `READY_FOR_NEW_B0_T` — preregistered, not run.

This is a new Rich (`Textualize/rich`) positive control at base revision `9d8f9a372cc5916fd4781fec207ced7ddac2f08f`, distinct from GS-T005's stackprinter problem while sharing its structured source-token/syntax-highlighting subject.

The mutation truncates `Syntax.highlight()`'s `line_tokenize()` iteration over `lexer.get_tokens(code)`. The trusted fail-to-pass boundary is `tests/test_syntax.py::test_python_render_line_range_indent_guides`; the pass-to-pass boundaries are `tests/test_syntax.py::test_pygments_syntax_theme` and `tests/test_highlighter.py::test_highlight_json_with_indent`. The expected recovery remains `edit_file/edit_file`.

The hardened applicability replay found exactly `recovery-pattern-fd7b65022b22dc5f2a42816f` eligible. The shared anchors are `syntax`, `highlighting`, `lexer`, `tokens`, and line-range token fragments. These are semantic/dataflow anchors: Rich's lexer token stream is partitioned into line fragments and consumed by span generation. Generic execution words did not establish eligibility.

No B0/T treatment run, model/provider call, or Neo4j write occurred before this registration.
