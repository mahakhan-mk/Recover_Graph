# GS-T018 preregistration

Status: `READY_FOR_NEW_B0_T` — preregistered, not run.

GS-T018 is the Pygments LassoLexer token-classification regression at base
revision `d53d06ab2c467be43d32f68881422698ee33adf6`. It is distinct from the
Rich and Patsy development cases. The mutation removes delimiter-state setup
and the built-in/member token reclassification branch from
`LassoLexer.get_tokens_unprocessed()`.

The deterministic FAIL_TO_PASS boundary is the pair of Lasso golden fixtures;
the PASS_TO_PASS boundary is pinned to the two token-type unit tests. The
expected first advisory opportunity is the failing test command, before the
first edit. The only expected eligible frozen pattern is
`recovery-pattern-fd7b65022b22dc5f2a42816f`, selected by the genuine shared
syntax-highlighting/lexer-token-flow anchors.

No B0/T run, model/provider call, agent run, or Neo4j write occurred before
registration.
