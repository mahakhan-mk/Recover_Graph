# GS-T018 pre-mutation advisory timing validation

This is an offline replay of frozen historical prefixes. No model/provider
call, agent run, Neo4j write, or experiment slot was performed.

The historical T018/T source mutation was `edit_file` action
`2f234c0a-9f38-4b46-908b-bcb8d264737a`, immediately preceded by completed
source inspection action `92b6c6bf-cb36-40bf-8dc1-36c15b7d4179`. The generic
hook now evaluates a bounded completed-prefix probe before the controlled edit
or write operation. It excludes current edit contents and future outcomes.

At T018, the probe uses the existing retrieval and hardened applicability
implementation with a test-execution action shape. The prefix contains the
lexer, token, and highlighting evidence needed by the frozen fd7 trigger. The
actual mutation has not executed at that boundary, so the advisory is
strictly pre-repair. The expected unique selection is
`recovery-pattern-fd7b65022b22dc5f2a42816f`.

The safety replay is recorded in the adjacent JSON. T007 remains silent;
T006, T008, and T017 retain only their historically related pattern, with no
unrelated-pattern false positives. This report does not use any B0 outcome.
