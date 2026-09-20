# GS-E003 Gate A1 — R3 Acquisition Decision

## Status

**Gate A1 retrieval evaluation: BLOCKED / NON-EVALUABLE under R3**

R3 acquisition completed all five canonical acquisition tasks, but produced zero eligible RecoveryPatterns. Because the retrieval corpus is empty, T006–T015 cannot meaningfully evaluate retrieval quality under this configuration.

This result is an acquisition-stage limitation. It is not evidence that Graph Swarm retrieval or behavioral treatment is ineffective, because neither retrieval nor behavioral treatment was executed.

## Frozen R3 Configuration

- Run revision: R3
- Namespace: `GS-E003/Gate-A1/acquisition-r3`
- Coding model: `cohere/north-mini-code:free`
- Recovery abstraction model: `cohere/north-mini-code:free`
- Temperature: 0
- Maximum actions/tool calls: 20
- Maximum requests: 24
- Agent wall-clock deadline: 300 seconds
- Prompt version: v1
- Configuration SHA-256: `c224720f3af29a5b1ccf68c3adcdd052eaade940cdd605f5b5f1551945510186`

Authoritative acquisition root:

`research/evidence/results/GS-E003/gate_a1/acquisition-r3-20260920T171809Z`

Authoritative manifest SHA-256:

`871824fc411f45d278b56a51bffaa38f47dcfe51e652f06ca87b56a7bda42330`

## Acquisition Results

| Task | Actions | Failures | Termination | Objective | Lineages | Recoveries | Patterns |
|---|---:|---:|---|---|---:|---:|---:|
| GS-T001 | 14 | 2 | Wall-clock timeout | Failed | 0 | 0 | 0 |
| GS-T002 | 20 | 3 | Request limit | Failed | 0 | 0 | 0 |
| GS-T003 | 11 | 3 | Wall-clock timeout | Failed | 0 | 0 | 0 |
| GS-T004 | 20 | 6 | Tool-call limit | Failed | 0 | 0 | 0 |
| GS-T005 | 20 | 3 | Tool-call limit | Failed | 0 | 0 | 0 |

Totals:

- Acquisition tasks attempted: 5 / 5
- Objective successes: 0 / 5
- Eligible recovery lineages: 0
- Verified recoveries: 0
- RecoveryPatterns: 0
- Persistence failures: 0
- Trusted PlannedAction provenance: valid for 5 / 5 tasks
- Prompt leakage violations: 0

All five attempts are retained as valid experimental outcomes and must not be selectively rerun.

## Retrieval Decision

T006–T015 retrieval evaluation was not executed.

Reason:

The R3 acquisition corpus contains zero eligible RecoveryPatterns. Running retrieval evaluation against an empty corpus would not meaningfully evaluate semantic retrieval quality and would not test the intended Gate A1 hypothesis.

Therefore the retrieval stage is recorded as:

**BLOCKED / NON-EVALUABLE UNDER R3 DUE TO EMPTY RECOVERY CORPUS**

No retrieval threshold was changed or invented after observing the result.

## Behavioral Evaluation

Behavioral treatment was not executed.

R3 therefore provides no evidence about:

- Graph Swarm advice effectiveness
- behavioral transfer
- treatment-vs-baseline performance
- B0/O1/T performance

Those questions remain unevaluated.

## Integrity

For all five R3 acquisition tasks:

- durable `started.json` and `completed.json` markers exist
- run IDs match their task records
- frozen configuration provenance is preserved
- `persistence_error` is null
- trusted PlannedAction provenance is true
- evaluator metadata was not exposed to the coding agent
- `family_id` was not exposed
- future-task metadata was not exposed
- gold patches were not exposed
- Graph Swarm advice was not injected

## Raw Step Database Checksums

The binary `steps.sqlite` databases are retained outside the normal Git evidence commit. Their SHA-256 hashes are recorded here for provenance.

### GS-T001

SHA-256:

`E4105A9ECDF777520DE7E9C13DD0D146E32327D123D448EA761B6C8625FD7148`

### GS-T002

SHA-256:

`7C9451C6B7602E753267A98E82F0155AE82CAF915EE2FC2B476E3404DF82202A`

### GS-T003

SHA-256:

`7577569D292DC3B332661BDDDDB8083EA5FA8B097E5730BC588EAC19A9DEC758`

### GS-T004

SHA-256:

`2A4904DCBC2D1FB97036CB20A7F6EDEAB8E6FE988F956052B17971C7CDD63A29`

### GS-T005

SHA-256:

`91C59281BFA87B05C25F78118B9DB12CF47CFEFEE67ED2043E1A538023727F08`

## Interpretation

The R3 result identifies memory acquisition as the current experimental bottleneck.

Under the frozen North Mini Code coding-agent configuration and frozen resource limits, none of the five acquisition tasks produced the required sequence of:

1. observed failure,
2. concrete change,
3. later trusted successful validation,
4. eligible verified recovery,
5. RecoveryPattern creation.

Because no RecoveryPatterns entered the memory substrate, downstream retrieval could not be meaningfully evaluated.

This result does not establish that the graph persistence layer, semantic retrieval mechanism, or Graph Swarm treatment is ineffective.

## Next Research Iteration

R3 is frozen and remains immutable.

Any subsequent attempt to improve acquisition must use a new experimental configuration, such as R4.

A coding-model change must not modify R3 retroactively.

If R4 changes the coding model, the following should remain fixed initially so that the coding model is the isolated variable:

- T001–T005 task set
- chronological order
- recovery abstraction model
- prompt policy
- evaluator
- environment definitions
- acquisition methodology
- 20-action limit
- 24-request limit
- 300-second wall-clock deadline

Resource limits should not be changed in the same revision as the coding model if the goal is to isolate model capability as the independent variable.
