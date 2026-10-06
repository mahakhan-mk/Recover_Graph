# Operational Graph Schema

Core nodes:

- Run
- Task
- Action
- Tool
- FailureEpisode
- Resolution
- Outcome
- Environment

Core relationships:

```text
(Task)-[:HAS_ACTION]->(Action)
(Action)-[:USED]->(Tool)
(Action)-[:PART_OF]->(Run)
(Action)-[:PART_OF_FAILURE]->(FailureEpisode)
(FailureEpisode)-[:OCCURRED_IN]->(Environment)
(FailureEpisode)-[:RESOLVED_BY]->(Resolution)
(Resolution)-[:VERIFIED_BY]->(Outcome)
```

A successful action after a change is recovery evidence. It is not automatically causal proof that the change produced the success.
