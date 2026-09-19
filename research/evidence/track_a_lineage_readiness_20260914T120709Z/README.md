# Track A lineage readiness

Conclusion: `BLOCKED_MISSING_TRUSTED_RECOVERY_LINEAGE`.

This audit records only currently durable Neo4j facts and the result of `get_recovery_evidence`; it does not infer missing task, run, environment, or runtime PlannedAction handoff data from summaries or source text.

The current graph has 10 nodes and 6 relationships. The available failure and action properties are not a complete trusted typed recovery lineage because the required graph links and runtime-owned PlannedAction provenance are absent.

No historical artifact was accepted as a legitimate reconstruction source: the available artifacts contain summaries/event fragments, not the complete trusted failed-action and recovery-action lineage required by Track A.

Exact per-failure requirements are in `lineage_readiness.jsonl`.
