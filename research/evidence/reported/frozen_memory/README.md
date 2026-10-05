# Frozen RecoverGraph memory

This package describes the five RecoveryPatterns distilled from the earlier objectively evaluated GS-T001–GS-T005 acquisition executions. Gate A1 R13B confirms that Cohere (`cohere/north-mini-code:free`) performed acquisition abstraction. The treatment Qwen runs consumed the fixed allowlist; they did not rebuild the patterns. The allowlist was frozen before transfer execution. The retained transfer runs are marked development-only in their current artifacts.

Open `recovery_patterns.jsonl` as UTF-8 JSON Lines: each line is one pattern. Stored embeddings are included for three patterns (384 values each). Two records have nulls where the available artifacts did not preserve the underlying RecoveryPattern fields. The Neo4j connection was unconfigured during this export, so this package is an offline inspection snapshot, not a full five-record Neo4j dump.

Provenance: `memory_manifest.json` lists the R13B closeout and selected acquisition artifacts, treatment configs/freezes, schema migrations, and checksums. Inspecting this export requires no LLM or provider call.