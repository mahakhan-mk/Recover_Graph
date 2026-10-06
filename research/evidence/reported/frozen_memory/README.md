# Frozen RecoverGraph memory

This package contains the fixed five-pattern recovery corpus used by treatment runs. The corpus was acquired from the objectively evaluated GS-T001–GS-T005 executions; Gate A1 R13B closeout confirms `cohere/north-mini-code:free` performed the abstraction. The treatment runs consumed this fixed allowlist and did not rebuild it.

Read `recovery_patterns.jsonl` as UTF-8 JSON Lines, with one pattern per line. Three patterns have stored 384-value embeddings. Two records preserve unavailable fields as null. The Neo4j connection was not configured during export, so this is an offline inspection snapshot rather than a complete five-record database dump.

Canonical acquisition records, closeout evidence, model/config provenance, freeze metadata, and pattern-to-execution lineage are retained in `provenance/`. `memory_manifest.json` lists their paths and SHA-256 hashes. Validation requires no provider call.
