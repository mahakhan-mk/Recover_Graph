# Track A Recovery Memory V2 model migration

Status: **blocked**.

GPT-OSS/Groq was replaced because the active Track A abstraction was frozen to North Mini Code/OpenRouter.

Previous abstraction: `openai/gpt-oss-120b` through Groq.
Current abstraction: `cohere/north-mini-code:free` through OpenRouter at temperature `0`.
Prompt history: v1 was the previous GPT-OSS development abstraction prompt; v2 is the North Mini Code abstraction prompt with tightened repository-specific leakage instructions.
Current abstraction prompt version: `v2`.

The backup is a complete logical Neo4j export with node/relationship JSONL, schema, counts, and SHA-256 checksums.
The regeneration bundle contains only trusted source lineage and a separate model-input projection; existing pattern semantics and embeddings are excluded from model input.
Source restoration uses the repository persistence methods and the existing migrations in `migrations/neo4j`.
Embeddings use `sentence-transformers/all-MiniLM-L6-v2`, package version `5.7.0`, Sprint 4 v1 text, normalized native 384-dimensional vectors, and the existing cosine index.

Database wipe performed: `False`.
Regeneration status: `blocked_no_complete_trusted_recovery_lineage`.
Neo4j status: `unchanged`.
Known deviation/blocker: `BLOCKED_MISSING_TRUSTED_RECOVERY_LINEAGE: no complete trusted recovery lineage exists`.

No credentials are stored in this evidence directory.
