# Track A Recovery Memory V2 model migration

Status: **blocked**.

GPT-OSS/Groq was replaced because the active Track A abstraction was frozen to North Mini Code/OpenRouter.

Previous abstraction: `openai/gpt-oss-120b` through Groq.
Current abstraction: `cohere/north-mini-code:free` through OpenRouter at temperature `0`.
The abstraction prompt version and typed `RecoveryAbstractionOutput` contract were preserved.

The backup is a complete logical Neo4j export with node/relationship JSONL, schema, counts, and SHA-256 checksums.
The regeneration bundle contains only trusted source lineage and a separate model-input projection; existing pattern semantics and embeddings are excluded from model input.
Source restoration uses the repository persistence methods and the existing migrations in `migrations/neo4j`.
Embeddings use `sentence-transformers/all-MiniLM-L6-v2`, package version `5.7.0`, Sprint 4 v1 text, normalized native 384-dimensional vectors, and the existing cosine index.

Database wiped: `False`.
Known deviation/blocker: `no trusted recovery lineage exists for the required exported-fixture smoke test`.

No credentials are stored in this evidence directory.
