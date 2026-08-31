# Graph Swarm

Graph Swarm is an experimental research system for studying **Verified Failure Memory**: whether empirically successful recovery experience from earlier agent runs can be retrieved before a related future action and reduce recurrence of known failures.

## Current research scope

This repository is intentionally scoped to the eight-week research program:

1. Observe and remember failures and recoveries.
2. Advise before a known failure recurs.
3. Evaluate the mechanism on a controlled recurring-failure benchmark.

Production SaaS features, dashboards, multi-framework adapters, autonomous recovery, and large-scale infrastructure are deliberately out of scope.

## Stack

- Python 3.12+
- PydanticAI experimental agent harness
- Groq-hosted inference
- Neo4j operational memory graph
- pytest
- Docker Compose

## Repository layout

```text
graph-swarm/
├── configs/            # Versioned model, graph, and experiment configuration
├── src/graph_swarm/    # Core implementation
├── migrations/neo4j/  # Explicit graph constraints and indexes
├── benchmark/          # Canonical benchmark inputs, derived data, and fixtures
├── experiments/        # Reproducible rollout runners
├── analysis/           # Metrics and statistical analysis
├── research/evidence/  # Versioned research evidence
├── artifacts/          # Temporary/generated local run output
├── tests/              # Unit, integration, and end-to-end tests
├── scripts/            # Developer utilities
└── docs/               # Architecture, protocol, and schema notes
```

## Setup

```bash
cp .env.example .env
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
pip install -e '.[dev]'
docker compose up -d
pytest
```

Add your `GROQ_API_KEY` to `.env` before running model-backed experiments.

## Research implementation order

Do not implement retrieval first. The intended sequence is:

```text
Domain models
  -> minimal PydanticAI agent
  -> controlled tools
  -> typed event capture
  -> deterministic failure detection
  -> Neo4j persistence
  -> resolution/outcome tracking
  -> GS-E001 vertical slice
  -> recurrence retrieval
  -> pre-action advice
  -> controlled evaluation
```

## Design rules

- Domain models must not depend on Neo4j or PydanticAI.
- Neo4j access stays behind a repository interface.
- A later success creates recovery evidence, not automatic causal attribution.
- Benchmark data stays separate from Graph Swarm implementation logic.
- Raw experiment artifacts are append-only and should not be manually rewritten.
- Graph Swarm remains advisory during the research phase. It must not silently rewrite or block tool actions.
