# Graph Swarm

Graph Swarm is an experimental research system for studying Verified Failure
Memory: whether successful recovery experience can be retrieved before a
related future action and reduce recurrence of known failures.

## Active stack

- Python 3.12 or 3.13 (`>=3.12,<3.14`)
- PydanticAI agent harness
- OpenRouter inference
- Neo4j operational memory graph
- pytest and pyright/ruff
- Docker for the isolated SWE-smith benchmark runtime

OpenRouter is the only supported live provider. Earlier provider configurations
and evidence remain only as historical research records.

## Teammate setup

From a clean clone, create the root development environment and configure only
your own credentials:

```powershell
py -3.13 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill in the Neo4j and OpenRouter values in `.env`. Never commit `.env`.
Benchmark-specific dependencies belong in the prepared benchmark containers,
not in the root virtual environment.

`AGENT_REQUEST_LIMIT=10` in `.env.example` is the general application
fallback/default for ordinary agent runs. Frozen research experiments do not
inherit that fallback: Gate B1 uses the versioned
`configs/experiments/gate_b1.yaml` contract with 24 model requests.

Run the safe, read-only setup audit:

```powershell
.\\.venv\\Scripts\\python.exe -m graph_swarm.research.verify_setup
```

The audit does not install, clone, pull/build/run Docker, call OpenRouter,
modify Neo4j or Git state, or run experiments. If benchmark repositories are
missing, acquire them explicitly with:

```powershell
.\\.venv\\Scripts\\python.exe -m graph_swarm.research.bootstrap_setup --clone
```

This command clones only missing repositories and pins the exact commits in
`benchmark/manifests/repositories.json`; it refuses existing dirty or
mismatched paths.

## Validation and benchmark flow

Keep setup, tests, runtime checks, and provider-backed research as separate
steps:

1. Run unit/static checks: `pytest`, `ruff`, and `pyright`.
2. Acquire the immutable SWE-smith base images and verify the entries in
   `configs/research/docker_environments.json`.
3. Run the existing `benchmark_prepare` tooling to build prepared images, or
   load a separately supplied prepared-image archive.
4. Run `benchmark_preflight` and then the zero-provider
   `benchmark_runtime_smoke`.
5. Run provider smoke only when explicitly needed to validate credentials and
   the OpenRouter path.
6. Run the frozen B0/O1 experiment only after all readiness barriers pass.

The canonical experiment controls are in
`configs/experiments/gate_b1.yaml`: OpenRouter, runtime-selected
`OPENROUTER_MODEL`, temperature 0, 20 actions, 24 model requests, 3 tool
retries, 300-second agent/model timeouts, five-second request-start pacing,
nominal 12 RPM, task-start O1 guidance, B0 without recovery advice, and T
disabled. These are research controls, not general-purpose defaults.

Prepared-image handoff supports two paths. The reproducible path acquires the
exact base images, verifies their digests, obtains the pinned repositories,
runs preparation, and verifies fingerprints. The convenience path transfers
an archive outside Git:

```powershell
# Sender
docker save -o graph-swarm-prepared-images.tar <required-image-tags>

# Receiver
docker load -i graph-swarm-prepared-images.tar
```

Archives are transport conveniences, not experiment definitions, and must not
be committed.

## Repository layout

```text
configs/                 Versioned model, experiment, and environment metadata
src/graph_swarm/         Core implementation and research runners
experiments/             Benchmark preparation and frozen evaluation boundaries
benchmark/manifests/     Pinned external repository identities
research/evidence/       Local/retained research evidence and runtime markers
migrations/neo4j/        Explicit graph constraints and indexes
tests/                   Unit and integration tests
docs/                    Architecture, protocol, and benchmark handoff notes
```

Generated workspaces, repositories, Docker archives, virtual environments,
caches, raw run directories, and secrets are intentionally excluded by Git.
