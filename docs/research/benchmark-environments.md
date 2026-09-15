# Sprint 3B benchmark environments

This document describes the reproducible environment workflow for the frozen
T006-T015 benchmark. Repository metadata and the frozen task manifest remain
authoritative for package requirements, Python constraints, selectors, and
container image identities. `configs/research/benchmark_environments.toml`
adds Graph Swarm's runtime and reproducibility policy without duplicating
those task facts.

## Clean clone and installation

From a clean clone, install Graph Swarm's development dependencies in a
development-only virtual environment:

```powershell
git clone <GRAPH_SWARM_REPOSITORY_URL> graph_swarm
Set-Location graph_swarm
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The root project requires `>=3.12,<3.14`: Python 3.12 through 3.13 are
supported, with Python 3.13 recommended for development. Benchmark packages
must not be installed into this root virtual environment.

## Authoritative runtime

The authoritative Sprint 3B T006-T015 evaluation uses the exact frozen
manifest container for all ten tasks. Local interpreters are diagnostic and
development-only; they are not an evaluation runtime and are never a fallback
for a failed container environment.

Docker Desktop (or an equivalent Docker-compatible runtime) with a working
Linux engine is required. Do not install system software automatically. Read
the exact image references from the frozen task manifest, then acquire and
verify each image without substituting an image or dependency:

```powershell
docker pull <manifest-image>
docker image inspect <manifest-image> --format '{{.Id}}'
docker version
```

The runner pins the immutable base-image digest and builds or reuses a
fingerprinted prepared image. Dependencies are installed only while preparing
that image. Objective execution is isolated, uses `--network none`, and runs
without `pip install`.

## Policy and fingerprints

`benchmark_environments.toml` declares the runtime policy, repository-metadata
authority, required validated environments, disabled execution networking,
expected Python assertions, and any task-scoped extras explicitly needed by
the frozen tests. The loader rejects unknown or malformed fields and rejects a
policy that is incompatible with repository Python constraints.

Each environment fingerprint includes the manifest contents, repository
commit, immutable base-image digest, selected interpreter identity/version,
dependency files and metadata, selected extras, overlays, tox requirements and
`setenv`, and the benchmark policy. Thus changing any relevant runtime input
invalidates reuse.

## Fresh-clone workflow

From a fresh clone, use the following sequence:

1. Install Graph Swarm using the development-only commands above.
2. Acquire and verify the exact frozen SWE-smith base images using the
   manifest image references and the Docker commands above.
3. Run `benchmark_prepare` to build or reuse the prepared images.
4. Run `benchmark_preflight` to validate the frozen mutation and objective.
5. Proceed to B0/O1 only after `READY_FOR_B0_O1` is emitted.

Preparation is the slow, network-dependent phase. It acquires any missing
frozen SWE-smith objective rows and installs repository-authoritative
dependencies in fingerprinted Docker images; it is normally needed only once
for each environment fingerprint. It never materializes a frozen mutation,
executes FAIL_TO_PASS tests, or calls a provider/model.

## Preparation

Run this command from the repository root after Docker is available and the
manifest images have been verified:

```powershell
& .\.venv\Scripts\python.exe -m graph_swarm.research.benchmark_prepare `
  --project-root (Get-Location).Path `
  --baseline-root (Join-Path (Get-Location).Path 'benchmark\workspaces') `
  --execution-root (Join-Path (Get-Location).Path 'research\evidence\workspaces') `
  --artifact-root (Join-Path (Get-Location).Path 'research\evidence\results')
```

Successful output is:

```text
BENCHMARK_ENVIRONMENTS_PREPARED <evidence-path>
```

Prepared environments have a matching environment fingerprint and image;
their `validated` marker may still be `false`. A matching prepared image is
reused regardless of that validation state.

## Preflight-only procedure

Run this command from the repository root after Docker is available and the
manifest images have been verified:

```powershell
& .\.venv\Scripts\python.exe -m graph_swarm.research.benchmark_preflight `
  --project-root (Get-Location).Path `
  --baseline-root (Join-Path (Get-Location).Path 'benchmark\workspaces') `
  --execution-root (Join-Path (Get-Location).Path 'research\evidence\workspaces') `
  --artifact-root (Join-Path (Get-Location).Path 'research\evidence\results')
```

Preflight only loads already prepared environments. It performs no Docker
build, package installation, dependency download, or base-image pull. It
applies the frozen mutations, collects each frozen FAIL_TO_PASS selector, and
executes each objective with network disabled. It makes zero provider/model
calls and does not launch B0, O1, or the treatment. Every stage must pass,
including distinguishing the expected mutated-test failure from infrastructure
failure. Successful output is:

```text
READY_FOR_B0_O1 <evidence-path>
```

Any setup, mutation, collection, or objective infrastructure failure stops the
barrier and reports `BLOCKED`; no readiness marker is created.

B0/O1 require the same prepared environments to have `validated=true`. They
never prepare benchmark environments implicitly. If either phase reports a
missing or unvalidated environment, run `benchmark_prepare` followed by
`benchmark_preflight` before retrying.

## Troubleshooting

- If Docker reports that the engine is unavailable, start the Linux engine and
  rerun the image verification commands. The runner does not fall back to the
  host interpreter.
- If an image is missing, compare the task manifest reference and immutable
  digest; do not substitute a package, image, or platform.
- If constraints conflict, inspect the frozen repository metadata and policy
  assertion. Do not weaken package constraints.
- If a prepared environment is invalid or partial, remove only that generated
  environment and rerun preparation. Invalid environments never receive a
  validated cache marker.

## Maintenance

When frozen task metadata, repository requirements, tox configuration, or
container digests change, update the corresponding frozen source first. Keep
the policy synchronized with those sources, then regenerate environments and
rerun preflight before any experiment. Keep generated environments, workspaces,
SQLite files, and temporary artifacts out of version control.
### Docker preparation timeout

Prepared Docker image builds use the validated benchmark-policy timeout rather
than a hard-coded limit. The current reproducible default is 7200 seconds
because some frozen environments, such as String2String, have large
repository-authoritative Torch dependencies.

GS-T014 uses Python 3.10 from the exact frozen String2String image because
its repository-authoritative `faiss-cpu==1.7.3` pin is not compatible with
Python 3.12. This is interpreter compatibility selection, not dependency
substitution.
