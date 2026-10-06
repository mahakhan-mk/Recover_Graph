# Reproducing the reported RecoverGraph evaluation

## Requirements

Use Python `>=3.12,<3.14`; the recorded benchmark runtime is Python 3.12.1. Install the project and development tools with `python -m pip install -e '.[dev]'`. Docker is required for benchmark execution; the task containers had network access disabled. The project requires the Neo4j Python client `neo4j>=6.2`; no server version is recorded. Stored-result validation and analysis work offline without Neo4j, Docker, or provider credentials.

Potential environment-variable names are `KILO_API_KEY`, `KILO_CODING_MODEL`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `HF_TOKEN` (for Hugging Face Inference API embeddings with `sentence-transformers/all-MiniLM-L6-v2`), and `OPENROUTER_API_KEY` (for acquisition abstraction). `MODEL_PROVIDER` is used by the general setup verifier. The stored-artifact analysis and reported runner validation need none of these. Never put secret values in command history or this repository.

## Validate stored research package

```powershell
python -m pytest tests/unit/memory/test_recovery_embeddings.py
python -m pytest tests/unit/research/test_reported_evaluation_package.py
python experiments/analysis/generate_results.py --output-dir experiments/analysis/generated
```

The public runner performs offline checks without initializing a model provider:

```powershell
python experiments/run_reported_evaluation.py --list
python experiments/run_reported_evaluation.py --case EXP1 --condition B0 --validate-only
python experiments/run_reported_evaluation.py --case EXP1 --condition T --dry-run
```

## Benchmark repositories and pinned revisions

The preparation mode clones the selected URL into the system temporary directory, checks out the recorded commit detached, and verifies `HEAD`. It refuses to replace an existing preparation directory.

| Case | Repository URL | Commit |
|---|---|---|
| EXP1 Tornado | https://github.com/swesmith/tornadoweb__tornado.d5ac65c1.git | `3a49fcc4520192175e1549dad76a8cf827630582` |
| EXP2 Jinja | https://github.com/swesmith/pallets__jinja.ada0a9a6.git | `83b73140ca308ee1c65870c78d11ebdff5a13d68` |
| EXP3 SQLGlot | https://github.com/swesmith/tobymao__sqlglot.036601ba.git | `11710cc8e44a685969087a8b1f24a5a06fde0161` |
| EXP4 Rich | https://github.com/Textualize/rich.git | `9d8f9a372cc5916fd4781fec207ced7ddac2f08f` |

Prepare one source tree with `python experiments/run_reported_evaluation.py --case EXP1 --prepare-only`; replace EXP1 with EXP2-EXP4 as needed. Equivalent manual Git steps are `git clone --no-checkout <URL> <DIR>`, `git -C <DIR> checkout --detach <COMMIT>`, and `git -C <DIR> rev-parse HEAD` (which must equal the commit above).

## Environment and objective evaluation

The recorded images are `swebench/swesmith.x86_64.tornadoweb_1776_tornado.d5ac65c1` (`sha256:6c145000a2bf5a7c946e66706dcf744ce99acc33c411f1e29ab0c01f11feb82f`), `swebench/swesmith.x86_64.pallets_1776_jinja.ada0a9a6` (`sha256:0416d3b73c08f501f96a47c769ab7bddb0f96f0cb4cc04efcc142577b4a59685`), and `swebench/swesmith.x86_64.tobymao_1776_sqlglot.036601ba` (`sha256:08bbf9a13106425b251b10cb7b77fe469c228d59d665a53086aab545b76c8b1f`). Recorded prepared-image digests are unavailable. T017's historical image digest is also unavailable. The run manifests specify isolated workspace mounting and network-disabled task execution. Success is judged by the frozen SWE-smith fail-to-pass objective evaluator; exact FAIL_TO_PASS tests are in each `benchmark/reported_cases/GS-T*.json` definition.

## Frozen memory

The exact five-pattern snapshot is `research/evidence/reported/frozen_memory/recovery_patterns.jsonl`; inspect it with a text/JSONL reader and validate its manifest offline. No Neo4j loader is provided for this export, so loading it into a server is not part of the reported transfer reproduction. No abstraction-model call is needed for transfer. Two records have incomplete fields, as described in `memory_manifest.json`.

## Run and regenerate

The public runner currently lists, validates, dry-runs, and prepares pinned repositories. It does not expose an experiment execution mode: actual B0/T execution requires the existing internal runtime adapter, Docker, Kilo/Qwen access, and credentials. A dry run never calls a provider. The canonical prompt snapshot is `configs/experiments/reported/system_prompt_r13b.txt`; its SHA-256 is `f3bf4187cc9bfabcf2551aa3988e592a2d0f2d0e61ad2984d7d90ddf13d54502`.

Regenerate tables offline:

```powershell
python experiments/analysis/generate_results.py --output-dir experiments/analysis/generated
```

Regenerate Figure 1:

```powershell
python experiments/analysis/generate_figure1.py --output research/figures/figure1_recovergraph.svg
```

## Known limitations

- T017 has no recorded historical image digest, and prepared-image digests are absent.
- The frozen memory contains three full records and two partial records with unavailable fields left null.
- All stored runs retain their recorded `VALID_DEVELOPMENT` classification.
- Historical freeze hashes for the evolving `src/graph_swarm/agent/prompts.py` remain unchanged. The exact immutable prompt is validated against reported run evidence instead.
- Reconstructing the recorded containers and executing the model requires external Docker images, network access for deliberate preparation, Kilo credentials, and the specified source repositories.

The EXP1 task definition is at `benchmark/reported_cases/GS-T006.json`; corresponding EXP2-EXP4 files are `GS-T007.json`, `GS-T008.json`, and `GS-T017.json` in the same directory.
