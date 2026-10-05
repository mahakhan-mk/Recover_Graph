# Reported benchmark cases

The paper labels map to GS-T006 (EXP1, Tornado), GS-T007 (EXP2, Jinja), GS-T008 (EXP3, SQLGlot), and GS-T017 (EXP4, Rich). These are SWE-smith-derived environments selected for this study; this package is not an official SWE-smith split.

`manifests/reported_evaluation.jsonl` is the canonical four-case manifest. Each case definition records its exact objective tests, pinned repository revision, and available environment/image metadata. B0/T artifacts and copied-file hashes are under `research/evidence/reported/`. The treatment corpus is at `research/evidence/reported/frozen_memory/`; inspecting these records requires no provider call.

Objective outcomes use stored frozen fail-to-pass test lists. Repository revisions are pinned by the SWE-smith repository manifest or the T017 task manifest and verified local remote. Docker image tags/digests, Python versions, and network-off execution policy are recorded per case; T017's image digest is absent from the retained metadata.

The selected run metadata classifies all four pairs as development evidence (`VALID_DEVELOPMENT`); the package preserves that classification.
