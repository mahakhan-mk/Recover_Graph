# Experimental design claim audit

## A. "GS-T006--GS-T015 were ten transfer tasks."

**TRUE.** v3 freeze explicitly lists T006-T015 and paired task plan.

Evidence: `research/evidence/results/GS-E003/sprint3b_kilo_v3/freeze.json`.

## B. "Each task had one B0 and one T primary run."

**FALSE.** One pair was planned per v3 task, but retained results include multiple revisions and invalid T008 attempts.

Evidence: `research/evidence/results/GS-E003/sprint3b_kilo_v3/freeze.json; research/paper_results_audit/run_inventory.csv`.

## C. "There were 20 total primary runs."

**PARTIALLY_TRUE.** Twenty slots were planned in v3; this is not the total retained run count or reported valid set.

Evidence: `research/evidence/results/GS-E003/sprint3b_kilo_v3/freeze.json`.

## D. "No selective reruns occurred."

**FALSE.** Duplicates and invalid attempts occur across revisions; no single-run-only history.

Evidence: `research/paper_results_audit/run_inventory.csv; research/evidence/results/GS-E003/sprint3b_kilo_v3/invalid_infrastructure_attempts/`.

## E. "Runs were replaced only for invalid infrastructure/provider failures."

**NOT_SUPPORTED.** An invalid T008 exists, but full replacement lineage is not established and later dev revisions repeat slots.

Evidence: `research/evidence/results/GS-E003/sprint3b_kilo_v3/invalid_infrastructure_attempts/; research/evidence/results/GS-E003/sprint3b_kilo_v4_qwen3_coder_canary/freeze.json`.

## F. "The treatment used a fixed five-pattern read-only recovery corpus."

**TRUE.** v3 freeze and v4 config specify five patterns and forbid memory/Neo4j writes.

Evidence: `research/evidence/results/GS-E003/sprint3b_kilo_v3/freeze.json; configs/experiments/sprint3b_kilo_v4_canary_qwen3_coder.yaml`.

