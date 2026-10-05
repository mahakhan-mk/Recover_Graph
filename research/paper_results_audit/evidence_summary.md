# Evidence summary

## A. Final admissible evidence set
No run is classified `VALID_REPORTED`. Eight descriptive candidates are selected: T006-T008 from the Qwen3-Coder development canary and T017 from hardened-development v1. Their results are not one methodologically comparable confirmatory set.

## B. Development-only runs
T006-T008 B0/T: `sprint3b_kilo_v4_qwen3_coder_canary` (6 slots). T017 B0/T: `sprint3b_hardened_development_v1` (2 slots). T018 is methodology/development history only and excluded; its hardened-development revisions used `frozen_recurrence_matcher_v3_exact_pytest_outcome`, while T006-T008 and T017 selected candidates used v2.

## C. Invalid runs
Two retained invalid-infrastructure T008/T attempts are under `sprint3b_kilo_v3/invalid_infrastructure_attempts/`; see inventory for IDs.

## D. Comparisons that must not be made
Do not combine these candidates as a confirmatory paired study. Qwen canary freeze says `development_canary_package_no_results`; T017 freeze says `development_hardened_revision_no_results`. Evaluator and protocol revisions differ. T008 repair precedes advice and cannot support causal repair. T017 v2 recurrence is not a stored v3 exact-outcome evaluation.

## E. RFR counts
The four preregistered occurrence candidates (T006-T008 and development-only T017) are shown with raw numerator/denominator in `aggregate_metrics.json`; T017 semantic recurrence is corrected to false from the exact passing target invocation while its historical v2 true value is preserved. A confirmatory aggregate RFR is **NOT SUPPORTED BY CURRENT ARTIFACTS** due development status and protocol/evaluator mismatch.

## F. FAR counts
FAR is **NOT SUPPORTED BY CURRENT ARTIFACTS**: no event-level objective incorrect/inapplicable-advice adjudication is stored. Silence is not a denominator. Event totals are in aggregate JSON.

## G. Task-success counts
Exact counts by condition appear in aggregate JSON. T006 and T008 succeed in both candidate conditions; T007 fails in both; T017 outcomes follow its stored metadata. These remain development evidence.

## H. Advice coverage
Selected canary candidates: T006/T and T008/T receive advice; T007/T receives none. T017 is development-only; its selected pattern is recorded in CSV.

## I. Behavior-change evidence
T006/T, T008/T, and T017/T each record changed behavior after advice, but all three materially correct source edits precede advice. These records show changed later actions, not advice-caused repair.

## J. T006 efficiency
Exact differences and percentages for requests, calls, tokens, wall time, and retries appear in `aggregate_metrics.json`; do not generalize beyond T006 canary.

## K. T008/T017 timing
T006/T source edit action `2f5312f8-446e-4efc-8465-d4a9daec1881` at 17:59:48Z preceded advice at 17:59:53Z. T008/T source edit action `ae3238d3-8221-4633-abab-fe5ad707cb9e` at 18:09:22Z preceded advice. T017/T source edit action `3a5bfaa2-e929-4ce2-8cd6-c586ca8f4635` at 08:14:54Z preceded advice at 08:15:00Z. The exact T017 target test was separately run and passed, while v2 recorded recurrence true from aggregate failure output; no v3 evaluator execution is stored.

## L. Final model/provider/settings
T006-T008 canary: Kilo, `qwen/qwen3-coder`, temperature 0, R13B system prompt hash in freeze, max actions 28, max requests 24, 3 retries, 300s task timeout, command 30s/tests 120s, five frozen patterns; memory and Neo4j writes forbidden. T017 uses a separate hardened-development freeze and recurrence v2; settings are not merged across revisions.

## M. Metrics marked NOT SUPPORTED BY CURRENT ARTIFACTS
Confirmatory aggregate RFR; FAR numerator/rate; T017 v3 evaluator execution (the exact target outcome is available from a separate passing invocation); advice-caused repair for T006/T008/T017; unified settings across cases; replacement-only rerun claim; retrieval latency; all-case efficiency aggregation.
