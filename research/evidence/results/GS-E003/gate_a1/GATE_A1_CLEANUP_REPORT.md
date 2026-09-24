# Gate A1 R13b Cleanup Report

## Deleted

- Repository cache directories outside `.venv` and `research/evidence`, comprising Python `__pycache__` directories plus the root `.pytest_cache` and `.ruff_cache` directories, were removed in three cleanup passes (29, 29, and 12 directories) because verification recreated some caches.
- Approximate space removed across all cleanup passes: 9,409,934 bytes (about 9.0 MiB).
- No research result, task artifact, workspace, configuration, objective, graph-persistence, retry, pattern-retry, or provenance file was deleted.

## Kept evidence categories

- The complete canonical R13b acquisition root, including all original and selected retry artifacts.
- All Gate A1 task definitions, frozen configurations, prompts, objectives, source code, tests, schemas, queries, and documentation.
- All `research/evidence/workspaces/` content, including workspaces associated with successful tasks, failed attempts, retries, validation, and environment reconstruction.
- Historical R8, R9, R10, R11, R12, R13, preflight, preparation, repair, and environment evidence outside the canonical R13b root.
- The R2, R4, and R7 configuration files present in the working tree because they are task/configuration history rather than disposable output.
- The new closeout and handoff artifacts.

## Ambiguous items intentionally retained

The untracked historical result directories under `research/evidence/results/GS-E003/gate_a1/`, the dated `research/evidence/gs_e003_gate_a1_*` directories, and all R13b workspaces were retained. Their current relationship to revision history, failure trajectories, reproducibility, or ledger auditability could not be disproven safely. The virtual environment was also left untouched because it is an execution environment, not repository evidence.

## Scope note

The cleanup did not alter tracked repository files or any file below the mandatory canonical R13b evidence root. The resulting working tree may still contain intentional prior code/evidence changes and the newly created closeout/handoff artifacts.
