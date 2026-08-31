# Graph Swarm Dataset Construction Phase

## Purpose

This document explains how the Graph Swarm pilot benchmark dataset was constructed, why each filtering and validation step exists, what artifacts were produced, and which information is allowed to be visible to the experimental agent.

It is intended for researchers, collaborators, reviewers, and future maintainers who need to understand the benchmark-construction methodology without reconstructing the full project history from scripts or chat logs.

The dataset construction phase is research-oriented. Its purpose is not to create a large generic software-engineering benchmark. Its purpose is to create a small, controlled recurring-failure benchmark suitable for testing the Graph Swarm research question:

> Does retrieval of empirically verified historical failure resolutions reduce recurring failures in agentic software-engineering workflows compared with the same agent without persistent failure memory?

The key requirement is recurrence. A later task must create a plausible opportunity for a recovery learned from an earlier task to be reused, while avoiding direct task leakage or exposure of ground-truth mutation information.

---

## 1. High-level construction pipeline

```text
SWE-bench/SWE-smith-py
50,908 train rows
        |
        v
Mutation-aware candidate extraction
105 candidates
7 mutation families x 15
        |
        v
Deterministic validation shortlist
35 candidates
7 mutation families x 5
        |
        v
Researcher-only mutation evidence extraction
35 evidence records
        |
        v
Manual recurrence review
5 approved recurrence families
3 tasks per family
        |
        v
Frozen 15-task pilot benchmark
```

The pipeline deliberately narrows the source dataset in multiple stages instead of directly sampling 15 tasks.

This separation is important because:

1. the source dataset is much larger than needed for the pilot;
2. mutation-family membership alone is not enough to prove useful recurrence;
3. issue text alone can be misleading;
4. researcher-only mutation information must remain separate from agent-visible task information;
5. the final benchmark needs a fixed execution order to prevent future-task leakage.

---

## 2. Source dataset

The source dataset is:

```text
SWE-bench/SWE-smith-py
```

Split:

```text
train
```

Rows observed during construction:

```text
50,908
```

The columns observed during evidence extraction were:

```text
instance_id
patch
FAIL_TO_PASS
PASS_TO_PASS
image_name
repo
problem_statement
```

The dataset version used here did not expose:

```text
test_patch
base_commit
version
```

No downstream script is allowed to silently assume that optional columns exist.

---

## 3. Why SWE-smith was used

Graph Swarm requires controlled software failures that can be grouped into recurrence opportunities.

SWE-smith is useful because the task identifier encodes the mutation process used to construct the faulty instance. This gives the researcher a stronger initial grouping signal than trying to infer recurrence from issue wording alone.

Examples of procedural mutation markers used during construction:

```text
func_pm_op_swap
func_pm_ctrl_invert_if
func_pm_remove_assign
func_pm_remove_cond
func_pm_ctrl_shuffle
func_pm_op_change
func_pm_remove_loop
```

These mutation names are researcher-side generation metadata. They are not the same thing as Graph Swarm's runtime failure taxonomy.

The project keeps three concepts separate:

```text
mutation_family
    = how SWE-smith generated the faulty state

runtime failure type
    = what Graph Swarm observes during execution
      such as TEST_FAILURE or COMMAND_FAILURE

recurrence family
    = a manually validated reusable recovery pattern
      defining a recurrence opportunity in the benchmark
```

This separation is methodologically important.

---

## 4. Initial heuristic candidate-pool attempt

An earlier extractor attempted to identify candidate tasks using issue-text keywords.

That first pass produced:

```text
250 candidates
```

with heuristic categories including:

```text
dependency_error
configuration_error
tool_parameter_error
test_failure
command_failure
```

This result was not used as the final benchmark-construction method.

### Why it was replaced

Keyword matching produced false positives and overly broad symptom categories.

Problems included:

- generic words such as `install`, `setting`, or `import` matching tasks whose actual bug had nothing to do with dependencies or configuration;
- substring matches creating misleading labels;
- reproduction snippets containing imports even when the failure itself was unrelated to imports;
- issue descriptions reflecting symptoms rather than reusable recovery structure.

The methodological conclusion was:

> Textual symptom similarity is not strong enough by itself to define recurring-failure opportunities for this experiment.

The candidate-construction method was therefore changed to use the SWE-smith mutation identifier as the primary initial grouping signal.

---

## 5. Mutation-aware candidate extraction

Script:

```text
scripts/build_candidate_pool.py
```

Outputs:

```text
benchmark/derived/swesmith_candidates.jsonl
benchmark/derived/swesmith_candidate_summary.txt
```

### Allowed mutation families

Exactly seven procedural mutation families are considered:

```text
func_pm_op_swap
func_pm_ctrl_invert_if
func_pm_remove_assign
func_pm_remove_cond
func_pm_ctrl_shuffle
func_pm_op_change
func_pm_remove_loop
```

Broader or less structurally constrained sources were excluded from the initial controlled pool, including:

```text
lm_rewrite
combine_file
combine_module
pr_*
```

### Candidate filters

A task must satisfy:

```text
FAIL_TO_PASS count >= 1
FAIL_TO_PASS count <= 30
PASS_TO_PASS count >= 1
```

No Docker image is downloaded and no task environment is executed during this stage.

### Deterministic sampling

The extractor uses:

```text
seed = 42
```

and retains at most:

```text
15 candidates per mutation family
105 candidates total
```

The resulting pool contained:

```text
func_pm_op_swap:          15
func_pm_ctrl_invert_if:  15
func_pm_remove_assign:   15
func_pm_remove_cond:     15
func_pm_ctrl_shuffle:    15
func_pm_op_change:       15
func_pm_remove_loop:     15
```

Total:

```text
105
```

### Candidate JSONL fields

Each candidate record contains:

```text
source
instance_id
repo
image_name
mutation_family
operation_hint
problem_statement
fail_to_pass_count
pass_to_pass_count
```

---

## 6. `operation_hint` is not a benchmark label

The candidate extractor emits a conservative `operation_hint` based on issue-text regexes.

Possible values include:

```text
parameter_or_argument
configuration
dependency_or_import
command_or_cli
test_or_assertion
null
```

This field was found to be too noisy to use for benchmark selection.

For example, reproduction snippets frequently contain import statements, which can trigger `dependency_or_import` even when the actual bug is arithmetic, control-flow, initialization, parsing, or traversal related.

Therefore:

> `operation_hint` is researcher-only convenience metadata and is not used as evidence for recurrence-family membership.

It must never be shown to the experimental agent.

---

## 7. Deterministic manual-review shortlist

Script:

```text
scripts/build_validation_shortlist.py
```

Input:

```text
benchmark/derived/swesmith_candidates.jsonl
```

Output:

```text
benchmark/annotations/recurrence_validation.csv
```

Observed result:

```text
105 input candidates
77 valid candidates after shortlist filters
35 selected review candidates
26 unique repositories
```

Each mutation family contributes exactly five tasks:

```text
7 families x 5 tasks = 35 review candidates
```

### Shortlist filters

A task remains eligible when:

- `problem_statement` is non-empty;
- the mutation family belongs to the seven allowed families;
- `1 <= fail_to_pass_count <= 30`;
- `pass_to_pass_count >= 1`.

### Deterministic selection order

Within each mutation family, candidates are sorted by:

1. `fail_to_pass_count` ascending;
2. `pass_to_pass_count` ascending;
3. repository ascending;
4. `instance_id` ascending.

The selector first prefers repositories not already selected for that family.

If fewer than five distinct repositories are available, it fills remaining positions from the deterministic sorted list.

### Why repository diversity is preferred

The research claim is stronger when a recovery principle is reusable beyond the exact same repository.

Repository diversity does not prove transferability. It only reduces the chance that a recurrence family is accidentally based on repository-specific names, APIs, conventions, or repeated local structure.

Manual validation is still required.

---

## 8. Manual-validation annotation schema

Researcher-only CSV:

```text
benchmark/annotations/recurrence_validation.csv
```

Columns:

```text
review_id
instance_id
mutation_family
repository
image_name
fail_to_pass_count
pass_to_pass_count
problem_statement
keep
transferable
recovery_pattern
transfer_rationale
chronological_index
notes
```

The first eight fields are populated automatically.

The following fields are intentionally left for manual review:

```text
keep
transferable
recovery_pattern
transfer_rationale
chronological_index
notes
```

The shortlist script is not allowed to infer transferability automatically from mutation-family equality.

---

## 9. Researcher-only mutation evidence

Issue descriptions alone are not enough to validate recurrence.

Two tasks may sound similar while having different code-level failure mechanisms. Conversely, different surface symptoms may share a reusable recovery principle.

A separate evidence stage was therefore added.

Script:

```text
scripts/build_validation_evidence.py
```

Input:

```text
benchmark/annotations/recurrence_validation.csv
```

Output:

```text
benchmark/annotations/recurrence_evidence.jsonl
```

Observed result:

```text
35 shortlist rows read
35 dataset matches found
35 non-empty patches found
35 unique instance IDs
```

Optional fields observed as absent:

```text
test_patch
base_commit
version
```

### Evidence record fields

```text
review_id
instance_id
mutation_family
repository
image_name
problem_statement
dataset_patch
test_patch
base_commit
version
fail_to_pass
pass_to_pass
patch_role
```

Every patch is explicitly labeled:

```text
patch_role = "unverified_swesmith_mutation_patch"
```

---

## 10. Critical patch interpretation

The SWE-smith `patch` field is treated as mutation evidence, not as a recovery.

It must not be called:

```text
gold_patch
solution_patch
resolution_patch
fix_patch
```

Manual inspection showed patches introducing faulty transformations such as:

- reversing an indexing expression;
- changing a condition or operator;
- removing a required assignment;
- reordering prerequisite statements;
- removing iteration.

Therefore:

> The dataset mutation patch is researcher-only evidence used to understand how a task was corrupted. It is not a resolution that may be injected into the agent.

This is a core leakage-control rule.

---

## 11. Manual recurrence-validation principle

A recurrence family is accepted only when multiple tasks support a plausible reusable recovery principle.

The family must not depend on:

- the same variable name;
- the same file;
- the same repository;
- the exact mutation patch;
- a copied code diff;
- future task knowledge;
- a ground-truth fix.

Instead, the recovery should be expressible operationally, for example:

```text
undefined state
    -> inspect initialization/dataflow

reversed behavior
    -> inspect branch polarity

use-before-definition
    -> restore prerequisite ordering

incorrect arithmetic/comparison
    -> verify operator semantics

incomplete sequence processing
    -> inspect iteration/traversal
```

This abstraction level is what makes later Graph Swarm retrieval meaningful.

---

## 12. Approved final recurrence families

Five recurrence families were manually approved for the pilot.

The other two mutation groups from the 35-task shortlist were not selected for the frozen 15-task pilot.

Not selected does not mean empirically non-transferable. It only means the five families below were judged clearer for the first controlled experiment.

### GS-F001: Conditional polarity error

Source mutation:

```text
func_pm_ctrl_invert_if
```

Canonical recovery pattern:

> When behavior is reversed around a boolean branch or guard, inspect condition polarity and restore the intended true/false branch mapping before changing downstream logic.

Transfer rationale:

> Across repositories the failure is produced by an inverted decision boundary. A previously verified recovery that checks branch polarity is plausibly reusable without depending on repository-specific names or implementation details.

Approved tasks:

```text
GS-R007
GS-R009
GS-R010
```

### GS-F002: Missing initialization or assignment

Source mutation:

```text
func_pm_remove_assign
```

Canonical recovery pattern:

> When a required value or state is undefined or missing, inspect its dataflow for a removed initialization or assignment and restore the required assignment before first use.

Transfer rationale:

> Across repositories the failure occurs because required state is no longer assigned before later use. The reusable recovery is to trace initialization and dataflow and restore the missing assignment.

Approved tasks:

```text
GS-R012
GS-R014
GS-R015
```

### GS-F003: Prerequisite ordering error

Source mutation:

```text
func_pm_ctrl_shuffle
```

Canonical recovery pattern:

> When execution references values, state, or checks before they are established, restore dependency order so prerequisite computation or validation occurs before use or return.

Transfer rationale:

> Across repositories statements are reordered so a prerequisite occurs after the operation that depends on it. A recovery based on prerequisite-before-use ordering can plausibly transfer across implementations.

Approved tasks:

```text
GS-R021
GS-R022
GS-R023
```

### GS-F004: Operator semantics error

Source mutation:

```text
func_pm_op_change
```

Canonical recovery pattern:

> When arithmetic, comparison, string, or other operator semantics become incorrect after a code change, verify the operator against the intended behavior and restore the semantically correct operator.

Transfer rationale:

> Across repositories the mutation replaces an operator with one that changes program semantics. A previously verified recovery that inspects operator intent and failing behavior is plausibly reusable.

Approved tasks:

```text
GS-R026
GS-R027
GS-R030
```

### GS-F005: Missing iteration

Source mutation:

```text
func_pm_remove_loop
```

Canonical recovery pattern:

> When a collection or candidate sequence is only partially processed or produces empty/incomplete output, inspect for removed iteration and restore traversal over all required elements.

Transfer rationale:

> Across repositories required iteration has been removed, producing incomplete processing. A recovery that checks collection traversal and restores iteration is plausibly reusable across implementations.

Approved tasks:

```text
GS-R032
GS-R033
GS-R034
```

---

## 13. Why `func_pm_op_swap` was not selected

`func_pm_op_swap` was useful during exploration but was not chosen for the first frozen pilot.

The reason is methodological conservatism.

Some operand reorderings are obviously harmful, while others are equivalent or highly context dependent. That makes it harder to define one clean recovery principle across all selected tasks without overgeneralizing.

The pilot favors recurrence families whose shared failure-recovery structure is clearer.

---

## 14. Why `func_pm_remove_cond` was not selected

`func_pm_remove_cond` also remained in the review pool but was not included in the five-family pilot.

This is not a claim that removed-condition failures are unimportant or non-transferable.

For the first controlled experiment, the benchmark was limited to five families with especially clear reusable recovery principles and strong cross-repository interpretation.

The family remains available for future benchmark expansion or ablation work.

---

## 15. Frozen 15-task pilot membership

Approved pilot size:

```text
5 recurrence families x 3 tasks = 15 tasks
```

| Family | Review ID | Repository |
|---|---|---|
| GS-F001 | GS-R007 | `swesmith/arrow-py__arrow.1d70d009` |
| GS-F001 | GS-R009 | `swesmith/tornadoweb__tornado.d5ac65c1` |
| GS-F001 | GS-R010 | `swesmith/tox-dev__pipdeptree.c31b6418` |
| GS-F002 | GS-R012 | `swesmith/pygments__pygments.27649ebb` |
| GS-F002 | GS-R014 | `swesmith/pallets__jinja.ada0a9a6` |
| GS-F002 | GS-R015 | `swesmith/stanfordnlp__dspy.651a4c71` |
| GS-F003 | GS-R021 | `swesmith/sunpy__sunpy.f8edfd5c` |
| GS-F003 | GS-R022 | `swesmith/tobymao__sqlglot.036601ba` |
| GS-F003 | GS-R023 | `swesmith/Cog-Creators__Red-DiscordBot.33e0eac7` |
| GS-F004 | GS-R026 | `swesmith/Project-MONAI__MONAI.a09c1f08` |
| GS-F004 | GS-R027 | `swesmith/pydata__patsy.a5d16484` |
| GS-F004 | GS-R030 | `swesmith/stanfordnlp__string2string.c4a72f59` |
| GS-F005 | GS-R032 | `swesmith/cknd__stackprinter.219fcc52` |
| GS-F005 | GS-R033 | `swesmith/tobymao__sqlglot.036601ba` |
| GS-F005 | GS-R034 | `swesmith/stanfordnlp__dspy.651a4c71` |

Expected unique repositories:

```text
13
```

SQLGlot and DSPy appear in more than one recurrence family, but no task instance is duplicated.

---

## 16. Frozen chronological execution order

This is benchmark execution chronology, not a claim about real-world historical repository dates.

| Global index | Review ID | Family | Family occurrence |
|---:|---|---|---:|
| 1 | GS-R007 | GS-F001 | 1 |
| 2 | GS-R012 | GS-F002 | 1 |
| 3 | GS-R021 | GS-F003 | 1 |
| 4 | GS-R026 | GS-F004 | 1 |
| 5 | GS-R032 | GS-F005 | 1 |
| 6 | GS-R009 | GS-F001 | 2 |
| 7 | GS-R014 | GS-F002 | 2 |
| 8 | GS-R022 | GS-F003 | 2 |
| 9 | GS-R027 | GS-F004 | 2 |
| 10 | GS-R033 | GS-F005 | 2 |
| 11 | GS-R010 | GS-F001 | 3 |
| 12 | GS-R015 | GS-F002 | 3 |
| 13 | GS-R023 | GS-F003 | 3 |
| 14 | GS-R030 | GS-F004 | 3 |
| 15 | GS-R034 | GS-F005 | 3 |

### Why this order is used

The first occurrence of every family appears before its second occurrence, and every second occurrence appears before its third occurrence.

This creates the intended memory opportunity:

```text
occurrence 1
    -> possible successful recovery stored in Graph Swarm

occurrence 2
    -> possible retrieval/reuse opportunity

occurrence 3
    -> later recurrence opportunity with more history available
```

---

## 17. Temporal leakage rule

For task `t`, Graph Swarm may use only memory produced by tasks satisfying:

```text
historical.chronological_index < t.chronological_index
```

It may not retrieve:

- a future task;
- a future task's mutation family;
- a future test result;
- a future recovery;
- a future success trajectory;
- future annotations;
- researcher-only evidence.

This must be enforced by the benchmark controller, not merely described in an agent prompt.

---

## 18. Researcher/controller metadata vs agent-visible information

### Researcher/controller-only

```text
review_id
family_id
family_label
mutation_family
occurrence_index
chronological_index
validated
dataset_patch
patch_role
transfer_rationale
manual recovery_pattern annotation
manual notes
future task membership
```

These fields must not automatically become agent prompt content.

### Agent-visible current-task information

The agent may eventually receive only information appropriate to solving the current task, for example:

```text
current problem statement
current repository state
available tools
test output produced during the run
command output produced during the run
eligible historical Graph Swarm memories from earlier tasks
```

The exact execution interface is implemented separately.

---

## 19. Curated recovery pattern vs experimental Graph Swarm memory

The canonical recovery pattern in the annotation CSV exists to justify why the benchmark family is considered transferable.

It must not simply be injected into the agent at benchmark start.

The experimental memory should come from the agent's own earlier observed failure and successful recovery episode.

```text
manual recurrence annotation
    = benchmark design evidence

Graph Swarm Resolution memory
    = experimental history generated during earlier runs
```

Confusing these would leak benchmark design into the treatment condition.

---

## 20. Finalization script

The intended finalization script is:

```text
scripts/finalize_pilot_manifest.py
```

Responsibilities:

1. apply the approved manual annotations to `recurrence_validation.csv`;
2. mark exactly 15 rows as `keep=yes`;
3. mark the other 20 as `keep=no`;
4. assign final family metadata;
5. assign the frozen chronological order;
6. generate the pilot manifest;
7. validate balance and uniqueness.

Output:

```text
benchmark/manifests/pilot.jsonl
```

The script should be deterministic and idempotent.

If it has not yet been executed in a fresh checkout, it must be run before benchmark execution.

---

## 21. Pilot manifest schema

The frozen manifest is intended to contain exactly:

```text
task_id
review_id
source
instance_id
repository
image_name
family_id
family_label
occurrence_index
chronological_index
split
validated
```

Expected task IDs:

```text
GS-T001
...
GS-T015
```

Expected split:

```text
pilot
```

Expected validation flag:

```text
true
```

---

## 22. Information intentionally excluded from `pilot.jsonl`

The manifest must not contain:

```text
dataset_patch
patch
test_patch
ground-truth fix
solution patch
recovery patch
FAIL_TO_PASS
PASS_TO_PASS
operation_hint
transfer_rationale
recovery_pattern
researcher notes
```

The manifest is controller metadata, not a combined research dump.

---

## 23. Repository artifacts and roles

| Path | Role | Agent visible? |
|---|---|---|
| `scripts/build_candidate_pool.py` | Build mutation-aware 105-task pool | No |
| `scripts/build_validation_shortlist.py` | Reduce 105 tasks to 35 review candidates | No |
| `scripts/build_validation_evidence.py` | Extract mutation evidence for 35 tasks | No |
| `scripts/finalize_pilot_manifest.py` | Materialize approved annotations and manifest | No |
| `benchmark/derived/swesmith_candidates.jsonl` | Derived candidate pool | No |
| `benchmark/derived/swesmith_candidate_summary.txt` | Derived candidate-pool summary | No |
| `benchmark/annotations/recurrence_validation.csv` | Manual recurrence QA record | No |
| `benchmark/annotations/recurrence_evidence.jsonl` | Researcher-only mutation evidence | Never |
| `benchmark/manifests/pilot.jsonl` | Frozen benchmark-controller manifest | Not directly |
| `benchmark/families/` | Reserved for family documentation if needed | No |

---

## 24. Recommended Git handling

Derived benchmark data should not be treated as canonical benchmark definitions.

In particular:

```text
benchmark/derived/
```

is versioned derived benchmark material and should remain separate from runtime output.

The durable research artifacts are the scripts, curated annotations, and frozen manifest.

Before publishing the repository, verify that no raw dataset cache or accidentally copied mutation patch is committed where it is not needed.

---

## 25. Reproducibility commands

From the repository root on Windows PowerShell.

### Build the 105-task candidate pool

```powershell
.\.venv\Scripts\python.exe scripts\build_candidate_pool.py
```

Expected:

```text
50,908 dataset rows scanned
105 candidate rows written
15 per allowed mutation family
```

### Build the 35-task validation shortlist

```powershell
.\.venv\Scripts\python.exe scripts\build_validation_shortlist.py
```

Expected:

```text
105 input candidates
77 valid candidates considered
35 selected
5 per mutation family
26 unique repositories
```

### Extract researcher-only mutation evidence

```powershell
.\.venv\Scripts\python.exe scripts\build_validation_evidence.py
```

Expected:

```text
35 shortlist rows
35 dataset matches
35 non-empty patches
35 unique IDs
```

### Finalize the approved pilot

```powershell
.\.venv\Scripts\python.exe scripts\finalize_pilot_manifest.py
```

Expected:

```text
15 keep=yes
20 keep=no
15 pilot manifest records
3 tasks per final family
5 first occurrences
5 second occurrences
5 third occurrences
13 unique repositories
```

---

## 26. Validation invariants

The pilot manifest should satisfy:

```text
rows = 15
unique instance IDs = 15
family count = 3 for each GS-F001..GS-F005
occurrence 1 count = 5
occurrence 2 count = 5
occurrence 3 count = 5
chronological indexes = 1..15
unique repositories = 13
```

The annotation CSV should satisfy:

```text
rows = 35
keep=yes = 15
keep=no = 20
transferable=yes among kept rows = 15
kept chronology = 1..15
```

If these conditions fail, benchmark execution should stop rather than silently continue with a changed dataset.

---

## 27. Why Docker images were not downloaded during construction

Dataset construction intentionally stops before environment execution.

Docker images can be large and are only needed for tasks that actually enter the executable benchmark.

Downloading them before pilot membership is frozen would:

- consume unnecessary disk and network resources;
- mix dataset QA with execution infrastructure;
- make curation failures harder to distinguish from environment failures.

The intended sequence is:

```text
curate first
freeze task IDs
then prepare only the environments needed for execution
```

---

## 28. Why the benchmark is small

The first pilot intentionally contains only 15 tasks.

The goal is not leaderboard coverage.

The goal is to validate the complete research mechanism:

```text
earlier failure
    ->
successful recovery
    ->
persistent memory
    ->
later comparable failure opportunity
    ->
retrieval before action
    ->
behavior change
    ->
measure recurrence prevention
```

A small benchmark makes it practical to inspect every recurrence relationship, trace every stored memory, diagnose false advice, and verify that no future leakage occurred.

A larger benchmark can be added after the mechanism works reliably.

---

## 29. Methodological limitations

### Synthetic mutation origin

The tasks are derived from SWE-smith mutation generation.

The pilot therefore tests recurring software-failure recovery under controlled synthetic faults rather than claiming full naturalistic coverage of production incidents.

### Manual family validation

The final recurrence families are manually curated.

This improves quality but introduces researcher judgment, so the rationale is preserved explicitly in the annotation file.

### Small sample size

Three tasks per family are suitable for an initial controlled pilot but not for broad population-level claims.

### Family membership is not causal evidence

A task belonging to a recurrence family does not prove that a specific historical resolution caused future success.

Graph Swarm V1 should prefer language such as:

```text
observed successful recovery
empirically verified historical resolution
reused recovery associated with success
```

and avoid unsupported causal claims.

### Mutation metadata is privileged

The benchmark remains valid only if mutation patches, future family information, and manual annotations stay outside the agent execution context.

---

## 30. Relationship to Graph Swarm runtime failure taxonomy

The benchmark recurrence families do not replace Graph Swarm's runtime failure detector.

The runtime system may observe operational categories such as:

```text
TEST_FAILURE
COMMAND_FAILURE
DEPENDENCY_ERROR
CONFIGURATION_ERROR
TOOL_PARAMETER_ERROR
```

A benchmark family such as:

```text
missing_initialization_or_assignment
```

describes reusable code-recovery structure.

The runtime event may still be:

```text
TEST_FAILURE
```

because a failing test is how the environment exposes the bug.

This separation lets Graph Swarm reason about both:

```text
what happened operationally
and
what historical recovery pattern may be applicable
```

---

## 31. What dataset construction does not prove

Dataset construction does not prove that Graph Swarm works.

It only creates controlled opportunities to test it.

Questions left for the experiments include:

- will the agent solve occurrence 1;
- will Graph Swarm capture the successful recovery correctly;
- will the resolution become verified;
- will later retrieval find the correct memory;
- will advice change behavior;
- will recurrence be prevented;
- will irrelevant advice remain low;
- will memory overhead remain acceptable.

---

## 32. Next implementation milestone

After the dataset is frozen, work moves to the first executable Graph Swarm vertical slice.

Conceptually:

```text
selected coding task
    ->
agent runs test
    ->
test fails
    ->
Graph Swarm records FailureEpisode
    ->
agent modifies code
    ->
test succeeds
    ->
Graph Swarm records Resolution
    ->
Outcome is persisted
    ->
complete lineage is queryable in Neo4j
```

This establishes the memory-producing half of the system before pre-action retrieval and recurrence-prevention experiments are introduced.

---

## 33. Expected lineage for the first executable task

At minimum, an executed task should eventually produce traceable entities equivalent to:

```text
Run
Task
Action
Tool
FailureEpisode
Resolution
Outcome
Environment
```

with lineage conceptually equivalent to:

```text
(Task)-[:HAS_ACTION]->(Action)
(Action)-[:USED]->(Tool)
(Action)-[:PART_OF]->(Run)
(Action)-[:PART_OF_FAILURE]->(FailureEpisode)
(FailureEpisode)-[:OCCURRED_IN]->(Environment)
(FailureEpisode)-[:RESOLVED_BY]->(Resolution)
(Resolution)-[:VERIFIED_BY]->(Outcome)
```

The dataset was constructed so later tasks create controlled opportunities to retrieve and reuse these historical resolutions.

---

## 34. Summary of the final methodology

1. Start from `SWE-bench/SWE-smith-py` train.
2. Use encoded procedural mutation families for candidate discovery rather than noisy issue-text labels.
3. Filter out tasks with no failing target test or no regression-preservation tests.
4. Keep a balanced deterministic pool of 105 candidates.
5. Reduce to 35 review tasks while preferring repository diversity and lower test cost.
6. Extract mutation patches into a separate researcher-only evidence file.
7. Treat those patches as bug-inducing mutation evidence, never as agent-visible fixes.
8. Manually evaluate whether tasks share a plausible reusable recovery principle.
9. Freeze five strong recurrence families with three tasks each.
10. Assign explicit experimental chronology so historical memory only flows forward.
11. Keep benchmark labels, future membership, mutation patches, and manual rationales outside the agent context.
12. Use the resulting 15-task pilot to test Graph Swarm's verified failure-memory mechanism.

---

## 35. Final construction numbers

```text
Source rows:                    50,908

Mutation-aware candidate pool:     105
Mutation families considered:        7
Candidates per mutation family:     15

Valid shortlist candidates:         77
Manual-review shortlist:            35
Review candidates per family:        5
Unique shortlist repositories:      26

Evidence records:                   35
Evidence matches:                   35
Non-empty mutation patches:         35

Final recurrence families:           5
Tasks per final family:              3
Final pilot tasks:                  15
Expected unique pilot repos:        13
```

---

## 36. One-sentence benchmark definition

> The Graph Swarm pilot is a manually validated, chronologically ordered 15-task SWE-smith-derived benchmark containing five cross-repository recurring recovery patterns, constructed specifically to test whether verified historical failure resolutions can reduce later recurrence without exposing future or mutation-ground-truth information to the agent.
