# Results

These are descriptive case-study findings, not inferential statistical claims. Values are regenerated from the canonical artifacts and explicit recurrence adjudications by `experiments/analysis/generate_results.py`.

## EXP1 — positive treatment case

Known failure recurrence: B0 = recurred; T = did not recur. Both conditions passed the objective task. T retrieved the intended Tornado recovery pattern. The advisory record documents a change from a planned `run_tests` action to a `read_file` action after advice. This temporal association does not establish that advice caused the repair. Descriptive efficiency changed from 20 to 13 model requests (-35.0%), 19 to 11 tool calls (-42.1%), 182,978 to 115,102 total tokens (-37.1%), and 101.1 to 71.9 seconds (-28.9%).

## EXP2 — no-advice recurrence case

GS-T007 was a manually validated recurrence opportunity. The validated failure signature appeared under both B0 and T. Treatment emitted no recovery advice, and both runs failed. The retained artifacts do not establish why the advisory pipeline emitted no recovery advice. EXP2 is a no-advice recurrence case; these records do not demonstrate selective abstention.

## EXP3 — retrieval versus useful intervention

T retrieved the intended SQLGlot recovery pattern. The validated failure did not recur under B0 or T (0/0 for this case). The advisory event followed the key corrective action in the step record, so the retrieved guidance arrived too late to contribute to that change. Retrieval alone does not guarantee a useful intervention.

## EXP4 — additional retrieval and timing case

Reported separately from EXP1–EXP3, T retrieved the intended Rich recovery pattern and the record shows late intervention. EXP4 is not included in the EXP1–EXP3 aggregate.

## Aggregate descriptive observations

Across the three validated recurrence opportunities, the known failure recurred in 2/3 B0 runs (66.7%) and 1/3 treatment runs (33.3%), a descriptive difference of 33.3 percentage points. Task success across EXP1–EXP3 remains 2/3 in both conditions. The false-advice rate (FAR) is **NOT COMPUTABLE FROM RETAINED ARTIFACTS**.
