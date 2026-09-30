"""Read-only GS-E003 pilot audit for corrected trigger applicability.

This is development evidence only. It reuses the retained v3 candidate scores
and trajectories, reads historical provenance, and writes a report under
``analysis/reports``. It does not call Kilo, rerun a benchmark, or write Neo4j.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from graph_swarm.domain.actions import PlannedAction  # noqa: E402
from graph_swarm.domain.environment import EnvironmentContext  # noqa: E402
from graph_swarm.domain.recovery_patterns import (  # noqa: E402
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.domain.tasks import Task  # noqa: E402
from graph_swarm.integration.advisory_runtime import (  # noqa: E402
    R13B_TREATMENT_PATTERN_IDS,
    create_neo4j_advisory_runtime,
)
from graph_swarm.retrieval.applicability import (  # noqa: E402
    ApplicabilityDecision,
    RecoveryPatternApplicabilityService,
)
from graph_swarm.settings import get_settings  # noqa: E402

EVIDENCE = PROJECT_ROOT / "research/evidence/results/GS-E003/sprint3b_kilo_v3"
INPUT = EVIDENCE / "pre_run_diagnosis/diagnosis.json"
OUTPUT = PROJECT_ROOT / "analysis/reports"
TASK_IDS = tuple(f"GS-T{i:03d}" for i in range(6, 16))
ANNOTATIONS = PROJECT_ROOT / "benchmark/annotations/recurrence_validation.csv"


def _policy_reasons(task: Any, pattern: RecoveryPattern) -> list[str]:
    reasons: list[str] = []
    if pattern.source_chronological_index >= task.chronological_index:
        reasons.append("not_strictly_historical")
    if pattern.invalidated_at is not None:
        reasons.append("invalidated")
    elif pattern.verification_status is RecoveryPatternStatus.CANDIDATE:
        reasons.append("candidate_not_eligible")
    elif pattern.verification_status is RecoveryPatternStatus.STALE:
        reasons.append("stale")
    elif pattern.verification_status is RecoveryPatternStatus.INVALIDATED:
        reasons.append("invalidated")
    return reasons


def _planned(task: Any, record: dict[str, object]) -> PlannedAction:
    return PlannedAction(
        id=str(record["tool_call_id"]),
        run_id=f"GS-E003-T-{task.id}",
        task_id=task.id,
        tool=str(record["tool"]),
        operation=str(record["operation"]),
        arguments=cast(dict[str, object], record.get("arguments", {})),
        planned_at=datetime.fromisoformat(str(record["planned_at"])),
    )


def _candidate_map(record: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], record["candidates"])


def _evaluate(
    task: Any,
    action: PlannedAction,
    environment: EnvironmentContext,
    pattern: RecoveryPattern,
    applicability: RecoveryPatternApplicabilityService,
    policy: str,
) -> tuple[bool, ApplicabilityDecision, list[str]]:
    if policy == "current":
        projected = pattern.model_copy(update={"trigger": None})
    elif policy == "source_failure_trigger":
        projected = pattern.model_copy(
            update={
                "trigger": None,
                "applicability_tool": pattern.source_tool,
                "applicability_operation": pattern.source_operation,
            }
        )
    elif policy == "corrected_trigger":
        projected = pattern
    else:
        raise ValueError(f"unknown policy: {policy}")
    decision = applicability.evaluate(action, environment, projected, task)
    reasons = _policy_reasons(task, projected) + list(decision.rejection_reasons)
    return not reasons, decision, reasons


def _task_report(
    task: Any,
    environment: EnvironmentContext,
    action_records: list[dict[str, object]],
    retrieval_records: list[dict[str, object]],
    patterns: dict[str, RecoveryPattern],
    expected_pattern: str | None,
    applicability: RecoveryPatternApplicabilityService,
) -> dict[str, object]:
    policies: dict[str, dict[str, object]] = {}
    for policy in ("current", "source_failure_trigger", "corrected_trigger"):
        actions: list[dict[str, object]] = []
        selected: list[str] = []
        first_eligible: dict[str, object] | None = None
        for action_record, retrieval_record in zip(action_records, retrieval_records, strict=True):
            action = _planned(task, action_record)
            evaluations: list[dict[str, object]] = []
            for candidate in _candidate_map(retrieval_record):
                pattern = patterns[str(candidate["pattern_id"])]
                eligible, decision, reasons = _evaluate(
                    task, action, environment, pattern, applicability, policy
                )
                evaluations.append(
                    {
                        "pattern_id": pattern.id,
                        "vector_rank": candidate["candidate_rank"],
                        "vector_score": candidate["vector_score"],
                        "eligible": eligible,
                        "matched_fields": list(decision.matched_fields),
                        "rejection_reasons": reasons,
                    }
                )
            eligible = [item for item in evaluations if item["eligible"] is True]
            selected_pattern = str(eligible[0]["pattern_id"]) if eligible else None
            if selected_pattern is not None:
                selected.append(selected_pattern)
                if first_eligible is None:
                    first_eligible = {
                        "action_sequence": action_record["action_sequence"],
                        "tool": action.tool,
                        "operation": action.operation,
                        "arguments": action.arguments,
                        "selected_pattern": selected_pattern,
                        "top_candidate_scores": [
                            {
                                "pattern_id": item["pattern_id"],
                                "vector_score": item["vector_score"],
                            }
                            for item in evaluations[:5]
                        ],
                    }
            actions.append(
                {
                    "action_sequence": action_record["action_sequence"],
                    "tool": action.tool,
                    "operation": action.operation,
                    "arguments": action.arguments,
                    "selected_pattern": selected_pattern,
                    "eligible_pattern_ids": [item["pattern_id"] for item in eligible],
                    "candidate_evaluations": evaluations,
                }
            )
        unique_selected = list(dict.fromkeys(selected))
        nonexpected = [item for item in unique_selected if item != expected_pattern]
        policies[policy] = {
            "first_eligible_action": first_eligible,
            "selected_pattern_ids_by_action": selected,
            "advice_count_after_pattern_deduplication": len(unique_selected),
            "actions_with_eligible_candidate": sum(
                bool(item["eligible_pattern_ids"]) for item in actions
            ),
            "actions_with_no_eligible_candidate": [
                item["action_sequence"] for item in actions if not item["eligible_pattern_ids"]
            ],
            "cases_selecting_non_expected_pattern": len(nonexpected),
            "non_expected_selected_pattern_ids": nonexpected,
            "actions": actions,
        }
    return {
        "task_id": task.id,
        "expected_pattern": expected_pattern,
        "actual_action_count": len(action_records),
        "policies": policies,
    }


def _manual_review(
    reports: list[dict[str, object]],
    patterns: dict[str, RecoveryPattern],
    tasks: dict[str, Any],
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for task_report in reports:
        task_id = str(task_report["task_id"])
        expected = task_report["expected_pattern"]
        corrected = cast(dict[str, object], task_report["policies"])["corrected_trigger"]
        actions = cast(list[dict[str, object]], cast(dict[str, object], corrected)["actions"])
        for action in actions:
            selected = action["selected_pattern"]
            if selected is None or selected == expected:
                continue
            pattern = patterns[str(selected)]
            evidence.append(
                {
                    "task_id": task_id,
                    "expected_pattern": expected,
                    "action_sequence": action["action_sequence"],
                    "selected_pattern": selected,
                    "task_problem_statement": tasks[task_id].problem_statement,
                    "planned_action": {
                        "tool": action["tool"],
                        "operation": action["operation"],
                        "arguments": action["arguments"],
                    },
                    "selected_pattern_text": {
                        "title": pattern.title,
                        "guidance": pattern.guidance,
                        "evidence_summary": pattern.evidence_summary,
                    },
                    "historical_trigger": (
                        pattern.trigger.model_dump(mode="json")
                        if pattern.trigger is not None
                        else None
                    ),
                    "review_question": (
                        "Does the selected recovery address the current task's failure mode "
                        "and the planned test intent, or is this only a vector-ranking artifact?"
                    ),
                }
            )
    return evidence


def _markdown(report: dict[str, object]) -> str:
    summary = cast(dict[str, object], report["summary"])
    lines = [
        "# GS-E003 trigger-applicability development audit",
        "",
        "Read-only replay of retained GS-T006..GS-T015 trajectories and v3 vector "
        "candidate scores. Kilo was not called; no benchmark or Neo4j data was modified.",
        "",
        "## Result",
        "",
        f"- Actions inspected: {summary['actions_inspected']}",
        f"- Corrected-trigger advice opportunities: {summary['corrected_advice_opportunities']}",
        "- Corrected expected-pattern task coverage: "
        f"{summary['corrected_expected_task_coverage']}",
        f"- Corrected non-expected selections: {summary['corrected_non_expected_selections']}",
        "",
        "## Policy comparison",
        "",
        "A uses the current edit_file recovery-action key. B uses the source failure "
        "action key. C requires source trigger action/intent, exact text anchors from "
        "the current task and historical context, chronology/status, runtime, and exact "
        "versions unless the trigger explicitly opts out.",
        "",
        "| Policy | Advice opportunities | Actions with no eligible "
        "candidate | Non-expected selections |",
        "|---|---:|---:|---:|",
    ]
    comparison = cast(dict[str, dict[str, int]], summary["policy_comparison"])
    for policy in ("current", "source_failure_trigger", "corrected_trigger"):
        lines.append(
            f"| {policy} | {comparison[policy]['advice_opportunities']} | "
            f"{comparison[policy]['actions_with_no_eligible_candidate']} | "
            f"{comparison[policy]['non_expected_selections']} |"
        )
    lines.extend(
        [
            "",
            "| Task | Expected pattern | A first eligible | B first eligible | "
            "C first eligible | C selected | C advice count | C non-expected |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for item in cast(list[dict[str, object]], report["tasks"]):
        policies = cast(dict[str, dict[str, object]], item["policies"])
        corrected = policies["corrected_trigger"]
        first = cast(dict[str, object] | None, corrected["first_eligible_action"])
        lines.append(
            f"| {item['task_id']} | `{item['expected_pattern']}` | "
            f"{_first(policies['current'])} | {_first(policies['source_failure_trigger'])} | "
            f"{first['action_sequence'] if first else 'none'} | "
            f"`{first['selected_pattern'] if first else 'none'}` | "
            f"{corrected['advice_count_after_pattern_deduplication']} | "
            f"{corrected['cases_selecting_non_expected_pattern']} |"
        )
    lines.extend(
        [
            "",
            "## Version applicability audit",
            "",
            "Exact Python-version equality remains enabled for the corrected trigger "
            "path. The retained pilot shows acquisition/runtime version mismatches for "
            "GS-T013 and GS-T014, but does not establish that either recovery is safe "
            "across Python versions; therefore C does not remove this gate.",
            "",
            "## Manual review of non-expected advice",
            "",
        ]
    )
    reviews = cast(list[dict[str, object]], report["manual_review_evidence"])
    if not reviews:
        lines.append("No corrected-trigger non-expected selections were produced.")
    else:
        for item in reviews:
            lines.append(
                f"- {item['task_id']} action {item['action_sequence']}: "
                f"selected `{item['selected_pattern']}`, expected `{item['expected_pattern']}`. "
                f"{item['review_question']}"
            )
    return "\n".join(lines) + "\n"


def _first(policy: dict[str, object]) -> str:
    first = cast(dict[str, object] | None, policy["first_eligible_action"])
    return str(first["action_sequence"]) if first else "none"


def _load_current_tasks(source: dict[str, object]) -> dict[str, Task]:
    """Load only the current problem statements used by the retained T tasks."""
    environment_contexts = cast(dict[str, dict[str, object]], source["environment_contexts"])
    repositories = {
        task_id: str(value["repository"]) for task_id, value in environment_contexts.items()
    }
    tasks: dict[str, Task] = {}
    with ANNOTATIONS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = row.get("chronological_index", "").strip()
            if not index.isdigit() or int(index) < 6 or int(index) > 15:
                continue
            task_id = f"GS-T{int(index):03d}"
            tasks[task_id] = Task(
                id=task_id,
                problem_statement=row["problem_statement"],
                family_id="offline-audit-placeholder",
                repository=repositories[task_id],
                chronological_index=int(index),
            )
    if set(tasks) != set(TASK_IDS):
        raise ValueError("retained GS-E003 task problem statements are incomplete")
    return tasks


def run_audit(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    retrieval = cast(dict[str, object], source["retrieval_replay"])
    retrieval_records = cast(list[dict[str, object]], retrieval["records"])
    actions = cast(list[dict[str, object]], source["actual_actions"])
    expected = {
        str(item["task_id"]): cast(str | None, item["expected_pattern_id"])
        for item in cast(list[dict[str, object]], source["expected_pattern_status"])
    }
    tasks = _load_current_tasks(source)

    runtime = create_neo4j_advisory_runtime(get_settings(), fail_closed_advisory=True)
    try:
        lineages = {
            pattern_id: runtime.treatment_repository.get_recovery_pattern(pattern_id)
            for pattern_id in sorted(R13B_TREATMENT_PATTERN_IDS)
        }
        patterns = {pattern_id: lineage.pattern for pattern_id, lineage in lineages.items()}
        environments: dict[str, EnvironmentContext] = {}
        raw_environments = cast(dict[str, object], source["environment_contexts"])
        for task_id, value in raw_environments.items():
            environments[task_id] = EnvironmentContext.model_validate(value)
        tasks_report: list[dict[str, object]] = []
        for task_id in TASK_IDS:
            task_actions = [item for item in actions if item["task_id"] == task_id]
            task_retrieval = [item for item in retrieval_records if item["task_id"] == task_id]
            tasks_report.append(
                _task_report(
                    tasks[task_id],
                    environments[task_id],
                    task_actions,
                    task_retrieval,
                    patterns,
                    expected[task_id],
                    RecoveryPatternApplicabilityService(),
                )
            )
        manual = _manual_review(tasks_report, patterns, tasks)
        corrected = [
            cast(dict[str, object], item["policies"])["corrected_trigger"] for item in tasks_report
        ]
        selected_expected_tasks = sum(
            any(
                pattern_id == item["expected_pattern"]
                for pattern_id in cast(
                    list[str],
                    cast(dict[str, object], policy)["selected_pattern_ids_by_action"],
                )
            )
            for item, policy in zip(tasks_report, corrected, strict=True)
        )
        summary = {
            "tasks_inspected": len(tasks_report),
            "actions_inspected": len(actions),
            "corrected_advice_opportunities": sum(
                int(cast(dict[str, object], policy)["advice_count_after_pattern_deduplication"])
                for policy in corrected
            ),
            "corrected_expected_task_coverage": selected_expected_tasks,
            "corrected_non_expected_selections": sum(
                int(cast(dict[str, object], policy)["cases_selecting_non_expected_pattern"])
                for policy in corrected
            ),
            "current_advice_opportunities": sum(
                int(
                    cast(dict[str, object], item["policies"])["current"][
                        "advice_count_after_pattern_deduplication"
                    ]
                )
                for item in tasks_report
            ),
            "source_failure_trigger_advice_opportunities": sum(
                int(
                    cast(dict[str, object], item["policies"])["source_failure_trigger"][
                        "advice_count_after_pattern_deduplication"
                    ]
                )
                for item in tasks_report
            ),
            "policy_comparison": {
                policy: {
                    "advice_opportunities": sum(
                        int(
                            cast(dict[str, object], item["policies"])[policy][
                                "advice_count_after_pattern_deduplication"
                            ]
                        )
                        for item in tasks_report
                    ),
                    "non_expected_selections": sum(
                        int(
                            cast(dict[str, object], item["policies"])[policy][
                                "cases_selecting_non_expected_pattern"
                            ]
                        )
                        for item in tasks_report
                    ),
                    "actions_with_no_eligible_candidate": sum(
                        len(
                            cast(dict[str, object], item["policies"])[policy][
                                "actions_with_no_eligible_candidate"
                            ]
                        )
                        for item in tasks_report
                    ),
                }
                for policy in ("current", "source_failure_trigger", "corrected_trigger")
            },
        }
        report: dict[str, object] = {
            "metadata": {
                "purpose": "GS-E003 development-pilot correction before GS-E004 freeze",
                "read_only": True,
                "kilo_called": False,
                "existing_evidence_immutable": True,
                "expected_pattern_annotations_used_only_for_offline_audit": True,
            },
            "summary": summary,
            "tasks": tasks_report,
            "manual_review_evidence": manual,
            "version_rule": {
                "default": "exact equality",
                "opt_out": "explicit RecoveryTrigger.version_sensitive=false only",
                "pilot_decision": "retain exact equality; no evidence-backed opt-out established",
            },
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "gs_e003_trigger_applicability.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (OUTPUT / "gs_e003_trigger_applicability.md").write_text(
            _markdown(report), encoding="utf-8"
        )
        return report
    finally:
        runtime.close()


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
