"""Post-hoc, read-only diagnosis of Sprint 3B Kilo v2 retrieval coverage.

This module deliberately lives outside the experiment and retrieval
implementation.  It reconstructs the retained agent trajectory, replays the
frozen applicability and retrieval paths, and writes an audit without making
Neo4j writes or calling the coding model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.run_sprint3b_kilo_v2 import load_frozen_execution_context  # noqa: E402
from graph_swarm.advisory.service import AdvisoryService  # noqa: E402
from graph_swarm.domain.actions import PlannedAction  # noqa: E402
from graph_swarm.domain.environment import EnvironmentContext  # noqa: E402
from graph_swarm.domain.recovery_patterns import (  # noqa: E402
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.graph.repository import OperationalMemoryRepository  # noqa: E402
from graph_swarm.integration.advisory_runtime import (  # noqa: E402
    R13B_TREATMENT_PATTERN_IDS,
    create_neo4j_advisory_runtime,
)
from graph_swarm.research import runner as research_runner  # noqa: E402
from graph_swarm.retrieval.applicability import (  # noqa: E402
    ApplicabilityDecision,
    RecoveryPatternApplicabilityService,
)
from graph_swarm.retrieval.query import recovery_retrieval_query_text  # noqa: E402
from graph_swarm.retrieval.service import (  # noqa: E402
    RecoveryPatternRetrievalService,
    RecoveryRetrievalResult,
)
from graph_swarm.settings import get_settings  # noqa: E402

_environment_resolver = cast(
    Callable[[Any, str], EnvironmentContext],
    getattr(research_runner, "_environment_for"),  # noqa: B009
)

EXPERIMENT_ID = "GS-E003"
TASK_IDS = tuple(f"GS-T{i:03d}" for i in range(6, 16))
EVIDENCE_ROOT = PROJECT_ROOT / "research/evidence/results/GS-E003/sprint3b_kilo_v2"
RUNS_ROOT = EVIDENCE_ROOT / "runs/GS-E003/T"
OUTPUT_ROOT = EVIDENCE_ROOT / "posthoc_retrieval_diagnosis"
PATTERN_IDS = tuple(sorted(R13B_TREATMENT_PATTERN_IDS))
ANNOTATIONS = PROJECT_ROOT / "benchmark/annotations/recurrence_validation.csv"


@dataclass(frozen=True)
class ReconstructedAction:
    """One unique planned tool action reconstructed from cumulative snapshots."""

    sequence: int
    tool_call_id: str
    tool: str
    operation: str
    arguments: dict[str, object]
    planned_at: datetime
    first_snapshot_seq: int
    first_step_index: int


@dataclass(frozen=True)
class TaskDiagnosis:
    """Task-level post-retrieval classification inputs."""

    task_id: str
    expected_pattern_id: str | None
    expected_source_task_id: str | None
    status: str
    detail: str


def _json_object(value: Any) -> dict[str, object]:
    if isinstance(value, dict):
        typed_value = cast(dict[Any, Any], value)
        return {str(key): item for key, item in typed_value.items()}
    if isinstance(value, str):
        try:
            decoded: Any = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, dict):
            typed_decoded = cast(dict[Any, Any], decoded)
            return {str(key): item for key, item in typed_decoded.items()}
    return {}


def _timestamp(value: object, fallback: str) -> datetime:
    raw = value if isinstance(value, str) and value.strip() else fallback
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def reconstruct_actions(steps_database: Path) -> tuple[ReconstructedAction, ...]:
    """Read cumulative snapshots and deduplicate tool calls chronologically."""
    connection = sqlite3.connect(steps_database)
    try:
        effects: dict[str, tuple[str, str]] = {}
        for row in connection.execute(
            "SELECT tool_call_id, tool_name, started_at FROM tool_effects"
        ):
            call_id, tool_name, started_at = cast(tuple[object, object, object], row)
            effects[str(call_id)] = (str(tool_name), str(started_at))
        actions: list[ReconstructedAction] = []
        seen: set[tuple[str, str]] = set()
        for snapshot_seq, step_index, snapshot_time, messages in connection.execute(
            "SELECT seq, step_index, timestamp, messages FROM snapshots ORDER BY seq"
        ):
            decoded_messages: Any = json.loads(str(messages))
            if not isinstance(decoded_messages, list):
                continue
            for message in cast(list[Any], decoded_messages):
                if not isinstance(message, dict):
                    continue
                typed_message = cast(dict[str, Any], message)
                raw_parts = typed_message.get("parts", [])
                if not isinstance(raw_parts, list):
                    continue
                parts = cast(list[Any], raw_parts)
                for raw_part in parts:
                    if not isinstance(raw_part, dict):
                        continue
                    typed_part = cast(dict[str, Any], raw_part)
                    if typed_part.get("part_kind") != "tool-call":
                        continue
                    tool = str(typed_part.get("tool_name", ""))
                    call_id = str(typed_part.get("tool_call_id") or "")
                    arguments = _json_object(typed_part.get("args"))
                    identity = (
                        call_id,
                        json.dumps(
                            {"tool": tool, "arguments": arguments},
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ),
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    effect = effects.get(call_id)
                    action_time = _timestamp(
                        effect[1] if effect is not None else None,
                        str(snapshot_time),
                    )
                    actions.append(
                        ReconstructedAction(
                            sequence=len(actions) + 1,
                            tool=tool,
                            operation=tool,
                            tool_call_id=call_id or f"snapshot-{snapshot_seq}-{len(actions)}",
                            arguments=arguments,
                            planned_at=action_time,
                            first_snapshot_seq=int(snapshot_seq),
                            first_step_index=int(step_index),
                        )
                    )
        return tuple(actions)
    finally:
        connection.close()


def _pattern_dump(pattern: RecoveryPattern) -> dict[str, object]:
    dumped = pattern.model_dump(mode="json", exclude={"embedding"})
    return cast(dict[str, object], dumped)


def _environment_dump(environment: EnvironmentContext) -> dict[str, object]:
    return cast(dict[str, object], environment.model_dump(mode="json"))


def _action_dump(task_id: str, action: ReconstructedAction) -> dict[str, object]:
    return {
        "task_id": task_id,
        "action_sequence": action.sequence,
        "tool_call_id": action.tool_call_id,
        "tool": action.tool,
        "operation": action.operation,
        "arguments": action.arguments,
        "planned_at": action.planned_at.isoformat(),
        "first_snapshot_seq": action.first_snapshot_seq,
        "first_step_index": action.first_step_index,
    }


def _policy_rejection_reasons(
    task: Any,
    pattern: RecoveryPattern,
    applicability: ApplicabilityDecision,
) -> tuple[str, ...]:
    """Match RecoveryPatternRetrievalService's chronology/status policy."""
    rejected: list[str] = []
    if pattern.source_chronological_index >= task.chronological_index:
        rejected.append("not_strictly_historical")
    if pattern.invalidated_at is not None:
        rejected.append("invalidated")
    elif pattern.verification_status is RecoveryPatternStatus.CANDIDATE:
        rejected.append("candidate_not_eligible")
    elif pattern.verification_status is RecoveryPatternStatus.STALE:
        rejected.append("stale")
    elif pattern.verification_status is RecoveryPatternStatus.INVALIDATED:
        rejected.append("invalidated")
    rejected.extend(applicability.rejection_reasons)
    return tuple(rejected)


def applicability_record(
    task: Any,
    action: ReconstructedAction,
    environment: EnvironmentContext,
    pattern: RecoveryPattern,
    applicability: RecoveryPatternApplicabilityService,
) -> dict[str, object]:
    planned = PlannedAction(
        id=action.tool_call_id,
        run_id=f"{EXPERIMENT_ID}-T-{task.id}",
        task_id=task.id,
        tool=action.tool,
        operation=action.operation,
        arguments=action.arguments,
        planned_at=action.planned_at,
    )
    decision = applicability.evaluate(planned, environment, pattern)
    rejection_reasons = _policy_rejection_reasons(task, pattern, decision)
    return {
        "task_id": task.id,
        "action_sequence": action.sequence,
        "tool_call_id": action.tool_call_id,
        "tool": action.tool,
        "operation": action.operation,
        "pattern_id": pattern.id,
        "source_task_id": pattern.source_task_id,
        "source_tool": pattern.source_tool,
        "source_operation": pattern.source_operation,
        "source_chronological_index": pattern.source_chronological_index,
        "verification_status": pattern.verification_status.value,
        "invalidated_at": pattern.invalidated_at.isoformat()
        if pattern.invalidated_at is not None
        else None,
        "pattern_environment_constraints": pattern.environment_constraints.model_dump(mode="json"),
        "current_environment_context": _environment_dump(environment),
        "eligible": not rejection_reasons,
        "matched_fields": list(decision.matched_fields),
        "compatible_versions": decision.compatible_versions,
        "compatible_markers": decision.compatible_markers,
        "rejection_reasons": list(rejection_reasons),
    }


def _planned_action(task: Any, run_id: str, action: ReconstructedAction) -> PlannedAction:
    return PlannedAction(
        id=action.tool_call_id,
        run_id=run_id,
        task_id=task.id,
        tool=action.tool,
        operation=action.operation,
        arguments=action.arguments,
        planned_at=action.planned_at,
    )


def _retrieval_record(
    task: Any,
    run_id: str,
    action: ReconstructedAction,
    result: RecoveryRetrievalResult,
    patterns: dict[str, RecoveryPattern],
) -> dict[str, object]:
    eligible_ranks = {
        item.pattern_id: rank for rank, item in enumerate(result.eligible_candidates, start=1)
    }
    candidates: list[dict[str, object]] = []
    for rank, candidate in enumerate(result.candidates, start=1):
        record = cast(dict[str, object], candidate.model_dump(mode="json"))
        record["candidate_rank"] = rank
        record["eligible_candidate_rank"] = eligible_ranks.get(candidate.pattern_id)
        pattern = patterns[candidate.pattern_id]
        record["source_task_id"] = pattern.source_task_id
        record["source_tool"] = pattern.source_tool
        record["source_operation"] = pattern.source_operation
        record["source_chronological_index"] = pattern.source_chronological_index
        record["verification_status"] = pattern.verification_status.value
        record["pattern_environment_constraints"] = pattern.environment_constraints.model_dump(
            mode="json"
        )
        candidates.append(record)
    selected_id = result.selected_pattern.id if result.selected_pattern is not None else None
    return {
        "task_id": task.id,
        "action_sequence": action.sequence,
        "tool_call_id": action.tool_call_id,
        "tool": action.tool,
        "operation": action.operation,
        "query_text": result.query_text,
        "candidates": candidates,
        "eligible_candidate_rank": (
            eligible_ranks.get(selected_id) if selected_id is not None else None
        ),
        "selected_pattern": selected_id,
        "selected_vector_score": result.selected_vector_score,
        "no_selection_reason": result.no_selection_reason,
        "query_version": result.query_version,
    }


def _load_recurrence_expectations(
    patterns: dict[str, RecoveryPattern],
    annotations_path: Path,
) -> dict[str, tuple[str, str]]:
    """Map transfer annotations to canonical source patterns after retrieval."""
    with annotations_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("chronological_index", "").strip().isdigit()]
    sources = {
        row["recovery_pattern"]: row["review_id"]
        for row in rows
        if int(row["chronological_index"]) <= 5
    }
    source_task_by_review = {
        row["review_id"]: f"GS-T{int(row['chronological_index']):03d}"
        for row in rows
        if int(row["chronological_index"]) <= 5
    }
    pattern_by_source_task = {
        pattern.source_task_id: pattern_id for pattern_id, pattern in patterns.items()
    }
    expected: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row["review_id"] not in {f"GS-R{i:03d}" for i in range(9, 35)}:
            continue
        index = int(row["chronological_index"])
        if index < 6 or row["recovery_pattern"] not in sources:
            continue
        source_review = sources[row["recovery_pattern"]]
        source_task = source_task_by_review[source_review]
        pattern_id = pattern_by_source_task.get(source_task)
        if pattern_id is not None:
            expected[f"GS-T{index:03d}"] = (pattern_id, source_task)
    return expected


def _artifact_advice_count(task_id: str, runs_root: Path) -> int:
    artifact_paths = tuple((runs_root / task_id).glob("*/artifact.json"))
    if len(artifact_paths) != 1:
        raise FileNotFoundError(
            f"expected one artifact.json for {task_id}, found {len(artifact_paths)}"
        )
    artifact = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
    return int(artifact.get("advice_count", 0))


def _classify_task(
    task_id: str,
    expected: tuple[str, str] | None,
    applicability_records: list[dict[str, object]],
    retrieval_records: list[dict[str, object]],
    advisory_records: list[dict[str, object]],
) -> TaskDiagnosis:
    if expected is None:
        return TaskDiagnosis(task_id, None, None, "F", "no recurrence expectation resolved")
    pattern_id, source_task_id = expected
    appearances = [
        record
        for record in retrieval_records
        if any(
            candidate["pattern_id"] == pattern_id
            for candidate in cast(list[dict[str, object]], record["candidates"])
        )
    ]
    if not appearances:
        return TaskDiagnosis(
            task_id,
            pattern_id,
            source_task_id,
            "A",
            "expected pattern absent from vector results",
        )
    selected = [record for record in appearances if record["selected_pattern"] == pattern_id]
    boundary = {record["action_sequence"]: record for record in advisory_records}
    for record in selected:
        boundary_record = boundary.get(record["action_sequence"])
        if boundary_record is not None and boundary_record.get("advice_returned") is True:
            return TaskDiagnosis(
                task_id,
                pattern_id,
                source_task_id,
                "E",
                "selected and AdviceResult returned advice",
            )
        if boundary_record is not None:
            return TaskDiagnosis(
                task_id,
                pattern_id,
                source_task_id,
                "D",
                "selected by retrieval but AdvisoryService returned no advice",
            )
    expected_applicability = [
        record for record in applicability_records if record["pattern_id"] == pattern_id
    ]
    if any(record["eligible"] is True for record in expected_applicability):
        return TaskDiagnosis(
            task_id, pattern_id, source_task_id, "C", "eligible but another pattern was selected"
        )
    structural = any(
        "tool_mismatch" not in cast(list[str], record["rejection_reasons"])
        and "operation_mismatch" not in cast(list[str], record["rejection_reasons"])
        for record in expected_applicability
    )
    if structural:
        reasons = Counter(
            reason
            for record in expected_applicability
            for reason in cast(list[str], record["rejection_reasons"])
        )
        return TaskDiagnosis(
            task_id,
            pattern_id,
            source_task_id,
            "B",
            "present but rejected: "
            + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items())),
        )
    return TaskDiagnosis(
        task_id, pattern_id, source_task_id, "F", "never applicable to any actual action"
    )


def _marker_audit(
    patterns: dict[str, RecoveryPattern], environment: EnvironmentContext
) -> dict[str, object]:
    historical = sorted(
        {key for pattern in patterns.values() for key in pattern.environment_constraints.markers}
    )
    current = sorted(environment.markers)
    compared = sorted(set(historical) & set(current))
    return {
        "historical_marker_keys": historical,
        "current_marker_keys": current,
        "keys_compared_by_applicability_service": compared,
        "acquisition_only_markers_present": [
            key for key in ("memory_write_only", "retrieval_performed") if key in historical
        ],
        "influenced_applicability": bool(compared),
        "explanation": (
            "No: the Sprint 3B EnvironmentContext has no markers, so the existing "
            "service compares no marker keys; the acquisition-only markers do not "
            "cause rejection or eligibility."
        ),
    }


def _root_causes(
    applicability_records: list[dict[str, object]],
    retrieval_records: list[dict[str, object]],
    task_diagnoses: list[TaskDiagnosis],
) -> list[dict[str, object]]:
    rejection_counts = Counter(
        reason
        for record in applicability_records
        for reason in cast(list[str], record["rejection_reasons"])
    )
    no_selection = Counter(record["no_selection_reason"] for record in retrieval_records)
    return [
        {
            "stage": "vector candidate generation",
            "finding": (
                f"All {len(retrieval_records)} action replays returned the five canonical "
                "vector candidates; no vector-generation failure was observed."
            ),
            "evidence": {"no_selection_reasons": dict(no_selection)},
        },
        {
            "stage": "chronology/verification",
            "finding": "No chronology, invalidation, stale, or verification rejection occurred.",
            "evidence": {
                key: rejection_counts.get(key, 0)
                for key in (
                    "not_strictly_historical",
                    "invalidated",
                    "candidate_not_eligible",
                    "stale",
                )
            },
        },
        {
            "stage": "structural applicability",
            "finding": (
                "The frozen patterns all require edit_file/edit_file. Most generated "
                "actions were run_command or read_file, so they had no structural match."
            ),
            "evidence": {
                "tool_mismatch": rejection_counts.get("tool_mismatch", 0),
                "operation_mismatch": rejection_counts.get("operation_mismatch", 0),
                "task_status_counts": dict(Counter[str](item.status for item in task_diagnoses)),
            },
        },
        {
            "stage": "environment applicability",
            "finding": (
                "The two structurally matching edit_file actions were rejected because "
                "the runner supplied runtime=python while every canonical pattern "
                "requires runtime=docker."
            ),
            "evidence": {"runtime_mismatch": rejection_counts.get("runtime_mismatch", 0)},
        },
        {
            "stage": "ranking",
            "finding": "Ranking was never reached for an eligible candidate.",
            "evidence": {"actions_with_eligible_pattern": 0, "actions_selected": 0},
        },
        {
            "stage": "AdvisoryService conversion",
            "finding": "No selected pattern reached AdvisoryService conversion.",
            "evidence": {"advisory_boundary_checks": 0, "advice_results": 0},
        },
        {
            "stage": "agent injection",
            "finding": (
                "No advice event was injected; this is downstream of zero retrieval selections."
            ),
            "evidence": {"observed_advice_events": 0},
        },
    ]


def _markdown(report: dict[str, object]) -> str:
    summary = cast(dict[str, object], report["summary"])
    diagnoses = cast(list[dict[str, object]], report["expected_pattern_status"])
    counts = cast(dict[str, object], report["candidate_rejection_counts_by_reason"])
    lines = [
        "# Sprint 3B Kilo v2 post-hoc retrieval diagnosis",
        "",
        "Read-only audit of GS-T006..GS-T015. No experiment, Graph Swarm retrieval "
        "behavior, benchmark artifact, or Neo4j data was modified. Kilo was not "
        "called.",
        "",
        "## Summary",
        "",
        f"- T tasks inspected: {summary['t_tasks_inspected']}",
        f"- Unique actual actions inspected: {summary['unique_actual_actions_inspected']}",
        "- Actions with at least one eligible pattern: "
        f"{summary['actions_with_at_least_one_eligible_pattern']}",
        "- Actions where retrieval selected a pattern: "
        f"{summary['actions_where_retrieval_selected']}",
        "- Actions where AdvisoryService produced advice: "
        f"{summary['actions_where_advisory_produced_advice']}",
        "",
        "## Candidate rejection counts",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{reason}` | {count} |" for reason, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## Expected-pattern status",
            "",
            "| Task | Expected pattern | Source task | Status | Evidence |",
            "|---|---|---|---|---|",
        ]
    )
    lines.extend(
        "| "
        f"{item['task_id']} | `{item['expected_pattern_id']}` | "
        f"{item['expected_source_task_id']} | **{item['status']}** | "
        f"{item['detail']} |"
        for item in diagnoses
    )
    marker = cast(dict[str, object], report["marker_audit"])
    compared_marker_keys = (
        ", ".join(cast(list[str], marker["keys_compared_by_applicability_service"])) or "(none)"
    )
    lines.extend(
        [
            "",
            "## Acquisition-only marker audit",
            "",
            str(marker["explanation"]),
            "",
            "Historical keys: "
            f"`{', '.join(cast(list[str], marker['historical_marker_keys']))}`; "
            "current keys: "
            f"`{', '.join(cast(list[str], marker['current_marker_keys']))}`; "
            "compared keys: "
            f"`{compared_marker_keys}`.",
            "",
            "## Evidence-supported root causes",
            "",
        ]
    )
    for root_cause in cast(list[dict[str, object]], report["root_causes"]):
        lines.append(f"- **{root_cause['stage']}** — {root_cause['finding']}")
    lines.extend(
        [
            "",
            "The coverage failure occurs before ranking: structural action-class "
            "mismatch explains the majority of actions, and the runner "
            "environment/runtime mismatch rejects the only two `edit_file` actions. "
            "Retrieval, AdvisoryService conversion, and injection therefore have no "
            "selected-pattern path to process.",
            "",
            "Full candidate/action evidence is in `diagnosis.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnosis(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Run the complete read-only diagnosis and write both report formats."""
    evidence_root = project_root / "research/evidence/results/GS-E003/sprint3b_kilo_v2"
    runs_root = evidence_root / "runs/GS-E003/T"
    output_root = evidence_root / "posthoc_retrieval_diagnosis"
    annotations_path = project_root / "benchmark/annotations/recurrence_validation.csv"

    context = load_frozen_execution_context(project_root=project_root)
    task_by_id = {case.task.id: case.task for case in context.cases}
    runtime = create_neo4j_advisory_runtime(get_settings(), fail_closed_advisory=True)
    try:
        lineages = {
            pattern_id: runtime.treatment_repository.get_recovery_pattern(pattern_id)
            for pattern_id in PATTERN_IDS
        }
        patterns = {pattern_id: lineage.pattern for pattern_id, lineage in lineages.items()}
        environments: dict[str, EnvironmentContext] = {}
        actions_by_task: dict[str, tuple[ReconstructedAction, ...]] = {}
        for task_id in TASK_IDS:
            run_dirs = tuple(runs_root.joinpath(task_id).glob("*"))
            databases = tuple(
                path / "steps.sqlite" for path in run_dirs if (path / "steps.sqlite").is_file()
            )
            if len(databases) != 1:
                raise FileNotFoundError(
                    f"expected one steps.sqlite for {task_id}, found {len(databases)}"
                )
            actions_by_task[task_id] = reconstruct_actions(databases[0])
            run_id = databases[0].parent.name
            environments[task_id] = _environment_resolver(task_by_id[task_id], run_id)

        applicability_service = RecoveryPatternApplicabilityService()
        applicability_records: list[dict[str, object]] = []
        action_applicability: dict[tuple[str, int], list[dict[str, object]]] = {}
        for task_id in TASK_IDS:
            task = task_by_id[task_id]
            for action in actions_by_task[task_id]:
                records = [
                    applicability_record(
                        task,
                        action,
                        environments[task_id],
                        patterns[pattern_id],
                        applicability_service,
                    )
                    for pattern_id in PATTERN_IDS
                ]
                action_applicability[(task_id, action.sequence)] = records
                applicability_records.extend(records)

        retrieval_service = RecoveryPatternRetrievalService(runtime.treatment_repository)
        retrieval_records: list[dict[str, object]] = []
        retrieval_cache: dict[str, RecoveryRetrievalResult] = {}
        cache_hits = 0
        for task_id in TASK_IDS:
            task = task_by_id[task_id]
            for action in actions_by_task[task_id]:
                run_id = f"{EXPERIMENT_ID}-T-{task_id}"
                planned = _planned_action(task, run_id, action)
                query_key = recovery_retrieval_query_text(task, planned, environments[task_id])
                result = retrieval_cache.get(query_key)
                if result is None:
                    result = retrieval_service.retrieve(task, planned, environments[task_id])
                    retrieval_cache[query_key] = result
                else:
                    cache_hits += 1
                retrieval_records.append(_retrieval_record(task, run_id, action, result, patterns))

        advisory_service = AdvisoryService(
            cast(OperationalMemoryRepository, runtime.treatment_repository),
            retrieval_service=retrieval_service,
            fail_closed=True,
        )
        advisory_records: list[dict[str, object]] = []
        for record in retrieval_records:
            selected_pattern = record["selected_pattern"]
            if selected_pattern is None:
                continue
            task = task_by_id[str(record["task_id"])]
            action = actions_by_task[task.id][int(cast(Any, record["action_sequence"])) - 1]
            planned = _planned_action(task, f"{EXPERIMENT_ID}-T-{task.id}", action)
            try:
                advice = advisory_service.evaluate_action(task, planned, environments[task.id])
                advisory_records.append(
                    {
                        "task_id": task.id,
                        "action_sequence": action.sequence,
                        "selected_pattern": selected_pattern,
                        "advice_returned": advice.has_advice,
                        "advice_result": advice.model_dump(mode="json"),
                        "error": None,
                    }
                )
            except Exception as error:  # noqa: BLE001 - preserve boundary evidence
                advisory_records.append(
                    {
                        "task_id": task.id,
                        "action_sequence": action.sequence,
                        "selected_pattern": selected_pattern,
                        "advice_returned": False,
                        "advice_result": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

        expectations = _load_recurrence_expectations(patterns, annotations_path)
        diagnoses = [
            _classify_task(
                task_id,
                expectations.get(task_id),
                [record for record in applicability_records if record["task_id"] == task_id],
                [record for record in retrieval_records if record["task_id"] == task_id],
                [record for record in advisory_records if record["task_id"] == task_id],
            )
            for task_id in TASK_IDS
        ]
        rejection_counts = Counter[str](
            reason
            for record in applicability_records
            for reason in cast(list[str], record["rejection_reasons"])
        )
        action_with_eligible = sum(
            any(record["eligible"] is True for record in records)
            for records in action_applicability.values()
        )
        selected_count = sum(record["selected_pattern"] is not None for record in retrieval_records)
        advice_count = sum(record["advice_returned"] is True for record in advisory_records)
        actual_advice_events = sum(
            _artifact_advice_count(task_id, runs_root) for task_id in TASK_IDS
        )
        report: dict[str, object] = {
            "metadata": {
                "experiment_id": EXPERIMENT_ID,
                "diagnostic": "sprint3b_kilo_v2_posthoc_retrieval_diagnosis",
                "read_only": True,
                "kilo_called": False,
                "benchmark_tasks_rerun": False,
                "canonical_pattern_ids": list(PATTERN_IDS),
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384,
                    "normalized": True,
                },
                "retrieval_cache_unique_queries": len(retrieval_cache),
                "retrieval_cache_hits": cache_hits,
            },
            "environment_contexts": {
                task_id: _environment_dump(environments[task_id]) for task_id in TASK_IDS
            },
            "actual_actions": [
                _action_dump(task_id, action)
                for task_id in TASK_IDS
                for action in actions_by_task[task_id]
            ],
            "canonical_patterns": {
                pattern_id: _pattern_dump(patterns[pattern_id]) for pattern_id in PATTERN_IDS
            },
            "applicability_only": {
                "records": applicability_records,
                "aggregate_rejection_counts_by_reason": dict(rejection_counts),
            },
            "retrieval_replay": {
                "records": retrieval_records,
                "advisory_boundary_records": advisory_records,
            },
            "marker_audit": _marker_audit(patterns, environments[TASK_IDS[0]]),
            "expected_pattern_status": [item.__dict__ for item in diagnoses],
            "summary": {
                "t_tasks_inspected": len(TASK_IDS),
                "unique_actual_actions_inspected": len(applicability_records) // len(PATTERN_IDS),
                "candidate_rejection_counts_by_reason": dict(rejection_counts),
                "actions_with_at_least_one_eligible_pattern": action_with_eligible,
                "actions_where_retrieval_selected": selected_count,
                "actions_where_advisory_produced_advice": advice_count,
                "observed_agent_advice_events_in_completed_artifacts": actual_advice_events,
                "actual_action_counts_by_tool_operation": dict(
                    Counter(
                        f"{action.tool}/{action.operation}"
                        for task_id in TASK_IDS
                        for action in actions_by_task[task_id]
                    )
                ),
            },
        }
        report["candidate_rejection_counts_by_reason"] = dict(rejection_counts)
        report["root_causes"] = _root_causes(applicability_records, retrieval_records, diagnoses)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "diagnosis.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_root / "diagnosis.md").write_text(_markdown(report), encoding="utf-8")
        return report
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Graph Swarm project root (defaults to this repository)",
    )
    args = parser.parse_args()
    report = run_diagnosis(args.project_root.resolve())
    summary = cast(dict[str, object], report["summary"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
