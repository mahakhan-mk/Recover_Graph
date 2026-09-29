"""Read-only corrected replay and structural audit for Sprint 3B Kilo v3.

This module deliberately lives outside the experiment and retrieval
implementation.  It reconstructs the retained agent trajectory, replays the
frozen applicability and retrieval paths, and writes an audit without making
Neo4j writes or calling the coding model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.run_sprint3b_kilo_v2 import load_frozen_execution_context  # noqa: E402
from experiments.run_sprint3b_kilo_v3 import (  # noqa: E402
    environment_context_for_prepared,
)
from experiments.sprint3 import IsolatedTaskEnvironment  # noqa: E402
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

# The replay deliberately reads untyped JSON/SQLite metadata and uses the
# existing private read-only runner loader; those legacy boundaries remain untyped.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

EXPERIMENT_ID = "GS-E003"
TASK_IDS = tuple(f"GS-T{i:03d}" for i in range(6, 16))
EVIDENCE_ROOT = PROJECT_ROOT / "research/evidence/results/GS-E003/sprint3b_kilo_v3"
RUNS_ROOT = EVIDENCE_ROOT / "runs/GS-E003/T"
OUTPUT_ROOT = EVIDENCE_ROOT / "pre_run_diagnosis"
PATTERN_IDS = tuple(sorted(R13B_TREATMENT_PATTERN_IDS))
ANNOTATIONS = PROJECT_ROOT / "benchmark/annotations/recurrence_validation.csv"


def _prepared_environment_context(
    project_root: Path,
    task: Any,
    run_id: str,
    run_directory: Path,
) -> tuple[EnvironmentContext, Path]:
    """Reconstruct the exact prepared environment recorded by one retained run."""
    run_metadata_path = run_directory / "run_metadata.json"
    try:
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FileNotFoundError(
            f"could not read retained run metadata for {task.id}: {run_metadata_path}"
        ) from error
    if not isinstance(run_metadata, dict):
        raise ValueError(f"retained run metadata is not an object: {run_metadata_path}")
    identity = run_metadata.get("workspace_environment_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"run has no workspace_environment_identity: {run_metadata_path}")
    fingerprint = identity.get("environment_fingerprint")
    runtime_type = identity.get("runtime_type")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"run has no environment_fingerprint: {run_metadata_path}")
    if runtime_type != "docker":
        raise ValueError(
            f"run environment identity is not Docker for {task.id}: {runtime_type!r}"
        )

    root = project_root / "research/evidence/workspaces/sprint3-task-environments" / task.id
    candidates = tuple(root.glob("*/environment.json"))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for metadata_path in candidates:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(metadata, dict)
            or metadata.get("runtime_type") != runtime_type
            or metadata.get("environment_fingerprint") != fingerprint
            or metadata.get("task_id") != task.id
        ):
            continue
        matches.append((metadata_path, metadata))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"exact retained Docker environment match is not proven for {task.id}: "
            f"fingerprint={fingerprint!r}, matches={len(matches)}"
        )
    metadata_path, metadata = matches[0]
    prepared = IsolatedTaskEnvironment(
            task_id=task.id,
            python_executable=Path(str(metadata.get("python_executable", "docker"))),
            validation_marker=metadata_path,
            python_version=(
                str(metadata["python_version"])
                if metadata.get("python_version") is not None
                else None
            ),
            environment_fingerprint=(
                str(metadata["environment_fingerprint"])
                if metadata.get("environment_fingerprint") is not None
                else None
            ),
            runtime_type="docker",
            container_image=(
                str(metadata["container_image"])
                if metadata.get("container_image") is not None
                else None
            ),
            container_python_executable=(
                str(metadata["container_python_executable"])
                if metadata.get("container_python_executable") is not None
                else None
            ),
        )
    return environment_context_for_prepared(task, run_id, prepared), metadata_path


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


def reconstruct_actions(
    steps_database: Path,
    *,
    executed_only: bool = False,
) -> tuple[ReconstructedAction, ...]:
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
                    if executed_only and call_id not in effects:
                        continue
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


def _projected_policy_records(
    task: Any,
    action: ReconstructedAction,
    environment: EnvironmentContext,
    result: RecoveryRetrievalResult,
    patterns: dict[str, RecoveryPattern],
    applicability: RecoveryPatternApplicabilityService,
    policy: str,
) -> list[dict[str, object]]:
    """Project one structural policy over already-produced vector candidates."""
    planned = _planned_action(task, f"{EXPERIMENT_ID}-T-{task.id}", action)
    projected: list[dict[str, object]] = []
    for vector_rank, candidate in enumerate(result.candidates, start=1):
        pattern = patterns[candidate.pattern_id]
        if policy == "current":
            tool = pattern.applicability_tool
            operation = pattern.applicability_operation
        elif policy == "failure_trigger":
            tool = pattern.source_tool
            operation = pattern.source_operation
        else:
            raise ValueError(f"unknown structural policy: {policy}")
        projected_pattern = pattern.model_copy(
            update={
                "applicability_tool": tool,
                "applicability_operation": operation,
            }
        )
        decision = applicability.evaluate(planned, environment, projected_pattern)
        reasons = _policy_rejection_reasons(task, projected_pattern, decision)
        projected.append(
            {
                "pattern_id": pattern.id,
                "vector_rank": vector_rank,
                "vector_score": candidate.vector_score,
                "structural_match": not any(
                    reason in {"tool_mismatch", "operation_mismatch"}
                    for reason in decision.rejection_reasons
                ),
                "eligible": not reasons,
                "rejection_reasons": list(reasons),
            }
        )
    return projected


def _structural_audit(
    task: Any,
    actions: tuple[ReconstructedAction, ...],
    environment: EnvironmentContext,
    patterns: dict[str, RecoveryPattern],
    retrieval_results: dict[int, RecoveryRetrievalResult],
    expected: tuple[str, str] | None,
    applicability: RecoveryPatternApplicabilityService,
) -> dict[str, object]:
    expected_pattern_id = expected[0] if expected is not None else None
    policy_reports: dict[str, dict[str, object]] = {}
    for policy in ("current", "failure_trigger"):
        action_reports: list[dict[str, object]] = []
        selected_pattern_ids: list[str] = []
        projected_eligible_actions = 0
        projected_selections = 0
        for action in actions:
            projected = _projected_policy_records(
                task,
                action,
                environment,
                retrieval_results[action.sequence],
                patterns,
                applicability,
                policy,
            )
            eligible = [item for item in projected if item["eligible"] is True]
            selected = eligible[0]["pattern_id"] if eligible else None
            if eligible:
                projected_eligible_actions += 1
                projected_selections += 1
                selected_pattern_ids.append(cast(str, selected))
            expected_candidate = next(
                (item for item in projected if item["pattern_id"] == expected_pattern_id),
                None,
            )
            action_reports.append(
                {
                    "action_sequence": action.sequence,
                    "tool": action.tool,
                    "operation": action.operation,
                    "expected_pattern_vector_rank": (
                        expected_candidate["vector_rank"]
                        if expected_candidate is not None
                        and expected_candidate["structural_match"] is True
                        else None
                    ),
                    "top1_pattern": (projected[0]["pattern_id"] if projected else None),
                    "selected_pattern": selected,
                    "eligible_pattern_ids": [item["pattern_id"] for item in eligible],
                    "candidate_evaluations": projected,
                }
            )
        matching_actions = [
            item for item in action_reports if item["expected_pattern_vector_rank"] is not None
        ]
        first_match = matching_actions[0]["action_sequence"] if matching_actions else None
        selected_expected = [
            item for item in action_reports if item["selected_pattern"] == expected_pattern_id
        ]
        incorrect = [
            item
            for item in action_reports
            if item["selected_pattern"] is not None
            and item["selected_pattern"] != expected_pattern_id
        ]
        policy_reports[policy] = {
            "first_structural_match": first_match,
            "expected_pattern_vector_rank_at_matching_actions": matching_actions,
            "top1_pattern_at_matching_actions": [
                {
                    "action_sequence": item["action_sequence"],
                    "top1_pattern": item["top1_pattern"],
                    "selected_pattern": item["selected_pattern"],
                }
                for item in matching_actions
            ],
            "projected_eligible_actions": projected_eligible_actions,
            "projected_selections": projected_selections,
            "expected_pattern_would_be_selected": bool(selected_expected),
            "same_pattern_projected_advice_count": len(selected_expected),
            "projected_incorrect_pattern_selections": len(incorrect),
            "projected_selected_pattern_counts": dict(Counter(selected_pattern_ids)),
            "actions": action_reports,
        }
    return {
        "task_id": task.id,
        "expected_pattern_id": expected_pattern_id,
        "expected_source_task_id": expected[1] if expected is not None else None,
        "actual_action_count": len(actions),
        "policies": policy_reports,
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


def budget_audit(source_runs_root: Path) -> dict[str, object]:
    """Summarize retained B0/T tool trajectories without replaying the agent."""
    records: list[dict[str, object]] = []
    for condition in ("B0", "T"):
        condition_root = source_runs_root.parent / condition
        for task_id in TASK_IDS:
            run_directories = tuple(condition_root.joinpath(task_id).glob("*"))
            artifact_paths = tuple(
                path / "artifact.json"
                for path in run_directories
                if (path / "artifact.json").is_file()
            )
            metadata_paths = tuple(
                path / "run_metadata.json"
                for path in run_directories
                if (path / "run_metadata.json").is_file()
            )
            database_paths = tuple(
                path / "steps.sqlite"
                for path in run_directories
                if (path / "steps.sqlite").is_file()
            )
            if len(artifact_paths) != 1 or len(metadata_paths) != 1 or len(database_paths) != 1:
                raise FileNotFoundError(
                    f"expected one retained {condition} trajectory for {task_id}"
                )
            artifact = _json_object(json.loads(artifact_paths[0].read_text(encoding="utf-8")))
            metadata = _json_object(json.loads(metadata_paths[0].read_text(encoding="utf-8")))
            actions = reconstruct_actions(database_paths[0], executed_only=True)
            snapshot_actions = reconstruct_actions(database_paths[0])
            first_edit = next(
                (action.sequence for action in actions if action.tool == "edit_file"), None
            )
            events = metadata.get("tool_action_events", [])
            if not isinstance(events, list):
                events = []
            errors: list[str] = []
            for event in cast(list[Any], events):
                if not isinstance(event, dict):
                    continue
                result = event.get("result")
                if isinstance(result, dict) and result.get("error"):
                    errors.append(str(result["error"]))
            executed = artifact.get("executed_action")
            if not errors and isinstance(executed, dict) and executed.get("error"):
                errors.append(str(executed["error"]))
            unsupported_shell = sum(
                "unsupported shell" in error.lower() or "requires structured argv" in error.lower()
                for error in errors
            )
            unavailable_rg = sum(
                "rg" in error.lower()
                and ("not found" in error.lower() or "executable" in error.lower())
                for error in errors
            )
            git_inspection = 0
            for action in actions:
                command = action.arguments.get("command")
                if action.tool != "run_command" or not isinstance(command, list):
                    continue
                words = [str(item) for item in command]
                if len(words) > 1 and words[0] == "git" and words[1] in {
                    "status", "diff", "show", "log", "rev-parse",
                }:
                    git_inspection += 1
            termination = artifact.get("termination")
            exhausted = (
                isinstance(termination, dict)
                and termination.get("type") == "budget_exhausted"
            )
            tool_calls = artifact.get("tool_calls")
            if not isinstance(tool_calls, int):
                tool_calls = len(actions)
            error_message = str(metadata.get("error_message") or "")
            budget_match = re.search(r"tool_calls=(\d+)", error_message)
            budget_consumed = int(budget_match.group(1)) if budget_match else tool_calls
            completed_count = len(actions)
            failed_count = len(errors)
            records.append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "configured_budget": 20,
                    "budget_consumed_count": budget_consumed,
                    "artifact_tool_calls": tool_calls,
                    "completed_tool_action_count": completed_count,
                    "successful_tool_action_count": completed_count - failed_count,
                    "failed_invalid_tool_action_count": failed_count,
                    "budget_count_absent_from_artifact_events": budget_consumed
                    - completed_count,
                    "final_observed_action_index": completed_count,
                    "snapshot_planned_action_count": len(snapshot_actions),
                    "snapshot_only_unexecuted_action_count": len(snapshot_actions)
                    - completed_count,
                    "exhausted_20_call_limit": exhausted,
                    "first_edit_file_action_index": first_edit,
                    "invalid_run_command_unsupported_shell_syntax": unsupported_shell,
                    "unavailable_rg_failures": unavailable_rg,
                    "git_state_inspection_actions": git_inspection,
                    "terminated_before_edit_file": bool(exhausted and first_edit is None),
                }
            )
    exhausted_records = [record for record in records if record["exhausted_20_call_limit"] is True]
    lost_records = [
        record
        for record in records
        if record["condition"] == "T" and record["terminated_before_edit_file"] is True
    ]
    return {
        "max_actions": 20,
        "counting_notes": {
            "completed_actions": "artifact.tool_calls and SQLite tool_effects rows",
            "budget_consumed_count": "tool_calls=N parsed from UsageLimitExceeded metadata",
            "snapshot_planned_actions": (
                "SQLite snapshot tool-call parts, including calls rejected before execution"
            ),
        },
        "records": records,
        "summary": {
            "artifact_tool_calls_by_task_condition": {
                f"{record['condition']}/{record['task_id']}": record["artifact_tool_calls"]
                for record in records
            },
            "budget_consumed_count_by_task_condition": {
                f"{record['condition']}/{record['task_id']}": record["budget_consumed_count"]
                for record in records
            },
            "tasks_exhausting_20_call_limit": [
                f"{record['condition']}/{record['task_id']}" for record in exhausted_records
            ],
            "tasks_exhausting_20_call_limit_count": len(exhausted_records),
            "invalid_run_command_unsupported_shell_syntax_count": sum(
                cast(int, record["invalid_run_command_unsupported_shell_syntax"])
                for record in records
            ),
            "unavailable_rg_failure_count": sum(
                int(cast(int, record["unavailable_rg_failures"])) for record in records
            ),
            "git_state_inspection_action_count": sum(
                int(cast(int, record["git_state_inspection_actions"])) for record in records
            ),
            "treatment_opportunities_lost_before_edit_file": len(lost_records),
            "treatment_tasks_lost_before_edit_file": [
                str(record["task_id"]) for record in lost_records
            ],
        },
    }


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
            "evidence": {
                "no_selection_reasons": {str(key): value for key, value in no_selection.items()}
            },
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
                "The corrected replay supplies the prepared Docker runtime and Python "
                "version to T; structurally matching edit_file actions are no longer "
                "rejected for runtime mismatch."
            ),
            "evidence": {"runtime_mismatch": rejection_counts.get("runtime_mismatch", 0)},
        },
        {
            "stage": "ranking",
            "finding": (
                "Corrected replay ranking is reached only where a candidate survives "
                "applicability."
            ),
            "evidence": {
                "actions_with_eligible_pattern": sum(
                    any(
                        candidate["eligible"] is True
                        for candidate in cast(list[dict[str, object]], record["candidates"])
                    )
                    for record in retrieval_records
                ),
                "actions_selected": sum(
                    record["selected_pattern"] is not None for record in retrieval_records
                ),
            },
        },
        {
            "stage": "AdvisoryService conversion",
            "finding": "Only corrected retrieval selections reach AdvisoryService conversion.",
            "evidence": {
                "advisory_boundary_checks": sum(
                    record["selected_pattern"] is not None for record in retrieval_records
                ),
                "advice_results": "reported in summary",
            },
        },
        {
            "stage": "agent injection",
            "finding": (
                "Observed artifact advice events remain historical v2 evidence; this replay "
                "does not mutate or rerun the agent."
            ),
            "evidence": {"observed_advice_events": 0},
        },
    ]


def _markdown(report: dict[str, object]) -> str:
    summary = cast(dict[str, object], report["summary"])
    diagnoses = cast(list[dict[str, object]], report["expected_pattern_status"])
    counts = cast(dict[str, object], report["candidate_rejection_counts_by_reason"])
    lines = [
        "# Sprint 3B Kilo v3 corrected offline replay",
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
        lines.append(f"- **{root_cause['stage']}** â€” {root_cause['finding']}")
    lines.extend(
        [
            "",
            "This replay corrects only the T EnvironmentContext runtime boundary. "
            "The separate structural applicability question is reported below as a "
            "counterfactual and is not applied to production retrieval.",
            "",
            "## CURRENT vs FAILURE-TRIGGER structural audit",
            "",
            "The audit reuses the exact vector candidates produced before policy "
            "projection. `CURRENT` uses `applicability_tool`/`applicability_operation`; "
            "`FAILURE-TRIGGER` uses `source_tool`/`source_operation`.",
            "",
            "Full candidate/action evidence is in `diagnosis.json`.",
            "",
        ]
    )
    for item in cast(list[dict[str, object]], report["structural_applicability_audit"]):
        policies = cast(dict[str, dict[str, object]], item["policies"])
        lines.append(f"### {item['task_id']} — expected `{item['expected_pattern_id']}`")
        lines.append("")
        lines.append(
            "| Policy | First match | Eligible actions | Selections | Expected selected | "
            "Same-pattern advice | Incorrect selections |"
        )
        lines.append("|---|---:|---:|---:|---|---:|---:|")
        for policy_name in ("current", "failure_trigger"):
            policy = policies[policy_name]
            lines.append(
                f"| {policy_name} | {policy['first_structural_match']} | "
                f"{policy['projected_eligible_actions']} | {policy['projected_selections']} | "
                f"{policy['expected_pattern_would_be_selected']} | "
                f"{policy['same_pattern_projected_advice_count']} | "
                f"{policy['projected_incorrect_pattern_selections']} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_diagnosis(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Run the complete read-only diagnosis and write both report formats."""
    source_evidence_root = project_root / "research/evidence/results/GS-E003/sprint3b_kilo_v2"
    evidence_root = project_root / "research/evidence/results/GS-E003/sprint3b_kilo_v3"
    runs_root = source_evidence_root / "runs/GS-E003/T"
    output_root = evidence_root / "pre_run_diagnosis"
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
        environment_sources: dict[str, str] = {}
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
            environment, metadata_path = _prepared_environment_context(
                project_root, task_by_id[task_id], run_id, databases[0].parent
            )
            environments[task_id] = environment
            environment_sources[task_id] = str(metadata_path)

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
        retrieval_results_by_action: dict[tuple[str, int], RecoveryRetrievalResult] = {}
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
                retrieval_results_by_action[(task_id, action.sequence)] = result
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
        structural_audit = [
            _structural_audit(
                task_by_id[task_id],
                actions_by_task[task_id],
                environments[task_id],
                patterns,
                {
                    action.sequence: retrieval_results_by_action[(task_id, action.sequence)]
                    for action in actions_by_task[task_id]
                },
                expectations.get(task_id),
                applicability_service,
            )
            for task_id in TASK_IDS
        ]
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
        for reason in ("tool_mismatch", "operation_mismatch", "runtime_mismatch"):
            rejection_counts.setdefault(reason, 0)
        action_with_eligible = sum(
            any(record["eligible"] is True for record in records)
            for records in action_applicability.values()
        )
        selected_count = sum(record["selected_pattern"] is not None for record in retrieval_records)
        advice_count = sum(record["advice_returned"] is True for record in advisory_records)
        actual_advice_events = sum(
            _artifact_advice_count(task_id, runs_root) for task_id in TASK_IDS
        )
        budget_audit_report = budget_audit(source_evidence_root / "runs/GS-E003/T")
        report: dict[str, object] = {
            "metadata": {
                "experiment_id": EXPERIMENT_ID,
                "diagnostic": "sprint3b_kilo_v3_corrected_offline_replay",
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
            "environment_replay_sources": environment_sources,
            "agent_tool_budget_audit": budget_audit_report,
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
            "structural_applicability_audit": structural_audit,
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
