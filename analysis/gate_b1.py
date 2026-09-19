"""Offline analysis for immutable Gate B1 attempt and run evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from experiments.sprint3b import (
    AttemptRecord,
    AttemptStatus,
    GateB1EvidenceStore,
    GateB1Manifest,
    GateB1PairStatus,
    load_gate_b1_manifest,
)
from graph_swarm.research.contracts import ExperimentCondition, ExperimentRunArtifact


def normalized_trajectory_signature(events: object) -> str | None:
    """Return a deterministic behavior-only signature for recorded events."""
    if not isinstance(events, list):
        return None
    normalized: list[dict[str, object]] = []
    for event in cast(list[Any], events):
        if not isinstance(event, dict):
            return None
        raw_event = cast(dict[str, Any], event)
        raw_result = raw_event.get("result")
        result = cast(dict[str, Any], raw_result) if isinstance(raw_result, dict) else {}
        tool = result.get("tool_name") or raw_event.get("tool")
        operation = raw_event.get("operation") or result.get("operation") or tool
        if not isinstance(tool, str) or not isinstance(operation, str):
            return None
        operation = re.sub(
            r"(?:run|action|environment|workspace)[-_][A-Za-z0-9-]+",
            "<id>",
            operation,
            flags=re.IGNORECASE,
        )
        operation = re.sub(
            r"\b\d{4}-\d{2}-\d{2}T[^ ]+",
            "<timestamp>",
            operation,
        )
        success = result.get("success")
        if not isinstance(success, bool):
            return None
        exit_code = result.get("exit_code")
        if not isinstance(exit_code, (int, type(None))):
            exit_code = None
        failure_class = raw_event.get("failure_type") or result.get("failure_type")
        if not isinstance(failure_class, str):
            failure_class = "error" if result.get("error") else None
        normalized.append(
            {
                "tool": tool,
                "operation": operation,
                "success": success,
                "exit_code": exit_code,
                "failure_class": failure_class,
            }
        )
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


class GateB1TaskAnalysis(BaseModel):
    """One factual task-level row; unavailable measurements remain null."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    b0_run_id: str | None = None
    o1_run_id: str | None = None
    pair_complete: bool
    b0_success: bool | None = None
    o1_success: bool | None = None
    b0_repeated_failure: bool | None = None
    o1_repeated_failure: bool | None = None
    b0_bounded_termination: bool | None = None
    o1_bounded_termination: bool | None = None
    b0_tool_calls: int | None = None
    o1_tool_calls: int | None = None
    b0_requests: int | None = None
    o1_requests: int | None = None
    b0_retries: int | None = None
    o1_retries: int | None = None
    b0_input_tokens: int | None = None
    o1_input_tokens: int | None = None
    b0_output_tokens: int | None = None
    o1_output_tokens: int | None = None
    b0_latency_ms: float | None = None
    o1_latency_ms: float | None = None
    b0_pacing_wait_seconds: float | None = None
    o1_pacing_wait_seconds: float | None = None
    o1_oracle_coverage: bool | None = None
    observable_trajectory_difference: bool | None = None
    trajectory_comparison: str | None = None
    anomaly: str | None = None


class GateB1Analysis(BaseModel):
    """Human-decision support without a scientific PASS/FAIL declaration."""

    manifest_id: str
    experiment_id: str
    expected_tasks: int
    complete_valid_pairs: int
    incomplete_pairs: int
    invalid_provider_attempts: int
    invalid_infrastructure_attempts: int
    b0_successes: int
    o1_successes: int
    b0_bounded_terminations: int
    o1_bounded_terminations: int
    b0_rfr_count: int | None
    o1_rfr_count: int | None
    o1_oracle_coverage: int
    tasks_with_observable_trajectory_difference: tuple[str, ...]
    tasks_with_identical_or_no_observable_difference: tuple[str, ...]
    missing_evidence_or_anomalies: tuple[str, ...]
    gate_b1_evaluated: bool
    gate_b1_ready_for_review: bool
    previous_provider_smoke_excluded: bool = True
    task_rows: tuple[GateB1TaskAnalysis, ...]


def _read_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return cast(dict[str, Any], raw) if isinstance(raw, dict) else None


def _artifact(record: AttemptRecord) -> ExperimentRunArtifact | None:
    if record.artifact_path is None:
        return None
    try:
        return ExperimentRunArtifact.model_validate_json(
            Path(record.artifact_path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError):
        return None


def _valid_records(
    records: Iterable[AttemptRecord],
    task_id: str,
    condition: ExperimentCondition,
) -> list[AttemptRecord]:
    return [
        record
        for record in records
        if record.task_id == task_id and record.condition is condition and record.valid_observation
    ]


def _metric(artifact: ExperimentRunArtifact | None, name: str) -> Any:
    return None if artifact is None else getattr(artifact, name, None)


def _requests(record: AttemptRecord | None) -> int | None:
    if record is None:
        return None
    raw = _read_json(record.raw_evidence_path)
    usage: Any = None if raw is None else raw.get("usage")
    if not isinstance(usage, dict):
        return None
    usage_record = cast(dict[str, Any], usage)
    value = usage_record.get("requests")
    return value if isinstance(value, int) else None


def _trajectory(record: AttemptRecord | None) -> tuple[str, bool] | None:
    if record is None:
        return None
    raw = _read_json(record.raw_evidence_path)
    if raw is None:
        return None
    signature = normalized_trajectory_signature(raw.get("events"))
    if signature is None:
        return None
    return signature, True


def _row(
    manifest: GateB1Manifest,
    task_id: str,
    records: tuple[AttemptRecord, ...],
    pair: GateB1PairStatus,
) -> GateB1TaskAnalysis:
    b0_records = _valid_records(records, task_id, ExperimentCondition.B0)
    o1_records = _valid_records(records, task_id, ExperimentCondition.O1)
    b0_record = b0_records[0] if b0_records else None
    o1_record = o1_records[0] if o1_records else None
    b0 = _artifact(b0_record) if b0_record else None
    o1 = _artifact(o1_record) if o1_record else None
    anomalies: list[str] = []
    if b0_record and b0 is None:
        anomalies.append("B0 artifact unavailable")
    if o1_record and o1 is None:
        anomalies.append("O1 artifact unavailable")
    if o1 is not None:
        raw = _read_json(o1_record.raw_evidence_path if o1_record else None)
        events: list[Any] | None = None
        if raw is not None and isinstance(raw.get("advice_events"), list):
            events = raw["advice_events"]
        if not isinstance(events, list) or len(events) != 1:
            anomalies.append("O1 Oracle event evidence is missing or not exactly one")
    for label, record, artifact in (
        ("B0", b0_record, b0),
        ("O1", o1_record, o1),
    ):
        if record is None or artifact is None:
            continue
        if record.actual_environment_fingerprint != manifest.task_map[
            task_id
        ].environment_fingerprint:
            anomalies.append(f"{label} environment fingerprint mismatch")
        raw = _read_json(record.raw_evidence_path)
        if raw is None:
            anomalies.append(f"{label} raw evidence unavailable")
        try:
            from experiments.sprint3b import validate_artifact_contract

            validate_artifact_contract(artifact, manifest, expected_condition=artifact.condition)
        except (OSError, ValueError):
            anomalies.append(f"{label} artifact contract invalid")

    b0_trajectory = _trajectory(b0_record)
    o1_trajectory = _trajectory(o1_record)
    trajectory_difference: bool | None = None
    comparison: str | None = None
    if b0_trajectory is not None and o1_trajectory is not None:
        trajectory_difference = b0_trajectory[0] != o1_trajectory[0]
        comparison = "different" if trajectory_difference else "identical"

    oracle_coverage = (
        o1 is not None
        and o1.advice_count == 1
        and o1.advice_intervention_boundary == manifest.oracle_intervention_boundary
        and o1.advice_delivery_timing == manifest.oracle_delivery
        and o1.advice_review_id == manifest.task_map[task_id].review_id
        and o1_record is not None
        and isinstance((_read_json(o1_record.raw_evidence_path) or {}).get("advice_events"), list)
        and len((_read_json(o1_record.raw_evidence_path) or {}).get("advice_events", [])) == 1
    )

    return GateB1TaskAnalysis(
        task_id=task_id,
        b0_run_id=pair.b0_run_id,
        o1_run_id=pair.o1_run_id,
        pair_complete=pair.pair_complete,
        b0_success=None if b0 is None else b0.task_success,
        o1_success=None if o1 is None else o1.task_success,
        b0_repeated_failure=None if b0 is None else b0.known_failure_repeated,
        o1_repeated_failure=None if o1 is None else o1.known_failure_repeated,
        b0_bounded_termination=None if b0 is None else b0.termination is not None,
        o1_bounded_termination=None if o1 is None else o1.termination is not None,
        b0_tool_calls=_metric(b0, "tool_calls"),
        o1_tool_calls=_metric(o1, "tool_calls"),
        b0_requests=_requests(b0_record),
        o1_requests=_requests(o1_record),
        b0_retries=_metric(b0, "retries"),
        o1_retries=_metric(o1, "retries"),
        b0_input_tokens=_metric(b0, "input_tokens"),
        o1_input_tokens=_metric(o1, "input_tokens"),
        b0_output_tokens=_metric(b0, "output_tokens"),
        o1_output_tokens=_metric(o1, "output_tokens"),
        b0_latency_ms=_metric(b0, "latency_ms"),
        o1_latency_ms=_metric(o1, "latency_ms"),
        b0_pacing_wait_seconds=_metric(b0, "provider_pacing_wait_seconds"),
        o1_pacing_wait_seconds=_metric(o1, "provider_pacing_wait_seconds"),
        o1_oracle_coverage=None if o1 is None else oracle_coverage,
        observable_trajectory_difference=trajectory_difference,
        trajectory_comparison=comparison,
        anomaly="; ".join(anomalies) if anomalies else None,
    )


def analyze_gate_b1(
    manifest: GateB1Manifest | Path,
    evidence_root: Path,
) -> GateB1Analysis:
    """Analyze only attempts indexed under the supplied Gate B1 manifest."""
    frozen = load_gate_b1_manifest(manifest) if isinstance(manifest, Path) else manifest
    store = GateB1EvidenceStore(evidence_root, frozen)
    records = store.attempts()
    pairs = store.pair_statuses()
    rows = tuple(
        _row(frozen, task_id, records, pair)
        for task_id, pair in zip(frozen.task_ids, pairs, strict=True)
    )
    complete = sum(row.pair_complete for row in rows)
    invalid_provider = sum(
        record.status is AttemptStatus.INVALID_PROVIDER_ATTEMPT for record in records
    )
    invalid_infrastructure = sum(
        record.status is AttemptStatus.INVALID_INFRASTRUCTURE_ATTEMPT for record in records
    )
    b0_successes = sum(row.b0_success is True for row in rows)
    o1_successes = sum(row.o1_success is True for row in rows)
    b0_rfr_values = [row.b0_repeated_failure for row in rows if row.b0_repeated_failure is not None]
    o1_rfr_values = [row.o1_repeated_failure for row in rows if row.o1_repeated_failure is not None]
    different = tuple(row.task_id for row in rows if row.observable_trajectory_difference is True)
    identical = tuple(
        row.task_id
        for row in rows
        if row.observable_trajectory_difference is False
        or row.trajectory_comparison == "identical"
    )
    anomalies = [row.task_id + ": " + row.anomaly for row in rows if row.anomaly]
    anomalies.extend(
        record.task_id + ": " + record.error_message
        for record in records
        if not record.valid_observation and record.error_message
    )
    complete_gate = complete == len(frozen.task_ids)
    valid_evidence_complete = complete_gate and all(row.anomaly is None for row in rows)
    return GateB1Analysis(
        manifest_id=frozen.manifest_id,
        experiment_id=frozen.experiment_id,
        expected_tasks=len(frozen.task_ids),
        complete_valid_pairs=complete,
        incomplete_pairs=len(frozen.task_ids) - complete,
        invalid_provider_attempts=invalid_provider,
        invalid_infrastructure_attempts=invalid_infrastructure,
        b0_successes=b0_successes,
        o1_successes=o1_successes,
        b0_bounded_terminations=sum(row.b0_bounded_termination is True for row in rows),
        o1_bounded_terminations=sum(row.o1_bounded_termination is True for row in rows),
        b0_rfr_count=(
            sum(b0_rfr_values)
            if complete and len(b0_rfr_values) == len(frozen.task_ids)
            else None
        ),
        o1_rfr_count=(
            sum(o1_rfr_values)
            if complete and len(o1_rfr_values) == len(frozen.task_ids)
            else None
        ),
        o1_oracle_coverage=sum(row.o1_oracle_coverage is True for row in rows),
        tasks_with_observable_trajectory_difference=different,
        tasks_with_identical_or_no_observable_difference=identical,
        missing_evidence_or_anomalies=tuple(anomalies),
        gate_b1_evaluated=False,
        gate_b1_ready_for_review=valid_evidence_complete,
        task_rows=rows,
    )


__all__ = [
    "GateB1Analysis",
    "GateB1TaskAnalysis",
    "analyze_gate_b1",
    "normalized_trajectory_signature",
]
