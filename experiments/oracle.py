"""Frozen Oracle recovery guidance for Track B O1 experiments.

This adapter reads only the manually validated recovery pattern and the
chronological transfer mapping.  It has no Graph Swarm, Neo4j, or retrieval
dependencies and is intentionally not part of the agent's persistent memory.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import (
    AdviceResult,
    ApplicabilityAssessment,
    RecoveryEvidence,
    RecoveryProvenance,
)
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.resolutions import ResolutionStatus
from graph_swarm.domain.tasks import Task
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    OracleEvidenceRequired,
)


@dataclass(frozen=True)
class OracleTransferEvidence:
    """Validated guidance and source provenance for one frozen transfer task."""

    task_id: str
    recovery_pattern: str
    review_id: str | None = None


class FrozenOracleResolver:
    """Resolve O1 guidance from validated transfer evidence only."""

    one_shot = True
    _TRANSFER_ACTION = ("run_tests", "run_tests")

    def __init__(self, transfers: Mapping[str, str | OracleTransferEvidence]) -> None:
        normalized: dict[str, OracleTransferEvidence] = {}
        for task_id, evidence in transfers.items():
            pattern = (
                evidence.recovery_pattern
                if isinstance(evidence, OracleTransferEvidence)
                else evidence
            )
            if not task_id.strip() or not pattern.strip():
                raise OracleEvidenceRequired(
                    "Oracle transfer evidence requires a task ID and recovery pattern"
                )
            review_id = evidence.review_id if isinstance(evidence, OracleTransferEvidence) else None
            if review_id is not None and not review_id.strip():
                raise OracleEvidenceRequired("Oracle transfer evidence has an empty review ID")
            normalized[task_id] = OracleTransferEvidence(
                task_id,
                pattern.strip(),
                review_id.strip() if review_id is not None else None,
            )
        self._transfers = normalized

    @classmethod
    def from_frozen_files(
        cls,
        manifest_path: Path,
        annotations_path: Path,
    ) -> FrozenOracleResolver:
        """Load validated recovery patterns and source review IDs for transfers."""
        try:
            manifest_records: list[dict[str, Any]] = []
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record: Any = json.loads(line)
                if not isinstance(record, dict):
                    raise OracleEvidenceRequired(
                        "frozen Oracle manifest contains a non-object record"
                    )
                manifest_records.append(cast(dict[str, Any], record))
            with annotations_path.open(encoding="utf-8", newline="") as handle:
                annotations: dict[str, dict[str, str]] = {}
                for row in csv.DictReader(handle):
                    review_id = row.get("review_id") or ""
                    annotations[review_id] = {key: value or "" for key, value in row.items()}
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as error:
            raise OracleEvidenceRequired(
                f"could not load frozen Oracle evidence: {error}"
            ) from error

        transfers: dict[str, OracleTransferEvidence] = {}
        for record in manifest_records:
            occurrence_index = record.get("occurrence_index")
            if not isinstance(occurrence_index, int) or isinstance(occurrence_index, bool):
                raise OracleEvidenceRequired(
                    "frozen Oracle manifest transfer mapping has an invalid occurrence index"
                )
            if occurrence_index <= 1:
                continue

            task_id = _required_text(record, "task_id")
            review_id = _required_text(record, "review_id")
            _required_text(record, "family_id")
            if record.get("validated") is not True:
                raise OracleEvidenceRequired(
                    f"transfer task {task_id} is not marked validated in the frozen manifest"
                )
            row = annotations.get(review_id)
            if (
                row is None
                or row.get("keep", "").strip().lower() != "yes"
                or row.get("transferable", "").strip().lower() != "yes"
            ):
                raise OracleEvidenceRequired(
                    f"transfer task {task_id} has no validated Oracle transfer mapping"
                )
            recovery_pattern = row.get("recovery_pattern", "")
            if not recovery_pattern.strip():
                raise OracleEvidenceRequired(
                    f"transfer task {task_id} has no validated recovery pattern"
                )
            transfers[task_id] = OracleTransferEvidence(
                task_id=task_id,
                recovery_pattern=recovery_pattern,
                review_id=review_id,
            )

        return cls(transfers)

    def validate_case(self, case: BenchmarkTaskCase) -> None:
        """Reject undeclared or incomplete transfer evidence before the model."""
        if case.occurrence_index <= 1:
            return
        if case.task.id not in self._transfers:
            raise OracleEvidenceRequired(
                f"O1 transfer task {case.task.id} has no validated Oracle evidence"
            )

    def review_id_for(self, task_id: str) -> str | None:
        """Return the frozen source review ID without exposing it to the model."""
        evidence = self._transfers.get(task_id)
        return None if evidence is None else evidence.review_id

    def recovery_pattern_for(self, task_id: str) -> str | None:
        """Return the frozen recovery pattern for evaluation-side comparison."""
        evidence = self._transfers.get(task_id)
        return None if evidence is None else evidence.recovery_pattern

    def evaluate_action(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
    ) -> AdviceResult:
        evidence = self._transfers.get(task.id)
        if evidence is None:
            return AdviceResult.no_advice("task is not a frozen Oracle transfer opportunity")
        if (planned_action.tool, planned_action.operation) != self._TRANSFER_ACTION:
            return AdviceResult.no_advice(
                "frozen Oracle guidance applies only before the transfer test action"
            )

        return AdviceResult.historical_recovery(
            matched_failure_episode_id=f"oracle-failure-{task.id}",
            matched_resolution_id=f"oracle-resolution-{task.id}",
            failed_tool=planned_action.tool,
            failed_operation=planned_action.operation,
            recovery_summary=evidence.recovery_pattern,
            resolution_status=ResolutionStatus.OBSERVED_SUCCESSFUL,
            recovery_evidence=RecoveryEvidence(
                successful_observations=1,
                failed_observations=0,
            ),
            applicability=ApplicabilityAssessment(
                matched_fields=("frozen_transfer_mapping",),
                repository=environment.repository,
                runtime=environment.runtime,
            ),
            provenance=RecoveryProvenance(
                failure_episode_id=f"oracle-failure-{task.id}",
                resolution_id=f"oracle-resolution-{task.id}",
                failed_action_id=planned_action.id,
                environment_id=environment.id,
            ),
        )

    def render_advice(self, advice: AdviceResult) -> str:
        """Render only the validated recovery pattern into the agent retry."""
        if not advice.has_advice or not advice.recovery_summary:
            raise OracleEvidenceRequired("Oracle advice has no recovery guidance")
        return f"Recovery guidance: {advice.recovery_summary}"


def _required_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OracleEvidenceRequired(f"frozen Oracle field {field!r} must be non-empty text")
    return value


__all__ = ["FrozenOracleResolver", "OracleTransferEvidence"]
