"""Deterministic applicability checks for historical recoveries."""

from pydantic import BaseModel, Field

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.recovery_patterns import RecoveryPattern
from graph_swarm.retrieval.candidates import HistoricalRecoveryCandidate


class ApplicabilityDecision(BaseModel):
    """Explainable result of one candidate applicability evaluation."""

    applicable: bool
    matched_fields: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    compatible_versions: dict[str, str] = Field(default_factory=dict)
    compatible_markers: dict[str, str] = Field(default_factory=dict)


class ApplicabilityService:
    """Apply exact structural and known-environment compatibility rules."""

    def evaluate(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        candidate: HistoricalRecoveryCandidate,
    ) -> ApplicabilityDecision:
        matched: list[str] = []
        rejected: list[str] = []

        if not planned_action.tool.strip() or not candidate.failed_action.tool.strip():
            rejected.append("tool applicability data is missing")
        elif planned_action.tool != candidate.failed_action.tool:
            rejected.append("tool does not match")
        else:
            matched.append("tool")

        if (
            not planned_action.operation.strip()
            or not candidate.failed_action.operation.strip()
        ):
            rejected.append("operation applicability data is missing")
        elif planned_action.operation != candidate.failed_action.operation:
            rejected.append("operation does not match")
        else:
            matched.append("operation")

        if not environment.runtime.strip() or not candidate.environment.runtime.strip():
            rejected.append("runtime applicability data is missing")
        elif environment.runtime != candidate.environment.runtime:
            rejected.append("runtime does not match")
        else:
            matched.append("runtime")

        compatible_versions, version_rejections = _compatible_values(
            environment.versions,
            candidate.environment.versions,
            "version",
        )
        rejected.extend(version_rejections)
        if not version_rejections and compatible_versions:
            matched.append("versions")

        compatible_markers, marker_rejections = _compatible_values(
            environment.markers,
            candidate.environment.markers,
            "marker",
        )
        rejected.extend(marker_rejections)
        if not marker_rejections and compatible_markers:
            matched.append("markers")

        return ApplicabilityDecision(
            applicable=not rejected,
            matched_fields=tuple(matched),
            rejection_reasons=tuple(rejected),
            compatible_versions=compatible_versions,
            compatible_markers=compatible_markers,
        )

    def is_applicable(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        candidate: HistoricalRecoveryCandidate,
    ) -> bool:
        """Return only the boolean decision for callers that need no detail."""
        return self.evaluate(planned_action, environment, candidate).applicable


class RecoveryPatternApplicabilityService:
    """Apply v1 structural and comparable-environment rules to patterns."""

    def evaluate(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        pattern: RecoveryPattern,
    ) -> ApplicabilityDecision:
        matched: list[str] = []
        rejected: list[str] = []

        effective_tool = (
            pattern.applicability_tool
            if pattern.applicability_tool is not None
            else pattern.source_tool
        )
        effective_operation = (
            pattern.applicability_operation
            if pattern.applicability_operation is not None
            else pattern.source_operation
        )

        if _normalize_action_value(effective_tool) != _normalize_action_value(
            planned_action.tool
        ):
            rejected.append("tool_mismatch")
        else:
            matched.append("tool")

        if _normalize_action_value(effective_operation) != _normalize_action_value(
            planned_action.operation
        ):
            rejected.append("operation_mismatch")
        else:
            matched.append("operation")

        runtime_constraint = pattern.environment_constraints.runtime
        if runtime_constraint and environment.runtime.strip():
            if _normalize_fact(runtime_constraint) != _normalize_fact(environment.runtime):
                rejected.append("runtime_mismatch")
            else:
                matched.append("runtime")

        compatible_versions, version_rejections = _compatible_values(
            environment.versions,
            pattern.environment_constraints.versions,
            "version",
        )
        rejected.extend(version_rejections)
        if not version_rejections and compatible_versions:
            matched.append("versions")

        compatible_markers, marker_rejections = _compatible_values(
            environment.markers,
            pattern.environment_constraints.markers,
            "marker",
        )
        rejected.extend(marker_rejections)
        if not marker_rejections and compatible_markers:
            matched.append("markers")

        # EnvironmentContext has no dependency mapping. Missing dependency facts
        # are intentionally not synthesized or treated as a mismatch.
        return ApplicabilityDecision(
            applicable=not rejected,
            matched_fields=tuple(matched),
            rejection_reasons=tuple(rejected),
            compatible_versions=compatible_versions,
            compatible_markers=compatible_markers,
        )


def _compatible_values(
    current: dict[str, str], historical: dict[str, str], label: str
) -> tuple[dict[str, str], list[str]]:
    compatible: dict[str, str] = {}
    rejected: list[str] = []
    for key in sorted(current.keys() & historical.keys()):
        current_value = current[key]
        historical_value = historical[key]
        if not key.strip() or not current_value.strip() or not historical_value.strip():
            rejected.append(f"{label}_mismatch:{key}")
        elif _normalize_fact(current_value) != _normalize_fact(historical_value):
            rejected.append(f"{label}_mismatch:{key}")
        else:
            compatible[key] = current_value
    return compatible, rejected


def _normalize_action_value(value: str) -> str:
    """Normalize only action whitespace; no undocumented tool equivalence."""
    return " ".join(value.split())


def _normalize_fact(value: str) -> str:
    """Normalize comparable environment facts deterministically for equality."""
    return " ".join(value.split()).casefold()
