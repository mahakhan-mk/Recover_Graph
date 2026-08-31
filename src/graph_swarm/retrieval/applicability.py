"""Deterministic applicability checks for historical recoveries."""

from pydantic import BaseModel, Field

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
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

        if not environment.repository.strip() or not candidate.environment.repository.strip():
            rejected.append("repository applicability data is missing")
        elif environment.repository != candidate.environment.repository:
            rejected.append("repository does not match")
        else:
            matched.append("repository")

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


def _compatible_values(
    current: dict[str, str], historical: dict[str, str], label: str
) -> tuple[dict[str, str], list[str]]:
    compatible: dict[str, str] = {}
    rejected: list[str] = []
    for key in sorted(current.keys() & historical.keys()):
        current_value = current[key]
        historical_value = historical[key]
        if not key.strip() or not current_value.strip() or not historical_value.strip():
            rejected.append(f"{label} applicability data for {key!r} is missing")
        elif current_value != historical_value:
            rejected.append(
                f"{label} {key!r} is incompatible ({current_value!r} != {historical_value!r})"
            )
        else:
            compatible[key] = current_value
    return compatible, rejected
