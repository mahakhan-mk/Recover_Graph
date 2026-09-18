"""Typed results returned by the deterministic advisory layer."""

from typing import Literal

from pydantic import BaseModel, Field

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import ResolutionStatus
from graph_swarm.domain.tasks import Task


class EvaluateActionRequest(BaseModel):
    """Inputs retained as a small domain contract for future adapters."""

    task: Task
    action: PlannedAction
    environment: EnvironmentContext


class EvaluateTaskStartRequest(BaseModel):
    """Inputs for guidance delivered at the frozen task-start boundary."""

    task: Task
    environment: EnvironmentContext


class ApplicabilityAssessment(BaseModel):
    """Structured facts that made a historical recovery applicable."""

    matched_fields: tuple[str, ...]
    repository: str
    runtime: str
    compatible_versions: dict[str, str] = Field(default_factory=dict)
    compatible_markers: dict[str, str] = Field(default_factory=dict)


class RecoveryEvidence(BaseModel):
    """Recovery evidence currently persisted by Rollout 1."""

    successful_observations: int
    failed_observations: int
    outcomes: tuple[Outcome, ...] = ()


class RecoveryProvenance(BaseModel):
    """Stable graph identities allowing an advice result to be traced back."""

    failure_episode_id: str
    resolution_id: str
    failed_action_id: str
    environment_id: str
    outcome_ids: tuple[str, ...] = ()


class NoApplicableAdvice(BaseModel):
    """The explicit no-advice branch of :class:`AdviceResult`."""

    kind: Literal["no_applicable_advice"] = "no_applicable_advice"
    reason: str
    considered_candidates: int = 0


class HistoricalRecoveryAdvice(BaseModel):
    """An applicable, persisted historical recovery recommendation."""

    kind: Literal["historical_recovery_found"] = "historical_recovery_found"
    matched_failure_episode_id: str
    matched_resolution_id: str
    failed_tool: str
    failed_operation: str
    recovery_summary: str
    resolution_status: ResolutionStatus
    recovery_evidence: RecoveryEvidence
    applicability: ApplicabilityAssessment
    provenance: RecoveryProvenance


class AdviceResult(BaseModel):
    """A discriminated result with explicit no-advice and advice outcomes.

    The variant is held in ``advice`` so neither branch needs placeholder null
    fields. Convenience properties expose the applicable branch without making
    the no-advice branch ambiguous.
    """

    advice: NoApplicableAdvice | HistoricalRecoveryAdvice = Field(
        discriminator="kind"
    )

    @classmethod
    def no_advice(cls, reason: str, considered_candidates: int = 0) -> "AdviceResult":
        return cls(
            advice=NoApplicableAdvice(
                reason=reason,
                considered_candidates=considered_candidates,
            )
        )

    @classmethod
    def historical_recovery(
        cls,
        *,
        matched_failure_episode_id: str,
        matched_resolution_id: str,
        failed_tool: str,
        failed_operation: str,
        recovery_summary: str,
        resolution_status: ResolutionStatus,
        recovery_evidence: RecoveryEvidence,
        applicability: ApplicabilityAssessment,
        provenance: RecoveryProvenance,
    ) -> "AdviceResult":
        return cls(
            advice=HistoricalRecoveryAdvice(
                matched_failure_episode_id=matched_failure_episode_id,
                matched_resolution_id=matched_resolution_id,
                failed_tool=failed_tool,
                failed_operation=failed_operation,
                recovery_summary=recovery_summary,
                resolution_status=resolution_status,
                recovery_evidence=recovery_evidence,
                applicability=applicability,
                provenance=provenance,
            )
        )

    @property
    def has_advice(self) -> bool:
        return isinstance(self.advice, HistoricalRecoveryAdvice)

    @property
    def kind(self) -> str:
        return self.advice.kind

    @property
    def matched_failure_episode_id(self) -> str | None:
        if isinstance(self.advice, HistoricalRecoveryAdvice):
            return self.advice.matched_failure_episode_id
        return None

    @property
    def matched_resolution_id(self) -> str | None:
        if isinstance(self.advice, HistoricalRecoveryAdvice):
            return self.advice.matched_resolution_id
        return None

    @property
    def recovery_summary(self) -> str | None:
        if isinstance(self.advice, HistoricalRecoveryAdvice):
            return self.advice.recovery_summary
        return None

    @property
    def resolution_status(self) -> ResolutionStatus | None:
        if isinstance(self.advice, HistoricalRecoveryAdvice):
            return self.advice.resolution_status
        return None

    @property
    def recovery_evidence(self) -> RecoveryEvidence | None:
        if isinstance(self.advice, HistoricalRecoveryAdvice):
            return self.advice.recovery_evidence
        return None

    @property
    def applicability(self) -> ApplicabilityAssessment | None:
        if isinstance(self.advice, HistoricalRecoveryAdvice):
            return self.advice.applicability
        return None

    @property
    def provenance(self) -> RecoveryProvenance | None:
        if isinstance(self.advice, HistoricalRecoveryAdvice):
            return self.advice.provenance
        return None
