"""Deterministic detection of failed test-suite executions."""

from uuid import uuid4

from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.failures import FailureEpisode, FailureType


class FailureDetector:
    """Classify canonical action events using explicit Rollout 1 rules."""

    def detect(self, event: AgentEvent) -> FailureEpisode | None:
        """Return one test failure for a non-zero failed ``run_tests`` action."""
        result = event.result
        if (
            result.tool_name != "run_tests"
            or result.success
            or result.exit_code is None
            or result.exit_code == 0
        ):
            return None

        evidence = _failure_evidence(result.error, result.output, result.exit_code)
        return FailureEpisode(
            id=str(uuid4()),
            action_id=event.action_id,
            failure_type=FailureType.TEST_FAILURE,
            signature=f"{result.tool_name}:exit_code={result.exit_code}",
            symptom=evidence,
            observed_at=event.occurred_at,
        )


def detect_failure(event: AgentEvent) -> FailureEpisode | None:
    """Detect a deterministic failure episode from one canonical agent event."""
    return FailureDetector().detect(event)


def _failure_evidence(error: str | None, output: str | None, exit_code: int) -> str:
    """Choose and normalize stable execution evidence for failure fields."""
    text = error or output or f"exit code {exit_code}"
    return " ".join(text.split())
