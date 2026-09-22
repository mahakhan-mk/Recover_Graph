"""Deterministic detection of failed trusted test and command executions."""

from uuid import NAMESPACE_URL, uuid5

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.memory.recovery_evidence import is_test_execution


class FailureDetector:
    """Classify canonical action events using explicit Rollout 1 rules."""

    def detect(
        self,
        event: AgentEvent,
        planned_action: PlannedAction | None = None,
    ) -> FailureEpisode | None:
        """Classify one failed trusted test or command execution."""
        result = event.result
        failure_type = failure_type_for_action(result, planned_action)
        if failure_type is None or result.exit_code is None:
            return None

        evidence = _failure_evidence(result.error, result.output, result.exit_code)
        return FailureEpisode(
            id=str(uuid5(NAMESPACE_URL, event.event_id)),
            action_id=event.action_id,
            failure_type=failure_type,
            signature=f"{result.tool_name}:exit_code={result.exit_code}",
            symptom=evidence,
            observed_at=event.occurred_at,
        )


def detect_failure(
    event: AgentEvent,
    planned_action: PlannedAction | None = None,
) -> FailureEpisode | None:
    """Detect a deterministic failure episode from one canonical agent event."""
    return FailureDetector().detect(event, planned_action)


def failure_type_for_action(
    result: ActionResult,
    planned_action: PlannedAction | None = None,
) -> FailureType | None:
    """Return the canonical failure type for one observed action result."""
    if result.success or result.exit_code in (None, 0):
        return None
    if result.tool_name == "run_tests":
        return FailureType.TEST_FAILURE
    if result.tool_name == "run_command":
        if planned_action is not None and planned_action.tool == result.tool_name:
            if is_test_execution(planned_action):
                return FailureType.TEST_FAILURE
        return FailureType.COMMAND_FAILURE
    return None


def _failure_evidence(error: str | None, output: str | None, exit_code: int) -> str:
    """Choose and normalize stable execution evidence for failure fields."""
    text = error or output or f"exit code {exit_code}"
    return " ".join(text.split())
