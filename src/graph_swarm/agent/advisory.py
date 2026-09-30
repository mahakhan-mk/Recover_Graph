"""The single agent-side boundary for pre-tool advisory evaluation."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from pydantic_ai import ModelRetry

from graph_swarm.advisory.formatting import format_advice
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.hooks import emit_advice_event
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.domain.behavior import BehaviorChangeEvidence
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.tasks import Task


class TreatmentAdvisoryInfrastructureError(RuntimeError):
    """Raised when a fail-closed treatment lookup cannot complete."""


def prepare_tool_action(
    dependencies: AgentDependencies,
    tool: str,
    operation: str,
    arguments: dict[str, object],
) -> PlannedAction:
    """Construct an action and issue advice before the controlled tool runs."""
    action = PlannedAction(
        id=str(uuid4()),
        run_id=dependencies.run_id,
        task_id=dependencies.task_id,
        tool=tool,
        operation=operation,
        arguments=arguments,
        planned_at=datetime.now(UTC),
    )
    # This is the trust boundary: retain the exact object before advisory
    # evaluation can retry, fail open, or otherwise alter control flow.
    dependencies.record_planned_action(action)
    if dependencies.before_tool_action is not None:
        dependencies.before_tool_action(action)

    service = dependencies.advisory_service
    task = dependencies.task
    environment = dependencies.environment
    if service is None or task is None or environment is None:
        return action

    # The frozen O1 adapter is deliberately a one-shot intervention.  Its
    # ModelRetry is the reconsideration boundary; later actions are evidence,
    # not additional opportunities to inject the same advice.
    if getattr(service, "one_shot", False) and dependencies.oracle_advice_issued:
        return action

    try:
        advice = service.evaluate_action(task, action, environment)
    except Exception as error:  # noqa: BLE001 - advisory failure must not mutate execution
        dependencies.advisory_errors.append(f"lookup failed for {tool}: {error}")
        if dependencies.fail_closed_advisory:
            raise TreatmentAdvisoryInfrastructureError(
                f"treatment advisory lookup failed for {tool}: {error}"
            ) from error
        return action

    if not advice.has_advice:
        return action

    try:
        oracle_renderer = getattr(service, "render_advice", None)
        rendered = (
            cast(str, oracle_renderer(advice))
            if callable(oracle_renderer)
            else format_advice(advice)
        )
        if not rendered:
            raise ValueError("applicable advice rendered as empty text")
        key = _recovery_key(action, advice)
        if dependencies.has_advised_action(key):
            return action
        event = emit_advice_event(dependencies, action, advice, rendered)
        if dependencies.capture_advisory_retrieval:
            pattern_id, vector_score, retrieval = _retrieval_snapshot(service)
            dependencies.record_advisory_retrieval_evidence(
                event,
                pattern_id=pattern_id,
                vector_score=vector_score,
                retrieval=retrieval,
            )
    except Exception as error:  # noqa: BLE001 - preserve normal tool execution
        dependencies.advisory_errors.append(f"formatting failed for {tool}: {error}")
        if dependencies.fail_closed_advisory:
            raise TreatmentAdvisoryInfrastructureError(
                f"treatment advisory rendering failed for {tool}: {error}"
            ) from error
        return action

    dependencies.remember_advised_action(key)
    if getattr(service, "one_shot", False):
        dependencies.oracle_advice_issued = True
    dependencies.queue_advice_event(event)
    raise ModelRetry(rendered)


def prepare_task_start_guidance(dependencies: AgentDependencies) -> str | None:
    """Resolve and record one guidance intervention before the first model request."""
    if dependencies.task_start_guidance_evaluated:
        return None
    dependencies.task_start_guidance_evaluated = True

    service = dependencies.advisory_service
    task = dependencies.task
    environment = dependencies.environment
    evaluator = cast(
        Callable[[Task, EnvironmentContext, PlannedAction], AdviceResult] | None,
        getattr(service, "evaluate_task_start", None),
    )
    if service is None or task is None or environment is None or not callable(evaluator):
        return None
    if dependencies.oracle_advice_issued:
        return None

    action = PlannedAction(
        id=str(uuid4()),
        run_id=dependencies.run_id,
        task_id=dependencies.task_id,
        tool="task_start",
        operation="task_start",
        arguments={},
        planned_at=datetime.now(UTC),
    )
    try:
        advice = evaluator(task, environment, action)
        if not advice.has_advice:
            return None
        renderer = getattr(service, "render_advice", None)
        rendered = cast(str, renderer(advice)) if callable(renderer) else format_advice(advice)
        if not rendered:
            raise ValueError("applicable task-start advice rendered as empty text")
        emit_advice_event(dependencies, action, advice, rendered)
    except Exception as error:  # noqa: BLE001 - advisory failure must not mutate execution
        dependencies.advisory_errors.append(f"task-start guidance failed: {error}")
        return None

    dependencies.oracle_advice_issued = True
    return rendered


def record_post_advice_action(
    dependencies: AgentDependencies,
    actual_action: PlannedAction,
    actual_event: AgentEvent | None = None,
) -> None:
    """Record direct before/after structural evidence after tool execution."""
    observed_at = datetime.now(UTC)
    pending = dependencies.take_pending_advice_events()
    for advice_event in pending:
        changed = _action_structure(advice_event.planned_action) != _action_structure(actual_action)
        dependencies.record_behavior_evidence(
            BehaviorChangeEvidence(
                advice_event_id=advice_event.event_id,
                planned_action_before_advice=advice_event.planned_action,
                actual_action_after_advice=actual_action,
                behavior_changed=changed,
                observation="changed" if changed else "unchanged",
                observed_at=observed_at,
            ),
            subsequent_event=actual_event,
        )


def finalize_pending_advice(dependencies: AgentDependencies) -> None:
    """Represent advice with no subsequent tool action without guessing acceptance."""
    observed_at = datetime.now(UTC)
    pending = dependencies.take_pending_advice_events()
    for advice_event in pending:
        dependencies.record_behavior_evidence(
            BehaviorChangeEvidence(
                advice_event_id=advice_event.event_id,
                planned_action_before_advice=advice_event.planned_action,
                observation="no_subsequent_action",
                observed_at=observed_at,
            )
        )


def _action_key(action: PlannedAction) -> str:
    return json.dumps(
        {
            "tool": action.tool,
            "operation": action.operation,
            "arguments": action.arguments,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _recovery_key(action: PlannedAction, advice: AdviceResult) -> str:
    if advice.provenance is None:
        raise ValueError("applicable advice has no provenance")
    provenance = advice.provenance
    return "|".join(
        (
            action.run_id,
            action.task_id,
            provenance.failure_episode_id,
            provenance.resolution_id,
        )
    )


def _action_structure(action: PlannedAction) -> tuple[str, str, str]:
    return action.tool, action.operation, _action_key(action)


def _retrieval_snapshot(
    service: object,
) -> tuple[str | None, float | None, dict[str, object] | None]:
    """Copy the service's current retrieval result before the next lookup."""
    result = getattr(service, "last_retrieval_result", None)
    pattern = getattr(getattr(result, "selected_pattern", None), "id", None)
    pattern_id = pattern if isinstance(pattern, str) and pattern.strip() else None
    score = getattr(result, "selected_vector_score", None)
    vector_score = (
        score if isinstance(score, (int, float)) and not isinstance(score, bool) else None
    )
    dump = getattr(result, "model_dump", None)
    retrieval: dict[str, object] | None = None
    try:
        dumped = dump(mode="json") if callable(dump) else None
        if isinstance(dumped, dict):
            retrieval = cast(dict[str, object], dumped)
    except Exception:  # noqa: BLE001 - retrieval evidence must not alter execution
        pass
    return pattern_id, vector_score, retrieval
