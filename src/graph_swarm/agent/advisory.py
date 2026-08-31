"""The single agent-side boundary for pre-tool advisory evaluation."""

import json
from datetime import UTC, datetime
from uuid import uuid4

from pydantic_ai import ModelRetry

from graph_swarm.advisory.formatting import format_advice
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.hooks import emit_advice_event
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.domain.behavior import BehaviorChangeEvidence
from graph_swarm.domain.events import AgentEvent


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

    service = dependencies.advisory_service
    task = dependencies.task
    environment = dependencies.environment
    if service is None or task is None or environment is None:
        return action

    try:
        advice = service.evaluate_action(task, action, environment)
    except Exception as error:  # noqa: BLE001 - advisory failure must not mutate execution
        dependencies.advisory_errors.append(f"lookup failed for {tool}: {error}")
        return action

    if not advice.has_advice:
        return action

    try:
        rendered = format_advice(advice)
        if not rendered:
            raise ValueError("applicable advice rendered as empty text")
        key = _recovery_key(action, advice)
        if dependencies.has_advised_action(key):
            return action
        event = emit_advice_event(dependencies, action, advice, rendered)
    except Exception as error:  # noqa: BLE001 - preserve normal tool execution
        dependencies.advisory_errors.append(f"formatting failed for {tool}: {error}")
        return action

    dependencies.remember_advised_action(key)
    dependencies.queue_advice_event(event)
    raise ModelRetry(rendered)


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
            _action_key(action),
            provenance.failure_episode_id,
            provenance.resolution_id,
        )
    )


def _action_structure(action: PlannedAction) -> tuple[str, str, str]:
    return action.tool, action.operation, _action_key(action)
