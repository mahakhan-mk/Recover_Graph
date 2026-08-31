"""Agent/tool lifecycle hooks used to capture completed action events."""

from datetime import UTC, datetime
from uuid import uuid4

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult, HistoricalRecoveryAdvice
from graph_swarm.domain.events import AdviceEvent, AgentEvent, AgentEventType


def emit_action_event(
    dependencies: AgentDependencies,
    result: ActionResult,
) -> AgentEvent:
    """Record and return one completed-action event for a canonical result."""
    event = AgentEvent(
        event_id=str(uuid4()),
        run_id=dependencies.run_id,
        task_id=dependencies.task_id,
        action_id=result.action_id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=datetime.now(UTC),
    )
    dependencies.events.append(event)
    return event


def emit_advice_event(
    dependencies: AgentDependencies,
    planned_action: PlannedAction,
    advice: AdviceResult,
    rendered_advice: str,
) -> AdviceEvent:
    """Record one applicable advice event with stable graph provenance."""
    if not isinstance(advice.advice, HistoricalRecoveryAdvice):
        raise ValueError("only applicable advice can be emitted as an advice event")
    event = AdviceEvent(
        event_id=str(uuid4()),
        run_id=dependencies.run_id,
        task_id=dependencies.task_id,
        planned_action=planned_action,
        advice=advice.advice,
        rendered_advice=rendered_advice,
        issued_at=datetime.now(UTC),
    )
    dependencies.record_advice_event(event)
    return event
