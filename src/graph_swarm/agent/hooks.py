"""Agent/tool lifecycle hooks used to capture completed action events."""

from datetime import UTC, datetime
from uuid import uuid4

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.events import AgentEvent, AgentEventType


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
