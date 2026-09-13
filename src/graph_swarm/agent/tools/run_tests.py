"""Controlled repository test-suite execution tool."""

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.agent.tools.run_command import execute_process
from graph_swarm.domain.action import ActionResult


def run_tests(
    dependencies: AgentDependencies,
    timeout_seconds: float | int = 120,
    *,
    action_id: str | None = None,
) -> ActionResult:
    """Run pytest with the current interpreter inside the configured workspace."""
    executable = dependencies.python_executable or _current_python()
    result = execute_process(
        dependencies,
        [str(executable), "-m", "pytest"],
        timeout_seconds,
        "run_tests",
        action_id=action_id,
    )
    emit_action_event(dependencies, result)
    return result


def _current_python() -> str:
    import sys

    return sys.executable
