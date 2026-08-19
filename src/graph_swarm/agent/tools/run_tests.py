"""Controlled repository test-suite execution tool."""

import sys

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.agent.tools.run_command import execute_process
from graph_swarm.domain.action import ActionResult


def run_tests(
    dependencies: AgentDependencies,
    timeout_seconds: float | int = 120,
) -> ActionResult:
    """Run pytest with the current interpreter inside the configured workspace."""
    result = execute_process(
        dependencies,
        [sys.executable, "-m", "pytest"],
        timeout_seconds,
        "run_tests",
    )
    emit_action_event(dependencies, result)
    return result
