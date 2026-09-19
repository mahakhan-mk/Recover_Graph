import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from graph_swarm.agent.advisory import prepare_tool_action
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.read_file import read_file
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph._validation import validate_action_persistence
from graph_swarm.graph.read_models import (
    ActionLineageRecord,
    RecoveryEvidenceLineage,
    RecoveryEvidenceTask,
)
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.integration.event_persistence import persist_agent_event_stream

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "fixtures"
    / "repositories"
    / "rollout1_agent_smoke"
)


class InMemoryLineageRepository:
    """Small graph-shaped fake used to validate the complete persistence handoff."""

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.runs: dict[str, Run] = {}
        self.actions: dict[str, PlannedAction] = {}
        self.results: dict[str, ActionResult] = {}
        self.tools: dict[str, Tool] = {}
        self.environments: dict[str, EnvironmentContext] = {}
        self.failures: dict[str, FailureEpisode] = {}
        self.resolutions: dict[str, Resolution] = {}
        self.outcomes: dict[str, Outcome] = {}
        self.edges: set[tuple[str, str, str]] = set()

    def save_task(self, task: Task) -> None:
        self.tasks[task.id] = task

    def save_run(self, run: Run) -> None:
        self.runs[run.id] = run

    def save_action(self, action: PlannedAction, result: ActionResult) -> None:
        validate_action_persistence(action, result)
        self.actions[action.id] = action
        self.results[action.id] = result

    def save_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def save_environment(self, environment: EnvironmentContext) -> None:
        self.environments[environment.id] = environment

    def save_failure(self, failure: FailureEpisode) -> None:
        self.failures[failure.id] = failure

    def save_resolution(self, resolution: Resolution) -> None:
        self.resolutions[resolution.id] = resolution

    def save_outcome(self, outcome: Outcome) -> None:
        self.outcomes[outcome.id] = outcome

    def link_task_action(self, task_id: str, action_id: str) -> None:
        self._link("Task", task_id, "HAS_ACTION", "Action", action_id)

    def link_action_tool(self, action_id: str, tool_name: str) -> None:
        self._link("Action", action_id, "USED", "Tool", tool_name)

    def link_action_run(self, action_id: str, run_id: str) -> None:
        self._link("Action", action_id, "PART_OF", "Run", run_id)

    def link_action_failure(self, action_id: str, failure_id: str) -> None:
        self._link("Action", action_id, "PART_OF_FAILURE", "FailureEpisode", failure_id)

    def link_failure_environment(self, failure_id: str, environment_id: str) -> None:
        self._link("FailureEpisode", failure_id, "OCCURRED_IN", "Environment", environment_id)

    def link_failure_resolution(self, failure_id: str, resolution_id: str) -> None:
        self._link("FailureEpisode", failure_id, "RESOLVED_BY", "Resolution", resolution_id)

    def link_resolution_outcome(self, resolution_id: str, outcome_id: str) -> None:
        self._link("Resolution", resolution_id, "VERIFIED_BY", "Outcome", outcome_id)

    def link_resolution_observed_change(self, resolution_id: str, action_id: str) -> None:
        self._link("Resolution", resolution_id, "OBSERVED_CHANGE", "Action", action_id)

    def get_recovery_evidence(self, failure_id: str) -> RecoveryEvidenceLineage:
        failure = self.failures[failure_id]
        resolution_id = self._target("FailureEpisode", failure_id, "RESOLVED_BY")
        resolution = self.resolutions[resolution_id]
        outcome_id = self._target("Resolution", resolution_id, "VERIFIED_BY")
        recovery_id = self._target("Resolution", resolution_id, "OBSERVED_CHANGE")
        failed_action = self.actions[failure.action_id]
        recovery_action = self.actions[recovery_id]
        task = self.tasks[next(iter(self._sources("Task", "HAS_ACTION", failure.action_id)))]
        environment_id = self._target("FailureEpisode", failure_id, "OCCURRED_IN")
        environment = self.environments[environment_id]
        return RecoveryEvidenceLineage(
            failure=failure,
            resolution=resolution,
            outcome=self.outcomes[outcome_id],
            task=RecoveryEvidenceTask(
                id=task.id,
                problem_statement=task.problem_statement,
                repository=task.repository,
                chronological_index=task.chronological_index,
            ),
            environment=environment,
            failed_action=ActionLineageRecord(
                planned_action=failed_action,
                result=self.results[failed_action.id],
            ),
            recovery_action=ActionLineageRecord(
                planned_action=recovery_action,
                result=self.results[recovery_action.id],
            ),
        )

    def _link(
        self,
        source_type: str,
        source_id: str,
        relation: str,
        target_type: str,
        target_id: str,
    ) -> None:
        self.edges.add((f"{source_type}:{source_id}", relation, f"{target_type}:{target_id}"))

    def _target(self, source_type: str, source_id: str, relation: str) -> str:
        prefix = f"{source_type}:{source_id}"
        matches = [
            target.split(":", 1)[1]
            for source, edge_relation, target in self.edges
            if source == prefix and edge_relation == relation
        ]
        assert len(matches) == 1
        return matches[0]

    def _sources(self, source_type: str, relation: str, target_id: str) -> list[str]:
        suffix = f":{target_id}"
        return [
            source.split(":", 1)[1]
            for source, edge_relation, target in self.edges
            if source.startswith(f"{source_type}:")
            and edge_relation == relation
            and target.endswith(suffix)
        ]


def test_runtime_planned_actions_create_complete_trusted_recovery_lineage(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    shutil.copytree(FIXTURE_PATH, workspace)
    task = Task(
        id="task-runtime-recovery",
        problem_statement="Repair the calculator.",
        family_id="family-runtime-recovery",
        repository="example/repository",
        chronological_index=3,
    )
    run = Run(
        id="run-runtime-recovery",
        task_id=task.id,
        started_at=datetime.now(UTC),
    )
    environment = EnvironmentContext(
        id="environment-runtime-recovery",
        repository=task.repository,
        runtime="python",
    )
    dependencies = AgentDependencies(workspace, run.id, task.id)

    failed_action = prepare_tool_action(dependencies, "run_tests", "run_tests", {})
    run_tests(dependencies, timeout_seconds=30, action_id=failed_action.id)
    read_action = prepare_tool_action(
        dependencies,
        "read_file",
        "read_file",
        {"path": "calculator.py", "offset": 0, "length": 200},
    )
    read_result = read_file(dependencies, "calculator.py", action_id=read_action.id)
    assert read_result.output is not None
    write_action = prepare_tool_action(
        dependencies,
        "write_file",
        "write_file",
        {"path": "calculator.py", "content": read_result.output.replace("a - b", "a + b")},
    )
    write_file(
        dependencies,
        "calculator.py",
        read_result.output.replace("a - b", "a + b"),
        action_id=write_action.id,
    )
    successful_action = prepare_tool_action(dependencies, "run_tests", "run_tests", {})
    run_tests(dependencies, timeout_seconds=30, action_id=successful_action.id)

    repository = InMemoryLineageRepository()
    persisted = persist_agent_event_stream(
        cast(OperationalMemoryRepository, repository),
        dependencies.events,
        task,
        run,
        environment,
        planned_actions=dependencies.planned_actions,
    )

    assert persisted is not None
    failure, resolution, outcome = persisted
    lineage = repository.get_recovery_evidence(failure.id)
    assert lineage.resolution.id == resolution.id
    assert lineage.outcome.id == outcome.id
    assert lineage.recovery_action.planned_action == write_action
    assert lineage.recovery_action.planned_action.arguments["content"]
    assert lineage.recovery_action.planned_action.arguments["content"] == write_action.arguments[
        "content"
    ]
    required_edges = {
        ("Task", "HAS_ACTION", "Action"),
        ("Action", "PART_OF", "Run"),
        ("Action", "PART_OF_FAILURE", "FailureEpisode"),
        ("FailureEpisode", "OCCURRED_IN", "Environment"),
        ("FailureEpisode", "RESOLVED_BY", "Resolution"),
        ("Resolution", "OBSERVED_CHANGE", "Action"),
        ("Resolution", "VERIFIED_BY", "Outcome"),
    }
    assert {
        (source.split(":", 1)[0], relation, target.split(":", 1)[0])
        for source, relation, target in repository.edges
    } >= required_edges
