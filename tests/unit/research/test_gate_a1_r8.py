# pyright: reportPrivateUsage=false

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from neo4j.exceptions import SessionExpired

from experiments.sprint3 import _remove_eol_only_worktree_noise
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.integration.event_persistence import persist_agent_event_stream
from graph_swarm.research import gate_a1
from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research.gate_a1_r8 import (
    InLoopPersistenceError,
    ObjectiveSatisfied,
    R8ObjectiveController,
    ShortLivedNeo4jRepository,
    attach_r8_objective_controller,
    classify_r8_acquisition,
    r8_acquisition_readiness,
    repository_state_fingerprint,
)
from graph_swarm.research.runner import load_experiment_configuration


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _event(
    dependencies: AgentDependencies,
    tool: str = "run_command",
    action_id: str = "action-1",
    *,
    success: bool = True,
    exit_code: int | None = 0,
) -> AgentEvent:
    action = dependencies.planned_action_for(action_id)
    if action is None:
        action = PlannedAction(
            id=action_id,
            run_id=dependencies.run_id,
            task_id=dependencies.task_id,
            tool=tool,
            operation=tool,
            arguments=(
                {"path": "arrow/locales.py", "content": "fixed"}
                if tool == "write_file"
                else {}
            ),
            planned_at=datetime.now(UTC),
        )
        dependencies.record_planned_action(action)
    result = ActionResult(
        action_id=action.id,
        tool_name=tool,
        success=success,
        exit_code=exit_code,
        output="durable output",
        started_at=action.planned_at,
        completed_at=action.planned_at,
    )
    return AgentEvent(
        event_id="event-1",
        run_id=dependencies.run_id,
        task_id=dependencies.task_id,
        action_id=action.id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=action.planned_at,
    )


def test_r8_configuration_disables_research_ceilings_and_keeps_timeout() -> None:
    root = Path(__file__).resolve().parents[3]
    configuration = load_experiment_configuration(
        root / "configs/experiments/gate_a1_acquisition_r8.yaml",
        project_root=root,
    )
    assert configuration.config.limits.max_actions is None
    assert configuration.config.limits.max_requests is None
    assert configuration.config.limits.timeout_seconds == 600
    assert configuration.config.stopping_policy == "objective_success_or_timeout_v1"


def test_r8_rejects_historical_resume_root_before_runtime_setup(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    resume = tmp_path / "acquisition-r8-20260922T000000Z"
    resume.mkdir()
    (resume / "manifest.json").write_text('{"run_revision":"R7"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="rejects resume roots"):
        acquisition.run_gate_a1_acquisition_r8(root, resume_root=resume)


@pytest.mark.parametrize("mode", ["--acquire-r8", "--resume-acquisition-r8"])
def test_canonical_cli_dispatches_r8_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    calls: list[tuple[Path, Path | None, int | None]] = []

    def fake_r8(
        project_root: Path,
        *,
        resume_root: Path | None = None,
        max_new_tasks: int | None = None,
    ) -> tuple[str, Path]:
        calls.append((project_root, resume_root, max_new_tasks))
        return "READY_TO_RESUME_GATE_A1_ACQUISITION_R8", tmp_path

    def fail_r7(**_kwargs: object) -> tuple[str, Path]:
        raise AssertionError("R7 dispatched")

    monkeypatch.setattr(acquisition, "run_gate_a1_acquisition_r8", fake_r8)
    monkeypatch.setattr(
        acquisition,
        "run_gate_a1_acquisition_r7",
        fail_r7,
    )
    arguments = ["gate-a1", mode]
    if mode == "--resume-acquisition-r8":
        arguments.append(str(tmp_path))
    arguments.extend(["--max-new-tasks", "1", "--project-root", str(tmp_path)])
    monkeypatch.setattr(sys, "argv", arguments)
    assert gate_a1.main() == 0
    assert calls == [
        (
            tmp_path.resolve(),
            tmp_path if mode == "--resume-acquisition-r8" else None,
            1,
        )
    ]


def test_mutation_controller_checks_only_after_mutation_and_continues_on_failure(
    tmp_path: Path,
) -> None:
    dependencies = AgentDependencies(tmp_path, "run-1", "GS-T001")
    checks: list[int] = []
    controller = R8ObjectiveController(
        task=SimpleNamespace(id="GS-T001"),
        workspace=tmp_path,
        objective=lambda task, workspace: checks.append(1) or False,
        dependencies=dependencies,
    )
    attach_r8_objective_controller(dependencies, controller)
    action = PlannedAction(
        id="action-1",
        run_id="run-1",
        task_id="GS-T001",
        tool="read_file",
        operation="read_file",
        arguments={},
        planned_at=datetime.now(UTC),
    )
    controller.before_action(action)
    read_event = _event(dependencies, "read_file")
    controller.after_event(read_event)
    assert checks == []

    mutating = PlannedAction(
        id="action-2",
        run_id="run-1",
        task_id="GS-T001",
        tool="write_file",
        operation="write_file",
        arguments={"path": "arrow/locales.py", "content": "fixed"},
        planned_at=action.planned_at,
    )
    dependencies.record_planned_action(mutating)
    controller.before_action(mutating)
    (tmp_path / "changed.txt").write_text("changed", encoding="utf-8")
    result = ActionResult(
        action_id="action-2",
        tool_name="write_file",
        success=True,
        output="full durable output",
        started_at=action.planned_at,
        completed_at=action.planned_at,
    )
    changed_event = read_event.model_copy(update={"action_id": "action-2", "result": result})
    controller.after_event(changed_event)
    assert checks == [1]
    assert controller.objective_success is False


def test_objective_success_persists_action_before_controlled_completion(tmp_path: Path) -> None:
    dependencies = AgentDependencies(tmp_path, "run-1", "GS-T001")
    persisted: list[str] = []

    class Memory:
        def __getattr__(self, _name: str):
            def save(*_args: object, **_kwargs: object) -> None:
                persisted.append("action")

            return save

    controller = R8ObjectiveController(
        task=SimpleNamespace(id="GS-T001"),
        workspace=tmp_path,
        objective=lambda task, workspace: True,
        dependencies=dependencies,
        repository=cast(Any, Memory()),
        run=SimpleNamespace(id="run-1", task_id="GS-T001"),
        environment=SimpleNamespace(id="env"),
    )
    attach_r8_objective_controller(dependencies, controller)
    action = PlannedAction(
        id="action-1",
        run_id="run-1",
        task_id="GS-T001",
        tool="write_file",
        operation="write_file",
        arguments={},
        planned_at=datetime.now(UTC),
    )
    dependencies.record_planned_action(action)
    controller.before_action(action)
    (tmp_path / "changed.txt").write_text("changed", encoding="utf-8")
    controller.after_event(_event(dependencies, "write_file", action_id="action-1"))
    assert persisted
    assert controller.objective_success
    assert controller.objective_success_pending_evidence


def test_trusted_recovery_evidence_triggers_controlled_stop(tmp_path: Path) -> None:
    dependencies = AgentDependencies(tmp_path, "run-1", "GS-T001")
    controller = R8ObjectiveController(
        task=SimpleNamespace(id="GS-T001"),
        workspace=tmp_path,
        objective=lambda task, workspace: True,
        dependencies=dependencies,
    )
    attach_r8_objective_controller(dependencies, controller)
    failed_action = PlannedAction(
        id="failed",
        run_id="run-1",
        task_id="GS-T001",
        tool="run_tests",
        operation="run_tests",
        arguments={},
        planned_at=datetime.now(UTC),
    )
    dependencies.record_planned_action(failed_action)
    dependencies.events.append(
        _event(dependencies, "run_tests", "failed", success=False, exit_code=1)
    )
    change = PlannedAction(
        id="change",
        run_id="run-1",
        task_id="GS-T001",
        tool="write_file",
        operation="write_file",
        arguments={"path": "arrow/locales.py", "content": "fixed"},
        planned_at=datetime.now(UTC),
    )
    dependencies.record_planned_action(change)
    controller.before_action(change)
    (tmp_path / "changed.txt").write_text("changed", encoding="utf-8")
    dependencies.events.append(_event(dependencies, "write_file", "change"))
    controller.after_event(dependencies.events[-1])
    dependencies.events.append(_event(dependencies, "run_tests", "verified", exit_code=0))
    with pytest.raises(ObjectiveSatisfied):
        controller.after_event(dependencies.events[-1])
    assert controller.termination_reason == "objective_satisfied_with_trusted_recovery_evidence"


def test_objective_success_without_lineage_is_not_acquisition_success() -> None:
    assert classify_r8_acquisition(
        task_success=True,
        complete_trusted_lineage=False,
        pattern_created=False,
        pattern_persisted=False,
        pattern_embedded=False,
        qualifying_failure_observed=False,
    ) == (False, "no_qualifying_failure_observed")


def test_r8_readiness_requires_completed_markers_and_eligible_corpus() -> None:
    task_ids = ("GS-T001", "GS-T002", "GS-T003", "GS-T004", "GS-T005")
    records = [{"task_id": task_ids[0], "acquisition_success": True}]
    assert r8_acquisition_readiness(records, (task_ids[0],), task_ids) == (
        "READY_TO_RESUME_GATE_A1_ACQUISITION_R8",
        False,
        1,
    )
    zero_pattern_records = [
        {"task_id": task_id, "acquisition_success": False} for task_id in task_ids
    ]
    assert r8_acquisition_readiness(zero_pattern_records, task_ids, task_ids) == (
        "BLOCKED_GATE_A1_ACQUISITION_R8_NON_EVALUABLE",
        False,
        0,
    )


def test_in_loop_persistence_failure_is_not_agent_failure(tmp_path: Path) -> None:
    class FailingRepository:
        def close(self) -> None:
            pass

        def save_task(self, *_args: object, **_kwargs: object) -> None:
            raise ValueError("invalid persistence payload")

    dependencies = AgentDependencies(tmp_path, "run-1", "GS-T001")
    controller = R8ObjectiveController(
        task=SimpleNamespace(id="GS-T001"),
        workspace=tmp_path,
        objective=lambda task, workspace: False,
        dependencies=dependencies,
        repository=ShortLivedNeo4jRepository(cast(Any, FailingRepository)),
        run=SimpleNamespace(id="run-1", task_id="GS-T001"),
        environment=SimpleNamespace(id="env"),
    )
    action = PlannedAction(
        id="action-1",
        run_id="run-1",
        task_id="GS-T001",
        tool="write_file",
        operation="write_file",
        arguments={"path": "changed.txt", "content": "changed"},
        planned_at=datetime.now(UTC),
    )
    dependencies.record_planned_action(action)
    controller.before_action(action)
    (tmp_path / "changed.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(InLoopPersistenceError):
        controller.after_event(_event(dependencies, "write_file", action_id=action.id))
    assert isinstance(controller.persistence_error, ValueError)
    assert controller.last_action_persisted is False
    assert controller.repository is not None
    telemetry = controller.repository.telemetry
    assert telemetry[0]["persistence_stage"] == "persist_task"
    assert telemetry[0]["exception_type"] == "ValueError"
    assert telemetry[0]["attempt_number"] == 1
    assert telemetry[0]["retryable"] is False
    assert telemetry[0]["retry_performed"] is False


def test_r8_eol_cleanup_preserves_staged_benchmark_mutation(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "R8 Test")
    target = tmp_path / "arrow" / "locales.py"
    target.parent.mkdir()
    target.write_bytes(b"base\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "baseline")
    target.write_bytes(b"base\nintentional mutation\n")
    _git(tmp_path, "add", "arrow/locales.py")
    target.write_bytes(b"base\r\nintentional mutation\r\n")
    assert _git(tmp_path, "status", "--short").startswith("MM")
    before_stage = _git(tmp_path, "ls-files", "-s", "--", "arrow/locales.py")
    before_blob = _git(tmp_path, "show", ":arrow/locales.py")
    _remove_eol_only_worktree_noise(tmp_path)
    assert _git(tmp_path, "ls-files", "-s", "--", "arrow/locales.py") == before_stage
    assert _git(tmp_path, "show", ":arrow/locales.py") == before_blob
    assert _git(tmp_path, "status", "--short").strip() == "M  arrow/locales.py"
    assert _git(tmp_path, "diff", "--", "arrow/locales.py") == ""
    assert "intentional mutation" in _git(tmp_path, "diff", "--cached")


def test_retryable_session_expired_uses_fresh_repository_and_succeeds() -> None:
    instances: list[object] = []

    class FakeRepository:
        def __init__(self) -> None:
            self.calls = 0
            instances.append(self)

        def save_action(self, *_args: object) -> None:
            self.calls += 1
            if len(instances) == 1:
                raise SessionExpired("session expired")

        def close(self) -> None:
            pass

    memory = ShortLivedNeo4jRepository(cast(Any, FakeRepository))
    memory.save_action("action", "result")
    assert len(instances) == 2
    assert memory.telemetry[0]["retryable"] is True
    assert memory.telemetry[0]["retry_performed"] is True


def test_non_retryable_persistence_error_is_not_retried() -> None:
    class FakeRepository:
        def close(self) -> None:
            pass

        def save_action(self, *_args: object) -> None:
            raise ValueError("bad data")

    calls = 0

    def factory() -> FakeRepository:
        nonlocal calls
        calls += 1
        return FakeRepository()

    memory = ShortLivedNeo4jRepository(cast(Any, factory))
    with pytest.raises(ValueError):
        memory.save_action("action", "result")
    assert calls == 1
    assert memory.telemetry[0]["retryable"] is False


def test_repository_fingerprint_changes_for_repository_mutation(tmp_path: Path) -> None:
    before = repository_state_fingerprint(tmp_path)
    (tmp_path / "file.bin").write_bytes(b"binary")
    assert repository_state_fingerprint(tmp_path) != before


def test_reprocessing_successful_event_stream_is_idempotent() -> None:
    class Memory:
        def __init__(self) -> None:
            self.actions: set[str] = set()
            self.failures: set[str] = set()
            self.resolutions: set[str] = set()
            self.outcomes: set[str] = set()
            self.edges: set[tuple[str, ...]] = set()

        def save_task(self, task: Task) -> None:
            pass

        def save_run(self, run: Run) -> None:
            pass

        def save_environment(self, environment: EnvironmentContext) -> None:
            pass

        def save_action(self, action: PlannedAction, result: ActionResult) -> None:
            self.actions.add(action.id)

        def save_tool(self, tool: object) -> None:
            pass

        def save_failure(self, failure: Any) -> None:
            self.failures.add(failure.id)

        def save_resolution(self, resolution: Any) -> None:
            self.resolutions.add(resolution.id)

        def save_outcome(self, outcome: Any) -> None:
            self.outcomes.add(outcome.id)

        def __getattr__(self, name: str):
            def link(*values: object) -> None:
                self.edges.add((name, *(str(value) for value in values)))

            return link

    task = Task(
        id="GS-T001",
        problem_statement="repair",
        family_id="family",
        repository="repo",
        chronological_index=1,
    )
    run = Run(id="run-1", task_id=task.id, started_at=datetime.now(UTC))
    environment = EnvironmentContext(
        id="environment-1",
        repository="repo",
        runtime="python",
    )
    dependencies = AgentDependencies(Path.cwd(), run.id, task.id)
    failure = PlannedAction(
        id="failure-action",
        run_id=run.id,
        task_id=task.id,
        tool="run_tests",
        operation="run_tests",
        arguments={},
        planned_at=datetime.now(UTC),
    )
    change = PlannedAction(
        id="change-action",
        run_id=run.id,
        task_id=task.id,
        tool="write_file",
        operation="write_file",
        arguments={"path": "arrow/locales.py", "content": "fixed"},
        planned_at=datetime.now(UTC),
    )
    verified = PlannedAction(
        id="verified-action",
        run_id=run.id,
        task_id=task.id,
        tool="run_tests",
        operation="run_tests",
        arguments={},
        planned_at=datetime.now(UTC),
    )
    for action in (failure, change, verified):
        dependencies.record_planned_action(action)
    events = (
        _event(dependencies, "run_tests", failure.id, success=False, exit_code=1),
        _event(dependencies, "write_file", change.id),
        _event(dependencies, "run_tests", verified.id, exit_code=0),
    )
    memory = Memory()
    first = persist_agent_event_stream(
        cast(Any, memory), events, task, run, environment,
        planned_actions=dependencies.planned_actions,
        require_trusted_planned_actions=True,
    )
    second = persist_agent_event_stream(
        cast(Any, memory), events, task, run, environment,
        planned_actions=dependencies.planned_actions,
        require_trusted_planned_actions=True,
    )
    assert first is not None and second is not None
    assert len(memory.actions) == 3
    assert len(memory.failures) == 1
    assert len(memory.resolutions) == 1
    assert len(memory.outcomes) == 1
    assert len(memory.edges) == len(set(memory.edges))
