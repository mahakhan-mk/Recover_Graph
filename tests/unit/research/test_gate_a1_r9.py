# pyright: reportPrivateUsage=false

import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.models.test import TestModel

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.read_models import (
    ActionLineageRecord,
    RecoveryEvidenceLineage,
    RecoveryEvidenceTask,
)
from graph_swarm.integration.event_persistence import (
    persist_agent_event_stream,
    select_recovery_event_chain,
)
from graph_swarm.memory.recovery_abstraction import (
    abstract_and_persist_recovery_pattern,
    build_recovery_evidence_package,
)
from graph_swarm.memory.recovery_evidence import (
    RepositoryMutationEvidence,
    is_test_execution,
)
from graph_swarm.research import gate_a1
from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research.gate_a1_r8 import (
    agent_timeout_seconds_for_configuration,
    git_worktree_content_fingerprint,
)
from graph_swarm.research.gate_a1_r9 import R9ObjectiveController
from graph_swarm.research.runner import load_experiment_configuration


def _action(
    action_id: str,
    tool: str,
    command: list[str],
    *,
    run_id: str = "run-1",
    task_id: str = "GS-T001",
) -> PlannedAction:
    return PlannedAction(
        id=action_id,
        run_id=run_id,
        task_id=task_id,
        tool=tool,
        operation=tool,
        arguments={"command": command} if tool == "run_command" else {},
        planned_at=datetime.now(UTC),
    )


def _event(
    action: PlannedAction,
    *,
    event_id: str,
    success: bool,
    exit_code: int | None,
) -> AgentEvent:
    now = action.planned_at
    return AgentEvent(
        event_id=event_id,
        run_id=action.run_id,
        task_id=action.task_id,
        action_id=action.id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=ActionResult(
            action_id=action.id,
            tool_name=action.tool,
            success=success,
            exit_code=exit_code,
            output="pytest output",
            started_at=now,
            completed_at=now,
        ),
        occurred_at=now,
    )


def _git_repository_with_source(root: Path) -> Path:
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "r9@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "R9 Test"],
        check=True,
    )
    (root / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\ncoverage.xml\n",
        encoding="utf-8",
    )
    source = root / "arrow" / "locales.py"
    source.parent.mkdir()
    source.write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "baseline"], check=True)
    return source


class Memory:
    def __init__(self) -> None:
        self.actions: dict[str, tuple[PlannedAction, ActionResult]] = {}
        self.failures: dict[str, Any] = {}
        self.resolutions: dict[str, Any] = {}
        self.outcomes: dict[str, Any] = {}
        self.patterns: dict[str, Any] = {}
        self.edges: set[tuple[str, ...]] = set()

    def save_task(self, _task: Task) -> None:
        pass

    def save_run(self, _run: Run) -> None:
        pass

    def save_environment(self, _environment: EnvironmentContext) -> None:
        pass

    def save_action(self, action: PlannedAction, result: ActionResult) -> None:
        self.actions[action.id] = (action, result)

    def save_tool(self, _tool: object) -> None:
        pass

    def save_failure(self, failure: Any) -> None:
        self.failures[failure.id] = failure

    def save_resolution(self, resolution: Any) -> None:
        self.resolutions[resolution.id] = resolution

    def save_outcome(self, outcome: Any) -> None:
        self.outcomes[outcome.id] = outcome

    def save_recovery_pattern(self, pattern: Any) -> None:
        self.patterns[pattern.id] = pattern

    def __getattr__(self, name: str):
        def link(*values: object) -> None:
            self.edges.add((name, *(str(value) for value in values)))

        return link


def _fixture() -> tuple[
    Task,
    Run,
    EnvironmentContext,
    dict[str, PlannedAction],
    tuple[AgentEvent, ...],
    RepositoryMutationEvidence,
]:
    task = Task(
        id="GS-T001",
        problem_statement="repair",
        family_id="family",
        repository="repo",
        chronological_index=1,
    )
    run = Run(id="run-1", task_id=task.id, started_at=datetime.now(UTC))
    environment = EnvironmentContext(id="env-1", repository="repo", runtime="python")
    failed = _action(
        "failed",
        "run_command",
        [
            "python",
            "-m",
            "pytest",
            "tests/test_locales.py::TestIcelandicLocale::test_format_timeframe",
            "-q",
        ],
    )
    changed = _action("changed", "run_command", ["python", "-c", "edit_repository"])
    verified = _action(
        "verified",
        "run_command",
        [
            "python",
            "-m",
            "pytest",
            "tests/test_locales.py::TestIcelandicLocale::test_format_timeframe",
            "-q",
        ],
    )
    actions = {action.id: action for action in (failed, changed, verified)}
    events = (
        _event(failed, event_id="event-failed", success=False, exit_code=1),
        _event(changed, event_id="event-changed", success=True, exit_code=0),
        _event(verified, event_id="event-verified", success=True, exit_code=0),
    )
    mutation = RepositoryMutationEvidence(
        action_id=changed.id,
        before_fingerprint="before",
        after_fingerprint="after",
    )
    return task, run, environment, actions, events, mutation


@pytest.mark.parametrize("mode", ["--acquire-r9", "--resume-acquisition-r9"])
def test_r9_config_and_cli_are_distinct_from_r8(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    root = Path(__file__).resolve().parents[3]
    configuration = load_experiment_configuration(
        root / "configs/experiments/gate_a1_acquisition_r9.yaml",
        project_root=root,
    )
    assert configuration.config.run_revision == "R9"
    assert configuration.config.recovery_event_semantics == "trusted_action_semantics_v2"
    assert (
        configuration.config.repository_mutation_evidence_policy
        == "git_worktree_content_fingerprint_v2"
    )
    assert configuration.config.limits.max_actions is None
    assert configuration.config.limits.max_requests is None
    assert configuration.config.limits.timeout_seconds == 600
    assert configuration.config.agent_timeout_seconds == 600
    assert configuration.config.objective_timeout_seconds == 900
    assert agent_timeout_seconds_for_configuration(configuration) == 600

    calls: list[tuple[Path, Path | None, int | None]] = []

    def fake_r9(
        project_root: Path,
        *,
        resume_root: Path | None = None,
        max_new_tasks: int | None = None,
    ) -> tuple[str, Path]:
        calls.append((project_root, resume_root, max_new_tasks))
        return "READY_TO_RESUME_GATE_A1_ACQUISITION_R9", tmp_path

    def fail_r8(**_kwargs: object) -> tuple[str, Path]:
        raise AssertionError("R8 dispatched")

    monkeypatch.setattr(acquisition, "run_gate_a1_acquisition_r9", fake_r9)
    monkeypatch.setattr(acquisition, "run_gate_a1_acquisition_r8", fail_r8)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate-a1", mode]
        + ([str(tmp_path)] if mode == "--resume-acquisition-r9" else [])
        + ["--max-new-tasks", "1", "--project-root", str(tmp_path)],
    )

    assert gate_a1.main() == 0
    assert calls == [
        (
            tmp_path.resolve(),
            tmp_path if mode == "--resume-acquisition-r9" else None,
            1,
        )
    ]


def test_r9_rejects_r8_resume_root_without_provider_setup(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    resume = tmp_path / "acquisition-r9-20260922T000000Z"
    resume.mkdir()
    (resume / "manifest.json").write_text('{"run_revision":"R8"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="rejects resume roots"):
        acquisition.run_gate_a1_acquisition_r9(root, resume_root=resume)


def test_r9_timeout_scopes_are_distinct_hash_inputs() -> None:
    root = Path(__file__).resolve().parents[3]
    configuration = load_experiment_configuration(
        root / "configs/experiments/gate_a1_acquisition_r9.yaml",
        project_root=root,
    )
    settings = SimpleNamespace(
        openrouter_coding_model="coding-model",
        openrouter_abstraction_model="abstraction-model",
    )
    environments = {
        task_id: SimpleNamespace(
            environment_fingerprint=f"environment-{task_id}",
            container_image="benchmark:image",
        )
        for task_id in acquisition.ACQUISITION_TASK_IDS
    }
    baseline_hash = acquisition._r8_configuration_hash(  # pyright: ignore[reportPrivateUsage]
        configuration, settings, cast(Any, environments)
    )
    changed_agent = replace(
        configuration,
        config=configuration.config.model_copy(update={"agent_timeout_seconds": 601}),
    )
    changed_objective = replace(
        configuration,
        config=configuration.config.model_copy(update={"objective_timeout_seconds": 901}),
    )

    assert (
        acquisition._r8_configuration_hash(  # pyright: ignore[reportPrivateUsage]
            changed_agent, settings, cast(Any, environments)
        )
        != baseline_hash
    )
    assert (
        acquisition._r8_configuration_hash(  # pyright: ignore[reportPrivateUsage]
            changed_objective, settings, cast(Any, environments)
        )
        != baseline_hash
    )


def test_r9_git_content_fingerprint_excludes_metadata_and_runtime_artifacts(
    tmp_path: Path,
) -> None:
    source = _git_repository_with_source(tmp_path)
    baseline = git_worktree_content_fingerprint(tmp_path)

    subprocess.run(["git", "-C", str(tmp_path), "status", "--short"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-index", "--refresh"],
        check=True,
    )
    assert git_worktree_content_fingerprint(tmp_path) == baseline

    (tmp_path / ".git" / "r9-runtime-metadata").write_text("metadata", encoding="utf-8")
    assert git_worktree_content_fingerprint(tmp_path) == baseline

    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "runtime.pyc").write_bytes(b"cache")
    (tmp_path / ".pytest_cache" / "v").mkdir(parents=True)
    (tmp_path / ".pytest_cache" / "v" / "cache").write_text("cache", encoding="utf-8")
    assert git_worktree_content_fingerprint(tmp_path) == baseline

    source.write_text("tracked edit\n", encoding="utf-8")
    edited = git_worktree_content_fingerprint(tmp_path)
    assert edited != baseline
    source.write_text("baseline\n", encoding="utf-8")
    assert git_worktree_content_fingerprint(tmp_path) == baseline

    new_source = tmp_path / "new_source.py"
    new_source.write_text("new source\n", encoding="utf-8")
    assert git_worktree_content_fingerprint(tmp_path) != baseline
    new_source.unlink()
    assert git_worktree_content_fingerprint(tmp_path) == baseline


def test_r9_controller_records_runtime_mutation_evidence(tmp_path: Path) -> None:
    source = _git_repository_with_source(tmp_path)
    dependencies = AgentDependencies(tmp_path, "run-1", "GS-T001")
    controller = R9ObjectiveController(
        task=SimpleNamespace(id="GS-T001"),
        workspace=tmp_path,
        objective=lambda task, workspace: False,
        dependencies=dependencies,
    )
    action = _action("changed", "run_command", ["python", "-c", "edit_repository"])
    dependencies.record_planned_action(action)
    controller.before_action(action)
    source.write_text("actual source mutation\n", encoding="utf-8")
    controller.after_event(
        _event(action, event_id="event-changed", success=True, exit_code=0)
    )

    evidence = dependencies.repository_mutation_evidence_for(action.id)
    assert evidence is not None
    assert evidence.action_id == action.id
    assert evidence.before_fingerprint != evidence.after_fingerprint


def test_r9_controller_does_not_record_cache_only_mutation(tmp_path: Path) -> None:
    _git_repository_with_source(tmp_path)
    dependencies = AgentDependencies(tmp_path, "run-1", "GS-T001")
    controller = R9ObjectiveController(
        task=SimpleNamespace(id="GS-T001"),
        workspace=tmp_path,
        objective=lambda task, workspace: False,
        dependencies=dependencies,
    )
    action = _action("cache", "run_command", ["python", "-m", "pytest", "--help"])
    dependencies.record_planned_action(action)
    controller.before_action(action)
    cache = tmp_path / ".pytest_cache" / "v"
    cache.mkdir(parents=True)
    (cache / "cache").write_text("runtime cache", encoding="utf-8")
    controller.after_event(_event(action, event_id="event-cache", success=True, exit_code=0))

    assert dependencies.repository_mutation_evidence_for(action.id) is None


def test_r9_recovers_pytest_run_command_mutation_chain_idempotently() -> None:
    task, run, environment, actions, events, mutation = _fixture()
    memory = Memory()
    first = persist_agent_event_stream(
        cast(Any, memory),
        events,
        task,
        run,
        environment,
        planned_actions=actions,
        require_trusted_planned_actions=True,
        mutation_evidence={mutation.action_id: mutation},
    )
    second = persist_agent_event_stream(
        cast(Any, memory),
        events,
        task,
        run,
        environment,
        planned_actions=actions,
        require_trusted_planned_actions=True,
        mutation_evidence={mutation.action_id: mutation},
    )

    assert first is not None and second is not None
    failure, resolution, outcome = first
    assert failure.failure_type.value == "test_failure"
    assert outcome.success is True
    chain = select_recovery_event_chain(events, actions, {mutation.action_id: mutation})
    assert chain is not None
    assert chain.change_action.id == mutation.action_id
    assert is_test_execution(actions["failed"])
    assert is_test_execution(actions["verified"])
    assert len(memory.actions) == 3
    assert len(memory.failures) == 1
    assert len(memory.resolutions) == 1
    assert len(memory.outcomes) == 1
    assert len(memory.edges) == len(set(memory.edges))

    lineage = RecoveryEvidenceLineage(
        failure=failure,
        resolution=resolution,
        outcome=outcome,
        task=RecoveryEvidenceTask(
            id=task.id,
            problem_statement=task.problem_statement,
            repository=task.repository,
            chronological_index=task.chronological_index,
        ),
        environment=environment,
        failed_action=ActionLineageRecord(
            planned_action=actions["failed"],
            result=events[0].result,
        ),
        recovery_action=ActionLineageRecord(
            planned_action=actions["changed"],
            result=events[1].result,
        ),
    )
    evidence = build_recovery_evidence_package(lineage)
    assert evidence.recovery_action_tool == "run_command"
    pattern_one = abstract_and_persist_recovery_pattern(
        lineage,
        cast(Any, memory),
        cast(Any, object()),
        model=TestModel(
            custom_output_args={
                "title": "Repair the failing locale behavior",
                "guidance": (
                    "Inspect the failing locale input and apply the smallest concrete "
                    "correction before rerunning the targeted test."
                ),
                "evidence_summary": (
                    "A pytest command failure was followed by an observed repository "
                    "change and a successful targeted pytest command."
                ),
            }
        ),
    )
    pattern_two = abstract_and_persist_recovery_pattern(
        lineage,
        cast(Any, memory),
        cast(Any, object()),
        model=TestModel(
            custom_output_args={
                "title": "Repair the failing locale behavior",
                "guidance": (
                    "Inspect the failing locale input and apply the smallest concrete "
                    "correction before rerunning the targeted test."
                ),
                "evidence_summary": (
                    "A pytest command failure was followed by an observed repository "
                    "change and a successful targeted pytest command."
                ),
            }
        ),
    )
    assert pattern_one.id == pattern_two.id
    assert len(memory.patterns) == 1


def test_command_failure_fallback_selects_latest_failure_before_mutation() -> None:
    task, run, environment, actions, base_events, mutation = _fixture()
    early_pwd = _action("early-pwd", "run_command", ["pwd", "-l"])
    early_rg = _action("early-rg", "run_command", ["rg", "--files"])
    relevant_python = _action(
        "relevant-python",
        "run_command",
        ["python", "-c", "raise ValueError('Icelandic locale')"],
    )
    actions.update(
        {
            early_pwd.id: early_pwd,
            early_rg.id: early_rg,
            relevant_python.id: relevant_python,
        }
    )
    events = (
        _event(early_pwd, event_id="event-early-pwd", success=False, exit_code=1),
        _event(early_rg, event_id="event-early-rg", success=False, exit_code=1),
        _event(
            relevant_python,
            event_id="event-relevant-python",
            success=False,
            exit_code=1,
        ),
        *base_events[1:],
    )

    memory = Memory()
    persisted = persist_agent_event_stream(
        cast(Any, memory),
        events,
        task,
        run,
        environment,
        planned_actions=actions,
        require_trusted_planned_actions=True,
        mutation_evidence={mutation.action_id: mutation},
    )

    assert persisted is not None
    failure, resolution, _outcome = persisted
    assert failure.action_id == relevant_python.id
    assert resolution.failure_id == failure.id
    assert f"source_failure_id={failure.id};" in resolution.description
    chain = select_recovery_event_chain(
        events, actions, {mutation.action_id: mutation}
    )
    assert chain is not None
    assert failure.id == chain.failure.id


def test_test_failure_still_outranks_generic_command_failures() -> None:
    _task, _run, _environment, actions, base_events, mutation = _fixture()
    early_command = _action("early-command", "run_command", ["pwd", "-l"])
    actions[early_command.id] = early_command
    events = (
        _event(early_command, event_id="event-early-command", success=False, exit_code=1),
        *base_events,
    )

    chain = select_recovery_event_chain(
        events,
        actions,
        {mutation.action_id: mutation},
    )

    assert chain is not None
    assert chain.failure_event.action_id == actions["failed"].id


def test_r9_non_mutating_command_cannot_form_recovery_change() -> None:
    _task, _run, _environment, actions, events, _mutation = _fixture()
    non_mutating = _action("non-mutating", "run_command", ["python", "-c", "inspect"])
    non_mutating_event = _event(
        non_mutating,
        event_id="event-non-mutating",
        success=True,
        exit_code=0,
    )
    events_without_change = (events[0], non_mutating_event, events[2])
    actions_without_change = {
        actions["failed"].id: actions["failed"],
        non_mutating.id: non_mutating,
        actions["verified"].id: actions["verified"],
    }

    assert (
        select_recovery_event_chain(events_without_change, actions_without_change)
        is None
    )
