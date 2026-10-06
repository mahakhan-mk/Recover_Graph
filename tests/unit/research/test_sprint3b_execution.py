from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from experiments.run_sprint3b_reduced_b0_t import (
    EXPECTED_CODING_MODEL,
    EXPECTED_CONDITIONS,
    EXPECTED_TASK_IDS,
    ExecutionSlot,
    PrimaryRunCollisionError,
    RunValidity,
    SlotStatus,
    build_execution_plan,
    execute_primary_plan,
    inspect_primary_slots,
    FrozenExecutionContext,
)
from experiments.run_sprint3b_kilo_v4_qwen3_coder_canary import (
    build_condition_runner as build_reported_condition_runner,
)
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.integration.advisory_runtime import (
    Neo4jAdvisoryRuntime,
    R13bTreatmentRepository,
)
from graph_swarm.research.contracts import ExperimentCondition, ExperimentRunArtifact
from graph_swarm.research.runner import (
    ExperimentRunArtifactStore,
    load_experiment_configuration,
    load_task_cases,
)
from graph_swarm.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/experiments/reported/exp1_exp3_qwen.yaml"


def _context(tmp_path: Path) -> FrozenExecutionContext:
    """Build the test execution context from the retained reported EXP1/EXP3 inputs."""
    configuration = load_experiment_configuration(CONFIG, project_root=ROOT)
    selected_ids = ("GS-T006", "GS-T007", "GS-T008")
    cases = tuple(
        case
        for case in load_task_cases(
            configuration.task_manifest_path,
            problem_statements_path=configuration.task_problems_path,
        )
        if case.task.id in selected_ids
    )
    assert tuple(case.task.id for case in cases) == selected_ids
    conditions = (ExperimentCondition.B0, ExperimentCondition.T)
    plan = tuple(
        ExecutionSlot(index, case.task.id, condition)
        for index, (case, condition) in enumerate(
            ((case, condition) for case in cases for condition in conditions)
        )
    )
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    protocol = {
        "provider": configuration.model.provider,
        "model": configuration.model.model,
        "protocol_revision": configuration.config.config_version,
        "prompt_version": configuration.model.prompt_version,
        "objective_evaluator": "frozen_swesmith_fail_to_pass",
        "objective_evaluator_version": "frozen_swesmith_objective_v1",
        "recurrence_evaluator": "frozen_fail_to_pass_recurrence_matcher",
        "recurrence_evaluator_version": "frozen_recurrence_matcher_v2_structured_pytest",
    }
    freeze = {
        "config_sha256": config_sha256,
        "model": {
            "system_prompt": configuration.config.system_prompt,
            "system_prompt_sha256": configuration.resolved_system_prompt_sha256,
        },
    }
    return FrozenExecutionContext(
        project_root=ROOT,
        config=configuration,
        protocol=protocol,
        freeze=freeze,
        freeze_hash=config_sha256,
        execution_commit_sha="test-commit",
        plan=plan,
        cases=cases,
        artifact_root=tmp_path / "runs",
    )


def _settings() -> Settings:
    return Settings(
        neo4j_uri="neo4j://test",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        openrouter_api_key="test-key",
        hf_token="test-token",
        model_provider="kilo",
        kilo_api_key="test-key",
        kilo_coding_model="qwen/qwen3-coder",
    )


class _FakeAdvisoryService:
    def evaluate_action(self, _task: Any, _action: Any, _environment: Any) -> None:
        return None


def _fake_execution(
    context: Any,
    slot: ExecutionSlot,
    store: ExperimentRunArtifactStore,
    run_number: int,
    error: Exception | None = None,
) -> Any:
    run_id = f"fake-{run_number}"
    now = datetime.now(UTC)
    action_id = f"{run_id}-action"
    planned = PlannedAction(
        id=action_id,
        run_id=run_id,
        task_id=slot.task_id,
        tool="agent",
        operation="task_completion",
        arguments={},
        planned_at=now,
    )
    result = ActionResult(
        action_id=action_id,
        tool_name="agent",
        success=error is None,
        error=None if error is None else type(error).__name__,
        started_at=now,
        completed_at=now,
    )
    case = next(item for item in context.cases if item.task.id == slot.task_id)
    artifact = ExperimentRunArtifact(
        experiment_id="GS-E003",
        condition=slot.condition,
        run_id=run_id,
        task_id=slot.task_id,
        family_id=case.task.family_id,
        chronological_index=case.task.chronological_index,
        model=context.config.model.model,
        model_settings=context.config.model.settings,
        prompt_version=context.config.model.prompt_version,
        planned_action=planned,
        executed_action=result,
        task_success=error is None,
        known_failure_repeated=False,
        tool_calls=0,
        retries=0,
    )
    artifact_path = store.write_artifact(artifact)
    return SimpleNamespace(
        artifact=artifact,
        artifact_path=artifact_path,
        dependencies=AgentDependencies(
            artifact_path.parent / "workspace",
            run_id,
            slot.task_id,
            task=case.task,
        ),
        agent_result=None,
        error=error,
    )


def test_exact_frozen_20_slot_plan() -> None:
    execution_plan: list[list[str]] = [
        [task_id, condition.value]
        for task_id in EXPECTED_TASK_IDS
        for condition in EXPECTED_CONDITIONS
    ]
    protocol: dict[str, Any] = {
        "task_order": list(EXPECTED_TASK_IDS),
        "conditions": [condition.value for condition in EXPECTED_CONDITIONS],
        "execution_plan": execution_plan,
        "planned_primary_runs": 20,
    }
    plan = build_execution_plan(protocol)
    assert len(plan) == 20
    assert [(slot.task_id, slot.condition.value) for slot in plan] == [
        tuple(item) for item in execution_plan
    ]


def test_b0_has_no_advisory_and_t_accepts_injected_runtime() -> None:
    runtime = Neo4jAdvisoryRuntime(
        repository=object(),  # type: ignore[arg-type]
        treatment_repository=R13bTreatmentRepository(cast(Any, object())),
        advisory_service=_FakeAdvisoryService(),  # type: ignore[arg-type]
    )
    def objective(_task: Any, _workspace: Path) -> bool:
        return True

    def recurrence(_case: Any, _events: Any, _result: Any, _workspace: Path) -> bool:
        return False

    def workspace(_task: Any) -> Path:
        return ROOT

    def resolve_runtime(_task: Any, _workspace: Path) -> None:
        return None

    common: dict[str, Any] = {
        "objective_evaluator": objective,
        "recurrence_evaluator": recurrence,
        "workspace_resolver": workspace,
        "execution_runtime_resolver": resolve_runtime,
        "settings": _settings(),
    }
    for config_path in (
        CONFIG,
        ROOT / "configs/experiments/reported/exp4_qwen.yaml",
    ):
        configuration = load_experiment_configuration(config_path, project_root=ROOT)
        b0 = build_reported_condition_runner(
            configuration, condition=ExperimentCondition.B0, **common
        )
        treatment = build_reported_condition_runner(
            configuration,
            condition=ExperimentCondition.T,
            advisory_runtime=runtime,
            **common,
        )
        assert b0.advisory_service is None
        assert treatment.advisory_service is runtime.advisory_service


def test_workspace_materialization_isolated_by_run(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline" / "repo"
    baseline.mkdir(parents=True)
    (baseline / "source.txt").write_text("baseline", encoding="utf-8")
    from graph_swarm.research.runner import BaselineWorkspaceManager

    manager = BaselineWorkspaceManager(tmp_path / "baseline", tmp_path / "execution")
    task = next(
        case.task
        for case in load_task_cases(
            ROOT / "benchmark/manifests/pilot.jsonl",
            problem_statements_path=ROOT / "benchmark/annotations/recurrence_validation.csv",
        )
        if case.task.id == "GS-T006"
    ).model_copy(update={"repository": "repo"})
    first = manager.materialize(task, "run-1")
    (first / "source.txt").write_text("changed", encoding="utf-8")
    second = manager.materialize(task, "run-2")
    assert (second / "source.txt").read_text(encoding="utf-8") == "baseline"


def test_resume_and_collision_refusal(tmp_path: Path) -> None:
    context = _context(tmp_path)
    store = ExperimentRunArtifactStore(context.artifact_root)
    order: list[tuple[str, str]] = []

    def execute(slot: ExecutionSlot, case: Any) -> Any:
        order.append((slot.task_id, slot.condition.value))
        return _fake_execution(context, slot, store, len(order))

    first = execute_primary_plan(context, execute, artifact_store=store)
    assert first.valid_completed_slots == len(context.plan) == 6
    assert order == [
        (slot.task_id, slot.condition.value) for slot in context.plan
    ]
    order.clear()
    resumed = execute_primary_plan(context, execute, artifact_store=store)
    assert resumed.valid_completed_slots == len(context.plan) == 6
    assert order == []

    first_slot = context.plan[0]
    collision_root = (
        tmp_path
        / "collision"
        / "GS-E003"
        / first_slot.condition.value
        / first_slot.task_id
        / "run"
    )
    collision_root.mkdir(parents=True)
    with pytest.raises(PrimaryRunCollisionError):
        inspect_primary_slots(tmp_path / "collision", context.plan)


def test_invalid_infrastructure_attempt_is_preserved_and_not_rerun(tmp_path: Path) -> None:
    context = _context(tmp_path)
    store = ExperimentRunArtifactStore(context.artifact_root)
    calls: list[tuple[str, str]] = []

    def execute(slot: ExecutionSlot, case: Any) -> Any:
        calls.append((slot.task_id, slot.condition.value))
        if len(calls) == 1:
            raise RuntimeError("provider transport failed")
        return _fake_execution(context, slot, store, len(calls))

    report = execute_primary_plan(context, execute, artifact_store=store)
    assert report.attempted_slots == 1
    assert report.invalid_infrastructure_slots
    assert report.valid_completed_slots == 0
    assert len(report.missing_slots) == len(context.plan) - 1
    assert calls == [("GS-T006", "B0")]
    preserved = inspect_primary_slots(context.artifact_root, context.plan)
    assert preserved[0].status is SlotStatus.INVALID_INFRASTRUCTURE
    calls.clear()
    resumed = execute_primary_plan(context, execute, artifact_store=store)
    assert resumed.invalid_infrastructure_slots
    assert calls == []


def test_treatment_runtime_surface_has_no_write_methods() -> None:
    repository = R13bTreatmentRepository(cast(Any, object()))
    public_methods = {
        name
        for name in dir(repository)
        if not name.startswith("_") and callable(getattr(repository, name))
    }
    assert not any(
        name.startswith(("save_", "update_", "link_", "ensure_")) for name in public_methods
    )
    assert RunValidity.VALID.value == "valid"
