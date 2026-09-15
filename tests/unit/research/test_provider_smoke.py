from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.oracle import FrozenOracleResolver, OracleTransferEvidence
from experiments.sprint3 import (
    BenchmarkPreflightError,
    FrozenSWEsmithCase,
    FrozenSWEsmithObjective,
    IsolatedTaskEnvironment,
    PreparedProviderSmokeContext,
    make_recurrence_matcher,
)
from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import HistoricalRecoveryAdvice
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AdviceEvent
from graph_swarm.domain.tasks import Task
from graph_swarm.research.contracts import ExperimentCondition, ExperimentRunArtifact
from graph_swarm.research.provider_smoke import (
    PROVIDER_SMOKE_TASK_ID,
    ProviderSmokePrerequisiteError,
    RuntimeSmokeArtifact,
    build_provider_smoke_summary,
    find_runtime_smoke_artifact,
    write_provider_smoke_summary,
)
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    ExperimentExecution,
    LoadedExperimentConfiguration,
    load_experiment_configuration,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "experiments" / "rollout_3a_pilot.yaml"
PILOT_MANIFEST = ROOT / "benchmark" / "manifests" / "pilot.jsonl"
PILOT_ANNOTATIONS = ROOT / "benchmark" / "annotations" / "recurrence_validation.csv"
PATTERN = "Inspect the failing dataflow and restore the required initialization."


def _task_case() -> BenchmarkTaskCase:
    return BenchmarkTaskCase(
        Task(
            id=PROVIDER_SMOKE_TASK_ID,
            problem_statement="Fix the frozen transfer task.",
            family_id="GS-F002",
            repository="offline-repository",
            chronological_index=7,
        ),
        occurrence_index=2,
    )


def _execution(
    tmp_path: Path,
    condition: ExperimentCondition,
    *,
    advice: bool = False,
    review_id: str | None = "GS-R014",
    recovery_pattern: str = PATTERN,
    error: Exception | None = None,
    run_id: str | None = None,
) -> ExperimentExecution:
    case = _task_case()
    now = datetime.now(UTC)
    selected_run_id = run_id or f"{condition.value}-run"
    planned = PlannedAction(
        id=f"{selected_run_id}-action",
        run_id=selected_run_id,
        task_id=case.task.id,
        tool="run_tests",
        operation="run_tests",
        planned_at=now,
    )
    result = ActionResult(
        action_id=planned.id,
        tool_name="run_tests",
        success=False,
        exit_code=1,
        started_at=now,
        completed_at=now,
    )
    artifact = ExperimentRunArtifact(
        experiment_id="GS-E003",
        condition=condition,
        run_id=selected_run_id,
        task_id=case.task.id,
        family_id=case.task.family_id,
        chronological_index=case.task.chronological_index,
        model="test-model",
        model_settings={"temperature": 0},
        prompt_version="v1",
        planned_action=planned,
        executed_action=result,
        advice_received=(
            FrozenOracleResolver(
                {
                    case.task.id: OracleTransferEvidence(
                        case.task.id,
                        recovery_pattern,
                        review_id,
                    )
                }
            ).evaluate_action(
                case.task,
                planned,
                EnvironmentContext(
                    id="environment",
                    repository=case.task.repository,
                    runtime="python",
                ),
            )
            if advice
            else None
        ),
        task_success=False,
        known_failure_repeated=False,
        tool_calls=1,
        retries=0,
    )
    dependencies = AgentDependencies(
        tmp_path / condition.value,
        selected_run_id,
        case.task.id,
        task=case.task,
    )
    if advice:
        advice_result = artifact.advice_received
        assert advice_result is not None
        assert isinstance(advice_result.advice, HistoricalRecoveryAdvice)
        dependencies.advice_events.append(
            AdviceEvent(
                event_id=f"{selected_run_id}-advice",
                run_id=selected_run_id,
                task_id=case.task.id,
                planned_action=planned,
                advice=advice_result.advice,
                rendered_advice=advice_result.advice.recovery_summary,
                issued_at=now,
            )
        )
    workspace = tmp_path / f"{condition.value}-{selected_run_id}-workspace"
    workspace.mkdir()
    (workspace / "artifact.json").write_text("{}", encoding="utf-8")
    (workspace / "raw_evidence.json").write_text("{}", encoding="utf-8")
    return ExperimentExecution(
        artifact=artifact,
        artifact_path=workspace / "artifact.json",
        raw_evidence_path=workspace / "raw_evidence.json",
        dependencies=dependencies,
        agent_result=None,
        error=error,
        step_database_path=workspace / "steps.sqlite",
    )


def _runtime_artifact() -> RuntimeSmokeArtifact:
    return RuntimeSmokeArtifact(
        path=Path("runtime-smoke-ready.json"),
        task_id=PROVIDER_SMOKE_TASK_ID,
        status="AGENT_RUNTIME_SMOKE_READY",
        provider_calls=0,
        b0_launched=False,
        o1_launched=False,
        environment_fingerprint="fingerprint-1",
    )


def _summary_inputs(
    tmp_path: Path,
    *,
    o1_advice: bool = True,
    o1_review_id: str | None = "GS-R014",
    o1_recovery_pattern: str = PATTERN,
    b0_advice: bool = False,
) -> tuple[
    list[ExperimentExecution],
    LoadedExperimentConfiguration,
    IsolatedTaskEnvironment,
    BenchmarkTaskCase,
]:
    executions = [
        _execution(tmp_path, ExperimentCondition.B0, advice=b0_advice, run_id="b0-fresh"),
        _execution(
            tmp_path,
            ExperimentCondition.O1,
            advice=o1_advice,
            review_id=o1_review_id,
            recovery_pattern=o1_recovery_pattern,
            run_id="o1-fresh",
        ),
    ]
    configuration = load_experiment_configuration(CONFIG, project_root=ROOT)
    environment = IsolatedTaskEnvironment(
        task_id=PROVIDER_SMOKE_TASK_ID,
        python_executable=Path("docker"),
        environment_fingerprint="fingerprint-1",
        runtime_type="docker",
        container_image="prepared:image",
        container_python_executable="/opt/python",
    )
    return (
        executions,
        configuration,
        environment,
        _task_case(),
    )


def test_summary_accepts_one_b0_and_one_o1_without_requiring_task_success(
    tmp_path: Path,
) -> None:
    executions, configuration, environment, case = _summary_inputs(tmp_path)
    summary = build_provider_smoke_summary(
        executions,
        objective_observations=[
            SimpleNamespace(condition="B0", status="test_failure", return_code=1),
            SimpleNamespace(condition="O1", status="test_failure", return_code=1),
        ],
        runtime_artifact=_runtime_artifact(),
        configuration=configuration,
        environment=environment,
        task=case,
        expected_oracle_review_id="GS-R014",
        expected_recovery_pattern=PATTERN,
    )

    assert summary["status"] == "PROVIDER_B0_O1_SMOKE_COMPLETE"
    assert summary["b0_runs"] == 1
    assert summary["o1_runs"] == 1
    assert summary["t_runs"] == 0
    assert summary["gate_b1_evaluated"] is False
    assert summary["b0_oracle_advice_count"] == 0
    assert summary["o1_oracle_advice_count"] == 1
    assert summary["b0_run_id"] != summary["o1_run_id"]
    assert summary["o1_oracle_review_id"] == "GS-R014"
    assert summary["o1_oracle_intervention"]["recovery_summary"] == PATTERN
    assert summary["prompt_config_identity"]["prompt_version"] == "v1"
    assert summary["environment_fingerprint"] == "fingerprint-1"


def test_frozen_t007_mapping_retains_gs_r014_and_advice_serialization_is_unchanged() -> None:
    resolver = FrozenOracleResolver.from_frozen_files(PILOT_MANIFEST, PILOT_ANNOTATIONS)

    assert resolver.review_id_for(PROVIDER_SMOKE_TASK_ID) == "GS-R014"
    assert resolver.recovery_pattern_for(PROVIDER_SMOKE_TASK_ID)
    advice = resolver.evaluate_action(
        _task_case().task,
        PlannedAction(
            id="planned",
            run_id="run",
            task_id=PROVIDER_SMOKE_TASK_ID,
            tool="run_tests",
            operation="run_tests",
            planned_at=datetime.now(UTC),
        ),
        EnvironmentContext(
            id="environment",
            repository="offline-repository",
            runtime="python",
        ),
    )

    assert "review_id" not in advice.model_dump(mode="json")
    assert advice.matched_failure_episode_id == "oracle-failure-GS-T007"

    legacy = FrozenOracleResolver({"GS-T007": PATTERN})
    assert legacy.review_id_for(PROVIDER_SMOKE_TASK_ID) is None
    assert legacy.recovery_pattern_for(PROVIDER_SMOKE_TASK_ID) == PATTERN


def test_wrong_frozen_review_id_fails_before_provider_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import graph_swarm.research.provider_smoke as provider_smoke

    configuration = load_experiment_configuration(CONFIG, project_root=ROOT)
    environment = IsolatedTaskEnvironment(
        task_id=PROVIDER_SMOKE_TASK_ID,
        python_executable=Path("docker"),
        environment_fingerprint="fingerprint-1",
        runtime_type="docker",
        container_image="prepared:image",
        container_python_executable="/opt/python",
    )

    class WrongReviewResolver(FrozenOracleResolver):
        def __init__(self) -> None:
            pass

        def review_id_for(self, task_id: str) -> str:
            return "GS-R015"

        def recovery_pattern_for(self, task_id: str) -> str:
            return PATTERN

    def fake_find_runtime_smoke_artifact(
        artifact_root: Path,
        *,
        task_id: str,
    ) -> RuntimeSmokeArtifact:
        return _runtime_artifact()

    def fake_prepare_provider_smoke_context(
        *,
        project_root: Path,
        baseline_root: Path,
        execution_root: Path,
        config_path: Path | None = None,
        task_id: str = PROVIDER_SMOKE_TASK_ID,
    ) -> PreparedProviderSmokeContext:
        frozen_case = FrozenSWEsmithCase(
            instance_id="GS-T007",
            fail_to_pass=(),
            patch="frozen patch",
        )
        frozen_cases = {task_id: frozen_case}
        return PreparedProviderSmokeContext(
            configuration=configuration,
            case=_task_case(),
            frozen_cases=frozen_cases,
            environment=environment,
            objective=FrozenSWEsmithObjective(frozen_cases, {task_id: environment}),
            recurrence_evaluator=make_recurrence_matcher(frozen_cases),
            baseline_root=baseline_root,
            execution_root=execution_root,
        )

    @classmethod
    def fake_from_frozen_files(
        cls: type[FrozenOracleResolver],
        manifest_path: Path,
        annotations_path: Path,
    ) -> WrongReviewResolver:
        return WrongReviewResolver()

    runner_called = False

    def unexpected_runner(*_args: object, **_kwargs: object) -> object:
        nonlocal runner_called
        runner_called = True
        raise AssertionError("provider runner should not be constructed")

    monkeypatch.setattr(
        provider_smoke,
        "find_runtime_smoke_artifact",
        fake_find_runtime_smoke_artifact,
    )
    monkeypatch.setattr(
        provider_smoke,
        "prepare_provider_smoke_context",
        fake_prepare_provider_smoke_context,
    )
    monkeypatch.setattr(
        provider_smoke.FrozenOracleResolver,
        "from_frozen_files",
        fake_from_frozen_files,
    )
    monkeypatch.setattr(provider_smoke, "ExperimentRunner", unexpected_runner)

    with pytest.raises(ProviderSmokePrerequisiteError, match="GS-R014"):
        provider_smoke.run_provider_smoke(
            project_root=ROOT,
            baseline_root=tmp_path / "baseline",
            execution_root=tmp_path / "execution",
            artifact_root=tmp_path,
        )
    assert runner_called is False


def test_wrong_delivered_recovery_pattern_fails_summary(tmp_path: Path) -> None:
    executions, configuration, environment, case = _summary_inputs(
        tmp_path,
        o1_recovery_pattern="different recovery pattern",
    )

    summary = build_provider_smoke_summary(
        executions,
        objective_observations=[],
        runtime_artifact=_runtime_artifact(),
        configuration=configuration,
        environment=environment,
        task=case,
        expected_oracle_review_id="GS-R014",
        expected_recovery_pattern=PATTERN,
    )

    assert summary["status"] == "PROVIDER_B0_O1_SMOKE_FAILED"
    assert "O1 delivered recovery pattern does not match frozen Oracle recovery pattern" in summary[
        "failures"
    ]


def test_missing_o1_advice_fails_summary(tmp_path: Path) -> None:
    executions, configuration, environment, case = _summary_inputs(tmp_path, o1_advice=False)

    summary = build_provider_smoke_summary(
        executions,
        objective_observations=[],
        runtime_artifact=_runtime_artifact(),
        configuration=configuration,
        environment=environment,
        task=case,
        expected_oracle_review_id="GS-R014",
        expected_recovery_pattern=PATTERN,
    )

    assert summary["status"] == "PROVIDER_B0_O1_SMOKE_FAILED"
    assert "O1 did not receive the expected frozen Oracle intervention" in summary["failures"]


def test_b0_advice_fails_summary(tmp_path: Path) -> None:
    executions, configuration, environment, case = _summary_inputs(tmp_path, b0_advice=True)

    summary = build_provider_smoke_summary(
        executions,
        objective_observations=[],
        runtime_artifact=_runtime_artifact(),
        configuration=configuration,
        environment=environment,
        task=case,
        expected_oracle_review_id="GS-R014",
        expected_recovery_pattern=PATTERN,
    )

    assert summary["status"] == "PROVIDER_B0_O1_SMOKE_FAILED"
    assert "B0 received Oracle advice" in summary["failures"]


def test_summary_records_provider_failure_without_replacement_or_retry(tmp_path: Path) -> None:
    executions, configuration, environment, case = _summary_inputs(tmp_path)
    executions[1] = _execution(
        tmp_path,
        ExperimentCondition.O1,
        error=RuntimeError("provider rejected request"),
        run_id="o1-failed",
    )
    summary = build_provider_smoke_summary(
        executions,
        objective_observations=[],
        runtime_artifact=_runtime_artifact(),
        configuration=configuration,
        environment=environment,
        task=case,
        expected_oracle_review_id="GS-R014",
        expected_recovery_pattern=PATTERN,
    )

    assert summary["status"] == "PROVIDER_B0_O1_SMOKE_FAILED"
    assert summary["o1_run_id"] == "o1-failed"
    assert summary["o1_retries"] == 0
    assert summary["error_classification"]["O1"]["classification"] == "infrastructure"
    assert summary["gate_b1_evaluated"] is False


def test_missing_runtime_smoke_prerequisite_fails_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ProviderSmokePrerequisiteError):
        find_runtime_smoke_artifact(tmp_path, task_id=PROVIDER_SMOKE_TASK_ID)


def test_unvalidated_preparation_fails_before_provider_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import graph_swarm.research.provider_smoke as provider_smoke

    runtime_root = tmp_path / "GS-E003" / "sprint3b"
    runtime_root.mkdir(parents=True)
    (runtime_root / "runtime-smoke-ready.json").write_text(
        json.dumps(
            {
                "task_id": PROVIDER_SMOKE_TASK_ID,
                "status": "AGENT_RUNTIME_SMOKE_READY",
                "provider_calls": 0,
                "b0_launched": False,
                "o1_launched": False,
                "environment_fingerprint": "fingerprint-1",
            }
        ),
        encoding="utf-8",
    )
    called = False

    def reject_unvalidated(**_kwargs: object) -> object:
        nonlocal called
        called = True
        raise BenchmarkPreflightError("environment is unvalidated")

    monkeypatch.setattr(provider_smoke, "prepare_provider_smoke_context", reject_unvalidated)
    with pytest.raises(ProviderSmokePrerequisiteError, match="unvalidated"):
        provider_smoke.run_provider_smoke(
            project_root=ROOT,
            baseline_root=tmp_path / "baseline",
            execution_root=tmp_path / "execution",
            artifact_root=tmp_path,
        )
    assert called is True


def test_provider_smoke_summary_is_append_only(tmp_path: Path) -> None:
    payload = {"status": "PROVIDER_B0_O1_SMOKE_FAILED", "gate_b1_evaluated": False}
    first = write_provider_smoke_summary(tmp_path, payload)
    second = write_provider_smoke_summary(tmp_path, payload)

    assert first != second
    assert first.is_file()
    assert second.is_file()


def test_docker_execution_runtime_is_carried_into_agent_dependencies(tmp_path: Path) -> None:
    runtime = ExecutionRuntime(
        runtime_type="docker",
        docker_executable=Path("docker"),
        container_image="prepared:image",
        container_python_executable="/opt/python",
    )
    dependencies = AgentDependencies(
        tmp_path,
        "run-docker",
        PROVIDER_SMOKE_TASK_ID,
        execution_runtime=runtime,
    )

    assert dependencies.execution_runtime is runtime
    selected_runtime = dependencies.execution_runtime
    assert selected_runtime is not None
    assert selected_runtime.runtime_type == "docker"
