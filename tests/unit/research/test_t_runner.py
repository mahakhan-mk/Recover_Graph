from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import (
    AdviceResult,
    ApplicabilityAssessment,
    RecoveryEvidence,
    RecoveryProvenance,
)
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.resolutions import ResolutionStatus
from graph_swarm.domain.tasks import Task
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    ExperimentRunArtifactStore,
    ExperimentRunner,
    LoadedExperimentConfiguration,
    load_experiment_configuration,
)
from graph_swarm.retrieval.query import recovery_retrieval_query_text
from graph_swarm.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
PILOT_CONFIG = ROOT / "configs" / "experiments" / "rollout_3a_pilot.yaml"


def make_settings() -> Settings:
    return Settings(
        neo4j_uri="neo4j+s://offline.example",
        neo4j_username="offline-user",
        neo4j_password="offline-password",
        neo4j_database="offline-db",
        agent_request_limit=4,
    )


def make_task() -> Task:
    return Task(
        id="GS-T006",
        problem_statement="Fix only the current task.",
        family_id="FORBIDDEN-FAMILY-METADATA",
        repository="offline-repository",
        chronological_index=6,
    )


def treatment_configuration(
    condition: ExperimentCondition = ExperimentCondition.T,
) -> LoadedExperimentConfiguration:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    return dataclasses.replace(
        configuration,
        config=configuration.config.model_copy(update={"conditions": (condition,)}),
    )


def advice_result(label: str = "earlier-task") -> AdviceResult:
    return AdviceResult.historical_recovery(
        matched_failure_episode_id=f"incident-{label}",
        matched_resolution_id=f"resolution-{label}",
        failed_tool="run_tests",
        failed_operation="run_tests",
        recovery_summary="Use the validated earlier recovery ordering.",
        resolution_status=ResolutionStatus.OBSERVED_SUCCESSFUL,
        recovery_evidence=RecoveryEvidence(successful_observations=1, failed_observations=0),
        applicability=ApplicabilityAssessment(
            matched_fields=("tool", "operation", "runtime"),
            repository="offline-repository",
            runtime="python",
        ),
        provenance=RecoveryProvenance(
            failure_episode_id=f"incident-{label}",
            resolution_id=f"resolution-{label}",
            failed_action_id=f"{label}-action",
            environment_id=f"{label}-environment",
        ),
    )


class FakeRetrievalResult:
    def __init__(self, pattern_id: str, score: float) -> None:
        self.selected_pattern = SimpleNamespace(id=pattern_id)
        self.selected_vector_score = score

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "query_version": "offline-test",
            "selected_pattern": {"id": self.selected_pattern.id},
            "selected_vector_score": self.selected_vector_score,
        }


class FakeAdvisoryService:
    def __init__(
        self,
        *,
        fail: bool = False,
        advice_results: tuple[AdviceResult, ...] = (),
        retrieval_results: tuple[FakeRetrievalResult | None, ...] = (),
    ) -> None:
        self.calls: list[tuple[Task, PlannedAction, EnvironmentContext]] = []
        self.fail = fail
        self._advice_results = advice_results
        self._retrieval_results = retrieval_results
        self._evaluation_count = 0
        self.last_retrieval_result: FakeRetrievalResult | None = None

    def evaluate_action(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
    ) -> AdviceResult:
        self.calls.append((task, planned_action, environment))
        if self.fail:
            raise RuntimeError("offline advisory unavailable")
        index = self._evaluation_count
        self._evaluation_count += 1
        if index < len(self._retrieval_results):
            retrieval = self._retrieval_results[index]
            if retrieval is not None:
                self.last_retrieval_result = retrieval
        if index < len(self._advice_results):
            return self._advice_results[index]
        return advice_result()


def build_runner(
    tmp_path: Path,
    configuration: LoadedExperimentConfiguration,
    service: FakeAdvisoryService | None = None,
    model: FunctionModel | TestModel | None = None,
) -> ExperimentRunner:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return ExperimentRunner(
        configuration,
        settings=make_settings(),
        model=model or TestModel(call_tools=["run_tests"], custom_output_text="complete"),
        workspace_resolver=lambda _task: workspace,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
        advisory_service=service,
        condition=configuration.config.conditions[0],
    )


def test_t_uses_injected_service_through_pre_tool_path_and_records_evidence(
    tmp_path: Path,
) -> None:
    service = FakeAdvisoryService()
    runner = build_runner(
        tmp_path,
        treatment_configuration(),
        service,
        retrying_tool_model(),
    )

    execution = runner.run_case(BenchmarkTaskCase(make_task(), occurrence_index=2))

    assert runner.condition is ExperimentCondition.T
    assert len(service.calls) >= 1
    advisory_task, planned_action, environment = service.calls[0]
    assert advisory_task.id == "GS-T006"
    assert advisory_task.problem_statement == "Fix only the current task."
    assert advisory_task.chronological_index == 6
    assert advisory_task.family_id == "FORBIDDEN-FAMILY-METADATA"
    retrieval_query = recovery_retrieval_query_text(
        advisory_task,
        planned_action,
        environment,
    )
    assert "FORBIDDEN-FAMILY-METADATA" not in retrieval_query
    assert execution.agent_result is not None
    assert "FORBIDDEN-FAMILY-METADATA" not in execution.agent_result.all_messages_json().decode()
    assert planned_action.tool == "run_tests"
    assert planned_action.task_id == "GS-T006"
    assert environment.repository == "offline-repository"
    assert environment.runtime == "python"

    assert execution.dependencies.advisory_service is service
    assert execution.artifact.advice_count == 1
    assert execution.artifact.advice_received is not None
    assert execution.artifact.retrieved_incident_id == "incident-earlier-task"
    assert execution.artifact.advice_intervention_boundary == "pre_tool"
    assert execution.artifact.advice_delivery_timing == "before_tool_execution"
    assert execution.artifact.advice_accepted is False
    assert execution.dependencies.behavior_evidence[0].actual_action_after_advice is not None

    raw = json.loads(execution.raw_evidence_path.read_text(encoding="utf-8"))
    assert raw["condition"] == "T"
    assert raw["advice_source"] == "T"
    assert raw["advice_fired"] is True
    assert raw["advice_payload"]["advice"]["applicability"]["runtime"] == "python"
    assert raw["advice_payload"]["advice"]["provenance"]["resolution_id"] == (
        "resolution-earlier-task"
    )
    assert len(raw["advisory_retrieval_evidence"]) == 1
    assert raw["advisory_retrieval_evidence"][0]["advice_event_id"] == (
        execution.dependencies.advice_events[0].event_id
    )
    assert raw["behavior_evidence"][0]["actual_action_after_advice"] is not None
    assert raw["objective_result"] is True


def test_b0_ignores_injected_persistent_service(tmp_path: Path) -> None:
    service = FakeAdvisoryService()
    runner = build_runner(tmp_path, treatment_configuration(ExperimentCondition.B0), service)

    execution = runner.run_task(make_task())

    assert service.calls == []
    assert execution.dependencies.advisory_service is None
    assert execution.dependencies.environment is None
    assert execution.dependencies.advice_events == []
    assert execution.artifact.advice_received is None


def test_t_advisory_failure_is_fail_open_and_is_recorded(tmp_path: Path) -> None:
    service = FakeAdvisoryService(fail=True)
    runner = build_runner(
        tmp_path,
        treatment_configuration(),
        service,
        retrying_tool_model(tool_calls=1),
    )

    execution = runner.run_task(make_task())

    assert len(service.calls) >= 1
    assert execution.error is None
    assert execution.dependencies.advice_events == []
    assert execution.dependencies.advisory_errors == [
        "lookup failed for run_tests: offline advisory unavailable"
    ]
    assert execution.artifact.advice_received is None
    assert execution.artifact.tool_calls == 1
    raw = json.loads(execution.raw_evidence_path.read_text(encoding="utf-8"))
    assert raw["advice_fired"] is False
    assert raw["advisory_errors"] == execution.dependencies.advisory_errors


def test_t_associates_each_advice_event_with_its_lookup_snapshot(tmp_path: Path) -> None:
    service = FakeAdvisoryService(
        advice_results=(
            advice_result("first"),
            advice_result("second"),
            AdviceResult.no_advice("done"),
        ),
        retrieval_results=(
            FakeRetrievalResult("pattern-first", 0.71),
            FakeRetrievalResult("pattern-second", 0.93),
        ),
    )
    runner = build_runner(
        tmp_path,
        treatment_configuration(),
        service,
        retrying_tool_model(tool_calls=3),
    )

    execution = runner.run_task(make_task())

    assert len(execution.dependencies.advice_events) == 2
    captured = execution.dependencies.advisory_retrieval_evidence
    assert [(item.pattern_id, item.vector_score) for item in captured] == [
        ("pattern-first", 0.71),
        ("pattern-second", 0.93),
    ]
    assert [item.action_id for item in captured] == [
        event.planned_action.id for event in execution.dependencies.advice_events
    ]
    assert [item.advice_event_id for item in captured] == [
        event.event_id for event in execution.dependencies.advice_events
    ]
    selected_pattern_ids: list[str] = []
    for item in captured:
        assert item.retrieval is not None
        selected_pattern = item.retrieval["selected_pattern"]
        assert isinstance(selected_pattern, dict)
        pattern_id = cast(str, selected_pattern["id"])
        assert isinstance(pattern_id, str)
        selected_pattern_ids.append(pattern_id)
    assert selected_pattern_ids == ["pattern-first", "pattern-second"]

    raw = json.loads(execution.raw_evidence_path.read_text(encoding="utf-8"))
    assert [
        (item["pattern_id"], item["vector_score"])
        for item in raw["advisory_retrieval_evidence"]
    ] == [("pattern-first", 0.71), ("pattern-second", 0.93)]


def test_t_advice_followed_by_changed_behavior_sets_acceptance(tmp_path: Path) -> None:
    service = FakeAdvisoryService(
        advice_results=(
            advice_result("changed"),
            AdviceResult.no_advice("execute changed action"),
        ),
    )
    runner = build_runner(
        tmp_path,
        treatment_configuration(),
        service,
        tool_sequence_model((("run_tests", {}), ("read_file", {"path": "missing.txt"}))),
    )

    execution = runner.run_task(make_task())

    assert execution.artifact.advice_accepted is True
    assert [item.behavior_changed for item in execution.dependencies.behavior_evidence] == [
        True
    ]


def test_t_advice_with_no_behavior_change_remains_unaccepted(tmp_path: Path) -> None:
    service = FakeAdvisoryService(
        advice_results=(
            advice_result("unchanged"),
            AdviceResult.no_advice("execute same action"),
        ),
    )
    runner = build_runner(
        tmp_path,
        treatment_configuration(),
        service,
        retrying_tool_model(tool_calls=2),
    )

    execution = runner.run_task(make_task())

    assert execution.artifact.advice_accepted is False
    assert [item.behavior_changed for item in execution.dependencies.behavior_evidence] == [
        False
    ]


def test_t_multiple_advice_events_aggregate_acceptance_deterministically(
    tmp_path: Path,
) -> None:
    service = FakeAdvisoryService(
        advice_results=(
            advice_result("same-action"),
            AdviceResult.no_advice("execute same action"),
            advice_result("changed-action"),
            AdviceResult.no_advice("execute changed action"),
        ),
        retrieval_results=(None, None, None, None),
    )
    runner = build_runner(
        tmp_path,
        treatment_configuration(),
        service,
        tool_sequence_model(
            (
                ("run_tests", {}),
                ("run_tests", {}),
                ("run_tests", {}),
                ("read_file", {"path": "missing.txt"}),
            )
        ),
    )

    execution = runner.run_task(make_task())

    assert len(execution.dependencies.advice_events) == 2
    assert [item.behavior_changed for item in execution.dependencies.behavior_evidence] == [
        False,
        True,
    ]
    assert execution.artifact.advice_accepted is True


def test_t_needs_an_injected_service_but_does_not_need_neo4j(tmp_path: Path) -> None:
    configuration = treatment_configuration()

    try:
        build_runner(tmp_path, configuration)
    except ValueError as error:
        assert "injected persistent advisory service" in str(error)
    else:
        raise AssertionError("T runner accepted a missing advisory service")


def test_t_dependency_shape_stays_within_runtime_inputs(tmp_path: Path) -> None:
    service = FakeAdvisoryService()
    runner = build_runner(tmp_path, treatment_configuration(), service)
    execution = runner.run_task(make_task())

    dependencies = execution.dependencies
    assert isinstance(dependencies, AgentDependencies)
    assert dependencies.task is not None
    assert dependencies.task.model_dump() == make_task().model_dump()


def retrying_tool_model(*, tool_calls: int = 2) -> FunctionModel:
    tools: tuple[tuple[str, dict[str, object]], ...] = tuple(
        ("run_tests", {}) for _ in range(tool_calls)
    )
    return tool_sequence_model(tools)


def tool_sequence_model(
    tools: tuple[tuple[str, dict[str, object]], ...],
) -> FunctionModel:
    calls = 0

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls <= len(tools):
            tool, arguments = tools[calls - 1]
            return ModelResponse(parts=[ToolCallPart(tool, arguments)])
        return ModelResponse(parts=[TextPart("complete")])

    return FunctionModel(respond)
