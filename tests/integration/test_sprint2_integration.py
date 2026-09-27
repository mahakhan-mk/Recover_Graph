"""Opt-in, read-only Integration Point 1 checks for the real advisory path."""

import dataclasses
import os
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.tasks import Task
from graph_swarm.integration.advisory_runtime import (
    R13B_TREATMENT_PATTERN_IDS,
    Neo4jAdvisoryRuntime,
    create_neo4j_advisory_runtime,
)
from graph_swarm.integration.sprint2_smoke import (
    R13B_CANONICAL_PATTERN_IDS,
    chronology_audit,
    preflight_r13b_corpus,
)
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    ExperimentRunArtifactStore,
    ExperimentRunner,
    load_experiment_configuration,
    load_task_cases,
)
from graph_swarm.settings import get_settings

RUN_SMOKE = os.getenv("GRAPH_SWARM_RUN_SPRINT2_NEO4J_SMOKE") == "1"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_SMOKE,
        reason="Set GRAPH_SWARM_RUN_SPRINT2_NEO4J_SMOKE=1 for the read-only smoke",
    ),
]

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime() -> Generator[Neo4jAdvisoryRuntime, None, None]:
    selected = create_neo4j_advisory_runtime(get_settings())
    try:
        selected.repository.verify_connectivity()
        yield selected
    finally:
        selected.close()


def _diagnostic_task() -> Task:
    source = next(
        case.task
        for case in load_task_cases(
            ROOT / "benchmark/manifests/pilot.jsonl",
            problem_statements_path=ROOT / "benchmark/annotations/recurrence_validation.csv",
        )
        if case.task.id == "GS-T006"
    )
    return source.model_copy(
        update={
            "id": "integration-point-1-diagnostic",
            "family_id": "evaluation-only-family-metadata",
        }
    )


def _environment(task: Task, run_id: str) -> EnvironmentContext:
    return EnvironmentContext(
        id=f"{run_id}-environment",
        repository=task.repository,
        runtime="docker",
        versions={"python": "3.12.1"},
    )


def _planned_action(task: Task, run_id: str) -> PlannedAction:
    return PlannedAction(
        id=f"{run_id}-action",
        run_id=run_id,
        task_id=task.id,
        tool="edit_file",
        operation="edit_file",
        arguments={"path": "README.md"},
        planned_at=datetime.now(UTC),
    )


def test_r13b_corpus_and_vector_index_are_readable(runtime: Neo4jAdvisoryRuntime) -> None:
    graph_count_before = runtime.repository.count_recovery_pattern_vectors()
    preflight = preflight_r13b_corpus(runtime.repository)
    graph_count_after = runtime.repository.count_recovery_pattern_vectors()
    treatment_preflight = preflight_r13b_corpus(runtime.treatment_repository)

    assert preflight.vector_index == "recovery_pattern_embedding_idx"
    assert graph_count_before == graph_count_after == preflight.embedded_pattern_count
    assert preflight.queryable_candidate_count == preflight.embedded_pattern_count
    assert treatment_preflight.embedded_pattern_count == 5
    assert treatment_preflight.queryable_candidate_count == 5
    assert {item.task_id for item in preflight.canonical_patterns} == set(
        R13B_CANONICAL_PATTERN_IDS
    )
    assert all(
        item.verification_status == "observed_successful"
        for item in preflight.canonical_patterns
    )
    assert [
        item.source_chronological_index for item in preflight.canonical_patterns
    ] == [1, 2, 3, 4, 5]
    assert all(item.embedding_dimension == 384 for item in preflight.canonical_patterns)


def test_real_retrieval_is_strictly_chronological_and_exposes_advice(
    runtime: Neo4jAdvisoryRuntime,
) -> None:
    task = _diagnostic_task()
    environment = _environment(task, "sprint2-direct")
    action = _planned_action(task, "sprint2-direct")

    advice = runtime.advisory_service.evaluate_action(task, action, environment)
    retrieval = runtime.advisory_service.last_retrieval_result
    assert retrieval is not None
    assert advice.advice is not None
    assert retrieval.selected_pattern is not None
    assert all(
        candidate.pattern_id in R13B_TREATMENT_PATTERN_IDS
        for candidate in retrieval.candidates
    )
    assert task.family_id not in retrieval.query_text
    assert retrieval.selected_pattern.source_chronological_index < task.chronological_index

    earlier_task = task.model_copy(
        update={"id": "integration-point-1-chronology", "chronological_index": 3}
    )
    runtime.advisory_service.evaluate_action(
        earlier_task,
        _planned_action(earlier_task, "sprint2-chronology"),
        _environment(earlier_task, "sprint2-chronology"),
    )
    earlier_retrieval = runtime.advisory_service.last_retrieval_result
    assert earlier_retrieval is not None
    lineages = {
        candidate.pattern_id: runtime.repository.get_recovery_pattern(candidate.pattern_id)
        for candidate in earlier_retrieval.candidates
    }
    audit = chronology_audit(
        earlier_retrieval,
        task_chronological_index=3,
        pattern_lineages=lineages,
    )
    assert all(index >= 3 for index in audit["not_strictly_historical"])


def test_real_advisory_runs_through_t_runner_before_tool_and_records_event(
    tmp_path: Path,
    runtime: Neo4jAdvisoryRuntime,
) -> None:
    task = _diagnostic_task()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("before\n", encoding="utf-8")
    model_calls = 0

    def model_response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "edit_file",
                        {
                            "path": "README.md",
                            "old_text": "before\n",
                            "new_text": "after\n",
                            "expected_replacements": 1,
                        },
                    )
                ]
            )
        if model_calls == 2:
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": "README.md"})])
        return ModelResponse(parts=[TextPart("complete")])

    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/rollout_3a_pilot.yaml",
        project_root=ROOT,
    )
    configuration = dataclasses.replace(
        configuration,
        config=configuration.config.model_copy(update={"conditions": (ExperimentCondition.T,)}),
    )
    execution = ExperimentRunner(
        configuration,
        settings=get_settings(),
        model=FunctionModel(model_response),
        workspace_resolver=lambda _task: workspace,
        environment_resolver=lambda current_task, run_id: _environment(current_task, run_id),
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
        advisory_service=runtime.advisory_service,
        condition=ExperimentCondition.T,
    ).run_task(task)

    assert execution.error is None
    assert execution.artifact.advice_count >= 1
    assert execution.dependencies.advice_events
    assert execution.dependencies.events
    assert execution.dependencies.advice_events[0].issued_at <= (
        execution.dependencies.events[0].result.completed_at
    )
    assert execution.dependencies.advisory_retrieval_evidence[0].pattern_id is not None
    assert (
        execution.dependencies.advisory_retrieval_evidence[0].pattern_id
        in R13B_TREATMENT_PATTERN_IDS
    )
    assert execution.dependencies.advisory_retrieval_evidence[0].vector_score is not None
    assert execution.artifact.advice_accepted is True
    assert task.family_id not in execution.raw_evidence_path.read_text(encoding="utf-8")


def test_b0_does_not_call_a_real_advisory_service(
    tmp_path: Path,
    runtime: Neo4jAdvisoryRuntime,
) -> None:
    task = _diagnostic_task()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = 0
    original = runtime.advisory_service.evaluate_action

    def counted(
        task_arg: Task,
        action_arg: PlannedAction,
        environment_arg: EnvironmentContext,
    ):
        nonlocal calls
        calls += 1
        return original(task_arg, action_arg, environment_arg)

    runtime.advisory_service.evaluate_action = counted  # type: ignore[method-assign]
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/rollout_3a_pilot.yaml",
        project_root=ROOT,
    )
    configuration = dataclasses.replace(
        configuration,
        config=configuration.config.model_copy(update={"conditions": (ExperimentCondition.B0,)}),
    )
    execution = ExperimentRunner(
        configuration,
        settings=get_settings(),
        model=FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("complete")])),
        workspace_resolver=lambda _task: workspace,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
        advisory_service=runtime.advisory_service,
        condition=ExperimentCondition.B0,
    ).run_task(task)

    assert calls == 0
    assert execution.dependencies.advisory_service is None
    assert execution.dependencies.advisory_retrieval_evidence == []
