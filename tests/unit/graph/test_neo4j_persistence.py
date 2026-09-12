from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import Mock, patch

import pytest

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph import queries
from graph_swarm.graph.neo4j_repository import (
    EntityNotFoundError,
    Neo4jRepository,
    _read_recovery_pattern,  # pyright: ignore[reportPrivateUsage]
)

STARTED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
COMPLETED_AT = STARTED_AT + timedelta(seconds=1)


class FakeResult:
    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self.records = records


def make_repository() -> Neo4jRepository:
    with patch("graph_swarm.graph.neo4j_repository.GraphDatabase.driver", return_value=Mock()):
        return Neo4jRepository("uri", "username", "password", "database")


def make_action_pair() -> tuple[PlannedAction, ActionResult]:
    return (
        PlannedAction(
            id="action-001",
            run_id="run-001",
            task_id="task-001",
            tool="write_file",
            operation="write_file",
            arguments={"paths": ["tests"], "options": {"quiet": True}},
            planned_at=STARTED_AT,
        ),
        ActionResult(
            action_id="action-001",
            tool_name="write_file",
            success=True,
            exit_code=0,
            output="passed",
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
        ),
    )


def test_entity_writes_use_canonical_merge_identity_and_parameters() -> None:
    repository = make_repository()
    action, result = make_action_pair()
    entities = (
        (
            repository.save_run,
            Run(id="run-001", task_id="task-001", started_at=STARTED_AT),
            queries.SAVE_RUN,
            "run-001",
        ),
        (
            repository.save_task,
            Task(
                id="task-001",
                problem_statement="Fix the example repository so its tests pass.",
                family_id="family-001",
                repository="example/repository",
                chronological_index=1,
            ),
            queries.SAVE_TASK,
            "task-001",
        ),
        (repository.save_tool, Tool(name="run_tests"), queries.SAVE_TOOL, "run_tests"),
        (
            repository.save_environment,
            EnvironmentContext(
                id="environment-001",
                repository="example/repository",
                runtime="python-3.13",
            ),
            queries.SAVE_ENVIRONMENT,
            "environment-001",
        ),
        (
            repository.save_failure,
            FailureEpisode(
                id="failure-001",
                action_id="action-001",
                failure_type=FailureType.TEST_FAILURE,
                signature="signature",
                symptom="symptom",
                observed_at=STARTED_AT,
            ),
            queries.SAVE_FAILURE,
            "failure-001",
        ),
        (
            repository.save_resolution,
            Resolution(
                id="resolution-001",
                failure_id="failure-001",
                description="fix",
                status=ResolutionStatus.CANDIDATE,
            ),
            queries.SAVE_RESOLUTION,
            "resolution-001",
        ),
        (
            repository.save_outcome,
            Outcome(
                id="outcome-001",
                action_id="action-001",
                success=True,
                observed_at=COMPLETED_AT,
            ),
            queries.SAVE_OUTCOME,
            "outcome-001",
        ),
        (
            repository.save_recovery_pattern,
            make_recovery_pattern(),
            queries.SAVE_RECOVERY_PATTERN,
            "pattern-001",
        ),
    )

    for save_method, entity, expected_query, identity in entities:
        with patch.object(repository, "execute_query") as execute_query:
            cast(Callable[[object], None], save_method)(entity)

        query = execute_query.call_args.args[0]
        parameters = execute_query.call_args.kwargs
        assert query == expected_query
        assert identity not in query
        assert identity in parameters.values()
        assert "MERGE" in query

    task_write = entities[1][1]
    with patch.object(repository, "execute_query") as execute_query:
        repository.save_task(task_write)
    assert execute_query.call_args.kwargs["problem_statement"] == (
        "Fix the example repository so its tests pass."
    )

    with patch.object(repository, "execute_query") as execute_query:
        repository.save_action(action, result)

    query = execute_query.call_args.args[0]
    parameters = execute_query.call_args.kwargs
    assert query == queries.SAVE_ACTION
    assert action.id not in query
    assert parameters["id"] == action.id
    assert parameters["arguments_json"] == '{"options":{"quiet":true},"paths":["tests"]}'

    with patch.object(repository, "execute_query") as execute_query:
        repository.save_recovery_pattern(make_recovery_pattern())

    pattern_parameters = execute_query.call_args.kwargs
    assert pattern_parameters["embedding"] is None
    assert pattern_parameters["environment_runtime"] == "python-3.13"
    assert pattern_parameters["environment_versions_json"] == '{"pytest":"8.0"}'
    assert pattern_parameters["environment_dependencies_json"] == '{"package":"1.2"}'
    assert pattern_parameters["environment_markers_json"] == '{"platform":"linux"}'


def test_save_action_rejects_mismatched_ids_before_database_write() -> None:
    repository = make_repository()
    action, result = make_action_pair()
    mismatched_result = result.model_copy(update={"action_id": "different-action"})

    with patch.object(repository, "execute_query") as execute_query:
        with pytest.raises(ValueError, match="must match"):
            repository.save_action(action, mismatched_result)

    execute_query.assert_not_called()


def test_relationship_writes_match_endpoints_and_merge_relationships() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"linked": 1}]),
    ) as execute_query:
        repository.link_action_tool("action-001", "run_tests")

    query = execute_query.call_args.args[0]
    assert query == queries.LINK_ACTION_TOOL
    assert "MATCH" in query
    assert "MERGE (action)-[:USED]->(tool)" in query
    assert execute_query.call_args.kwargs == {
        "action_id": "action-001",
        "tool_name": "run_tests",
    }


def test_resolution_observed_change_relationship_is_idempotent_and_typed() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"linked": 1}]),
    ) as execute_query:
        repository.link_resolution_observed_change("resolution-001", "action-001")

    assert execute_query.call_args.args[0] == queries.LINK_RESOLUTION_OBSERVED_CHANGE
    assert "MATCH" in execute_query.call_args.args[0]
    assert "MERGE (resolution)-[:OBSERVED_CHANGE]->(action)" in (
        execute_query.call_args.args[0]
    )
    assert execute_query.call_args.kwargs == {
        "resolution_id": "resolution-001",
        "action_id": "action-001",
    }


def make_recovery_pattern() -> RecoveryPattern:
    return RecoveryPattern(
        id="pattern-001",
        title="Restore the branch condition",
        guidance="Inspect branch polarity and restore the intended condition.",
        source_failure_id="failure-001",
        source_resolution_id="resolution-001",
        source_outcome_id="outcome-001",
        source_task_id="task-001",
        source_chronological_index=1,
        source_tool="run_tests",
        source_operation="pytest",
        source_failure_type="test_failure",
        environment_constraints=EnvironmentConstraints(
            runtime="python-3.13",
            versions={"pytest": "8.0"},
            dependencies={"package": "1.2"},
            markers={"platform": "linux"},
        ),
        verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        evidence_count=1,
        evidence_summary="One observed change preceded one successful outcome.",
        created_at=STARTED_AT,
    )


def test_relationship_write_rejects_missing_endpoint_and_empty_ids() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query", return_value=FakeResult([])):
        with pytest.raises(EntityNotFoundError, match="endpoint"):
            repository.link_action_run("action-001", "missing-run")

    with patch.object(repository, "execute_query") as execute_query:
        with pytest.raises(ValueError, match="action_id"):
            repository.link_action_run("", "run-001")

    execute_query.assert_not_called()


def test_recovery_pattern_persists_native_embedding_and_provenance_atomically() -> None:
    repository = make_repository()
    pattern = make_recovery_pattern().model_copy(update={"embedding": [0.1, 0.2]})

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"provenance_matches": True}]),
    ) as execute_query:
        repository.save_recovery_pattern(pattern)

    query = execute_query.call_args.args[0]
    parameters = execute_query.call_args.kwargs
    assert query == queries.SAVE_RECOVERY_PATTERN
    assert parameters["embedding"] == [0.1, 0.2]
    assert "embedding_json" not in parameters
    assert "MATCH (failure:FailureEpisode" in query
    assert "MATCH (failure)-[:RESOLVED_BY]->(resolution)" in query
    assert "MATCH (resolution)-[:VERIFIED_BY]->(outcome)" in query
    assert "MATCH (resolution)-[:OBSERVED_CHANGE]->(recovery_action:Action)" in query
    assert "MATCH (task)-[:HAS_ACTION]->(recovery_action)" in query
    assert "MERGE (pattern)-[:SOURCE_FAILURE]->(failure)" in query
    assert "MERGE (pattern)-[:SOURCE_RESOLUTION]->(resolution)" in query
    assert "MERGE (pattern)-[:SOURCE_OUTCOME]->(outcome)" in query
    assert "MERGE (pattern)-[:SOURCE_TASK]->(task)" in query


def test_recovery_pattern_save_does_not_create_orphan_when_source_chain_is_missing() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([]),
    ) as execute_query:
        with pytest.raises(EntityNotFoundError, match="historical provenance"):
            repository.save_recovery_pattern(make_recovery_pattern())

    execute_query.assert_called_once_with(
        queries.SAVE_RECOVERY_PATTERN,
        **execute_query.call_args.kwargs,
    )
    assert "MERGE (pattern:RecoveryPattern" in execute_query.call_args.args[0]


def test_recovery_pattern_save_rejects_conflicting_existing_provenance() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"provenance_matches": False}]),
    ) as execute_query:
        with pytest.raises(ValueError, match="conflicting immutable provenance"):
            repository.save_recovery_pattern(make_recovery_pattern())

    execute_query.assert_called_once()


@pytest.mark.parametrize(
    ("missing_requirement", "query_fragment"),
    (
        ("FailureEpisode", "MATCH (failure:FailureEpisode {id: $source_failure_id})"),
        ("Resolution", "MATCH (resolution:Resolution {id: $source_resolution_id})"),
        ("Outcome", "MATCH (outcome:Outcome {id: $source_outcome_id})"),
        ("Task", "MATCH (task:Task {id: $source_task_id})"),
        ("FailureEpisode -> Resolution", "MATCH (failure)-[:RESOLVED_BY]->(resolution)"),
        ("Resolution -> Outcome", "MATCH (resolution)-[:VERIFIED_BY]->(outcome)"),
        (
            "Resolution -> observed change",
            "MATCH (resolution)-[:OBSERVED_CHANGE]->(recovery_action:Action)",
        ),
        ("Task -> historical action", "MATCH (task)-[:HAS_ACTION]->(recovery_action)"),
    ),
)
def test_recovery_pattern_save_requires_every_historical_provenance_requirement(
    missing_requirement: str,
    query_fragment: str,
) -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([]),
    ) as execute_query:
        with pytest.raises(EntityNotFoundError, match="historical provenance"):
            repository.save_recovery_pattern(make_recovery_pattern())

    assert missing_requirement
    assert query_fragment in execute_query.call_args.args[0]


@pytest.mark.parametrize(
    ("field", "value", "query_fragment"),
    (
        (
            "source_chronological_index",
            99,
            "WHERE task.chronological_index = $source_chronological_index",
        ),
        (
            "source_failure_type",
            "syntax_error",
            "AND failure.failure_type = $source_failure_type",
        ),
        (
            "source_tool",
            "write_file",
            "AND failed_action.tool = $source_tool",
        ),
        (
            "source_operation",
            "write_file",
            "AND failed_action.operation = $source_operation",
        ),
    ),
)
def test_recovery_pattern_save_rejects_mismatched_source_metadata(
    field: str,
    value: object,
    query_fragment: str,
) -> None:
    repository = make_repository()
    pattern = make_recovery_pattern().model_copy(update={field: value})

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([]),
    ) as execute_query:
        with pytest.raises(EntityNotFoundError, match="historical provenance"):
            repository.save_recovery_pattern(pattern)

    assert execute_query.call_count == 1
    query = execute_query.call_args.args[0]
    assert query.index(query_fragment) < query.index("MERGE (pattern:RecoveryPattern")


def test_recovery_pattern_save_is_idempotent_for_same_provenance() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"provenance_matches": True}]),
    ) as execute_query:
        repository.save_recovery_pattern(make_recovery_pattern())
        repository.save_recovery_pattern(make_recovery_pattern())

    assert execute_query.call_count == 2
    assert "MERGE (pattern:RecoveryPattern {id: $id})" in (
        execute_query.call_args.args[0]
    )
    assert execute_query.call_args.kwargs["source_failure_id"] == "failure-001"

    with patch.object(repository, "execute_query", return_value=FakeResult([])):
        with pytest.raises(EntityNotFoundError, match="endpoint"):
            repository.link_resolution_observed_change(
                "resolution-001",
                "missing-action",
            )


def test_missing_incident_raises_clear_not_found_error() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query", return_value=FakeResult([])):
        with pytest.raises(EntityNotFoundError, match="was not found"):
            repository.get_incident_lineage("unknown-failure")


def test_missing_recovery_pattern_raises_clear_not_found_error() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query", return_value=FakeResult([])):
        with pytest.raises(EntityNotFoundError, match="RecoveryPattern"):
            repository.get_recovery_pattern("unknown-pattern")


def test_missing_recovery_evidence_raises_without_matching_patterns() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query", return_value=FakeResult([])) as execute_query:
        with pytest.raises(EntityNotFoundError, match="Recovery evidence"):
            repository.get_recovery_evidence("unknown-failure")

    assert execute_query.call_args.args[0] == queries.GET_RECOVERY_EVIDENCE
    assert "RecoveryPattern" not in execute_query.call_args.args[0]


def test_recovery_evidence_readback_reconstructs_complete_pre_pattern_chain() -> None:
    repository = make_repository()
    recovery_action, recovery_result = make_action_pair()
    repository.execute_query = Mock(
        return_value=FakeResult(
            [
                {
                    "failure": {
                        "id": "failure-001",
                        "action_id": "failed-action-001",
                        "failure_type": "test_failure",
                        "signature": "run_tests:exit_code=1",
                        "symptom": "failed",
                        "observed_at": STARTED_AT.isoformat(),
                    },
                    "resolution": {
                        "id": "resolution-001",
                        "failure_id": "failure-001",
                        "description": "Observed concrete change",
                        "status": "observed_successful",
                        "successful_observations": 1,
                        "failed_observations": 0,
                        "observed_at": COMPLETED_AT.isoformat(),
                    },
                    "outcome": {
                        "id": "outcome-001",
                        "action_id": recovery_action.id,
                        "success": True,
                        "tests_passed": 1,
                        "tests_failed": 0,
                        "exit_code": 0,
                        "observed_at": COMPLETED_AT.isoformat(),
                    },
                    "task": {
                        "id": "task-001",
                        "problem_statement": "Fix the repository.",
                        "repository": "example/repository",
                        "chronological_index": 1,
                    },
                    "environment": {
                        "id": "environment-001",
                        "repository": "example/repository",
                        "runtime": "python-3.13",
                        "versions_json": '{"pytest":"8.0"}',
                        "markers_json": "{}",
                    },
                    "failed_action": {
                        "id": "failed-action-001",
                        "run_id": "run-001",
                        "task_id": "task-001",
                        "tool": "run_tests",
                        "operation": "pytest",
                        "arguments_json": '{"paths":["tests"]}',
                        "planned_at": STARTED_AT.isoformat(),
                        "result_tool_name": "run_tests",
                        "success": False,
                        "exit_code": 1,
                        "output": "failed",
                        "error": "tests failed",
                        "started_at": STARTED_AT.isoformat(),
                        "completed_at": COMPLETED_AT.isoformat(),
                    },
                    "recovery_action": {
                        "id": recovery_action.id,
                        "run_id": recovery_action.run_id,
                        "task_id": recovery_action.task_id,
                        "tool": recovery_action.tool,
                        "operation": recovery_action.operation,
                        "arguments_json": '{"options":{"quiet":true},"paths":["tests"]}',
                        "planned_at": STARTED_AT.isoformat(),
                        "result_tool_name": recovery_result.tool_name,
                        "success": recovery_result.success,
                        "exit_code": recovery_result.exit_code,
                        "output": recovery_result.output,
                        "error": recovery_result.error,
                        "started_at": STARTED_AT.isoformat(),
                        "completed_at": COMPLETED_AT.isoformat(),
                    },
                }
            ]
        )
    )

    lineage = repository.get_recovery_evidence("failure-001")

    assert lineage.failure.id == "failure-001"
    assert lineage.resolution.failure_id == lineage.failure.id
    assert lineage.outcome.success is True
    assert lineage.task.id == "task-001"
    assert lineage.failed_action.planned_action.tool == "run_tests"
    assert lineage.recovery_action.planned_action.tool == "write_file"
    assert "RecoveryPattern" not in repository.execute_query.call_args.args[0]


def test_recovery_pattern_readback_reconstructs_typed_source_evidence() -> None:
    repository = make_repository()
    pattern = make_recovery_pattern()
    action, result = make_action_pair()
    repository.execute_query = Mock(
        return_value=FakeResult(
            [
                {
                    "pattern": {
                        "id": pattern.id,
                        "title": pattern.title,
                        "guidance": pattern.guidance,
                        "source_failure_id": pattern.source_failure_id,
                        "source_resolution_id": pattern.source_resolution_id,
                        "source_outcome_id": pattern.source_outcome_id,
                        "source_task_id": pattern.source_task_id,
                        "source_chronological_index": pattern.source_chronological_index,
                        "source_tool": pattern.source_tool,
                        "source_operation": pattern.source_operation,
                        "source_failure_type": pattern.source_failure_type,
                        "environment_runtime": "python-3.13",
                        "environment_versions_json": '{"pytest":"8.0"}',
                        "environment_dependencies_json": '{"package":"1.2"}',
                        "environment_markers_json": '{"platform":"linux"}',
                        "verification_status": pattern.verification_status.value,
                        "evidence_count": pattern.evidence_count,
                        "evidence_summary": pattern.evidence_summary,
                        "embedding": [0.1, 0.2],
                        "created_at": STARTED_AT.isoformat(),
                        "invalidated_at": None,
                    },
                    "failure": {
                        "id": "failure-001",
                        "action_id": "failed-action-001",
                        "failure_type": "test_failure",
                        "signature": "run_tests:exit_code=1",
                        "symptom": "failed",
                        "observed_at": STARTED_AT.isoformat(),
                    },
                    "resolution": {
                        "id": "resolution-001",
                        "failure_id": "failure-001",
                        "description": "Observed concrete change",
                        "status": "observed_successful",
                        "successful_observations": 1,
                        "failed_observations": 0,
                        "observed_at": COMPLETED_AT.isoformat(),
                    },
                    "outcome": {
                        "id": "outcome-001",
                        "action_id": "successful-action-001",
                        "success": True,
                        "tests_passed": 1,
                        "tests_failed": 0,
                        "exit_code": 0,
                        "observed_at": COMPLETED_AT.isoformat(),
                    },
                    "task": {
                        "id": "task-001",
                        "problem_statement": "Fix the repository.",
                        "repository": "example/repository",
                        "chronological_index": 1,
                        "family_id": "must-not-enter-pattern-read-model",
                    },
                    "environment": {
                        "id": "environment-001",
                        "repository": "example/repository",
                        "runtime": "python-3.13",
                        "versions_json": '{"pytest":"8.0"}',
                        "markers_json": "{}",
                    },
                    "failed_action": {
                        "id": "failed-action-001",
                        "run_id": "run-001",
                        "task_id": "task-001",
                        "tool": "run_tests",
                        "operation": "pytest",
                        "arguments_json": "{}",
                        "planned_at": STARTED_AT.isoformat(),
                        "result_tool_name": "run_tests",
                        "success": False,
                        "exit_code": 1,
                        "output": "failed",
                        "error": "tests failed",
                        "started_at": STARTED_AT.isoformat(),
                        "completed_at": COMPLETED_AT.isoformat(),
                    },
                    "recovery_action": {
                        "id": action.id,
                        "run_id": action.run_id,
                        "task_id": action.task_id,
                        "tool": action.tool,
                        "operation": action.operation,
                        "arguments_json": '{"options":{"quiet":true},"paths":["tests"]}',
                        "planned_at": STARTED_AT.isoformat(),
                        "result_tool_name": result.tool_name,
                        "success": result.success,
                        "exit_code": result.exit_code,
                        "output": result.output,
                        "error": result.error,
                        "started_at": STARTED_AT.isoformat(),
                        "completed_at": COMPLETED_AT.isoformat(),
                    },
                }
            ]
        )
    )

    lineage = repository.get_recovery_pattern("pattern-001")

    assert lineage.pattern.embedding == [0.1, 0.2]
    assert lineage.pattern.source_chronological_index == 1
    assert lineage.task.chronological_index == 1
    assert "family_id" not in lineage.task.model_dump()
    assert lineage.recovery_action.planned_action.arguments == action.arguments
    repository.execute_query.assert_called_once_with(
        queries.GET_RECOVERY_PATTERN,
        pattern_id="pattern-001",
    )
    assert "MATCH (failure)-[:RESOLVED_BY]->(resolution)" in (
        repository.execute_query.call_args.args[0]
    )
    assert "MATCH (resolution)-[:VERIFIED_BY]->(outcome)" in (
        repository.execute_query.call_args.args[0]
    )
    assert "MATCH (task)-[:HAS_ACTION]->(recovery_action)" in (
        repository.execute_query.call_args.args[0]
    )
    assert "pattern.source_failure_type = failure.failure_type" in (
        repository.execute_query.call_args.args[0]
    )
    assert "pattern.source_tool = failed_action.tool" in (
        repository.execute_query.call_args.args[0]
    )
    assert "pattern.source_operation = failed_action.operation" in (
        repository.execute_query.call_args.args[0]
    )


def test_recovery_pattern_readback_rejects_json_embedding_property() -> None:
    pattern = make_recovery_pattern()
    properties: dict[str, object] = {
        "id": pattern.id,
        "title": pattern.title,
        "guidance": pattern.guidance,
        "source_failure_id": pattern.source_failure_id,
        "source_resolution_id": pattern.source_resolution_id,
        "source_outcome_id": pattern.source_outcome_id,
        "source_task_id": pattern.source_task_id,
        "source_chronological_index": pattern.source_chronological_index,
        "source_tool": pattern.source_tool,
        "source_operation": pattern.source_operation,
        "source_failure_type": pattern.source_failure_type,
        "environment_runtime": "python-3.13",
        "environment_versions_json": "{}",
        "environment_dependencies_json": "{}",
        "environment_markers_json": "{}",
        "verification_status": pattern.verification_status.value,
        "evidence_count": pattern.evidence_count,
        "evidence_summary": pattern.evidence_summary,
        "embedding": "[0.1,0.2]",
        "created_at": STARTED_AT.isoformat(),
        "invalidated_at": None,
    }

    with pytest.raises(ValueError, match="numeric list"):
        _read_recovery_pattern(properties)


def test_recovery_pattern_readback_rejects_inconsistent_pattern_properties() -> None:
    repository = make_repository()
    pattern = make_recovery_pattern()
    action, result = make_action_pair()
    record = {
        "pattern": {
            "id": pattern.id,
            "title": pattern.title,
            "guidance": pattern.guidance,
            "source_failure_id": "different-failure",
            "source_resolution_id": pattern.source_resolution_id,
            "source_outcome_id": pattern.source_outcome_id,
            "source_task_id": pattern.source_task_id,
            "source_chronological_index": pattern.source_chronological_index,
            "source_tool": pattern.source_tool,
            "source_operation": pattern.source_operation,
            "source_failure_type": pattern.source_failure_type,
            "environment_runtime": "python-3.13",
            "environment_versions_json": "{}",
            "environment_dependencies_json": "{}",
            "environment_markers_json": "{}",
            "verification_status": pattern.verification_status.value,
            "evidence_count": pattern.evidence_count,
            "evidence_summary": pattern.evidence_summary,
            "embedding": None,
            "created_at": STARTED_AT.isoformat(),
            "invalidated_at": None,
        },
        "failure": {
            "id": "failure-001",
            "action_id": "failed-action-001",
            "failure_type": "test_failure",
            "signature": "signature",
            "symptom": "failed",
            "observed_at": STARTED_AT.isoformat(),
        },
        "resolution": {
            "id": "resolution-001",
            "failure_id": "failure-001",
            "description": "Observed concrete change",
            "status": "observed_successful",
            "successful_observations": 1,
            "failed_observations": 0,
            "observed_at": COMPLETED_AT.isoformat(),
        },
        "outcome": {
            "id": "outcome-001",
            "action_id": "successful-action-001",
            "success": True,
            "tests_passed": 1,
            "tests_failed": 0,
            "exit_code": 0,
            "observed_at": COMPLETED_AT.isoformat(),
        },
        "task": {
            "id": "task-001",
            "problem_statement": "Fix the repository.",
            "repository": "example/repository",
            "chronological_index": 1,
        },
        "environment": {
            "id": "environment-001",
            "repository": "example/repository",
            "runtime": "python-3.13",
            "versions_json": "{}",
            "markers_json": "{}",
        },
        "failed_action": {
            "id": "failed-action-001",
            "run_id": "run-001",
            "task_id": "task-001",
            "tool": "run_tests",
            "operation": "pytest",
            "arguments_json": "{}",
            "planned_at": STARTED_AT.isoformat(),
            "result_tool_name": "run_tests",
            "success": False,
            "exit_code": 1,
            "output": "failed",
            "error": "tests failed",
            "started_at": STARTED_AT.isoformat(),
            "completed_at": COMPLETED_AT.isoformat(),
        },
        "recovery_action": {
            "id": action.id,
            "run_id": action.run_id,
            "task_id": action.task_id,
            "tool": action.tool,
            "operation": action.operation,
            "arguments_json": '{"options":{"quiet":true},"paths":["tests"]}',
            "planned_at": STARTED_AT.isoformat(),
            "result_tool_name": result.tool_name,
            "success": result.success,
            "exit_code": result.exit_code,
            "output": result.output,
            "error": result.error,
            "started_at": STARTED_AT.isoformat(),
            "completed_at": COMPLETED_AT.isoformat(),
        },
    }
    repository.execute_query = Mock(return_value=FakeResult([record]))

    with pytest.raises(ValueError, match="inconsistent historical provenance"):
        repository.get_recovery_pattern(pattern.id)


def test_lineage_query_converts_nodes_without_cross_product_duplicates() -> None:
    repository = make_repository()
    action, result = make_action_pair()
    repository.execute_query = Mock(
        side_effect=[
            FakeResult(
                [
                    {
                        "failure": {
                            "id": "failure-001",
                            "action_id": action.id,
                            "failure_type": "test_failure",
                            "signature": "signature",
                            "symptom": "symptom",
                            "observed_at": STARTED_AT.isoformat(),
                        }
                    }
                ]
            ),
            FakeResult(
                [
                    {
                        "environment": {
                            "id": "environment-001",
                            "repository": "example/repository",
                            "runtime": "python-3.13",
                            "versions_json": "{}",
                            "markers_json": "{}",
                        }
                    }
                ]
            ),
            FakeResult(
                [
                    {
                        "action": {
                            "id": action.id,
                            "run_id": action.run_id,
                            "task_id": action.task_id,
                            "tool": action.tool,
                            "operation": action.operation,
                            "arguments_json": '{"paths":["tests"],"options":{"quiet":true}}',
                            "planned_at": STARTED_AT.isoformat(),
                            "result_tool_name": result.tool_name,
                            "success": result.success,
                            "exit_code": result.exit_code,
                            "output": result.output,
                            "error": result.error,
                            "started_at": STARTED_AT.isoformat(),
                            "completed_at": COMPLETED_AT.isoformat(),
                        },
                        "task": {
                            "id": "task-001",
                            "problem_statement": "Fix the example repository so its tests pass.",
                            "family_id": "family-001",
                            "repository": "example/repository",
                            "chronological_index": 1,
                        },
                        "run": {
                            "id": "run-001",
                            "task_id": "task-001",
                            "started_at": STARTED_AT.isoformat(),
                        },
                    },
                    {
                        "action": {
                            "id": action.id,
                            "run_id": action.run_id,
                            "task_id": action.task_id,
                            "tool": action.tool,
                            "operation": action.operation,
                            "arguments_json": '{"paths":["tests"],"options":{"quiet":true}}',
                            "planned_at": STARTED_AT.isoformat(),
                            "result_tool_name": result.tool_name,
                            "success": result.success,
                            "exit_code": result.exit_code,
                            "output": result.output,
                            "error": result.error,
                            "started_at": STARTED_AT.isoformat(),
                            "completed_at": COMPLETED_AT.isoformat(),
                        },
                        "task": {
                            "id": "task-001",
                            "problem_statement": "Fix the example repository so its tests pass.",
                            "family_id": "family-001",
                            "repository": "example/repository",
                            "chronological_index": 1,
                        },
                        "run": {
                            "id": "run-001",
                            "task_id": "task-001",
                            "started_at": STARTED_AT.isoformat(),
                        },
                    },
                ]
            ),
            FakeResult([{"tool": {"name": "run_tests"}}]),
            FakeResult(
                [
                    {
                        "resolution": {
                            "id": "resolution-001",
                            "failure_id": "failure-001",
                            "description": "fix",
                            "status": "candidate",
                            "successful_observations": 0,
                            "failed_observations": 0,
                        },
                        "outcomes": [
                            {
                                "id": "outcome-001",
                                "action_id": action.id,
                                "success": True,
                                "tests_passed": 1,
                                "tests_failed": None,
                                "exit_code": 0,
                                "observed_at": COMPLETED_AT.isoformat(),
                            }
                        ],
                        "observed_changes": [
                            {
                                "id": action.id,
                                "run_id": action.run_id,
                                "task_id": action.task_id,
                                "tool": action.tool,
                                "operation": action.operation,
                                "arguments_json": (
                                    '{"options":{"quiet":true},"paths":["tests"]}'
                                ),
                                "planned_at": STARTED_AT.isoformat(),
                                "result_tool_name": result.tool_name,
                                "success": result.success,
                                "exit_code": result.exit_code,
                                "output": result.output,
                                "error": result.error,
                                "started_at": STARTED_AT.isoformat(),
                                "completed_at": COMPLETED_AT.isoformat(),
                            }
                        ],
                    }
                ]
            ),
        ]
    )

    lineage = repository.get_incident_lineage("failure-001")

    assert lineage.failure.id == "failure-001"
    assert lineage.run.id == "run-001"
    assert lineage.task.id == "task-001"
    assert lineage.task.problem_statement == "Fix the example repository so its tests pass."
    assert len(lineage.actions) == 1
    assert len(lineage.tools) == 1
    assert lineage.environment.id == "environment-001"
    assert len(lineage.resolutions) == 1
    assert len(lineage.outcomes) == 1
    assert len(lineage.recovery_actions) == 1
    assert lineage.recovery_actions[0].planned_action.arguments == action.arguments
    assert lineage.actions[0].result.action_id == lineage.actions[0].planned_action.id
