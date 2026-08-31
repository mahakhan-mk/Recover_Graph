from datetime import UTC, datetime
from unittest.mock import Mock, patch

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution
from graph_swarm.graph import queries
from graph_swarm.graph.neo4j_repository import Neo4jRepository
from graph_swarm.retrieval.candidates import HistoricalRecoveryCandidate

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class FakeResult:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records


def make_repository() -> Neo4jRepository:
    with patch("graph_swarm.graph.neo4j_repository.GraphDatabase.driver", return_value=Mock()):
        return Neo4jRepository("uri", "username", "password", "database")


def test_candidate_lookup_maps_graph_rows_to_domain_read_models() -> None:
    repository = make_repository()
    repository.execute_query = Mock(
        return_value=FakeResult(
            [
                {
                    "failed_action_id": "failed-action-001",
                    "tool": "run_tests",
                    "operation": "pytest",
                    "planned_at": NOW.isoformat(),
                    "failure_id": "failure-001",
                    "failure_action_id": "failed-action-001",
                    "failure_type": FailureType.TEST_FAILURE.value,
                    "failure_signature": "run_tests:exit_code=1",
                    "symptom": "one test failed",
                    "failure_observed_at": NOW.isoformat(),
                    "environment_id": "environment-001",
                    "repository": "example/repository",
                    "runtime": "python-3.13",
                    "versions_json": '{"pytest":"8.0"}',
                    "markers_json": "{}",
                    "resolution_id": "resolution-001",
                    "resolution_failure_id": "failure-001",
                    "resolution_description": "Restore the expected branch condition",
                    "resolution_status": "observed_successful",
                    "successful_observations": 1,
                    "failed_observations": 0,
                    "outcomes": [
                        {
                            "id": "outcome-001",
                            "action_id": "successful-action-001",
                            "success": True,
                            "tests_passed": 3,
                            "tests_failed": 0,
                            "exit_code": 0,
                            "observed_at": NOW.isoformat(),
                        }
                    ],
                }
            ]
        )
    )

    candidates = repository.find_historical_recovery_candidates(
        PlannedAction(
            id="planned-action",
            run_id="run-current",
            task_id="task-current",
            tool="run_tests",
            operation="pytest",
            planned_at=NOW,
        ),
        EnvironmentContext(
            id="environment-current",
            repository="example/repository",
            runtime="python-3.13",
        ),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.failure.id == "failure-001"
    assert candidate.resolution.id == "resolution-001"
    assert candidate.failed_action.tool == "run_tests"
    assert candidate.outcomes[0].id == "outcome-001"
    assert isinstance(candidate.resolution, Resolution)
    assert isinstance(candidate.outcomes[0], Outcome)
    assert isinstance(candidate, HistoricalRecoveryCandidate)
    assert not isinstance(candidate, dict)
    repository.execute_query.assert_called_once_with(
        queries.FIND_HISTORICAL_RECOVERY_CANDIDATES,
        tool="run_tests",
        operation="pytest",
        repository="example/repository",
    )
