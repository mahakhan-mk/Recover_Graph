from datetime import UTC, datetime

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.tasks import Task
from graph_swarm.retrieval.query import (
    RECOVERY_RETRIEVAL_QUERY_VERSION,
    recovery_retrieval_query_text,
)


def make_context() -> tuple[Task, PlannedAction, EnvironmentContext]:
    return (
        Task(
            id="task-current-001",
            problem_statement="Restore the intended branch condition.",
            family_id="family-secret-001",
            repository="current/repository",
            chronological_index=8,
        ),
        PlannedAction(
            id="action-current-001",
            run_id="run-current-001",
            task_id="task-current-001",
            tool="run_tests",
            operation="pytest",
            arguments={"future": "must not enter query"},
            planned_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        EnvironmentContext(
            id="environment-current-001",
            repository="current/repository",
            runtime="python-3.13",
            versions={"z-package": "2", "a-package": "1"},
            markers={"platform": "linux", "ci": "true"},
        ),
    )


def test_retrieval_query_v1_contains_only_legitimate_deterministic_context() -> None:
    task, action, environment = make_context()

    text = recovery_retrieval_query_text(task, action, environment)

    assert RECOVERY_RETRIEVAL_QUERY_VERSION == "v1"
    assert "Restore the intended branch condition." in text
    assert "Planned tool: run_tests" in text
    assert "Planned operation: pytest" in text
    assert "Runtime: python-3.13" in text
    assert "marker:ci=true\nmarker:platform=linux" in text
    assert "version:a-package=1\nversion:z-package=2" in text
    for excluded in (
        environment.repository,
        task.id,
        str(task.chronological_index),
        task.family_id,
        "future",
        "2026-01-01",
    ):
        assert excluded not in text


def test_retrieval_query_text_is_stable_for_identical_context() -> None:
    first = recovery_retrieval_query_text(*make_context())
    second = recovery_retrieval_query_text(*make_context())

    assert first == second
