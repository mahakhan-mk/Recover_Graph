from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from experiments.sprint3 import (
    BenchmarkPreflightError,
    FrozenSWEsmithCase,
    FrozenSWEsmithObjective,
    IsolatedTaskEnvironment,
    make_recurrence_matcher,
)
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.tasks import Task
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    RecurrenceEvaluationRequired,
)


def _case(occurrence_index: int) -> BenchmarkTaskCase:
    return BenchmarkTaskCase(
        task=Task(
            id="GS-T006",
            problem_statement="Fix the benchmark issue.",
            family_id="GS-F001",
            repository="swesmith/example__repo.abc12345",
            chronological_index=6,
        ),
        occurrence_index=occurrence_index,
    )


def _test_event(output: str | None, *, success: bool = False) -> AgentEvent:
    started = datetime.now(UTC)
    result = ActionResult(
        action_id="action-1",
        tool_name="run_tests",
        success=success,
        exit_code=0 if success else 1,
        output=output,
        started_at=started,
        completed_at=started,
    )
    return AgentEvent(
        event_id="event-1",
        run_id="run-1",
        task_id="GS-T006",
        action_id="action-1",
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=started,
    )


def test_sprint3_recurrence_requires_the_frozen_failure_signature() -> None:
    frozen = {
        "GS-T006": FrozenSWEsmithCase(
            "example__repo.abc12345",
            ("test_historical (example.test_recurrence.TestCase)",),
            "",
        )
    }
    matcher = make_recurrence_matcher(frozen)

    assert (
        matcher(
            _case(2),
            [_test_event("FAILED example/test_recurrence.py::TestCase::test_historical")],
            None,
            Path("workspace"),
        )
        is True
    )
    assert (
        matcher(
            _case(2),
            [_test_event("FAILED example/test_recurrence.py::test_unrelated")],
            None,
            Path("workspace"),
        )
        is False
    )
    assert matcher(_case(2), [_test_event(None)], None, Path("workspace")) is False
    assert (
        matcher(
            _case(1),
            [_test_event("FAILED example/test_recurrence.py::TestCase::test_historical")],
            None,
            Path("workspace"),
        )
        is False
    )


def test_sprint3_recurrence_missing_frozen_evidence_fails_explicitly() -> None:
    matcher = make_recurrence_matcher({})

    with pytest.raises(RecurrenceEvaluationRequired):
        matcher(_case(2), [], None, Path("workspace"))


def test_benchmark_preflight_requires_a_task_isolated_executable(tmp_path: Path) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_missing.py",), "")},
        {},
    )

    with pytest.raises(BenchmarkPreflightError, match="isolated SWE-smith"):
        objective.preflight(task_case.task, tmp_path)


def test_objective_collection_failure_is_infrastructure_evidence(tmp_path: Path) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_missing.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path(sys.executable))},
    )

    assert objective(task_case.task, tmp_path) is False
    assert objective.observations[-1].status == "objective_infrastructure_failure"


def test_missing_repository_dependency_cannot_pass_preflight(tmp_path: Path) -> None:
    (tmp_path / "test_dependency.py").write_text(
        "import package_that_is_not_declared_or_installed\n\n"
        "def test_dependency():\n    assert True\n",
        encoding="utf-8",
    )
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_dependency.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path(sys.executable))},
    )

    with pytest.raises(BenchmarkPreflightError, match="preflight failed"):
        objective.preflight(task_case.task, tmp_path)
