from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import ModelRetry

from graph_swarm.advisory.service import AdvisoryService
from graph_swarm.agent.advisory import prepare_tool_action
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
    RecoveryTrigger,
)
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.read_models import (
    ActionLineageRecord,
    RecoveryPatternLineage,
    RecoveryPatternTask,
    RecoveryPatternVectorCandidate,
)
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.memory.recovery_embeddings import RecoveryPatternEmbedder
from graph_swarm.retrieval.service import RecoveryPatternRetrievalService

NOW = datetime(2026, 1, 1, tzinfo=UTC)
FD7 = "recovery-pattern-fd7b65022b22dc5f2a42816f"


class FakeEncoder:
    def encode(self, text: str, *, normalize_embeddings: bool) -> object:
        assert normalize_embeddings is True
        assert text
        return [0.25] * 384


class FrozenRecoveryRepository:
    def __init__(self, candidates: tuple[RecoveryPatternVectorCandidate, ...]) -> None:
        self.candidates = candidates
        self.lineages = {
            candidate.pattern.id: _lineage(candidate.pattern) for candidate in candidates
        }

    def count_recovery_pattern_vectors(self) -> int:
        return len(self.candidates)

    def query_recovery_pattern_vectors(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> tuple[RecoveryPatternVectorCandidate, ...]:
        assert query_embedding
        return self.candidates[:limit]

    def get_recovery_pattern(self, pattern_id: str) -> RecoveryPatternLineage:
        return self.lineages[pattern_id]


def _pattern(
    pattern_id: str = FD7,
    *,
    anchor: str = "LassoLexer Name.Builtin token highlighting",
    score: float = 0.99,
) -> RecoveryPatternVectorCandidate:
    pattern = RecoveryPattern(
        id=pattern_id,
        title="Restore lexer token classification",
        guidance="Restore the validated lexer token classification path.",
        source_failure_id=f"{pattern_id}-failure",
        source_resolution_id=f"{pattern_id}-resolution",
        source_outcome_id=f"{pattern_id}-outcome",
        source_task_id=f"{pattern_id}-task",
        source_chronological_index=5,
        source_tool="run_tests",
        source_operation="run_tests",
        applicability_tool="edit_file",
        applicability_operation="edit_file",
        source_failure_type="test_failure",
        trigger=RecoveryTrigger(
            failure_type="test_failure",
            failure_signature="run_tests:exit_code=1",
            failure_context=f"failure in {anchor}",
            source_task_problem_statement=f"The implementation breaks {anchor}.",
            source_tool="run_tests",
            source_operation="run_tests",
            source_action_arguments={"command": ["python", "-m", "pytest"]},
            version_sensitive=True,
        ),
        environment_constraints=EnvironmentConstraints(
            runtime="docker",
            versions={"python": "3.12.1"},
        ),
        verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        evidence_count=1,
        evidence_summary="Observed successful recovery.",
        created_at=NOW,
    )
    return RecoveryPatternVectorCandidate(pattern=pattern, vector_score=score)


def _lineage(pattern: RecoveryPattern) -> RecoveryPatternLineage:
    trigger = pattern.trigger
    assert trigger is not None
    failed = PlannedAction(
        id=f"{pattern.id}-failed-action",
        run_id="run-source",
        task_id=pattern.source_task_id,
        tool="run_tests",
        operation="run_tests",
        arguments={"command": ["python", "-m", "pytest"]},
        planned_at=NOW,
    )
    recovery = failed.model_copy(
        update={
            "id": f"{pattern.id}-recovery-action",
            "tool": "edit_file",
            "operation": "edit_file",
        }
    )
    failure = FailureEpisode(
        id=pattern.source_failure_id,
        action_id=failed.id,
        failure_type=FailureType.TEST_FAILURE,
        signature="run_tests:exit_code=1",
        symptom="historical test failure",
        observed_at=NOW,
    )
    resolution = Resolution(
        id=pattern.source_resolution_id,
        failure_id=failure.id,
        description="Observed lexer correction.",
        status=ResolutionStatus.OBSERVED_SUCCESSFUL,
        successful_observations=1,
        observed_at=NOW,
    )
    outcome = Outcome(
        id=pattern.source_outcome_id,
        action_id=recovery.id,
        success=True,
        exit_code=0,
        observed_at=NOW,
    )
    source_task = RecoveryPatternTask(
        id=pattern.source_task_id,
        problem_statement=trigger.source_task_problem_statement,
        repository="source/repository",
        chronological_index=pattern.source_chronological_index,
    )
    environment = EnvironmentContext(
        id=f"{pattern.id}-environment",
        repository="source/repository",
        runtime="docker",
        versions={"python": "3.12.1"},
    )
    return RecoveryPatternLineage(
        pattern=pattern,
        failure=failure,
        resolution=resolution,
        outcome=outcome,
        task=source_task,
        environment=environment,
        failed_action=ActionLineageRecord(
            planned_action=failed,
            result=ActionResult(
                action_id=failed.id,
                tool_name="run_tests",
                success=False,
                exit_code=1,
                output="historical failure",
                started_at=NOW,
                completed_at=NOW,
            ),
        ),
        recovery_action=ActionLineageRecord(
            planned_action=recovery,
            result=ActionResult(
                action_id=recovery.id,
                tool_name="edit_file",
                success=True,
                exit_code=0,
                started_at=NOW,
                completed_at=NOW,
            ),
        ),
    )


def _service(
    candidates: tuple[RecoveryPatternVectorCandidate, ...],
) -> AdvisoryService:
    repository = FrozenRecoveryRepository(candidates)
    retrieval = RecoveryPatternRetrievalService(
        repository,
        RecoveryPatternEmbedder(encoder=FakeEncoder()),
    )
    return AdvisoryService(
        cast(OperationalMemoryRepository, repository),
        retrieval_service=retrieval,
    )


def _context(problem_statement: str) -> tuple[Task, EnvironmentContext]:
    task = Task(
        id="GS-T018",
        problem_statement=problem_statement,
        family_id="syntax_token_classification",
        repository="current/repository",
        chronological_index=18,
    )
    environment = EnvironmentContext(
        id="current-environment",
        repository=task.repository,
        runtime="docker",
        versions={"python": "3.12.1"},
    )
    return task, environment


def _failed_prefix(dependencies: AgentDependencies, text: str) -> None:
    prior = PlannedAction(
        id="prior-failed-test",
        run_id=dependencies.run_id,
        task_id=dependencies.task_id,
        tool="run_tests",
        operation="run_tests",
        arguments={"command": ["python", "-m", "pytest", "tests/test_token.py"]},
        planned_at=NOW,
    )
    dependencies.record_planned_action(prior)
    emit_action_event(
        dependencies,
        ActionResult(
            action_id=prior.id,
            tool_name="run_tests",
            success=False,
            exit_code=1,
            output=text,
            started_at=NOW,
            completed_at=NOW,
        ),
    )


def test_real_advisory_service_advises_before_real_write_and_preserves_action(
    tmp_path: Path,
) -> None:
    task, environment = _context(
        "LassoLexer built-in functions are emitted as Name.Other instead of Name.Builtin."
    )
    service = _service((_pattern(),))
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=environment,
        advisory_service=service,
        capture_advisory_retrieval=True,
    )
    _failed_prefix(
        dependencies,
        "LassoLexer token highlighting returns Name.Other, not Name.Builtin.",
    )

    with pytest.raises(ModelRetry):
        prepare_tool_action(
            dependencies,
            "write_file",
            "write_file",
            {"path": "lexer.py", "content": "current source mutation"},
        )

    assert not (tmp_path / "lexer.py").exists()
    assert len(dependencies.advice_events) == 1
    advice_event = dependencies.advice_events[0]
    assert advice_event.planned_action.tool == "write_file"
    assert advice_event.planned_action.operation == "write_file"
    assert advice_event.advice.matched_failure_episode_id == f"{FD7}-failure"
    assert dependencies.advisory_retrieval_evidence[0].pattern_id == FD7
    assert "recovery_action_equivalence" in advice_event.advice.applicability.matched_fields


def test_real_advisory_service_rejects_same_write_boundary_without_matching_context(
    tmp_path: Path,
) -> None:
    task, environment = _context("A CSV formatter has an unrelated delimiter issue.")
    service = _service((_pattern(),))
    dependencies = AgentDependencies(
        tmp_path,
        "run-unrelated",
        task.id,
        task=task,
        environment=environment,
        advisory_service=service,
    )
    _failed_prefix(dependencies, "unrelated delimiter assertion failed")

    action = prepare_tool_action(
        dependencies,
        "write_file",
        "write_file",
        {"path": "formatter.py", "content": "unrelated source mutation"},
    )

    assert action.tool == "write_file"
    assert dependencies.advice_events == []


@pytest.mark.parametrize(
    ("task_id", "problem", "prefix", "expected"),
    (
        (
            "GS-T006",
            "UTF-8 BOM encoding is misread as UTF-16.",
            "UTF-8 BOM encoding was classified as UTF-16.",
            "recovery-pattern-ad07a6a45718b848a30ad377",
        ),
        (
            "GS-T007",
            "The unrelated template branch needs a formatting adjustment.",
            "The template formatting assertion failed.",
            None,
        ),
        (
            "GS-T008",
            "An UnboundLocalError occurs during variable initialization.",
            "UnboundLocalError occurred during variable initialization.",
            "recovery-pattern-f490f62ab931191c6eac6db1",
        ),
        (
            "GS-T017",
            "The lexer token highlighting loses Name.Builtin classification.",
            "Lexer token highlighting emits Name.Other instead of Name.Builtin.",
            FD7,
        ),
        (
            "GS-T018",
            "The lexer token highlighting loses Name.Builtin classification.",
            "Lexer token highlighting emits Name.Other instead of Name.Builtin.",
            FD7,
        ),
    ),
)
def test_pre_mutation_safety_matrix_has_no_unrelated_patterns(
    task_id: str,
    problem: str,
    prefix: str,
    expected: str | None,
) -> None:
    patterns = (
        _pattern(
            "recovery-pattern-ad07a6a45718b848a30ad377",
            anchor="UTF-8 BOM encoding UTF-16",
            score=0.95,
        ),
        _pattern(
            "recovery-pattern-f490f62ab931191c6eac6db1",
            anchor="UnboundLocalError variable initialization",
            score=0.94,
        ),
        _pattern(score=0.93),
    )
    current_task = Task(
        id=task_id,
        problem_statement=problem,
        family_id="offline-safety-matrix",
        repository="current/repository",
        chronological_index=20,
    )
    current_action = PlannedAction(
        id=f"{task_id}-write",
        run_id=f"run-{task_id}",
        task_id=task_id,
        tool="write_file",
        operation="write_file",
        arguments={"path": "target.py", "content": "not used for applicability"},
        planned_at=NOW,
    )
    environment = EnvironmentContext(
        id=f"{task_id}-environment",
        repository="current/repository",
        runtime="docker",
        versions={"python": "3.12.1"},
    )
    prefix_context: dict[str, object] = {
        "completed_prefix": [
            {
                "tool": "run_tests",
                "operation": "run_tests",
                "arguments": {"command": ["python", "-m", "pytest"]},
                "result": {"success": False, "output": prefix},
            }
        ]
    }
    repository = FrozenRecoveryRepository(patterns)
    result = RecoveryPatternRetrievalService(
        repository,
        RecoveryPatternEmbedder(encoder=FakeEncoder()),
    ).retrieve(
        current_task,
        current_action,
        environment,
        prefix_context=prefix_context,
    )

    eligible = [candidate.pattern_id for candidate in result.eligible_candidates]
    assert eligible == ([] if expected is None else [expected])
