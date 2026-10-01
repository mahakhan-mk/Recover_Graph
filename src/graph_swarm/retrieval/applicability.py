"""Deterministic applicability checks for historical recoveries.

Recovery-action metadata and failure-trigger metadata are intentionally
separate. A successful historical ``edit_file`` action is provenance for the
repair; it is not, by itself, a reason to issue advice before a current edit.
Likewise, a generic ``run_command`` is only a trigger when its normalized
intent and task context match the historical failure.
"""

import re
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel, Field

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.recovery_patterns import RecoveryPattern
from graph_swarm.domain.tasks import Task
from graph_swarm.retrieval.candidates import HistoricalRecoveryCandidate


class ApplicabilityDecision(BaseModel):
    """Explainable result of one candidate applicability evaluation."""

    applicable: bool
    matched_fields: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    compatible_versions: dict[str, str] = Field(default_factory=dict)
    compatible_markers: dict[str, str] = Field(default_factory=dict)


class ApplicabilityService:
    """Apply exact structural and known-environment compatibility rules."""

    def evaluate(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        candidate: HistoricalRecoveryCandidate,
    ) -> ApplicabilityDecision:
        matched: list[str] = []
        rejected: list[str] = []
        if not planned_action.tool.strip() or not candidate.failed_action.tool.strip():
            rejected.append("tool applicability data is missing")
        elif planned_action.tool != candidate.failed_action.tool:
            rejected.append("tool does not match")
        else:
            matched.append("tool")
        if not planned_action.operation.strip() or not candidate.failed_action.operation.strip():
            rejected.append("operation applicability data is missing")
        elif planned_action.operation != candidate.failed_action.operation:
            rejected.append("operation does not match")
        else:
            matched.append("operation")
        if not environment.runtime.strip() or not candidate.environment.runtime.strip():
            rejected.append("runtime applicability data is missing")
        elif environment.runtime != candidate.environment.runtime:
            rejected.append("runtime does not match")
        else:
            matched.append("runtime")
        compatible_versions, version_rejections = _compatible_values(
            environment.versions, candidate.environment.versions, "version"
        )
        rejected.extend(version_rejections)
        if not version_rejections and compatible_versions:
            matched.append("versions")
        compatible_markers, marker_rejections = _compatible_values(
            environment.markers, candidate.environment.markers, "marker"
        )
        rejected.extend(marker_rejections)
        if not marker_rejections and compatible_markers:
            matched.append("markers")
        return ApplicabilityDecision(
            applicable=not rejected,
            matched_fields=tuple(matched),
            rejection_reasons=tuple(rejected),
            compatible_versions=compatible_versions,
            compatible_markers=compatible_markers,
        )

    def is_applicable(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        candidate: HistoricalRecoveryCandidate,
    ) -> bool:
        return self.evaluate(planned_action, environment, candidate).applicable


class RecoveryPatternApplicabilityService:
    """Apply trigger semantics and comparable-environment rules to patterns."""

    def evaluate(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        pattern: RecoveryPattern,
        task: Task | None = None,
    ) -> ApplicabilityDecision:
        """Evaluate one pattern while preserving the legacy three-arg contract."""
        if pattern.trigger is None:
            return self._evaluate_legacy(planned_action, environment, pattern)
        return self._evaluate_trigger(planned_action, environment, pattern, task)

    def _evaluate_legacy(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        pattern: RecoveryPattern,
    ) -> ApplicabilityDecision:
        """Read legacy patterns without rewriting immutable historical records."""
        matched: list[str] = []
        rejected: list[str] = []
        effective_tool = pattern.applicability_tool or pattern.source_tool
        effective_operation = pattern.applicability_operation or pattern.source_operation
        if _normalize_action_value(effective_tool) != _normalize_action_value(planned_action.tool):
            rejected.append("tool_mismatch")
        else:
            matched.append("tool")
        if _normalize_action_value(effective_operation) != _normalize_action_value(
            planned_action.operation
        ):
            rejected.append("operation_mismatch")
        else:
            matched.append("operation")
        compatible_versions, version_rejections = _compatible_values(
            environment.versions, pattern.environment_constraints.versions, "version"
        )
        rejected.extend(version_rejections)
        if not version_rejections and compatible_versions:
            matched.append("versions")
        compatible_markers, marker_rejections = _compatible_values(
            environment.markers, pattern.environment_constraints.markers, "marker"
        )
        rejected.extend(marker_rejections)
        if not marker_rejections and compatible_markers:
            matched.append("markers")
        _apply_runtime(environment, pattern, matched, rejected)
        return ApplicabilityDecision(
            applicable=not rejected,
            matched_fields=tuple(matched),
            rejection_reasons=tuple(rejected),
            compatible_versions=compatible_versions,
            compatible_markers=compatible_markers,
        )

    def _evaluate_trigger(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        pattern: RecoveryPattern,
        task: Task | None,
    ) -> ApplicabilityDecision:
        trigger = pattern.trigger
        if trigger is None:  # pragma: no cover - guarded by evaluate
            raise AssertionError("triggerred evaluation requires a trigger")
        matched: list[str] = []
        rejected: list[str] = []
        historical_intent = _planned_intent(
            trigger.source_tool,
            trigger.source_operation,
            trigger.source_action_arguments,
        )
        current_intent = _planned_intent(
            planned_action.tool, planned_action.operation, planned_action.arguments
        )
        exact_tool = _normalize_action_value(trigger.source_tool) == _normalize_action_value(
            planned_action.tool
        )
        exact_operation = _normalize_action_value(
            trigger.source_operation
        ) == _normalize_action_value(planned_action.operation)
        test_tool_bridge = (
            historical_intent == frozenset({"test_execution"})
            and current_intent == frozenset({"test_execution"})
            and planned_action.tool == "run_tests"
        )
        if not exact_tool and not test_tool_bridge:
            rejected.append("trigger_tool_mismatch")
        else:
            matched.append("trigger_tool")
        if not exact_operation and not test_tool_bridge:
            rejected.append("trigger_operation_mismatch")
        else:
            matched.append("trigger_operation")
        if not historical_intent or not current_intent:
            rejected.append("trigger_intent_missing")
        elif not historical_intent & current_intent:
            rejected.append("trigger_intent_mismatch")
        else:
            matched.append("trigger_intent")

        if task is None:
            rejected.append("current_task_context_missing")
        else:
            historical_trigger_terms = _trigger_context_terms(
                " ".join(
                    (
                        trigger.failure_type,
                        trigger.failure_signature,
                        trigger.failure_context,
                        _problem_summary(trigger.source_task_problem_statement),
                    )
                )
            )
            current_terms = _trigger_context_terms(
                " ".join((task.problem_statement, _argument_text(planned_action.arguments)))
            )
            # Only historical failure-trigger text can establish subject
            # matter. Recovery guidance/title/evidence describe the repair,
            # not the incident that made it applicable. Exact anchor
            # intersection is deliberately not a tunable vector threshold.
            if not historical_trigger_terms & current_terms:
                rejected.append("trigger_context_mismatch")
            else:
                matched.append("trigger_context")

        _apply_runtime(environment, pattern, matched, rejected)
        if trigger.version_sensitive:
            compatible_versions, version_rejections = _compatible_values(
                environment.versions,
                pattern.environment_constraints.versions,
                "version",
            )
            rejected.extend(version_rejections)
            if not version_rejections and compatible_versions:
                matched.append("versions")
        else:
            # Exact equality remains the conservative default. Only this
            # explicit trigger flag may classify acquisition version data as
            # non-applicability metadata.
            compatible_versions = {}
        compatible_markers, marker_rejections = _compatible_values(
            environment.markers, pattern.environment_constraints.markers, "marker"
        )
        rejected.extend(marker_rejections)
        if not marker_rejections and compatible_markers:
            matched.append("markers")
        return ApplicabilityDecision(
            applicable=not rejected,
            matched_fields=tuple(matched),
            rejection_reasons=tuple(rejected),
            compatible_versions=compatible_versions,
            compatible_markers=compatible_markers,
        )


def _apply_runtime(
    environment: EnvironmentContext,
    pattern: RecoveryPattern,
    matched: list[str],
    rejected: list[str],
) -> None:
    runtime_constraint = pattern.environment_constraints.runtime
    if runtime_constraint and environment.runtime.strip():
        if _normalize_fact(runtime_constraint) != _normalize_fact(environment.runtime):
            rejected.append("runtime_mismatch")
        else:
            matched.append("runtime")


def _compatible_values(
    current: dict[str, str], historical: dict[str, str], label: str
) -> tuple[dict[str, str], list[str]]:
    compatible: dict[str, str] = {}
    rejected: list[str] = []
    for key in sorted(current.keys() & historical.keys()):
        current_value = current[key]
        historical_value = historical[key]
        if not key.strip() or not current_value.strip() or not historical_value.strip():
            rejected.append(f"{label}_mismatch:{key}")
        elif _normalize_fact(current_value) != _normalize_fact(historical_value):
            rejected.append(f"{label}_mismatch:{key}")
        else:
            compatible[key] = current_value
    return compatible, rejected


def _normalize_action_value(value: str) -> str:
    return " ".join(value.split())


def _normalize_fact(value: str) -> str:
    return " ".join(value.split()).casefold()


_SEMANTIC_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "all",
        "and",
        "are",
        "because",
        "before",
        "being",
        "but",
        "cause",
        "causing",
        "code",
        "due",
        "error",
        "expected",
        "fails",
        "failed",
        "failure",
        "for",
        "from",
        "function",
        "issue",
        "method",
        "not",
        "once",
        "problem",
        "pytest",
        "return",
        "running",
        "should",
        "succeeded",
        "test",
        "tests",
        "the",
        "this",
        "when",
        "with",
    }
)

# These terms describe the shape of a software defect or its report rather
# than the failure's subject matter. They cannot establish transfer
# applicability: otherwise a historical variable-initialization repair can
# match an unrelated variable-scope task merely because both reports mention
# variables and assignments. This is a deterministic anchor filter, not a
# similarity threshold.
_TRIGGER_CONTEXT_GENERIC_TERMS = frozenset(
    {
        "actual",
        "appear",
        "appears",
        "assignment",
        "assignments",
        "affect",
        "affects",
        "behavior",
        "broken",
        "code",
        "condition",
        "conditional",
        "context",
        "correctly",
        "defined",
        "description",
        "does",
        "error",
        "expected",
        "function",
        "import",
        "instead",
        "incorrect",
        "incorrectly",
        "issue",
        "local",
        "method",
        "problem",
        "properly",
        "print",
        "python",
        "raise",
        "raises",
        "reproduce",
        "result",
        "return",
        "returned",
        "returns",
        "running",
        "step",
        "steps",
        "test",
        "tests",
        "that",
        "trying",
        "undefined",
        "used",
        "using",
        "value",
        "values",
        "variable",
        "variables",
        "where",
        "working",
        "works",
        "wrong",
        "wrongly",
    }
)

# Execution mechanics identify how a failure was observed, not what failed.
# They must not establish transfer applicability on their own. Keep the
# vocabulary explicit so adding a trusted test-intent form does not silently
# turn its command spelling into a subject-matter anchor.
_TRIGGER_CONTEXT_EXECUTION_TERMS = frozenset(
    {
        "command",
        "commands",
        "execute",
        "executed",
        "executes",
        "execution",
        "exit",
        "exited",
        "exits",
        "exit_code",
        "fail",
        "failed",
        "failing",
        "failure",
        "failures",
        "pytest",
        "python",
        "run_command",
        "run_tests",
        "test_execution",
        "test_failure",
        "testing",
    }
)


def _semantic_terms(value: str) -> frozenset[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_+-]*", value.casefold()):
        if len(token) < 4 or token in _SEMANTIC_STOPWORDS:
            continue
        terms.add(token)
        # Keep a conservative singular alias for ordinary plural words so
        # "variables" and "variable" express the same text anchor.
        if token.endswith("s") and len(token) > 5:
            terms.add(token[:-1])
    return frozenset(terms)


def _trigger_context_terms(value: str) -> frozenset[str]:
    """Return subject-matter anchors suitable for trigger transfer."""
    return _semantic_terms(value) - (
        _TRIGGER_CONTEXT_GENERIC_TERMS | _TRIGGER_CONTEXT_EXECUTION_TERMS
    )


def _problem_summary(value: str) -> str:
    """Use the problem's summary line, excluding reproduction boilerplate."""
    for line in value.splitlines():
        summary = line.lstrip("#").strip()
        if summary:
            return summary
    return value


def _argument_text(arguments: Mapping[str, object]) -> str:
    values: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            for key in sorted(mapping, key=str):
                values.append(str(key))
                visit(mapping[key])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            sequence = cast(Sequence[object], value)
            for item in sequence:
                visit(item)

    visit(arguments)
    return " ".join(values)


def _planned_intent(
    tool: str,
    operation: str,
    arguments: Mapping[str, object],
) -> frozenset[str]:
    """Recognize test execution, while leaving generic commands distinct."""
    if tool == "run_tests" or operation == "run_tests":
        return frozenset({"test_execution"})
    if tool != "run_command" and operation != "run_command":
        return frozenset()
    tokens = tuple(re.findall(r"[a-z0-9_.-]+", _argument_text(arguments).casefold()))
    if not tokens:
        return frozenset()
    for index, token in enumerate(tokens):
        executable = token.replace("\\", "/").rsplit("/", 1)[-1]
        if executable in {"pytest", "py.test", "pytest.exe", "py.test.exe"}:
            return frozenset({"test_execution"})
        if (
            index + 2 < len(tokens)
            and tokens[index + 1] == "-m"
            and tokens[index + 2] in {"pytest", "py.test", "unittest"}
        ):
            return frozenset({"test_execution"})
    return frozenset()


__all__ = [
    "ApplicabilityDecision",
    "ApplicabilityService",
    "RecoveryPatternApplicabilityService",
]
