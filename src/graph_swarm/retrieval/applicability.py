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
from graph_swarm.domain.recovery_patterns import RecoveryPattern, RecoveryTrigger
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
        prefix_context: Mapping[str, object] | None = None,
    ) -> ApplicabilityDecision:
        """Evaluate one pattern while preserving the legacy three-arg contract."""
        if pattern.trigger is None:
            return self._evaluate_legacy(planned_action, environment, pattern)
        return self._evaluate_trigger(
            planned_action,
            environment,
            pattern,
            task,
            prefix_context,
        )

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

    def _recovery_action_matches(
        self,
        planned_action: PlannedAction,
        pattern: RecoveryPattern,
        *,
        pre_mutation: bool,
    ) -> tuple[bool, bool]:
        """Compare the live recovery boundary with historical action metadata.

        ``edit_file/edit_file`` and ``write_file/write_file`` are equivalent
        only while evaluating a real source-mutation boundary.  The second
        return value records that the equivalence, rather than an exact pair,
        made the structural check pass; it never changes trigger or
        environment applicability.
        """
        if pattern.applicability_tool is None or pattern.applicability_operation is None:
            return False, False
        historical = (pattern.applicability_tool, pattern.applicability_operation)
        current = (planned_action.tool, planned_action.operation)
        tool_matches = _normalize_action_value(current[0]) == _normalize_action_value(
            historical[0]
        )
        operation_matches = _normalize_action_value(current[1]) == _normalize_action_value(
            historical[1]
        )
        exact = tool_matches and operation_matches
        if exact:
            return True, False
        equivalent = (
            pre_mutation
            and _is_mutation_action_pair(current)
            and _is_mutation_action_pair(historical)
        )
        if equivalent:
            return True, True
        return False, False

    def _evaluate_trigger(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        pattern: RecoveryPattern,
        task: Task | None,
        prefix_context: Mapping[str, object] | None = None,
    ) -> ApplicabilityDecision:
        trigger = pattern.trigger
        if trigger is None:  # pragma: no cover - guarded by evaluate
            raise AssertionError("triggered evaluation requires a trigger")
        matched: list[str] = []
        rejected: list[str] = []

        if prefix_context is None:
            trigger_actions = ((planned_action, ""),)
            prefix_evidence = ""
        else:
            recovery_matches, recovery_equivalent = self._recovery_action_matches(
                planned_action,
                pattern,
                pre_mutation=True,
            )
            if not recovery_matches:
                recovery_tool = pattern.applicability_tool
                recovery_operation = pattern.applicability_operation
                if recovery_tool is None or _normalize_action_value(
                    recovery_tool
                ) != _normalize_action_value(planned_action.tool):
                    rejected.append("recovery_tool_mismatch")
                if recovery_operation is None or _normalize_action_value(
                    recovery_operation
                ) != _normalize_action_value(planned_action.operation):
                    rejected.append("recovery_operation_mismatch")
            else:
                matched.append("recovery_tool")
                matched.append("recovery_operation")
                if recovery_equivalent:
                    matched.append("recovery_action_equivalence")
            trigger_actions = _completed_prefix_actions(prefix_context, planned_action)
            if not trigger_actions:
                rejected.append("completed_prefix_trigger_missing")
            prefix_evidence = _completed_prefix_evidence_text(prefix_context)

        trigger_decisions = tuple(
            self._evaluate_trigger_action(
                trigger,
                trigger_action,
                " ".join((diagnostic_text, prefix_evidence)),
                task,
            )
            for trigger_action, diagnostic_text in trigger_actions
        )
        if trigger_decisions:
            selected = next(
                (decision for decision in trigger_decisions if decision.applicable),
                trigger_decisions[0],
            )
            matched.extend(selected.matched_fields)
            rejected.extend(selected.rejection_reasons)

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

    def _evaluate_trigger_action(
        self,
        trigger: RecoveryTrigger,
        planned_action: PlannedAction,
        diagnostic_text: str,
        task: Task | None,
    ) -> ApplicabilityDecision:
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
        exact_tool = _normalize_action_value(
            trigger.source_tool
        ) == _normalize_action_value(
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
                " ".join(
                    (
                        task.problem_statement,
                        _argument_text(planned_action.arguments),
                        diagnostic_text,
                    )
                )
            )
            # Only historical failure-trigger text can establish subject
            # matter. Recovery guidance/title/evidence describe the repair,
            # not the incident that made it applicable. Exact anchor
            # intersection is deliberately not a tunable vector threshold.
            if not historical_trigger_terms & current_terms:
                rejected.append("trigger_context_mismatch")
            else:
                matched.append("trigger_context")

        return ApplicabilityDecision(
            applicable=not rejected,
            matched_fields=tuple(matched),
            rejection_reasons=tuple(rejected),
        )


def _completed_prefix_actions(
    prefix_context: Mapping[str, object],
    prototype: PlannedAction,
) -> tuple[tuple[PlannedAction, str], ...]:
    """Return only real failed actions recorded before the mutation boundary."""
    raw_prefix = prefix_context.get("completed_prefix")
    if not isinstance(raw_prefix, Sequence) or isinstance(raw_prefix, (str, bytes)):
        return ()
    actions: list[tuple[PlannedAction, str]] = []
    for entry in cast(Sequence[object], raw_prefix):
        if not isinstance(entry, Mapping):
            continue
        typed_entry = cast(Mapping[str, object], entry)
        raw_result = typed_entry.get("result")
        if not isinstance(raw_result, Mapping):
            continue
        typed_result = cast(Mapping[str, object], raw_result)
        if typed_result.get("success") is not False:
            continue
        tool = typed_entry.get("tool")
        operation = typed_entry.get("operation")
        arguments = typed_entry.get("arguments", {})
        if not isinstance(tool, str) or not isinstance(operation, str):
            continue
        if not isinstance(arguments, Mapping):
            continue
        action = prototype.model_copy(
            update={
                "tool": tool,
                "operation": operation,
                "arguments": {
                    str(key): value
                    for key, value in cast(Mapping[str, object], arguments).items()
                },
            }
        )
        diagnostics = " ".join(
            value
            for key in ("output", "error")
            for value in (typed_result.get(key),)
            if isinstance(value, str) and value
        )
        actions.append((action, diagnostics))
    return tuple(actions)


def _completed_prefix_evidence_text(prefix_context: Mapping[str, object]) -> str:
    """Collect safe prior reads and diagnostics without mutation payloads."""
    raw_prefix = prefix_context.get("completed_prefix")
    if not isinstance(raw_prefix, Sequence) or isinstance(raw_prefix, (str, bytes)):
        return ""
    evidence: list[str] = []
    blocked = ("expected_pattern", "fail_to_pass", "gold patch", "benchmark label")
    for entry in cast(Sequence[object], raw_prefix):
        if not isinstance(entry, Mapping):
            continue
        typed_entry = cast(Mapping[str, object], entry)
        tool = typed_entry.get("tool")
        arguments = typed_entry.get("arguments")
        result = typed_entry.get("result")
        if not isinstance(result, Mapping):
            continue
        typed_result = cast(Mapping[str, object], result)
        values: list[object] = [typed_result.get("output"), typed_result.get("error")]
        if tool == "read_file" and isinstance(arguments, Mapping):
            values.append(_argument_text(cast(Mapping[str, object], arguments)))
        for value in values:
            if not isinstance(value, str) or not value:
                continue
            if any(term in value.casefold() for term in blocked):
                continue
            evidence.append(value)
    return " ".join(evidence)


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


def _is_mutation_action_pair(action: tuple[str, str]) -> bool:
    return action in {
        ("edit_file", "edit_file"),
        ("write_file", "write_file"),
    }


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
