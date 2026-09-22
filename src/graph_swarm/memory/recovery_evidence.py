"""Deterministic, observational evidence for one concrete recovery change."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from graph_swarm.domain._validation import require_non_empty
from graph_swarm.domain.actions import PlannedAction


class RepositoryMutationEvidence(BaseModel):
    """Runtime-observed repository state change for one trusted action."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    before_fingerprint: str
    after_fingerprint: str

    @field_validator("action_id", "before_fingerprint", "after_fingerprint")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        return require_non_empty(value)

    @model_validator(mode="after")
    def require_changed_repository(self) -> RepositoryMutationEvidence:
        if self.before_fingerprint == self.after_fingerprint:
            raise ValueError("repository mutation evidence requires a changed fingerprint")
        return self


def is_test_execution(action: PlannedAction) -> bool:
    """Identify pytest execution from trusted PlannedAction semantics."""
    if action.tool == "run_tests":
        return True
    if action.tool != "run_command":
        return False
    command = action.arguments.get("command")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return False
    command_values = tuple(cast(Sequence[object], command))
    argv = tuple(item for item in command_values if isinstance(item, str))
    if len(argv) != len(command_values) or not argv:
        return False
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if executable in {"pytest", "pytest.exe", "py.test", "py.test.exe"}:
        return True
    if len(argv) >= 3 and argv[1] == "-m":
        module = argv[2].lower()
        if module not in {"pytest", "py.test"}:
            return False
        return executable.startswith("python") or executable in {"py", "py.exe"}
    return False


def _normalize_value(value: object) -> object:
    """Convert supported action arguments into stable JSON-shaped values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        normalized: dict[str, object] = {}
        for key in sorted(mapping, key=str):
            if not isinstance(key, str):
                raise ValueError("action argument mapping keys must be strings")
            normalized[key] = _normalize_value(mapping[key])
        return normalized
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [_normalize_value(item) for item in sequence]
    if isinstance(value, (set, frozenset)):
        items = cast(set[object] | frozenset[object], value)
        normalized_items: list[object] = [_normalize_value(item) for item in items]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
        )
    raise ValueError(
        f"unsupported action argument value type: {type(value).__name__}"
    )


def normalize_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Normalize action arguments without interpreting their meaning."""
    normalized = _normalize_value(arguments)
    if not isinstance(normalized, dict):
        raise TypeError("action arguments must normalize to an object")
    # Validate the exact representation that will be persisted in Neo4j.
    json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return cast(dict[str, object], normalized)


def normalize_planned_action(action: PlannedAction) -> PlannedAction:
    """Return a copy of an action with deterministic persisted arguments."""
    arguments = normalize_arguments(action.arguments)
    if action.tool == "write_file":
        path = arguments.get("path")
        if isinstance(path, str):
            arguments["path"] = path.replace("\\", "/")
    return action.model_copy(update={"arguments": arguments})


def has_concrete_change(
    action: PlannedAction,
    *,
    mutation_evidence: RepositoryMutationEvidence | None = None,
) -> bool:
    """Recognize only action shapes that identify an observable change.

    A successful result is not itself a change. The explicit shapes and
    runtime evidence keep recovery creation deterministic and prevent result
    prose from becoming a fabricated recovery description. ``write_file``
    retains its deterministic argument rule; other actions require valid
    before/after repository evidence.
    """
    if mutation_evidence is not None and mutation_evidence.action_id == action.id:
        return True
    arguments = action.arguments
    if action.tool == "write_file":
        path = arguments.get("path")
        content = arguments.get("content")
        return (
            isinstance(path, str)
            and bool(path.strip())
            and isinstance(content, str)
        )
    return False


class ConcreteRecoveryEvidence(BaseModel):
    """Typed observational links needed to reconstruct one recovery episode."""

    recovery_action: PlannedAction
    source_failure_id: str
    resolution_id: str
    objective_outcome_id: str
    task_id: str
    source_chronological_index: int
    environment_id: str
    repository_mutation_evidence: RepositoryMutationEvidence | None = None

    @field_validator(
        "source_failure_id",
        "resolution_id",
        "objective_outcome_id",
        "task_id",
        "environment_id",
    )
    @classmethod
    def require_non_empty_references(cls, value: str) -> str:
        return require_non_empty(value)

    @model_validator(mode="after")
    def require_action_task_identity(self) -> ConcreteRecoveryEvidence:
        action = self.recovery_action
        for field_name in ("id", "run_id", "task_id", "tool", "operation"):
            value = getattr(action, field_name)
            if not value.strip():
                raise ValueError(f"recovery_action.{field_name} must be non-empty")
        if action.planned_at.tzinfo is None or action.planned_at.utcoffset() is None:
            raise ValueError("recovery_action.planned_at must be timezone-aware")
        if not has_concrete_change(
            action,
            mutation_evidence=self.repository_mutation_evidence,
        ):
            raise ValueError("recovery_action must identify a supported concrete change")
        if (
            self.repository_mutation_evidence is not None
            and self.repository_mutation_evidence.action_id != action.id
        ):
            raise ValueError("repository mutation evidence action_id must match recovery_action")
        if self.recovery_action.task_id != self.task_id:
            raise ValueError("recovery action task_id must match evidence task_id")
        return self


def arguments_json(action: PlannedAction) -> str:
    """Serialize already-normalized action arguments canonically."""
    return json.dumps(
        normalize_arguments(action.arguments),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


__all__ = [
    "ConcreteRecoveryEvidence",
    "RepositoryMutationEvidence",
    "arguments_json",
    "has_concrete_change",
    "is_test_execution",
    "normalize_arguments",
    "normalize_planned_action",
]
