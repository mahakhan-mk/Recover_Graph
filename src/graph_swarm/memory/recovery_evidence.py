"""Deterministic, observational evidence for one concrete recovery change."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, field_validator, model_validator

from graph_swarm.domain._validation import require_non_empty
from graph_swarm.domain.actions import PlannedAction


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


def has_concrete_change(action: PlannedAction) -> bool:
    """Recognize only action shapes that identify an observable change.

    A successful result is not itself a change. The explicit shapes here keep
    recovery creation deterministic and prevent result prose from becoming a
    fabricated recovery description. Sprint 1 supports only ``write_file``:
    its path must be non-empty and its content must be present as a string.
    Read operations, test execution, and generic commands are not classified
    as mutations by this module.
    """
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
        if not has_concrete_change(action):
            raise ValueError("recovery_action must identify a supported concrete change")
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
    "arguments_json",
    "has_concrete_change",
    "normalize_arguments",
    "normalize_planned_action",
]
