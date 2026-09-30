"""Grounded RecoveryPattern abstraction through the frozen OpenRouter model."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import Agent, ModelSettings, PromptedOutput
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
    RecoveryTrigger,
)
from graph_swarm.graph.read_models import RecoveryEvidenceLineage
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.memory.recovery_evidence import normalize_arguments
from graph_swarm.memory.recovery_prompt import (
    RECOVERY_ABSTRACTION_PROMPT_VERSION,
    RECOVERY_ABSTRACTION_SYSTEM_PROMPT,
)
from graph_swarm.settings import Settings

FROZEN_RECOVERY_MODEL = "cohere/north-mini-code:free"
RECOVERY_MODEL_SETTINGS: ModelSettings = {"temperature": 0}


class RecoveryAbstractionModelError(RuntimeError):
    """Raised when the configured model/provider cannot produce an output."""


class RecoveryAbstractionStructuredOutputError(RuntimeError):
    """Raised when the model response cannot satisfy the output schema."""


class RecoveryAbstractionValidationError(ValueError):
    """Raised when a structured abstraction is not grounded and reusable."""


OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE = "objective_anchored_v1"


class RecoveryOutcomeEvidence(BaseModel):
    """Objective outcome facts supplied to the abstraction model."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    tests_passed: int | None = None
    tests_failed: int | None = None
    exit_code: int | None = None


class RecoveryEvidencePackage(BaseModel):
    """The smallest trusted historical package needed for abstraction."""

    model_config = ConfigDict(extra="forbid")

    source_failure_id: str
    source_resolution_id: str
    source_outcome_id: str
    source_task_id: str
    source_chronological_index: int
    source_task_problem_statement: str
    failed_action_tool: str
    failed_action_operation: str
    failed_action_arguments: dict[str, object] = Field(default_factory=dict)
    failure_type: str
    failure_signature: str
    failure_symptom: str
    recovery_action_tool: str
    recovery_action_operation: str
    recovery_action_id: str
    trusted_recovery_action_id: str | None = None
    recovery_evidence_source: str | None = None
    recovery_action_arguments: dict[str, object] = Field(default_factory=dict)
    outcome: RecoveryOutcomeEvidence
    source_environment_id: str
    source_runtime: str
    source_versions: dict[str, str] = Field(default_factory=dict)
    source_markers: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "source_task_id",
        "source_failure_id",
        "source_resolution_id",
        "source_outcome_id",
        "source_task_problem_statement",
        "failed_action_tool",
        "failed_action_operation",
        "failure_type",
        "failure_signature",
        "failure_symptom",
        "recovery_action_tool",
        "recovery_action_operation",
        "recovery_action_id",
        "source_environment_id",
        "source_runtime",
    )
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence text fields must be non-empty")
        return value

    @field_validator("source_chronological_index")
    @classmethod
    def require_non_negative_chronology(cls, value: int) -> int:
        if value < 0:
            raise ValueError("source_chronological_index must be non-negative")
        return value


class RecoveryAbstractionOutput(BaseModel):
    """Only the semantic fields the model is allowed to generate."""

    model_config = ConfigDict(extra="forbid")

    title: str
    guidance: str
    evidence_summary: str


def build_recovery_evidence_package(
    lineage: RecoveryEvidenceLineage,
) -> RecoveryEvidencePackage:
    """Build a bounded package from trusted source lineage only."""
    failed_action = lineage.failed_action.planned_action
    recovery_action_record = cast(Any, lineage.recovery_action)
    if recovery_action_record is None:
        raise RecoveryAbstractionValidationError(
            "trusted recovery action is unavailable in observed-change lineage"
        )
    if len(lineage.recovery_action_candidates) > 1:
        raise RecoveryAbstractionValidationError(
            "objective-anchored recovery lineage has ambiguous observed-change actions"
        )
    recovery_action = recovery_action_record.planned_action
    if not recovery_action.tool.strip() or not recovery_action.operation.strip():
        raise RecoveryAbstractionValidationError(
            "trusted recovery action tool and operation are required"
        )
    outcome = lineage.outcome
    return RecoveryEvidencePackage(
        source_failure_id=lineage.failure.id,
        source_resolution_id=lineage.resolution.id,
        source_outcome_id=lineage.outcome.id,
        source_task_id=lineage.task.id,
        source_chronological_index=lineage.task.chronological_index,
        source_task_problem_statement=lineage.task.problem_statement,
        failed_action_tool=failed_action.tool,
        failed_action_operation=failed_action.operation,
        failed_action_arguments=normalize_arguments(failed_action.arguments),
        failure_type=lineage.failure.failure_type.value,
        failure_signature=lineage.failure.signature,
        failure_symptom=lineage.failure.symptom,
        recovery_action_tool=recovery_action.tool,
        recovery_action_operation=recovery_action.operation,
        recovery_action_id=recovery_action.id,
        trusted_recovery_action_id=lineage.trusted_recovery_action_id,
        recovery_evidence_source=lineage.recovery_evidence_source,
        recovery_action_arguments=normalize_arguments(recovery_action.arguments),
        outcome=RecoveryOutcomeEvidence(
            success=outcome.success,
            tests_passed=outcome.tests_passed,
            tests_failed=outcome.tests_failed,
            exit_code=outcome.exit_code,
        ),
        source_environment_id=lineage.environment.id,
        source_runtime=lineage.environment.runtime,
        source_versions=dict(lineage.environment.versions),
        source_markers=dict(lineage.environment.markers),
    )


def create_recovery_abstraction_agent(
    settings: Settings,
    *,
    model: Model | None = None,
) -> Agent[None, RecoveryAbstractionOutput]:
    """Create the OpenRouter abstraction agent with test injection."""
    selected_model = model
    if selected_model is None:
        configured_model = settings.openrouter_abstraction_model
        if not configured_model or not configured_model.strip():
            raise RecoveryAbstractionModelError(
                "openrouter_abstraction_model must be supplied through Settings or "
                "OPENROUTER_ABSTRACTION_MODEL"
            )
        api_key = settings.openrouter_api_key
        if not api_key or not api_key.strip():
            raise RecoveryAbstractionModelError(
                "OpenRouter provider unavailable: OPENROUTER_API_KEY must be supplied "
                "through Settings or the environment"
            )
        selected_model = OpenAIChatModel(
            configured_model,
            provider=OpenRouterProvider(api_key=api_key),
        )
    return Agent(
        selected_model,
        output_type=PromptedOutput(RecoveryAbstractionOutput),
        system_prompt=RECOVERY_ABSTRACTION_SYSTEM_PROMPT,
        model_settings=RECOVERY_MODEL_SETTINGS,
        retries=0,
    )


def _evidence_prompt(evidence: RecoveryEvidencePackage) -> str:
    payload = json.dumps(
        evidence.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return (
        f"Recovery abstraction prompt version: {RECOVERY_ABSTRACTION_PROMPT_VERSION}\n"
        "Use this bounded historical evidence package:\n"
        f"{payload}"
    )


def _run_structured_abstraction(
    agent: Agent[None, RecoveryAbstractionOutput],
    evidence: RecoveryEvidencePackage,
) -> RecoveryAbstractionOutput:
    try:
        return agent.run_sync(_evidence_prompt(evidence)).output
    except ModelAPIError as error:
        raise RecoveryAbstractionModelError(
            f"North Mini Code/OpenRouter abstraction failed: {error}"
        ) from error
    except UnexpectedModelBehavior as error:
        raise RecoveryAbstractionStructuredOutputError(
            f"North Mini Code returned invalid structured abstraction output: {error}"
        ) from error


_FORBIDDEN_METADATA = (
    "family_id",
    "family label",
    "mutation family",
    "mutation_label",
    "benchmark recovery annotation",
    "expected solution",
    "gold patch",
    "future-task",
    "future task",
    "transfer-task",
    "transfer task",
    "later task",
)
_GENERIC_GUIDANCE = {"try again", "fix the issue", "be careful"}
_PATH_ARGUMENT_KEYS = {
    "path",
    "file",
    "filename",
    "file_path",
    "filepath",
    "source_path",
    "target_path",
    "patch_path",
}
_IDENTIFIER_ARGUMENT_KEYS = {
    "id",
    "identifier",
    "task_id",
    "failure_id",
    "run_id",
    "repository",
    "repository_id",
}
_CONTENT_ARGUMENT_KEYS = {"content", "code", "patch", "diff", "source"}


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def validate_recovery_abstraction(
    output: RecoveryAbstractionOutput,
    evidence: RecoveryEvidencePackage,
) -> RecoveryAbstractionOutput:
    """Apply conservative deterministic groundedness checks."""
    for field_name in ("title", "guidance", "evidence_summary"):
        value = getattr(output, field_name)
        if not isinstance(value, str) or not value.strip():
            raise RecoveryAbstractionValidationError(f"abstraction {field_name} must be non-empty")

    normalized_guidance = _normalized_text(output.guidance)
    if normalized_guidance in _GENERIC_GUIDANCE:
        raise RecoveryAbstractionValidationError(
            "abstraction guidance is too generic to be operational"
        )

    generated_text = _normalized_text(
        " ".join((output.title, output.guidance, output.evidence_summary))
    )
    for forbidden in _FORBIDDEN_METADATA:
        if _normalized_text(forbidden) in generated_text:
            raise RecoveryAbstractionValidationError(
                f"abstraction contains forbidden benchmark/future metadata: {forbidden}"
            )

    forbidden_source_fragments: dict[str, set[str]] = {
        "source identifier": {
            evidence.source_task_id,
            evidence.source_failure_id,
            evidence.source_resolution_id,
            evidence.source_outcome_id,
            evidence.source_environment_id,
        },
        "source path": set(),
        "copied source fragment": set(),
        "large literal action payload": set(),
    }

    def iter_string_values(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            strings: list[str] = []
            for nested in mapping.values():
                strings.extend(iter_string_values(nested))
            return strings
        if isinstance(value, list):
            values = cast(list[object], value)
            strings = []
            for nested in values:
                strings.extend(iter_string_values(nested))
            return strings
        return []

    def is_substantial_copied_material(value: str) -> bool:
        """Recognize multiline code/patch shape without scoring semantics.

        Two or more code markers across at least three non-empty lines, or a
        long value with three markers, is treated as copied source material.
        Short operational phrases such as ``return value`` do not qualify.
        """
        text = value.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        code_markers = re.findall(
            r"(?:\bdef\b|\bclass\b|\bimport\b|\bfrom\b|\breturn\b|"
            r"\bif\b|\bfor\b|\bwhile\b|->|[={}();])",
            text,
        )
        return (len(lines) >= 3 and len(code_markers) >= 2) or (
            len(text) >= 120 and len(code_markers) >= 3
        )

    def collect_argument_leakage(value: object, key: str | None = None) -> None:
        normalized_key = key.casefold().replace("-", "_") if key else None
        if normalized_key in _PATH_ARGUMENT_KEYS:
            forbidden_source_fragments["source path"].update(
                item for item in iter_string_values(value) if item.strip()
            )
        elif normalized_key in _IDENTIFIER_ARGUMENT_KEYS:
            forbidden_source_fragments["source identifier"].update(
                item for item in iter_string_values(value) if item.strip()
            )
        elif normalized_key in _CONTENT_ARGUMENT_KEYS:
            for item in iter_string_values(value):
                if is_substantial_copied_material(item):
                    forbidden_source_fragments["copied source fragment"].add(item)
        for item in iter_string_values(value):
            if len(item.strip()) >= 120 or len(item.splitlines()) >= 3:
                forbidden_source_fragments["large literal action payload"].add(item)
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            for nested_key, nested_value in mapping.items():
                if isinstance(nested_key, str):
                    collect_argument_leakage(nested_value, nested_key)

    collect_argument_leakage(evidence.failed_action_arguments)
    collect_argument_leakage(evidence.recovery_action_arguments)
    raw_generated_text = " ".join(
        (output.title, output.guidance, output.evidence_summary)
    ).casefold()
    normalized_generated_text = _normalized_text(raw_generated_text)
    for category, fragments in forbidden_source_fragments.items():
        for fragment in fragments:
            if (
                fragment.strip().casefold() in raw_generated_text
                or _normalized_text(fragment) in normalized_generated_text
            ):
                raise RecoveryAbstractionValidationError(
                    f"abstraction leakage rejected: {category}"
                )
    return output


def deterministic_recovery_pattern_id(evidence: RecoveryEvidencePackage) -> str:
    """Return the stable identity for one exact source evidence chain."""
    provenance = {
        "failure_id": evidence.source_failure_id,
        "resolution_id": evidence.source_resolution_id,
        "outcome_id": evidence.source_outcome_id,
        "task_id": evidence.source_task_id,
        "chronological_index": evidence.source_chronological_index,
        "failure_type": evidence.failure_type,
        "failed_action_tool": evidence.failed_action_tool,
        "failed_action_operation": evidence.failed_action_operation,
    }
    digest = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"recovery-pattern-{digest}"


def construct_recovery_pattern(
    output: RecoveryAbstractionOutput,
    evidence: RecoveryEvidencePackage,
    *,
    source_failure_id: str,
    source_resolution_id: str,
    source_outcome_id: str,
    created_at: datetime,
) -> RecoveryPattern:
    """Combine model semantics with deterministic trusted provenance."""
    validate_recovery_abstraction(output, evidence)
    if (
        source_failure_id != evidence.source_failure_id
        or source_resolution_id != evidence.source_resolution_id
        or source_outcome_id != evidence.source_outcome_id
    ):
        raise RecoveryAbstractionValidationError(
            "supplied provenance does not match the trusted evidence package"
        )
    if not evidence.outcome.success:
        raise RecoveryAbstractionValidationError(
            "objective outcome is not successful recovery evidence"
        )
    applicability_tool: str | None = None
    applicability_operation: str | None = None
    if evidence.recovery_evidence_source == OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE:
        trusted_action_id = evidence.trusted_recovery_action_id
        if not trusted_action_id or trusted_action_id != evidence.recovery_action_id:
            raise RecoveryAbstractionValidationError(
                "objective-anchored recovery lineage lacks one trusted observed mutation action"
            )
        if (
            not evidence.recovery_action_tool.strip()
            or not evidence.recovery_action_operation.strip()
        ):
            raise RecoveryAbstractionValidationError(
                "objective-anchored recovery action tool and operation are required"
            )
        applicability_tool = evidence.recovery_action_tool
        applicability_operation = evidence.recovery_action_operation
    elif evidence.recovery_evidence_source is not None:
        raise RecoveryAbstractionValidationError(
            "unsupported recovery evidence source cannot derive applicability"
        )
    return RecoveryPattern(
        id=deterministic_recovery_pattern_id(evidence),
        title=output.title.strip(),
        guidance=output.guidance.strip(),
        source_failure_id=source_failure_id,
        source_resolution_id=source_resolution_id,
        source_outcome_id=source_outcome_id,
        source_task_id=evidence.source_task_id,
        source_chronological_index=evidence.source_chronological_index,
        source_tool=evidence.failed_action_tool,
        source_operation=evidence.failed_action_operation,
        applicability_tool=applicability_tool,
        applicability_operation=applicability_operation,
        source_failure_type=evidence.failure_type,
        trigger=RecoveryTrigger(
            failure_type=evidence.failure_type,
            failure_signature=evidence.failure_signature,
            failure_context=evidence.failure_symptom,
            source_task_problem_statement=evidence.source_task_problem_statement,
            source_tool=evidence.failed_action_tool,
            source_operation=evidence.failed_action_operation,
            source_action_arguments=dict(evidence.failed_action_arguments),
            # Exact equality remains the conservative default.  A later
            # evidence-backed pattern may explicitly opt out per trigger.
            version_sensitive=True,
        ),
        environment_constraints=EnvironmentConstraints(
            runtime=evidence.source_runtime,
            versions=evidence.source_versions,
            markers=evidence.source_markers,
        ),
        verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        evidence_count=1,
        evidence_summary=output.evidence_summary.strip(),
        created_at=created_at,
    )


def abstract_and_persist_recovery_pattern(
    lineage: RecoveryEvidenceLineage,
    repository: OperationalMemoryRepository,
    settings: Settings,
    *,
    model: Model | None = None,
    created_at: datetime | None = None,
) -> RecoveryPattern:
    """Abstract one bounded lineage and persist it only after validation."""
    evidence = build_recovery_evidence_package(lineage)
    agent = create_recovery_abstraction_agent(settings, model=model)
    output = _run_structured_abstraction(agent, evidence)
    pattern = construct_recovery_pattern(
        output,
        evidence,
        source_failure_id=lineage.failure.id,
        source_resolution_id=lineage.resolution.id,
        source_outcome_id=lineage.outcome.id,
        created_at=created_at or datetime.now(UTC),
    )
    repository.save_recovery_pattern(pattern)
    return pattern


__all__ = [
    "FROZEN_RECOVERY_MODEL",
    "OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE",
    "RECOVERY_MODEL_SETTINGS",
    "RecoveryAbstractionModelError",
    "RecoveryAbstractionOutput",
    "RecoveryAbstractionStructuredOutputError",
    "RecoveryAbstractionValidationError",
    "RecoveryEvidencePackage",
    "RecoveryOutcomeEvidence",
    "abstract_and_persist_recovery_pattern",
    "build_recovery_evidence_package",
    "construct_recovery_pattern",
    "create_recovery_abstraction_agent",
    "deterministic_recovery_pattern_id",
    "validate_recovery_abstraction",
]
