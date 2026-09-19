"""Freeze and operate the Track B Gate B1 B0-vs-O1 harness.

This module deliberately separates Gate B1 bookkeeping from provider
execution.  ``freeze_gate_b1`` and ``build_execution_plan`` are safe to run
without Docker or a provider.  ``execute_gate_b1`` accepts an injected
condition runner so production execution can reuse the existing Sprint 3D-A
runner without adding a second model/runtime implementation here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from experiments.oracle import FrozenOracleResolver
from graph_swarm.agent.coding_agent import (
    CODING_AGENT_TOOL_RETRIES,
    MODEL_REQUEST_TIMEOUT_SECONDS,
)
from graph_swarm.agent.pacing import ProviderRequestPacing
from graph_swarm.research.benchmark_environments import (
    BenchmarkEnvironmentConfigurationError,
    load_benchmark_environment_policy,
)
from graph_swarm.research.contracts import ExperimentCondition, ExperimentRunArtifact
from graph_swarm.research.runner import (
    ExperimentConfigurationError,
    ExperimentExecution,
    LoadedExperimentConfiguration,
    load_experiment_configuration,
    load_task_cases,
)

GATE_B1_ID = "B1"
GATE_B1_EXPERIMENT_ID = "GS-E003"
GATE_B1_TASK_IDS: tuple[str, ...] = tuple(f"GS-T{i:03d}" for i in range(6, 16))
GATE_B1_CONDITIONS: tuple[ExperimentCondition, ...] = (
    ExperimentCondition.B0,
    ExperimentCondition.O1,
)
GATE_B1_EXPECTED_RUNS = len(GATE_B1_TASK_IDS) * len(GATE_B1_CONDITIONS)
GATE_B1_MAX_ACTIONS = 20
GATE_B1_MAX_REQUESTS = 24
GATE_B1_TOOL_RETRIES = 3
GATE_B1_TIMEOUT_SECONDS = 300.0
GATE_B1_PACING_SECONDS = 5.0
GATE_B1_NOMINAL_RPM = 12.0


class GateB1Error(ValueError):
    """Base error for a frozen Gate B1 contract or evidence violation."""


class GateB1MetadataError(GateB1Error):
    """Raised when frozen task, environment, or Oracle metadata is incomplete."""


class GateB1DuplicateError(GateB1Error):
    """Raised when an immutable valid observation would be duplicated."""


class GateB1ReplacementError(GateB1Error):
    """Raised when a manual replacement does not reference an invalid attempt."""


class GateB1EnvironmentMismatch(GateB1Error):
    """Raised when a prepared runtime fingerprint differs from the freeze."""

    def __init__(self, expected: str, actual: str | None) -> None:
        self.expected = expected
        self.actual = actual
        self.actual_environment_fingerprint = actual
        super().__init__(
            f"environment fingerprint mismatch: expected {expected}, actual {actual}"
        )


class AttemptStatus(StrEnum):
    """Status of one immutable attempt in the Gate B1 evidence index."""

    VALID_EXPERIMENTAL_RUN = "valid_experimental_run"
    BOUNDED_TERMINATION = "bounded_termination"
    INVALID_PROVIDER_ATTEMPT = "invalid_provider_attempt"
    INVALID_INFRASTRUCTURE_ATTEMPT = "invalid_infrastructure_attempt"

    @property
    def valid_observation(self) -> bool:
        return self in {
            AttemptStatus.VALID_EXPERIMENTAL_RUN,
            AttemptStatus.BOUNDED_TERMINATION,
        }


class FrozenGateB1Task(BaseModel):
    """Task metadata copied into the Gate B1 freeze record."""

    task_id: str
    chronological_index: int
    family_id: str
    repository: str
    image_name: str
    review_id: str
    occurrence_index: int
    environment_task_id: str
    environment_fingerprint: str

    @field_validator(
        "task_id",
        "family_id",
        "repository",
        "image_name",
        "review_id",
        "environment_task_id",
        "environment_fingerprint",
    )
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Gate B1 frozen task metadata must be non-empty")
        return value


class GateB1Manifest(BaseModel):
    """Machine-readable execution identity frozen before provider execution."""

    model_config = ConfigDict(frozen=True)

    manifest_id: str
    experiment_id: str = GATE_B1_EXPERIMENT_ID
    gate_id: str = GATE_B1_ID
    task_ids: tuple[str, ...] = GATE_B1_TASK_IDS
    conditions: tuple[ExperimentCondition, ...] = GATE_B1_CONDITIONS
    expected_runs: int = GATE_B1_EXPECTED_RUNS
    model_config_path: str
    prompt_version: str
    provider: Literal["openrouter"] = "openrouter"
    model: str = "runtime-selected-from-OPENROUTER_MODEL"
    temperature: float = 0.0
    max_actions: int = GATE_B1_MAX_ACTIONS
    max_requests: int = GATE_B1_MAX_REQUESTS
    tool_retries: int = GATE_B1_TOOL_RETRIES
    agent_timeout_seconds: float = GATE_B1_TIMEOUT_SECONDS
    model_request_timeout_seconds: float = GATE_B1_TIMEOUT_SECONDS
    pacing_interval_seconds: float = GATE_B1_PACING_SECONDS
    nominal_requests_per_minute: float = GATE_B1_NOMINAL_RPM
    runtime_environment_manifest: str
    runtime_environment_manifest_identity: str
    oracle_intervention_boundary: Literal["task_start"] = "task_start"
    oracle_delivery: Literal["pre_first_model_request"] = "pre_first_model_request"
    objective_evaluator: str = "frozen_swesmith_fail_to_pass"
    repetition_count: int = 1
    t_enabled: bool = False
    created_at: datetime
    tasks: tuple[FrozenGateB1Task, ...]

    @model_validator(mode="after")
    def validate_freeze(self) -> GateB1Manifest:
        if self.gate_id != GATE_B1_ID:
            raise ValueError("Gate B1 manifest has the wrong gate identifier")
        if self.experiment_id != GATE_B1_EXPERIMENT_ID:
            raise ValueError("Gate B1 manifest has the wrong experiment identity")
        if self.task_ids != GATE_B1_TASK_IDS:
            raise ValueError("Gate B1 task order must be exactly GS-T006 through GS-T015")
        if self.conditions != GATE_B1_CONDITIONS:
            raise ValueError("Gate B1 conditions must be exactly B0 then O1")
        if self.expected_runs != GATE_B1_EXPECTED_RUNS:
            raise ValueError("Gate B1 expected run count must be 20")
        if tuple(task.task_id for task in self.tasks) != self.task_ids:
            raise ValueError("Gate B1 frozen task metadata is incomplete or out of order")
        if self.repetition_count != 1 or self.t_enabled:
            raise ValueError("Gate B1 permits one repetition and no T condition")
        return self

    @property
    def task_map(self) -> dict[str, FrozenGateB1Task]:
        return {task.task_id: task for task in self.tasks}


class PlannedGateB1Execution(BaseModel):
    """One deterministic condition execution in the frozen plan."""

    plan_index: int
    task_id: str
    chronological_index: int
    condition: ExperimentCondition
    pair_key: str
    experiment_id: str
    manifest_id: str


class AttemptRecord(BaseModel):
    """Append-only index row for valid, bounded, or invalid evidence."""

    model_config = ConfigDict(frozen=True)

    manifest_id: str
    experiment_id: str
    gate_id: str = GATE_B1_ID
    task_id: str
    condition: ExperimentCondition
    run_id: str
    environment_fingerprint: str | None = None
    expected_environment_fingerprint: str | None = None
    actual_environment_fingerprint: str | None = None
    status: AttemptStatus
    valid_observation: bool
    artifact_path: str | None = None
    raw_evidence_path: str | None = None
    replacement_of: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    objective_result: bool | None = None
    termination: dict[str, Any] | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def status_matches_validity(self) -> AttemptRecord:
        if self.valid_observation != self.status.valid_observation:
            raise ValueError("attempt validity does not match attempt status")
        return self


class GateB1PairStatus(BaseModel):
    """Current pair view reconstructed from immutable attempt rows."""

    task_id: str
    b0_run_id: str | None = None
    b0_validity: AttemptStatus | None = None
    o1_run_id: str | None = None
    o1_validity: AttemptStatus | None = None
    pair_complete: bool = False


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_manifest_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw: Any = json.loads(line)
            if not isinstance(raw, dict):
                raise GateB1MetadataError(f"benchmark manifest contains a non-object: {path}")
            record = cast(dict[str, Any], raw)
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise GateB1MetadataError("benchmark manifest task ID is missing")
            records[task_id] = record
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateB1MetadataError(
            f"could not read frozen benchmark manifest {path}: {error}"
        ) from error
    return records


def _read_environment_records(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateB1MetadataError(
            f"could not read frozen Gate B1 environment manifest {path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise GateB1MetadataError("Gate B1 environment manifest has no task mapping")
    environment_manifest = cast(dict[str, Any], raw)
    if not isinstance(environment_manifest.get("tasks"), dict):
        raise GateB1MetadataError("Gate B1 environment manifest has no task mapping")
    return cast(dict[str, dict[str, Any]], environment_manifest["tasks"])


def _config_for_gate(
    project_root: Path,
    config_path: Path,
) -> LoadedExperimentConfiguration:
    configuration = load_experiment_configuration(config_path, project_root=project_root)
    if configuration.config.experiment_id != GATE_B1_EXPERIMENT_ID:
        raise GateB1MetadataError("Gate B1 must use experiment GS-E003")
    if configuration.model.provider != "openrouter":
        raise GateB1MetadataError("Gate B1 provider is frozen to OpenRouter")
    if configuration.model.settings.get("temperature") != 0:
        raise GateB1MetadataError("Gate B1 temperature is frozen to zero")
    limits = configuration.config.limits
    if (
        limits.max_actions != GATE_B1_MAX_ACTIONS
        or limits.max_requests != GATE_B1_MAX_REQUESTS
        or limits.timeout_seconds != GATE_B1_TIMEOUT_SECONDS
    ):
        raise GateB1MetadataError("Gate B1 run limits do not match the frozen 20/24/300 contract")
    return configuration


def freeze_gate_b1(
    *,
    project_root: Path,
    output_path: Path,
    config_path: Path | None = None,
    environment_manifest_path: Path | None = None,
    created_at: datetime | None = None,
) -> GateB1Manifest:
    """Create an exclusive Gate B1 freeze record without executing anything."""
    root = project_root.expanduser().resolve()
    selected_config = (config_path or root / "configs/experiments/rollout_3a_pilot.yaml").resolve()
    configuration = _config_for_gate(root, selected_config)
    try:
        policy_path = root / "configs/research/benchmark_environments.toml"
        policy = load_benchmark_environment_policy(policy_path)
    except (OSError, BenchmarkEnvironmentConfigurationError) as error:
        raise GateB1MetadataError(f"could not load Gate B1 environment policy: {error}") from error
    if policy.task_order != GATE_B1_TASK_IDS:
        raise GateB1MetadataError("environment policy task order is not the frozen T006-T015 order")

    cases = load_task_cases(
        configuration.task_manifest_path,
        problem_statements_path=configuration.task_problems_path,
    )
    by_id = {case.task.id: case for case in cases}
    if (
        tuple(case.task.id for case in cases if case.task.id in GATE_B1_TASK_IDS)
        != GATE_B1_TASK_IDS
    ):
        raise GateB1MetadataError(
            "frozen benchmark metadata does not contain T006-T015 chronologically"
        )
    records = _read_manifest_records(configuration.task_manifest_path)
    try:
        oracle = FrozenOracleResolver.from_frozen_files(
            configuration.task_manifest_path,
            configuration.task_problems_path,
        )
    except ExperimentConfigurationError as error:
        raise GateB1MetadataError(str(error)) from error
    environment_path = (
        environment_manifest_path
        or root / "configs/research/gate_b1_environments.json"
    ).resolve()
    environment_records = _read_environment_records(environment_path)
    frozen_tasks: list[FrozenGateB1Task] = []
    for task_id in GATE_B1_TASK_IDS:
        case = by_id.get(task_id)
        record = records.get(task_id)
        if case is None or record is None:
            raise GateB1MetadataError(f"missing frozen metadata for {task_id}")
        if record.get("repository") != case.task.repository:
            raise GateB1MetadataError(f"repository metadata mismatch for {task_id}")
        image_name = record.get("image_name")
        review_id = record.get("review_id")
        if not isinstance(image_name, str) or not image_name.strip():
            raise GateB1MetadataError(f"missing frozen environment image for {task_id}")
        if not isinstance(review_id, str) or not review_id.strip():
            raise GateB1MetadataError(f"missing frozen Oracle review ID for {task_id}")
        try:
            oracle.validate_case(case)
        except ExperimentConfigurationError as error:
            raise GateB1MetadataError(str(error)) from error
        if oracle.review_id_for(task_id) != review_id:
            raise GateB1MetadataError(f"Oracle review ID mismatch for {task_id}")
        if not (oracle.recovery_pattern_for(task_id) or "").strip():
            raise GateB1MetadataError(f"Oracle recovery pattern is empty for {task_id}")
        task_policy = policy.task(task_id)
        if (
            task_policy.runtime != "manifest_container_required"
            or not task_policy.require_validated_environment
        ):
            raise GateB1MetadataError(
                f"runtime policy is not frozen prepared-container policy for {task_id}"
            )
        environment = environment_records.get(task_id)
        fingerprint = None if environment is None else environment.get("environment_fingerprint")
        if (
            environment is None
            or
            not isinstance(fingerprint, str)
            or not fingerprint.strip()
            or environment.get("validated") is not True
        ):
            raise GateB1MetadataError(
                f"validated environment fingerprint is missing for {task_id}"
            )
        frozen_tasks.append(
            FrozenGateB1Task(
                task_id=task_id,
                chronological_index=case.task.chronological_index,
                family_id=case.task.family_id,
                repository=case.task.repository,
                image_name=image_name,
                review_id=review_id,
                occurrence_index=case.occurrence_index,
                environment_task_id=task_id,
                environment_fingerprint=fingerprint,
            )
        )

    model_config_path = configuration.config.model_config_path
    identity = _canonical_hash(
        {
            "benchmark_manifest": configuration.task_manifest_path.read_bytes().decode("utf-8"),
            "environment_policy": policy_path.read_bytes().decode("utf-8"),
            "environment_manifest": environment_path.read_bytes().decode("utf-8"),
            "model_config": configuration.model.model_dump(mode="json"),
        }
    )
    payload = {
        "experiment_id": GATE_B1_EXPERIMENT_ID,
        "gate_id": GATE_B1_ID,
        "task_ids": GATE_B1_TASK_IDS,
        "conditions": [condition.value for condition in GATE_B1_CONDITIONS],
        "model_config_path": model_config_path,
        "prompt_version": configuration.model.prompt_version,
        "runtime_environment_manifest": str(environment_path),
        "runtime_environment_manifest_identity": identity,
        "tasks": [task.model_dump(mode="json") for task in frozen_tasks],
    }
    manifest = GateB1Manifest(
        manifest_id=f"{GATE_B1_EXPERIMENT_ID}-{GATE_B1_ID}-{_canonical_hash(payload)[:16]}",
        model_config_path=model_config_path,
        prompt_version=configuration.model.prompt_version,
        runtime_environment_manifest=str(environment_path),
        runtime_environment_manifest_identity=identity,
        created_at=created_at or datetime.now(UTC),
        tasks=tuple(frozen_tasks),
    )
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if output.exists():
        try:
            existing = GateB1Manifest.model_validate_json(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise GateB1MetadataError(
                f"existing Gate B1 freeze record is invalid: {output}"
            ) from error
        if existing.manifest_id != manifest.manifest_id:
            raise GateB1MetadataError(
                "refusing to replace a different immutable Gate B1 freeze record"
            )
        return existing
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    return manifest


def load_gate_b1_manifest(path: Path) -> GateB1Manifest:
    try:
        return GateB1Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise GateB1MetadataError(f"could not load Gate B1 manifest {path}: {error}") from error


def build_execution_plan(
    manifest: GateB1Manifest,
    *,
    task_id: str | None = None,
    start_task: str | None = None,
    end_task: str | None = None,
    conditions: Sequence[ExperimentCondition] | None = None,
) -> tuple[PlannedGateB1Execution, ...]:
    """Build a deterministic chronological B0-then-O1 plan."""
    if task_id is not None and (start_task is not None or end_task is not None):
        raise GateB1Error("--task-id cannot be combined with --start-task/--end-task")
    selected_conditions = tuple(conditions or GATE_B1_CONDITIONS)
    if any(condition not in GATE_B1_CONDITIONS for condition in selected_conditions):
        raise GateB1Error("Gate B1 supports only B0 and O1; T cannot execute")
    if len(set(selected_conditions)) != len(selected_conditions):
        raise GateB1Error("Gate B1 condition selection must not duplicate a condition")
    if selected_conditions not in (
        GATE_B1_CONDITIONS,
        (ExperimentCondition.O1,),
        (ExperimentCondition.B0,),
    ):
        raise GateB1Error("within-task condition order must be B0 then O1")

    ids = list(manifest.task_ids)
    if task_id is not None:
        if task_id not in ids:
            raise GateB1Error(f"unknown frozen Gate B1 task: {task_id}")
        selected_ids = [task_id]
    else:
        start = ids.index(start_task) if start_task is not None and start_task in ids else 0
        if start_task is not None and start_task not in ids:
            raise GateB1Error(f"unknown frozen Gate B1 start task: {start_task}")
        end = ids.index(end_task) if end_task is not None and end_task in ids else len(ids) - 1
        if end_task is not None and end_task not in ids:
            raise GateB1Error(f"unknown frozen Gate B1 end task: {end_task}")
        if start > end:
            raise GateB1Error("Gate B1 task range is reversed")
        selected_ids = ids[start : end + 1]

    plan: list[PlannedGateB1Execution] = []
    for selected_id in selected_ids:
        task = manifest.task_map[selected_id]
        for condition in selected_conditions:
            plan.append(
                PlannedGateB1Execution(
                    plan_index=len(plan),
                    task_id=selected_id,
                    chronological_index=task.chronological_index,
                    condition=condition,
                    pair_key=f"{manifest.manifest_id}:{selected_id}",
                    experiment_id=manifest.experiment_id,
                    manifest_id=manifest.manifest_id,
                )
            )
    return tuple(plan)


def validate_execution_settings(
    manifest: GateB1Manifest,
    configurations: Iterable[LoadedExperimentConfiguration],
    *,
    pacing: ProviderRequestPacing | None = None,
    selected_model: str | None = None,
) -> None:
    """Validate effective run configuration immediately before execution."""
    configs = tuple(configurations)
    if len(configs) != 2:
        raise GateB1Error("Gate B1 requires one B0 and one O1 configuration")
    expected_conditions = {ExperimentCondition.B0, ExperimentCondition.O1}
    observed_conditions: set[ExperimentCondition] = set()
    for configuration in configs:
        _config_for_gate(configuration.project_root, configuration.config_path)
        if len(configuration.config.conditions) != 1:
            raise GateB1Error("each Gate B1 configuration must enable exactly one condition")
        observed_conditions.add(configuration.config.conditions[0])
        if configuration.model.prompt_version != manifest.prompt_version:
            raise GateB1Error("prompt version differs from the frozen Gate B1 manifest")
        actual_model_config = (
            configuration.project_root / configuration.config.model_config_path
        ).resolve()
        frozen_model_config = (
            configuration.project_root / manifest.model_config_path
        ).resolve()
        if actual_model_config != frozen_model_config:
            raise GateB1Error("model configuration path differs from the frozen Gate B1 manifest")
    if observed_conditions != expected_conditions:
        raise GateB1Error("Gate B1 requires exactly one B0 and one O1 configuration")
    if selected_model is not None and not selected_model.strip():
        raise GateB1Error("runtime-selected model identity cannot be empty")
    if pacing is not None:
        config = pacing.config
        if config.min_interval_seconds != manifest.pacing_interval_seconds:
            raise GateB1Error("provider pacing interval differs from the frozen Gate B1 manifest")
        if config.max_nominal_requests_per_minute != manifest.nominal_requests_per_minute:
            raise GateB1Error("provider pacing ceiling differs from the frozen Gate B1 manifest")
    if MODEL_REQUEST_TIMEOUT_SECONDS != manifest.model_request_timeout_seconds:
        raise GateB1Error("model request timeout differs from the frozen Gate B1 manifest")
    if CODING_AGENT_TOOL_RETRIES != manifest.tool_retries:
        raise GateB1Error("tool retry count differs from the frozen Gate B1 manifest")


def validate_artifact_contract(
    artifact: ExperimentRunArtifact,
    manifest: GateB1Manifest,
    *,
    expected_condition: ExperimentCondition | None = None,
) -> None:
    """Validate advice and frozen identity fields in a recorded run artifact."""
    if (
        artifact.experiment_id != manifest.experiment_id
        or artifact.gate_id != manifest.gate_id
        or artifact.execution_manifest_id != manifest.manifest_id
        or (
        expected_condition is not None and artifact.condition != expected_condition
        )
    ):
        raise GateB1Error("run artifact does not belong to the selected Gate B1 contract")
    task = manifest.task_map.get(artifact.task_id)
    if task is None or artifact.chronological_index != task.chronological_index:
        raise GateB1Error(f"run artifact task metadata is not frozen for {artifact.task_id}")
    if artifact.condition is ExperimentCondition.B0:
        if artifact.advice_count != 0 or artifact.advice_review_id is not None:
            raise GateB1Error("B0 must record advice_count=0 and advice_review_id=null")
        return
    if artifact.condition is not ExperimentCondition.O1:
        raise GateB1Error("T is disabled at Gate B1")
    if (
        artifact.advice_count != 1
        or artifact.advice_review_id is None
        or artifact.advice_review_id != task.review_id
        or artifact.advice_received is None
        or not (artifact.advice_received.recovery_summary or "").strip()
        or artifact.advice_intervention_boundary != manifest.oracle_intervention_boundary
        or artifact.advice_delivery_timing != manifest.oracle_delivery
    ):
        raise GateB1Error(
            f"O1 artifact for {artifact.task_id} is missing the exact frozen Oracle intervention"
        )


def validate_o1_pre_execution(manifest: GateB1Manifest, task_id: str) -> None:
    """Validate the frozen O1 mapping immediately before an O1 callback runs."""
    task = manifest.task_map.get(task_id)
    if task is None:
        raise GateB1MetadataError(f"O1 task is not in the frozen Gate B1 task list: {task_id}")
    if not task.review_id.strip():
        raise GateB1MetadataError(f"O1 task {task_id} has no frozen review/provenance ID")
    if manifest.oracle_intervention_boundary != "task_start":
        raise GateB1MetadataError("O1 intervention boundary is not task_start")
    if manifest.oracle_delivery != "pre_first_model_request":
        raise GateB1MetadataError("O1 delivery is not pre_first_model_request")


def classify_attempt(
    *,
    artifact: ExperimentRunArtifact | None,
    error: BaseException | None = None,
) -> AttemptStatus:
    """Classify an attempt without turning provider/infrastructure failure into data."""
    if artifact is not None and artifact.termination is not None and error is not None:
        return AttemptStatus.BOUNDED_TERMINATION
    if error is None:
        return (
            AttemptStatus.BOUNDED_TERMINATION
            if artifact is not None and artifact.termination is not None
            else AttemptStatus.VALID_EXPERIMENTAL_RUN
        )
    message = f"{type(error).__name__}: {error}".lower()
    provider_markers = (
        "429",
        "rate limit",
        "provider",
        "openrouter",
        "transport",
        "api error",
        "connection reset",
        "http 4",
        "http 5",
        "model request",
        "provider unavailable",
    )
    if any(marker in message for marker in provider_markers):
        return AttemptStatus.INVALID_PROVIDER_ATTEMPT
    return AttemptStatus.INVALID_INFRASTRUCTURE_ATTEMPT


def _record_from_execution(
    manifest: GateB1Manifest,
    planned: PlannedGateB1Execution,
    execution: ExperimentExecution,
    *,
    replacement_of: str | None,
) -> AttemptRecord:
    status = classify_attempt(artifact=execution.artifact, error=execution.error)
    expected_fingerprint = manifest.task_map[planned.task_id].environment_fingerprint
    actual_fingerprint = execution.actual_environment_fingerprint
    if actual_fingerprint != expected_fingerprint:
        status = AttemptStatus.INVALID_INFRASTRUCTURE_ATTEMPT
    if status.valid_observation:
        validate_artifact_contract(
            execution.artifact,
            manifest,
            expected_condition=planned.condition,
        )
    return AttemptRecord(
        manifest_id=manifest.manifest_id,
        experiment_id=manifest.experiment_id,
        task_id=planned.task_id,
        condition=planned.condition,
        run_id=execution.artifact.run_id,
        environment_fingerprint=expected_fingerprint,
        expected_environment_fingerprint=expected_fingerprint,
        actual_environment_fingerprint=actual_fingerprint,
        status=status,
        valid_observation=status.valid_observation,
        artifact_path=str(execution.artifact_path),
        raw_evidence_path=str(execution.raw_evidence_path),
        replacement_of=replacement_of,
        error_type=(
            "GateB1EnvironmentMismatch"
            if actual_fingerprint != expected_fingerprint
            else None if execution.error is None else type(execution.error).__name__
        ),
        error_message=(
            str(GateB1EnvironmentMismatch(expected_fingerprint, actual_fingerprint))
            if actual_fingerprint != expected_fingerprint
            else None if execution.error is None else str(execution.error)
        ),
        objective_result=execution.artifact.task_success,
        termination=(
            None
            if execution.artifact.termination is None
            else execution.artifact.termination.model_dump(mode="json")
        ),
    )


class GateB1EvidenceStore:
    """Append-only Gate B1 attempt index and pair reconstruction."""

    def __init__(self, root: Path, manifest: GateB1Manifest) -> None:
        self.root = root.expanduser().resolve()
        self.manifest = manifest
        self.index_path = self.root / "attempts.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    def attempts(self) -> tuple[AttemptRecord, ...]:
        if not self.index_path.is_file():
            return ()
        rows: list[AttemptRecord] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(AttemptRecord.model_validate_json(line))
        return tuple(rows)

    def assert_can_attempt(
        self,
        task_id: str,
        condition: ExperimentCondition,
        *,
        replacement_of: str | None = None,
    ) -> None:
        """Reject duplicate or unlinked reruns before the provider is called."""
        existing = self.attempts()
        valid = [
            item
            for item in existing
            if item.task_id == task_id
            and item.condition is condition
            and item.valid_observation
        ]
        if valid:
            raise GateB1DuplicateError(
                f"completed valid {condition.value} observation already exists for {task_id}"
            )
        invalid = [
            item
            for item in existing
            if (
                item.task_id == task_id
                and item.condition is condition
                and not item.valid_observation
            )
        ]
        if invalid and replacement_of is None:
            raise GateB1ReplacementError(
                f"rerun for {task_id}/{condition.value} requires explicit replacement_of"
            )
        if replacement_of is not None:
            replaced = next((item for item in existing if item.run_id == replacement_of), None)
            if replaced is None or replaced.valid_observation:
                raise GateB1ReplacementError(
                    "replacement_of must reference an existing invalid attempt"
                )
            if replaced.task_id != task_id or replaced.condition is not condition:
                raise GateB1ReplacementError(
                    "replacement attempt must retain task and condition"
                )

    def append(self, record: AttemptRecord, *, replacement_of: str | None = None) -> AttemptRecord:
        if record.manifest_id != self.manifest.manifest_id:
            raise GateB1Error("attempt belongs to a different immutable Gate B1 manifest")
        frozen_task = self.manifest.task_map.get(record.task_id)
        if frozen_task is None:
            raise GateB1MetadataError(f"attempt task is not frozen for Gate B1: {record.task_id}")
        expected_fingerprint = frozen_task.environment_fingerprint
        if (
            record.environment_fingerprint != expected_fingerprint
            or (
                record.expected_environment_fingerprint is not None
                and record.expected_environment_fingerprint != expected_fingerprint
            )
        ):
            raise GateB1MetadataError(
                f"environment fingerprint does not match the frozen contract for {record.task_id}"
            )
        self.assert_can_attempt(
            record.task_id,
            record.condition,
            replacement_of=replacement_of,
        )
        existing = self.attempts()
        if any(item.run_id == record.run_id for item in existing):
            raise GateB1DuplicateError(
                f"run ID already exists in Gate B1 evidence: {record.run_id}"
            )
        if record.replacement_of != replacement_of:
            raise GateB1ReplacementError(
                "replacement metadata does not match the referenced attempt"
            )
        if record.valid_observation and record.replacement_of is None:
            invalid = [
                item
                for item in existing
                if item.task_id == record.task_id
                and item.condition is record.condition
                and not item.valid_observation
            ]
            if invalid:
                raise GateB1ReplacementError(
                    "valid rerun of an invalid attempt must include replacement_of"
                )
        with self.index_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record.model_dump_json() + "\n")
        return record

    def pair_statuses(self) -> tuple[GateB1PairStatus, ...]:
        rows = self.attempts()
        statuses: list[GateB1PairStatus] = []
        for task_id in self.manifest.task_ids:
            values: dict[ExperimentCondition, AttemptRecord | None] = {}
            for condition in GATE_B1_CONDITIONS:
                valid = [
                    item
                    for item in rows
                    if item.task_id == task_id
                    and item.condition is condition
                    and item.valid_observation
                ]
                values[condition] = valid[0] if valid else None
            b0 = values[ExperimentCondition.B0]
            o1 = values[ExperimentCondition.O1]
            statuses.append(
                GateB1PairStatus(
                    task_id=task_id,
                    b0_run_id=None if b0 is None else b0.run_id,
                    b0_validity=None if b0 is None else b0.status,
                    o1_run_id=None if o1 is None else o1.run_id,
                    o1_validity=None if o1 is None else o1.status,
                    pair_complete=b0 is not None and o1 is not None,
                )
            )
        return tuple(statuses)


ExecutionCallback = Callable[[PlannedGateB1Execution], ExperimentExecution]


def execute_gate_b1(
    manifest: GateB1Manifest,
    plan: Sequence[PlannedGateB1Execution],
    *,
    store: GateB1EvidenceStore,
    execute_condition: ExecutionCallback,
    replacements: Mapping[tuple[str, ExperimentCondition], str] | None = None,
) -> tuple[AttemptRecord, ...]:
    """Execute an explicit plan with no retry or outcome-dependent selection."""
    replacements = replacements or {}
    records: list[AttemptRecord] = []
    for planned in plan:
        if planned.condition is ExperimentCondition.O1:
            validate_o1_pre_execution(manifest, planned.task_id)
        replacement = replacements.get((planned.task_id, planned.condition))
        store.assert_can_attempt(
            planned.task_id,
            planned.condition,
            replacement_of=replacement,
        )
        try:
            execution = execute_condition(planned)
        except Exception as error:  # preserve a pre-artifact infrastructure attempt
            status = classify_attempt(artifact=None, error=error)
            record = AttemptRecord(
                manifest_id=manifest.manifest_id,
                experiment_id=manifest.experiment_id,
                task_id=planned.task_id,
                condition=planned.condition,
                run_id=(
                    f"{manifest.experiment_id}-{planned.condition.value}-"
                    f"{planned.task_id}-{uuid.uuid4().hex}"
                ),
                environment_fingerprint=manifest.task_map[
                    planned.task_id
                ].environment_fingerprint,
                expected_environment_fingerprint=manifest.task_map[
                    planned.task_id
                ].environment_fingerprint,
                actual_environment_fingerprint=getattr(
                    error, "actual_environment_fingerprint", None
                ),
                status=status,
                valid_observation=False,
                replacement_of=replacement,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        else:
            record = _record_from_execution(
                manifest,
                planned,
                execution,
                replacement_of=replacement,
            )
        records.append(store.append(record, replacement_of=replacement))
    return tuple(records)


def execute_gate_b1_with_existing_stack(
    manifest: GateB1Manifest,
    plan: Sequence[PlannedGateB1Execution],
    *,
    project_root: Path,
    baseline_root: Path | None = None,
    execution_root: Path | None = None,
    artifact_root: Path,
    replacements: Mapping[tuple[str, ExperimentCondition], str] | None = None,
) -> tuple[AttemptRecord, ...]:
    """Run an explicit plan through the existing Sprint 3D-A stack.

    Environment preparation is preflight-only here: the existing helper loads
    and validates already-prepared Docker environments and never builds them.
    """
    from experiments import sprint3
    from graph_swarm.research.runner import ExperimentRunArtifactStore, ExperimentRunner

    root = project_root.expanduser().resolve()
    b0_config = _config_for_gate(
        root,
        root / "configs/experiments/rollout_3a_pilot.yaml",
    )
    o1_config = _config_for_gate(
        root,
        root / "configs/experiments/rollout_3a_o1.yaml",
    )
    baseline = (
        baseline_root.expanduser().resolve()
        if baseline_root is not None
        else b0_config.workspace_baseline_root_path
    )
    execution = (
        execution_root.expanduser().resolve()
        if execution_root is not None
        else b0_config.workspace_execution_root_path
    )
    b0_config = sprint3._configured_runtime(  # pyright: ignore[reportPrivateUsage]
        b0_config, baseline, execution
    )
    o1_config = sprint3._configured_runtime(  # pyright: ignore[reportPrivateUsage]
        o1_config, baseline, execution
    )
    validate_execution_settings(
        manifest,
        (b0_config, o1_config),
        pacing=(pacing := ProviderRequestPacing()),
    )
    selected_ids = tuple(dict.fromkeys(item.task_id for item in plan))
    all_cases = load_task_cases(
        b0_config.task_manifest_path,
        problem_statements_path=b0_config.task_problems_path,
    )
    cases = [case for case in all_cases if case.task.id in selected_ids]
    if tuple(case.task.id for case in cases) != selected_ids:
        raise GateB1MetadataError("selected Gate B1 tasks are missing from the frozen manifest")
    _verify_manifest_task_identity(manifest, cases, b0_config.task_manifest_path)
    sprint3._verify_baselines(cases, baseline)  # pyright: ignore[reportPrivateUsage]
    instance_ids = sprint3._manifest_instance_ids(  # pyright: ignore[reportPrivateUsage]
        b0_config.task_manifest_path, selected_ids
    )
    container_images = sprint3._manifest_image_names(  # pyright: ignore[reportPrivateUsage]
        b0_config.task_manifest_path,
        selected_ids,
    )
    frozen_by_instance = sprint3.load_frozen_swesmith_cases(
        tuple(instance_ids.values()),
        allow_network=False,
    )
    frozen_cases = {
        task_id: frozen_by_instance[instance_ids[task_id].lower()]
        for task_id in selected_ids
    }
    sprint3._verify_frozen_patches(  # pyright: ignore[reportPrivateUsage]
        cases, baseline, frozen_cases
    )
    policy = load_benchmark_environment_policy(
        root / "configs/research/benchmark_environments.toml"
    )
    environments = sprint3._prepare_task_environments(  # pyright: ignore[reportPrivateUsage]
        cases,
        environment_root=execution / "sprint3-task-environments",
        source_root=baseline,
        dependency_overlays=sprint3.TASK_DEPENDENCY_OVERLAYS,
        container_images=container_images,
        benchmark_policy=policy,
        benchmark_manifest_path=b0_config.task_manifest_path,
        mode="preflight",
    )
    sprint3._require_validated_environments(  # pyright: ignore[reportPrivateUsage]
        cases, environments
    )
    frozen_cases_by_task = dict(frozen_cases)
    objective = sprint3.FrozenSWEsmithObjective(frozen_cases_by_task, environments)
    recurrence = sprint3.make_recurrence_matcher(frozen_cases_by_task)
    oracle = FrozenOracleResolver.from_frozen_files(
        o1_config.task_manifest_path,
        o1_config.task_problems_path,
    )
    case_by_id = {case.task.id: case for case in cases}
    runners: dict[ExperimentCondition, ExperimentRunner] = {}
    run_artifacts = artifact_root.expanduser().resolve() / "runs"
    for configuration, condition, resolver in (
        (b0_config, ExperimentCondition.B0, None),
        (o1_config, ExperimentCondition.O1, oracle),
    ):
        runners[condition] = ExperimentRunner(
            configuration,
            objective_evaluator=objective,
            recurrence_evaluator=recurrence,
            oracle_resolver=resolver,
            condition=condition,
            execution_runtime_resolver=lambda task, _workspace: (
                environments[task.id].agent_execution_runtime()
            ),
            workspace_resolver=sprint3._make_workspace_resolver(  # pyright: ignore[reportPrivateUsage]
                source_root=baseline,
                execution_root=execution,
                frozen_cases=frozen_cases_by_task,
                condition=condition,
            ),
            artifact_store=ExperimentRunArtifactStore(run_artifacts),
            request_pacing=pacing,
            gate_id=manifest.gate_id,
            execution_manifest_id=manifest.manifest_id,
        )

    def execute_condition(item: PlannedGateB1Execution) -> ExperimentExecution:
        actual = environments[item.task_id].environment_fingerprint
        expected = manifest.task_map[item.task_id].environment_fingerprint
        if actual != expected:
            raise GateB1EnvironmentMismatch(expected, actual)
        result = runners[item.condition].run_case(case_by_id[item.task_id])
        return replace(result, actual_environment_fingerprint=actual)

    resolved_artifact_root = artifact_root
    if not resolved_artifact_root.is_absolute():
        resolved_artifact_root = root / resolved_artifact_root
    store = GateB1EvidenceStore(resolved_artifact_root, manifest)
    return execute_gate_b1(
        manifest,
        plan,
        store=store,
        execute_condition=execute_condition,
        replacements=replacements,
    )


def _verify_manifest_task_identity(
    manifest: GateB1Manifest,
    cases: Sequence[Any],
    manifest_path: Path,
) -> None:
    records = _read_manifest_records(manifest_path)
    for case in cases:
        frozen = manifest.task_map.get(case.task.id)
        record = records.get(case.task.id)
        if frozen is None or record is None:
            raise GateB1MetadataError(f"missing frozen task identity for {case.task.id}")
        if record.get("repository") != frozen.repository:
            raise GateB1MetadataError(f"repository identity mismatch for {case.task.id}")


def _parse_replacement(
    plan: Sequence[PlannedGateB1Execution],
    replacement_of: str | None,
) -> Mapping[tuple[str, ExperimentCondition], str] | None:
    if replacement_of is None:
        return None
    if len(plan) != 1:
        raise GateB1Error("--replacement-of requires exactly one selected task/condition")
    item = plan[0]
    return {(item.task_id, item.condition): replacement_of}


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--task-id")
    parser.add_argument("--start-task")
    parser.add_argument("--end-task")
    parser.add_argument("--condition", choices=("B0", "O1"), action="append")
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--replacement-of")
    return parser


def main() -> None:
    args = make_cli_parser().parse_args()
    if args.freeze and args.execute:
        raise SystemExit("--freeze and --execute are mutually exclusive")
    if args.freeze:
        output = args.output or Path("research/evidence/results/GS-E003/gate_b1/freeze.json")
        manifest = freeze_gate_b1(
            project_root=args.project_root,
            output_path=output,
        )
        print(manifest.manifest_id)
        return
    if args.manifest is None:
        raise SystemExit("--manifest is required unless --freeze is supplied")
    manifest = load_gate_b1_manifest(args.manifest)
    conditions = (
        None
        if args.condition is None
        else tuple(ExperimentCondition(item) for item in args.condition)
    )
    plan = build_execution_plan(
        manifest,
        task_id=args.task_id,
        start_task=args.start_task,
        end_task=args.end_task,
        conditions=conditions,
    )
    if args.execute:
        if args.freeze:
            raise SystemExit("--execute cannot be combined with --freeze")
        artifact_root = args.artifact_root or (
            args.project_root / "research/evidence/results/GS-E003/gate_b1"
        )
        records = execute_gate_b1_with_existing_stack(
            manifest,
            plan,
            project_root=args.project_root,
            baseline_root=args.baseline_root,
            execution_root=args.execution_root,
            artifact_root=artifact_root,
            replacements=_parse_replacement(plan, args.replacement_of),
        )
        print(json.dumps([record.model_dump(mode="json") for record in records], indent=2))
        return
    print(json.dumps([item.model_dump(mode="json") for item in plan], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "AttemptRecord",
    "AttemptStatus",
    "GATE_B1_CONDITIONS",
    "GATE_B1_EXPECTED_RUNS",
    "GATE_B1_ID",
    "GATE_B1_MAX_ACTIONS",
    "GATE_B1_MAX_REQUESTS",
    "GATE_B1_NOMINAL_RPM",
    "GATE_B1_PACING_SECONDS",
    "GATE_B1_TASK_IDS",
    "GATE_B1_TIMEOUT_SECONDS",
    "GATE_B1_TOOL_RETRIES",
    "GateB1DuplicateError",
    "GateB1EnvironmentMismatch",
    "GateB1Error",
    "GateB1EvidenceStore",
    "GateB1Manifest",
    "GateB1MetadataError",
    "GateB1PairStatus",
    "GateB1ReplacementError",
    "FrozenGateB1Task",
    "PlannedGateB1Execution",
    "build_execution_plan",
    "classify_attempt",
    "execute_gate_b1",
    "execute_gate_b1_with_existing_stack",
    "freeze_gate_b1",
    "load_gate_b1_manifest",
    "validate_artifact_contract",
    "validate_execution_settings",
    "validate_o1_pre_execution",
]
