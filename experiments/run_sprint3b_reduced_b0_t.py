"""Execute the frozen Sprint 3A reduced B0-vs-T primary plan.

This module is intentionally an orchestration boundary.  Agent execution,
tool interception, objective evaluation, recurrence evaluation, workspace
materialization, and advisory retrieval remain in their existing components.
The executor adds only the frozen-plan checks, immutable slot bookkeeping, and
resume/invalid-attempt policy required for Sprint 3B.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from experiments import sprint3, sprint3a
from graph_swarm.agent.coding_agent import preflight_kilo_provider
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.tasks import Task
from graph_swarm.integration.advisory_runtime import (
    R13B_TREATMENT_PATTERN_IDS,
    Neo4jAdvisoryRuntime,
    R13bTreatmentRepository,
    create_neo4j_advisory_runtime,
)
from graph_swarm.research.contracts import (
    ExperimentCondition,
    ExperimentRunArtifact,
    TimeoutContract,
)
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    ExperimentExecution,
    ExperimentRunArtifactStore,
    ExperimentRunner,
    LoadedExperimentConfiguration,
    ObjectiveTaskEvaluator,
    RecurrenceEvaluator,
    load_experiment_configuration,
    load_task_cases,
)
from graph_swarm.settings import Settings, get_settings

EXPERIMENT_ID = "GS-E003"
PROTOCOL_REVISION = "sprint3b-kilo-reduced-b0-t-v1"
EXPECTED_CODING_MODEL = "nex-agi/nex-n2.5-pro"
EXPECTED_PROVIDER = "kilo"
KILO_BASE_URL = "https://api.kilo.ai/api/gateway"
LEGACY_PROTOCOL_REVISION = sprint3a.SPRINT3A_PROTOCOL_REVISION
LEGACY_CODING_MODEL = sprint3a.EXPECTED_CODING_MODEL
LEGACY_PROVIDER = "openrouter"
EXPECTED_TASK_IDS: tuple[str, ...] = tuple(f"GS-T{i:03d}" for i in range(6, 16))
EXPECTED_CONDITIONS: tuple[ExperimentCondition, ...] = (
    ExperimentCondition.B0,
    ExperimentCondition.T,
)
EXPECTED_PRIMARY_RUNS = 20
EXPECTED_MAX_ACTIONS = 20
EXPECTED_MAX_REQUESTS = 24
EXPECTED_TIMEOUT_SECONDS = 300.0
EXPECTED_TOOL_RETRIES = 3
EXPECTED_PATTERN_IDS = frozenset(sprint3a.SPRINT3A_PATTERN_IDS)
DEFAULT_CONFIG = Path("configs/experiments/sprint3b_kilo.yaml")
DEFAULT_FREEZE = Path("research/evidence/results/GS-E003/sprint3b_kilo/freeze.json")
LEGACY_CONFIG = Path("configs/experiments/sprint3a_reduced_b0_t_v1.yaml")
LEGACY_FREEZE = Path("research/evidence/results/GS-E003/sprint3a_reduced_b0_t/freeze.json")
KILO_FREEZE_INPUT_PATHS: tuple[str, ...] = (
    "configs/experiments/sprint3b_kilo.yaml",
    "configs/models/kilo_coding.yaml",
    "experiments/run_sprint3b_reduced_b0_t.py",
    "benchmark/manifests/pilot.jsonl",
    "benchmark/annotations/recurrence_validation.csv",
    "src/graph_swarm/agent/coding_agent.py",
    "src/graph_swarm/agent/prompts.py",
    "src/graph_swarm/settings.py",
    "configs/research/gate_b1_environments.json",
)


class Sprint3BExecutionError(ValueError):
    """Base error for frozen Sprint 3B execution contract violations."""


class FreezeValidationError(Sprint3BExecutionError):
    """Raised when the frozen protocol or one of its inputs has changed."""


class PrimaryRunCollisionError(Sprint3BExecutionError):
    """Raised when a planned slot cannot be resumed safely."""


class InvalidInfrastructureError(Sprint3BExecutionError):
    """Raised only when an infrastructure/provider attempt is invalid."""


class RunValidity(StrEnum):
    VALID = "valid"
    INVALID_INFRASTRUCTURE = "invalid_infrastructure"


class SlotStatus(StrEnum):
    MISSING = "missing"
    VALID = "valid"
    INVALID_INFRASTRUCTURE = "invalid_infrastructure"


@dataclass(frozen=True)
class ExecutionSlot:
    """One immutable task-condition position in the frozen primary plan."""

    index: int
    task_id: str
    condition: ExperimentCondition

    @property
    def key(self) -> tuple[str, ExperimentCondition]:
        return self.task_id, self.condition


@dataclass(frozen=True)
class SlotObservation:
    slot: ExecutionSlot
    status: SlotStatus
    run_id: str | None = None
    artifact_path: Path | None = None
    metadata_path: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class FrozenExecutionContext:
    project_root: Path
    config: LoadedExperimentConfiguration
    protocol: dict[str, Any]
    freeze: dict[str, Any]
    freeze_hash: str
    execution_commit_sha: str
    plan: tuple[ExecutionSlot, ...]
    cases: tuple[BenchmarkTaskCase, ...]
    artifact_root: Path


@dataclass(frozen=True)
class ExecutionReport:
    planned_slots: int
    attempted_slots: int
    valid_completed_slots: int
    invalid_infrastructure_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]
    b0_count: int
    t_count: int
    artifact_root: Path
    execution_commit_sha: str
    stopped_reason: str | None = None


class SlotExecutor(Protocol):
    def __call__(self, slot: ExecutionSlot, case: BenchmarkTaskCase) -> ExperimentExecution: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _project_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise FreezeValidationError(f"could not resolve execution commit SHA: {result.stderr}")
    return result.stdout.strip()


def _dotenv_value(root: Path, variable: str) -> str | None:
    value = os.environ.get(variable)
    if value is not None:
        return value.strip()
    env_path = root / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{variable}="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return None


def _is_legacy_protocol(protocol: Mapping[str, Any]) -> bool:
    return protocol.get("protocol_revision") == LEGACY_PROTOCOL_REVISION


def _profile(protocol: Mapping[str, Any]) -> tuple[str, str, str]:
    if _is_legacy_protocol(protocol):
        return LEGACY_PROVIDER, LEGACY_CODING_MODEL, "OPENROUTER_CODING_MODEL"
    return EXPECTED_PROVIDER, EXPECTED_CODING_MODEL, "KILO_CODING_MODEL"


def _kilo_freeze_input_hashes(root: Path) -> dict[str, str]:
    return {
        relative_path: _sha256(root / relative_path)
        for relative_path in KILO_FREEZE_INPUT_PATHS
    }


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise FreezeValidationError(f"{label} differs from the frozen protocol: {actual!r}")


def load_frozen_execution_context(
    *,
    project_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
    require_credentials: bool = False,
    allow_existing_artifacts: bool = True,
) -> FrozenExecutionContext:
    """Load and validate the frozen protocol without provider/Neo4j calls."""
    root = project_root.expanduser().resolve()
    resolved_config = _project_path(config_path, root)
    selected_freeze = freeze_path
    if freeze_path == DEFAULT_FREEZE and resolved_config.name == LEGACY_CONFIG.name:
        selected_freeze = LEGACY_FREEZE
    resolved_freeze = _project_path(selected_freeze, root)
    try:
        protocol = sprint3a.load_protocol(resolved_config)
        freeze = json.loads(resolved_freeze.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, sprint3a.Sprint3AProtocolError) as error:
        raise FreezeValidationError(f"could not load frozen Sprint 3B inputs: {error}") from error
    if not isinstance(freeze, dict):
        raise FreezeValidationError("freeze artifact must be a JSON object")

    expected_provider, expected_model, model_variable = _profile(protocol)
    expected_revision = (
        LEGACY_PROTOCOL_REVISION if expected_provider == LEGACY_PROVIDER else PROTOCOL_REVISION
    )

    _require_equal(freeze.get("status"), "READY", "freeze status")
    _require_equal(freeze.get("experiment_id"), EXPERIMENT_ID, "experiment ID")
    _require_equal(freeze.get("protocol_revision"), expected_revision, "protocol revision")
    _require_equal(_sha256(resolved_config), freeze.get("config_sha256"), "config hash")
    current_hashes = (
        sprint3a.freeze_input_hashes(root)
        if expected_provider == LEGACY_PROVIDER
        else _kilo_freeze_input_hashes(root)
    )
    _require_equal(current_hashes, freeze.get("freeze_input_hashes"), "freeze-input hashes")
    _require_equal(
        sprint3a.freeze_inputs_hash(current_hashes),
        freeze.get("freeze_inputs_sha256"),
        "aggregate freeze-input hash",
    )

    plan_values = sprint3a.build_execution_plan(protocol)
    _require_equal(
        tuple(protocol.get("conditions", ())),
        tuple(c.value for c in EXPECTED_CONDITIONS),
        "conditions",
    )
    _require_equal(tuple(protocol.get("task_order", ())), EXPECTED_TASK_IDS, "task order")
    _require_equal(len(plan_values), EXPECTED_PRIMARY_RUNS, "planned primary runs")
    _require_equal(
        tuple(tuple(item) for item in freeze.get("execution_plan", ())),
        plan_values,
        "execution plan",
    )
    _require_equal(
        freeze.get("planned_primary_runs"), EXPECTED_PRIMARY_RUNS, "freeze planned primary runs"
    )
    _require_equal(protocol.get("provider"), expected_provider, "provider")
    _require_equal(protocol.get("model"), expected_model, "coding model")
    _require_equal(
        protocol.get("model_resolution_variable"),
        model_variable,
        "model resolution variable",
    )
    _require_equal(protocol.get("model_settings", {}).get("temperature"), 0, "temperature")
    _require_equal(
        protocol.get("limits", {}).get("max_actions"), EXPECTED_MAX_ACTIONS, "max actions"
    )
    _require_equal(
        protocol.get("limits", {}).get("max_requests"), EXPECTED_MAX_REQUESTS, "max requests"
    )
    _require_equal(
        protocol.get("limits", {}).get("task_timeout_seconds"),
        EXPECTED_TIMEOUT_SECONDS,
        "task timeout",
    )
    _require_equal(
        protocol.get("limits", {}).get("model_request_timeout_seconds"),
        EXPECTED_TIMEOUT_SECONDS,
        "model request timeout",
    )
    _require_equal(
        protocol.get("limits", {}).get("tool_retries"), EXPECTED_TOOL_RETRIES, "tool retries"
    )
    _require_equal(protocol.get("memory_writes"), "forbidden", "memory writes")
    _require_equal(protocol.get("neo4j_write_policy"), "forbidden", "Neo4j write policy")
    _require_equal(
        protocol.get("abstraction_model", {}).get("used_during_transfer_execution"),
        False,
        "abstraction execution",
    )
    pattern_ids = frozenset(protocol.get("treatment_pattern_ids", ()))
    if (
        pattern_ids != EXPECTED_PATTERN_IDS
        or pattern_ids != R13B_TREATMENT_PATTERN_IDS
        or len(pattern_ids) != 5
    ):
        raise FreezeValidationError(
            "treatment-visible pattern IDs differ from the frozen canonical five"
        )

    configuration = load_experiment_configuration(resolved_config, project_root=root)
    if (
        configuration.model.provider != expected_provider
        or configuration.model.settings.get("temperature") != 0
    ):
        raise FreezeValidationError(
            "loaded model configuration differs from the frozen comparison contract"
        )
    # The protocol, rather than the historical model YAML label, owns the
    # experiment prompt identity.  The agent's actual system prompt remains
    # the existing ROLLOUT1_SYSTEM_PROMPT in create_coding_agent.
    frozen_model = configuration.model.model_copy(
        update={
            "model": expected_model,
            "prompt_version": str(protocol.get("prompt_version", "")),
            "settings": {"temperature": 0},
        }
    )
    configuration = replace(configuration, model=frozen_model)

    if (
        expected_provider == LEGACY_PROVIDER
        and _dotenv_value(root, model_variable) != expected_model
    ):
        raise FreezeValidationError(
            f"{model_variable} must exactly match the frozen coding model"
        )
    if require_credentials:
        required = (
            "OPENROUTER_API_KEY" if expected_provider == LEGACY_PROVIDER else "KILO_API_KEY",
            "HF_TOKEN",
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "NEO4J_DATABASE",
        )
        missing = [name for name in required if not (_dotenv_value(root, name) or "").strip()]
        if missing:
            raise FreezeValidationError(
                "required runtime credentials are missing: " + ", ".join(missing)
            )

    cases = tuple(
        case
        for case in load_task_cases(
            configuration.task_manifest_path,
            problem_statements_path=configuration.task_problems_path,
        )
        if case.task.id in EXPECTED_TASK_IDS
    )
    if tuple(case.task.id for case in cases) != EXPECTED_TASK_IDS:
        raise FreezeValidationError(
            "the ten frozen transfer task cases are incomplete or out of order"
        )
    if tuple(case.task.chronological_index for case in cases) != tuple(range(6, 16)):
        raise FreezeValidationError("frozen transfer task indexes must be 6 through 15")

    artifact_root = _project_path(str(protocol["artifact_root"]), root)
    if not allow_existing_artifacts and artifact_root.exists():
        raise FreezeValidationError(f"primary artifact root already exists: {artifact_root}")
    slots = tuple(
        ExecutionSlot(index=index, task_id=task_id, condition=ExperimentCondition(condition))
        for index, (task_id, condition) in enumerate(plan_values, start=1)
    )
    return FrozenExecutionContext(
        project_root=root,
        config=configuration,
        protocol=protocol,
        freeze=freeze,
        freeze_hash=_sha256(resolved_freeze),
        execution_commit_sha=_git_head(root),
        plan=slots,
        cases=cases,
        artifact_root=artifact_root,
    )


def build_execution_plan(protocol: Mapping[str, Any]) -> tuple[ExecutionSlot, ...]:
    """Build exactly the order written in the frozen protocol."""
    values = sprint3a.build_execution_plan(protocol)
    return tuple(
        ExecutionSlot(index=index, task_id=task_id, condition=ExperimentCondition(condition))
        for index, (task_id, condition) in enumerate(values, start=1)
    )


def _slot_directory(root: Path, slot: ExecutionSlot) -> Path:
    return root / EXPERIMENT_ID / slot.condition.value / slot.task_id


def inspect_primary_slots(
    artifact_root: Path,
    plan: Sequence[ExecutionSlot],
) -> tuple[SlotObservation, ...]:
    """Inspect existing artifacts without changing them."""
    observations: list[SlotObservation] = []
    for slot in plan:
        slot_root = _slot_directory(artifact_root, slot)
        if not slot_root.exists():
            observations.append(SlotObservation(slot, SlotStatus.MISSING))
            continue
        if not slot_root.is_dir():
            raise PrimaryRunCollisionError(f"planned slot path is not a directory: {slot_root}")
        candidates = [item for item in slot_root.iterdir() if item.is_dir()]
        if not candidates:
            raise PrimaryRunCollisionError(
                f"planned slot contains no immutable run artifact: {slot_root}"
            )
        completed: list[SlotObservation] = []
        for run_dir in candidates:
            artifact_path = run_dir / "artifact.json"
            metadata_path = run_dir / "run_metadata.json"
            if not artifact_path.is_file() and not metadata_path.is_file():
                raise PrimaryRunCollisionError(f"unrecognized partial run artifact: {run_dir}")
            if artifact_path.is_file():
                try:
                    artifact = ExperimentRunArtifact.model_validate_json(
                        artifact_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, ValueError) as error:
                    raise PrimaryRunCollisionError(
                        f"unreadable immutable run artifact: {artifact_path}"
                    ) from error
                if (artifact.experiment_id, artifact.task_id, artifact.condition) != (
                    EXPERIMENT_ID,
                    slot.task_id,
                    slot.condition,
                ):
                    raise PrimaryRunCollisionError(
                        f"artifact identity does not match planned slot: {artifact_path}"
                    )
                status = SlotStatus.VALID
                error = None
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if (
                        metadata.get("final_run_validity")
                        == RunValidity.INVALID_INFRASTRUCTURE.value
                    ):
                        status = SlotStatus.INVALID_INFRASTRUCTURE
                        error = str(
                            metadata.get("error_message") or "invalid infrastructure attempt"
                        )
                completed.append(
                    SlotObservation(
                        slot, status, artifact.run_id, artifact_path, metadata_path, error
                    )
                )
            else:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("final_run_validity") != RunValidity.INVALID_INFRASTRUCTURE.value:
                    raise PrimaryRunCollisionError(
                        f"metadata-only run is not an invalid preserved attempt: {metadata_path}"
                    )
                completed.append(
                    SlotObservation(
                        slot,
                        SlotStatus.INVALID_INFRASTRUCTURE,
                        str(metadata.get("run_id")),
                        None,
                        metadata_path,
                        str(metadata.get("error_message")),
                    )
                )
        if len(completed) != 1:
            raise PrimaryRunCollisionError(f"planned slot has multiple attempts: {slot_root}")
        observations.append(completed[0])
    return tuple(observations)


def _settings_for_frozen_model(
    settings: Settings | None,
    configuration: LoadedExperimentConfiguration | None = None,
) -> Settings:
    base = settings or get_settings()
    provider = base.model_provider if configuration is None else configuration.model.provider
    if provider == EXPECTED_PROVIDER:
        return base.model_copy(
            update={
                "model_provider": EXPECTED_PROVIDER,
                "kilo_coding_model": EXPECTED_CODING_MODEL,
                "kilo_api_key": _dotenv_value(Path.cwd(), "KILO_API_KEY")
                or base.kilo_api_key,
            }
        )
    return base.model_copy(
        update={
            "model_provider": LEGACY_PROVIDER,
            "openrouter_coding_model": LEGACY_CODING_MODEL,
            "openrouter_api_key": _dotenv_value(Path.cwd(), "OPENROUTER_API_KEY")
            or base.openrouter_api_key,
        }
    )


def build_condition_runner(
    configuration: LoadedExperimentConfiguration,
    *,
    condition: ExperimentCondition,
    objective_evaluator: ObjectiveTaskEvaluator,
    recurrence_evaluator: RecurrenceEvaluator,
    workspace_resolver: Callable[[Task], Path],
    execution_runtime_resolver: Callable[[Task, Path], Any],
    advisory_runtime: Neo4jAdvisoryRuntime | None = None,
    settings: Settings | None = None,
    artifact_store: ExperimentRunArtifactStore | None = None,
) -> ExperimentRunner:
    """Construct B0 or T through the existing ExperimentRunner boundary."""
    if condition is ExperimentCondition.B0:
        if advisory_runtime is not None:
            raise Sprint3BExecutionError("B0 cannot be constructed with an advisory runtime")
        advisory_service = None
    elif condition is ExperimentCondition.T:
        if advisory_runtime is None:
            raise Sprint3BExecutionError("T requires the real injected Neo4j advisory runtime")
        if not isinstance(advisory_runtime.treatment_repository, R13bTreatmentRepository):
            raise Sprint3BExecutionError("T must use R13bTreatmentRepository")
        if set(R13B_TREATMENT_PATTERN_IDS) != EXPECTED_PATTERN_IDS:
            raise Sprint3BExecutionError("T treatment corpus is not the canonical five-pattern set")
        advisory_service = advisory_runtime.advisory_service
    else:
        raise Sprint3BExecutionError(f"unsupported Sprint 3B condition: {condition}")
    return ExperimentRunner(
        configuration,
        settings=_settings_for_frozen_model(settings, configuration),
        objective_evaluator=objective_evaluator,
        recurrence_evaluator=recurrence_evaluator,
        workspace_resolver=workspace_resolver,
        execution_runtime_resolver=execution_runtime_resolver,
        advisory_service=advisory_service,
        condition=condition,
        artifact_store=artifact_store,
    )


def _is_infrastructure_failure(execution: ExperimentExecution, objective: object | None) -> bool:
    if execution.error is not None:
        # Usage limits are bounded experimental outcomes, not invalid attempts.
        if type(execution.error).__name__ != "UsageLimitExceeded":
            return True
    observations = getattr(objective, "observations", ())
    for observation in reversed(tuple(observations)):
        if getattr(observation, "task_id", None) != execution.artifact.task_id:
            continue
        if getattr(observation, "status", None) == "objective_infrastructure_failure":
            return True
        break
    return False


def _metadata_for_execution(
    context: FrozenExecutionContext,
    slot: ExecutionSlot,
    execution: ExperimentExecution,
    *,
    environment_identity: Mapping[str, object] | None,
    validity: RunValidity,
) -> dict[str, object]:
    artifact = execution.artifact
    provider = str(context.protocol.get("provider", EXPECTED_PROVIDER))
    coding_model = str(context.protocol.get("model", EXPECTED_CODING_MODEL))
    completed_at = datetime.now(UTC)
    started_at = artifact.executed_action.started_at
    usage = (
        None if execution.agent_result is None else dataclasses.asdict(execution.agent_result.usage)
    )
    retrieval = [
        dataclasses.asdict(item) for item in execution.dependencies.advisory_retrieval_evidence
    ]
    selected = retrieval[0] if retrieval else None
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_revision": context.protocol.get("protocol_revision", PROTOCOL_REVISION),
        "sprint3a_freeze_hash": context.freeze_hash,
        "execution_commit_sha": context.execution_commit_sha,
        "task_id": slot.task_id,
        "condition": slot.condition.value,
        "run_id": artifact.run_id,
        "plan_index": slot.index,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "coding_model": coding_model,
        "provider": provider,
        "prompt_config_identity": {
            "config_sha256": context.freeze.get("config_sha256"),
            "prompt_version": artifact.prompt_version,
            "system_prompt": context.freeze.get("model", {}).get("system_prompt"),
            "system_prompt_source_sha256": context.freeze.get("model", {}).get(
                "system_prompt_source_sha256"
            ),
        },
        "workspace_environment_identity": dict(environment_identity or {}),
        "objective_evaluator": {
            "name": context.protocol.get("objective_evaluator"),
            "version": context.protocol.get("objective_evaluator_version"),
            "result": artifact.task_success,
        },
        "recurrence_evaluator": {
            "name": context.protocol.get("recurrence_evaluator"),
            "version": context.protocol.get("recurrence_evaluator_version"),
            "result": artifact.known_failure_repeated,
        },
        "tool_action_events": [
            event.model_dump(mode="json") for event in execution.dependencies.events
        ],
        "advice_retrieval_events": retrieval,
        "selected_recovery_pattern_id": None if selected is None else selected.get("pattern_id"),
        "retrieval_score": None if selected is None else selected.get("vector_score"),
        "behavior_change_evidence": [
            item.model_dump(mode="json") for item in execution.dependencies.behavior_evidence
        ],
        "task_success": artifact.task_success,
        "repeated_failure_result": artifact.known_failure_repeated,
        "usage": usage,
        "latency_ms": artifact.latency_ms,
        "timeout_provenance": artifact.timeout_provenance,
        "error_type": None if execution.error is None else type(execution.error).__name__,
        "error_message": None if execution.error is None else str(execution.error),
        "final_run_validity": validity.value,
        "treatment_memory_writes": 0,
        "abstraction_model_called": False,
    }


def _write_metadata(path: Path, metadata: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        )
    return path


def _write_invalid_attempt(
    context: FrozenExecutionContext,
    slot: ExecutionSlot,
    case: BenchmarkTaskCase,
    error: BaseException,
    store: ExperimentRunArtifactStore,
) -> SlotObservation:
    provider = str(context.protocol.get("provider", EXPECTED_PROVIDER))
    coding_model = str(context.protocol.get("model", EXPECTED_CODING_MODEL))
    run_id = f"{EXPERIMENT_ID}-{slot.condition.value}-{case.task.id}-{uuid.uuid4().hex}"
    run_dir = store.prepare_run_directory(EXPERIMENT_ID, slot.condition, case.task.id, run_id)
    now = datetime.now(UTC)
    action_id = f"{run_id}-infrastructure"
    action = PlannedAction(
        id=action_id,
        run_id=run_id,
        task_id=case.task.id,
        tool="executor",
        operation="infrastructure_failure",
        arguments={},
        planned_at=now,
    )
    result = ActionResult(
        action_id=action_id,
        tool_name="executor",
        success=False,
        error=type(error).__name__,
        started_at=now,
        completed_at=now,
    )
    artifact = ExperimentRunArtifact(
        experiment_id=EXPERIMENT_ID,
        condition=slot.condition,
        run_id=run_id,
        task_id=case.task.id,
        family_id=case.task.family_id,
        chronological_index=case.task.chronological_index,
        model=coding_model,
        model_settings={"temperature": 0},
        prompt_version=str(context.protocol.get("prompt_version", "v1")),
        planned_action=action,
        executed_action=result,
        task_success=False,
        known_failure_repeated=False,
        tool_calls=0,
        retries=0,
        timeout_contract=TimeoutContract(
            agent_wall_clock_seconds=EXPECTED_TIMEOUT_SECONDS,
            model_request_timeout_seconds=EXPECTED_TIMEOUT_SECONDS,
            tool_timeout_seconds={"run_command": 30, "run_tests": 120},
            experiment_timeout_seconds=EXPECTED_TIMEOUT_SECONDS,
        ),
        timeout_provenance={
            "layer": "executor",
            "error_type": type(error).__name__,
            "message": str(error),
        },
    )
    artifact_path = store.write_artifact(artifact)
    metadata_path = _write_metadata(
        run_dir / "run_metadata.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_revision": context.protocol.get("protocol_revision", PROTOCOL_REVISION),
            "sprint3a_freeze_hash": context.freeze_hash,
            "execution_commit_sha": context.execution_commit_sha,
            "task_id": case.task.id,
            "condition": slot.condition.value,
            "run_id": run_id,
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "coding_model": coding_model,
            "provider": provider,
            "objective_evaluator": {"result": False},
            "recurrence_evaluator": {"result": False},
            "tool_action_events": [],
            "advice_retrieval_events": [],
            "behavior_change_evidence": [],
            "task_success": False,
            "repeated_failure_result": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "final_run_validity": RunValidity.INVALID_INFRASTRUCTURE.value,
            "treatment_memory_writes": 0,
            "abstraction_model_called": False,
        },
    )
    return SlotObservation(
        slot, SlotStatus.INVALID_INFRASTRUCTURE, run_id, artifact_path, metadata_path, str(error)
    )


def execute_primary_plan(
    context: FrozenExecutionContext,
    slot_executor: SlotExecutor,
    *,
    artifact_store: ExperimentRunArtifactStore | None = None,
    environment_identities: Mapping[str, Mapping[str, object]] | None = None,
    infrastructure_checker: Callable[[ExperimentExecution], bool] | None = None,
) -> ExecutionReport:
    """Run remaining slots sequentially; stop immediately on invalid infrastructure."""
    store = artifact_store or ExperimentRunArtifactStore(context.artifact_root)
    existing = inspect_primary_slots(context.artifact_root, context.plan)
    existing_by_key = {item.slot.key: item for item in existing}
    cases = {case.task.id: case for case in context.cases}
    attempted = 0
    invalid: list[str] = [
        f"{item.slot.task_id}/{item.slot.condition.value}: "
        f"{item.error or 'invalid infrastructure attempt'}"
        for item in existing
        if item.status is SlotStatus.INVALID_INFRASTRUCTURE
    ]
    stopped_reason = invalid[0] if invalid else None
    if stopped_reason is None:
        for slot in context.plan:
            previous = existing_by_key[slot.key]
            if previous.status is SlotStatus.VALID:
                continue
            case = cases[slot.task_id]
            attempted += 1
            try:
                execution = slot_executor(slot, case)
                checker = infrastructure_checker or (
                    lambda item: _is_infrastructure_failure(item, None)
                )
                validity = (
                    RunValidity.INVALID_INFRASTRUCTURE if checker(execution) else RunValidity.VALID
                )
                _write_metadata(
                    execution.artifact_path.parent / "run_metadata.json",
                    _metadata_for_execution(
                        context,
                        slot,
                        execution,
                        environment_identity=(environment_identities or {}).get(slot.task_id),
                        validity=validity,
                    ),
                )
                if validity is RunValidity.INVALID_INFRASTRUCTURE:
                    reason = (
                        f"{slot.task_id}/{slot.condition.value}: "
                        f"{execution.error or 'objective/runtime infrastructure failure'}"
                    )
                    invalid.append(reason)
                    stopped_reason = reason
                    break
            except PrimaryRunCollisionError:
                raise
            except Exception as error:  # preserve an invalid attempt before stopping
                observation = _write_invalid_attempt(context, slot, case, error, store)
                invalid.append(f"{slot.task_id}/{slot.condition.value}: {observation.error}")
                stopped_reason = invalid[-1]
                break
        # The just-completed slots are re-scanned below; this also verifies
        # that each immutable metadata file was persisted.
    final_states = inspect_primary_slots(context.artifact_root, context.plan)
    valid_count = sum(item.status is SlotStatus.VALID for item in final_states)
    b0_count = sum(
        item.status is SlotStatus.VALID and item.slot.condition is ExperimentCondition.B0
        for item in final_states
    )
    t_count = sum(
        item.status is SlotStatus.VALID and item.slot.condition is ExperimentCondition.T
        for item in final_states
    )
    return ExecutionReport(
        planned_slots=len(context.plan),
        attempted_slots=attempted,
        valid_completed_slots=valid_count,
        invalid_infrastructure_slots=tuple(invalid),
        missing_slots=tuple(
            f"{item.slot.task_id}/{item.slot.condition.value}"
            for item in final_states
            if item.status is SlotStatus.MISSING
        ),
        b0_count=b0_count,
        t_count=t_count,
        artifact_root=context.artifact_root,
        execution_commit_sha=context.execution_commit_sha,
        stopped_reason=stopped_reason,
    )


def validate_live_preflight(context: FrozenExecutionContext) -> None:
    """Run provider-free checks that are specific to the real primary run."""
    cases = list(context.cases)
    baseline = context.config.workspace_baseline_root_path
    execution = context.config.workspace_execution_root_path
    try:
        sprint3._verify_baselines(cases, baseline)  # pyright: ignore[reportPrivateUsage]
        policy = sprint3.load_benchmark_environment_policy(
            context.project_root / "configs/research/benchmark_environments.toml"
        )
        instance_ids = sprint3._manifest_instance_ids(
            context.config.task_manifest_path, EXPECTED_TASK_IDS
        )  # pyright: ignore[reportPrivateUsage]
        images = sprint3._manifest_image_names(context.config.task_manifest_path, EXPECTED_TASK_IDS)  # pyright: ignore[reportPrivateUsage]
        frozen_by_instance = sprint3.load_frozen_swesmith_cases(
            tuple(instance_ids.values()), allow_network=False
        )
        frozen_cases = {
            task_id: frozen_by_instance[instance_ids[task_id].lower()]
            for task_id in EXPECTED_TASK_IDS
        }
        sprint3._verify_frozen_patches(cases, baseline, frozen_cases)  # pyright: ignore[reportPrivateUsage]
        environments = sprint3._prepare_task_environments(  # pyright: ignore[reportPrivateUsage]
            cases,
            environment_root=execution / "sprint3-task-environments",
            source_root=baseline,
            dependency_overlays=sprint3.TASK_DEPENDENCY_OVERLAYS,
            container_images=images,
            benchmark_policy=policy,
            benchmark_manifest_path=context.config.task_manifest_path,
            mode="preflight",
        )
        sprint3._require_validated_environments(cases, environments)  # pyright: ignore[reportPrivateUsage]
    except Exception as error:
        raise FreezeValidationError(
            f"prepared task environment preflight failed: {error}"
        ) from error
    inspect_primary_slots(context.artifact_root, context.plan)


def validate_kilo_provider_preflight(context: FrozenExecutionContext) -> str:
    """Run the sole provider request used by ``--preflight-only``."""
    if context.protocol.get("provider") != EXPECTED_PROVIDER:
        raise FreezeValidationError("Kilo provider preflight requires the Kilo protocol")
    settings = _settings_for_frozen_model(get_settings(), context.config)
    return preflight_kilo_provider(settings)


def execute_frozen_primary_runs(
    *,
    project_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
) -> ExecutionReport:
    """Preflight and execute the real frozen 20-slot plan."""
    context = load_frozen_execution_context(
        project_root=project_root,
        config_path=config_path,
        freeze_path=freeze_path,
        require_credentials=True,
        allow_existing_artifacts=True,
    )
    validate_live_preflight(context)
    settings = _settings_for_frozen_model(get_settings(), context.config)
    all_cases = list(context.cases)
    instance_ids = sprint3._manifest_instance_ids(
        context.config.task_manifest_path, EXPECTED_TASK_IDS
    )  # pyright: ignore[reportPrivateUsage]
    images = sprint3._manifest_image_names(context.config.task_manifest_path, EXPECTED_TASK_IDS)  # pyright: ignore[reportPrivateUsage]
    frozen_by_instance = sprint3.load_frozen_swesmith_cases(
        tuple(instance_ids.values()), allow_network=False
    )
    frozen_cases = {
        task_id: frozen_by_instance[instance_ids[task_id].lower()] for task_id in EXPECTED_TASK_IDS
    }
    policy = sprint3.load_benchmark_environment_policy(
        context.project_root / "configs/research/benchmark_environments.toml"
    )
    environments = sprint3._prepare_task_environments(  # pyright: ignore[reportPrivateUsage]
        all_cases,
        environment_root=context.config.workspace_execution_root_path / "sprint3-task-environments",
        source_root=context.config.workspace_baseline_root_path,
        dependency_overlays=sprint3.TASK_DEPENDENCY_OVERLAYS,
        container_images=images,
        benchmark_policy=policy,
        benchmark_manifest_path=context.config.task_manifest_path,
        mode="preflight",
    )
    objective = sprint3.FrozenSWEsmithObjective(
        frozen_cases,
        environments,
        objective_coverage_policy="no_cov",
        coverage_policy_selection_version=(
            sprint3.OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION
        ),
    )
    recurrence = sprint3.make_recurrence_matcher(frozen_cases)
    artifact_store = ExperimentRunArtifactStore(context.artifact_root)
    runners: dict[ExperimentCondition, ExperimentRunner] = {}
    runtimes: dict[ExperimentCondition, Neo4jAdvisoryRuntime | None] = {
        ExperimentCondition.B0: None
    }
    runtimes[ExperimentCondition.T] = create_neo4j_advisory_runtime(settings)
    for condition in EXPECTED_CONDITIONS:
        runtime = runtimes[condition]
        runners[condition] = build_condition_runner(
            context.config,
            condition=condition,
            objective_evaluator=cast(ObjectiveTaskEvaluator, objective),
            recurrence_evaluator=cast(RecurrenceEvaluator, recurrence),
            workspace_resolver=sprint3._make_workspace_resolver(  # pyright: ignore[reportPrivateUsage]
                source_root=context.config.workspace_baseline_root_path,
                execution_root=context.config.workspace_execution_root_path,
                frozen_cases=frozen_cases,
                condition=condition,
            ),
            execution_runtime_resolver=lambda task, _workspace: environments[
                task.id
            ].agent_execution_runtime(),
            advisory_runtime=runtime,
            settings=settings,
            artifact_store=artifact_store,
        )

    def run_slot(slot: ExecutionSlot, case: BenchmarkTaskCase) -> ExperimentExecution:
        return runners[slot.condition].run_case(case)

    identities = {
        task_id: {
            "environment_fingerprint": environments[task_id].environment_fingerprint,
            "runtime_type": environments[task_id].runtime_type,
        }
        for task_id in EXPECTED_TASK_IDS
    }
    try:
        return execute_primary_plan(
            context,
            run_slot,
            artifact_store=artifact_store,
            environment_identities=identities,
            infrastructure_checker=lambda execution: _is_infrastructure_failure(
                execution, objective
            ),
        )
    finally:
        t_runtime = runtimes[ExperimentCondition.T]
        if t_runtime is not None:
            t_runtime.close()


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    args = make_cli_parser().parse_args()
    context = load_frozen_execution_context(
        project_root=args.project_root,
        config_path=args.config,
        freeze_path=args.freeze,
        require_credentials=not args.preflight_only,
        allow_existing_artifacts=True,
    )
    if args.preflight_only:
        validate_kilo_provider_preflight(context)
        validate_live_preflight(context)
        print(json.dumps({"status": "READY", "planned_slots": len(context.plan)}, sort_keys=True))
        return
    report = execute_frozen_primary_runs(
        project_root=args.project_root,
        config_path=args.config,
        freeze_path=args.freeze,
    )
    print(json.dumps(dataclasses.asdict(report), default=str, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_FREEZE",
    "EXPECTED_CODING_MODEL",
    "EXPECTED_CONDITIONS",
    "EXPECTED_PATTERN_IDS",
    "EXPECTED_PRIMARY_RUNS",
    "EXPECTED_PROVIDER",
    "KILO_BASE_URL",
    "EXPECTED_TASK_IDS",
    "ExecutionReport",
    "ExecutionSlot",
    "FreezeValidationError",
    "FrozenExecutionContext",
    "PrimaryRunCollisionError",
    "RunValidity",
    "SlotObservation",
    "SlotStatus",
    "Sprint3BExecutionError",
    "build_condition_runner",
    "build_execution_plan",
    "execute_frozen_primary_runs",
    "execute_primary_plan",
    "inspect_primary_slots",
    "load_frozen_execution_context",
    "validate_live_preflight",
    "validate_kilo_provider_preflight",
]
