"""Execute the corrected, retrieval-validity-safe Sprint 3B Kilo protocol."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from experiments import run_sprint3b_reduced_b0_t as v1
from experiments import sprint3, sprint3a
from graph_swarm.agent import coding_agent
from graph_swarm.integration.advisory_runtime import (
    R13B_TREATMENT_PATTERN_IDS,
    Neo4jAdvisoryRuntime,
    R13bTreatmentRepository,
    create_neo4j_advisory_runtime,
)
from graph_swarm.memory.recovery_embeddings import (
    RECOVERY_PATTERN_EMBEDDING_DIMENSION,
    RECOVERY_PATTERN_EMBEDDING_MODEL,
    RECOVERY_PATTERN_EMBEDDING_NORMALIZED,
    RecoveryPatternEmbedder,
)
from graph_swarm.research.contracts import ExperimentCondition
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
PROTOCOL_REVISION = "sprint3b-kilo-reduced-b0-t-v2"
EXPECTED_CODING_MODEL = v1.EXPECTED_CODING_MODEL
EXPECTED_PROVIDER = v1.EXPECTED_PROVIDER
EXPECTED_TASK_IDS = v1.EXPECTED_TASK_IDS
EXPECTED_CONDITIONS = v1.EXPECTED_CONDITIONS
EXPECTED_PRIMARY_RUNS = v1.EXPECTED_PRIMARY_RUNS
EXPECTED_MAX_ACTIONS = v1.EXPECTED_MAX_ACTIONS
EXPECTED_MAX_REQUESTS = v1.EXPECTED_MAX_REQUESTS
EXPECTED_TIMEOUT_SECONDS = v1.EXPECTED_TIMEOUT_SECONDS
EXPECTED_TOOL_RETRIES = v1.EXPECTED_TOOL_RETRIES
EXPECTED_PATTERN_IDS = v1.EXPECTED_PATTERN_IDS
FROZEN_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FROZEN_EMBEDDING_DIMENSION = 384
FROZEN_EMBEDDING_NORMALIZED = True
DEFAULT_CONFIG = Path("configs/experiments/sprint3b_kilo_v2.yaml")
DEFAULT_FREEZE = Path("research/evidence/results/GS-E003/sprint3b_kilo_v2/freeze.json")

V2_FREEZE_INPUT_PATHS: tuple[str, ...] = (
    "configs/experiments/sprint3b_kilo_v2.yaml",
    "configs/models/kilo_coding.yaml",
    "experiments/run_sprint3b_kilo_v2.py",
    "benchmark/manifests/pilot.jsonl",
    "benchmark/annotations/recurrence_validation.csv",
    "src/graph_swarm/agent/coding_agent.py",
    "src/graph_swarm/agent/advisory.py",
    "src/graph_swarm/agent/dependencies.py",
    "src/graph_swarm/agent/prompts.py",
    "src/graph_swarm/advisory/service.py",
    "src/graph_swarm/integration/advisory_runtime.py",
    "src/graph_swarm/memory/recovery_embeddings.py",
    "src/graph_swarm/research/runner.py",
    "src/graph_swarm/settings.py",
    "configs/research/gate_b1_environments.json",
)


class Sprint3BExecutionError(v1.Sprint3BExecutionError):
    """Base error for corrected Sprint 3B execution violations."""


class FreezeValidationError(Sprint3BExecutionError):
    """Raised when the v2 protocol or runtime freeze is not exact."""


FrozenExecutionContext = v1.FrozenExecutionContext
ExecutionReport = v1.ExecutionReport
ExecutionSlot = v1.ExecutionSlot
RunValidity = v1.RunValidity
SlotStatus = v1.SlotStatus
SlotObservation = v1.SlotObservation
PrimaryRunCollisionError = v1.PrimaryRunCollisionError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise FreezeValidationError(f"{label} differs from the frozen protocol: {actual!r}")


def _freeze_input_hashes(root: Path) -> dict[str, str]:
    return {relative: _sha256(root / relative) for relative in V2_FREEZE_INPUT_PATHS}


def _validate_protocol_embedding(protocol: dict[str, Any]) -> None:
    embedding = protocol.get("embedding")
    if not isinstance(embedding, dict):
        raise FreezeValidationError("v2 protocol has no frozen treatment embedding configuration")
    _require_equal(embedding.get("model"), FROZEN_EMBEDDING_MODEL, "embedding model")
    _require_equal(embedding.get("dimension"), FROZEN_EMBEDDING_DIMENSION, "embedding dimension")
    _require_equal(
        embedding.get("normalized"), FROZEN_EMBEDDING_NORMALIZED, "embedding normalization"
    )


def validate_treatment_embedding_configuration(
    protocol: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> None:
    """Require the runtime embedding identity before any experimental slot."""
    if protocol is not None:
        _validate_protocol_embedding(protocol)
    selected = settings or get_settings()
    if selected.hf_embedding_model != FROZEN_EMBEDDING_MODEL:
        raise FreezeValidationError(
            "HF_EMBEDDING_MODEL must exactly match the frozen treatment model "
            f"{FROZEN_EMBEDDING_MODEL!r}; resolved {selected.hf_embedding_model!r}"
        )
    if (
        RECOVERY_PATTERN_EMBEDDING_MODEL != FROZEN_EMBEDDING_MODEL
        or RECOVERY_PATTERN_EMBEDDING_DIMENSION != FROZEN_EMBEDDING_DIMENSION
        or RECOVERY_PATTERN_EMBEDDING_NORMALIZED is not FROZEN_EMBEDDING_NORMALIZED
    ):
        raise FreezeValidationError(
            "runtime RecoveryPattern embedding constants differ from v2 freeze"
        )


def load_frozen_execution_context(
    *,
    project_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
    require_credentials: bool = False,
    allow_existing_artifacts: bool = True,
) -> FrozenExecutionContext:
    """Load v2 protocol inputs without provider, Hugging Face, or Neo4j calls."""
    root = project_root.expanduser().resolve()
    resolved_config = _project_path(config_path, root)
    resolved_freeze = _project_path(freeze_path, root)
    try:
        protocol = sprint3a.load_protocol(resolved_config)
        freeze = json.loads(resolved_freeze.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, sprint3a.Sprint3AProtocolError) as error:
        raise FreezeValidationError(
            f"could not load frozen Sprint 3B v2 inputs: {error}"
        ) from error
    if not isinstance(freeze, dict):
        raise FreezeValidationError("freeze artifact must be a JSON object")

    _require_equal(freeze.get("status"), "READY", "freeze status")
    _require_equal(freeze.get("experiment_id"), EXPERIMENT_ID, "experiment ID")
    _require_equal(freeze.get("protocol_revision"), PROTOCOL_REVISION, "protocol revision")
    _require_equal(protocol.get("protocol_revision"), PROTOCOL_REVISION, "protocol revision")
    _require_equal(_sha256(resolved_config), freeze.get("config_sha256"), "config hash")
    input_hashes = _freeze_input_hashes(root)
    _require_equal(input_hashes, freeze.get("freeze_input_hashes"), "freeze-input hashes")
    _require_equal(
        sprint3a.freeze_inputs_hash(input_hashes),
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
    _require_equal(protocol.get("provider"), EXPECTED_PROVIDER, "provider")
    _require_equal(protocol.get("model"), EXPECTED_CODING_MODEL, "coding model")
    _require_equal(
        protocol.get("model_resolution_variable"), "KILO_CODING_MODEL", "model resolution variable"
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
    if frozenset(protocol.get("treatment_pattern_ids", ())) != EXPECTED_PATTERN_IDS:
        raise FreezeValidationError(
            "treatment-visible pattern IDs differ from the frozen canonical five"
        )
    if EXPECTED_PATTERN_IDS != R13B_TREATMENT_PATTERN_IDS or len(EXPECTED_PATTERN_IDS) != 5:
        raise FreezeValidationError(
            "runtime canonical treatment pattern set is not the frozen five"
        )
    _validate_protocol_embedding(protocol)

    configuration = load_experiment_configuration(resolved_config, project_root=root)
    if (
        configuration.model.provider != EXPECTED_PROVIDER
        or configuration.model.settings.get("temperature") != 0
    ):
        raise FreezeValidationError(
            "loaded model configuration differs from the frozen comparison contract"
        )
    configuration = dataclasses.replace(
        configuration,
        model=configuration.model.model_copy(
            update={
                "model": EXPECTED_CODING_MODEL,
                "prompt_version": str(protocol.get("prompt_version", "")),
                "settings": {"temperature": 0},
            }
        ),
    )
    if require_credentials:
        required = (
            "KILO_API_KEY",
            "HF_TOKEN",
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "NEO4J_DATABASE",
        )
        missing = [name for name in required if not v1._dotenv_value(root, name)]
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
        v1.ExecutionSlot(index=index, task_id=task_id, condition=ExperimentCondition(condition))
        for index, (task_id, condition) in enumerate(plan_values, start=1)
    )
    return v1.FrozenExecutionContext(
        project_root=root,
        config=configuration,
        protocol=protocol,
        freeze=freeze,
        freeze_hash=_sha256(resolved_freeze),
        execution_commit_sha=v1._git_head(root),
        plan=slots,
        cases=cases,
        artifact_root=artifact_root,
    )


def build_execution_plan(protocol: dict[str, Any]) -> tuple[ExecutionSlot, ...]:
    return v1.build_execution_plan(protocol)


def inspect_primary_slots(
    artifact_root: Path, plan: tuple[ExecutionSlot, ...]
) -> tuple[SlotObservation, ...]:
    return v1.inspect_primary_slots(artifact_root, plan)


def build_condition_runner(
    configuration: LoadedExperimentConfiguration,
    *,
    condition: ExperimentCondition,
    objective_evaluator: ObjectiveTaskEvaluator,
    recurrence_evaluator: RecurrenceEvaluator,
    workspace_resolver: Any,
    execution_runtime_resolver: Any,
    advisory_runtime: Neo4jAdvisoryRuntime | None = None,
    settings: Settings | None = None,
    artifact_store: ExperimentRunArtifactStore | None = None,
) -> ExperimentRunner:
    if condition is ExperimentCondition.B0:
        if advisory_runtime is not None:
            raise Sprint3BExecutionError("B0 cannot be constructed with an advisory runtime")
        advisory_service = None
    elif condition is ExperimentCondition.T:
        if advisory_runtime is None or not isinstance(
            advisory_runtime.treatment_repository, R13bTreatmentRepository
        ):
            raise Sprint3BExecutionError("T must use the real injected R13bTreatmentRepository")
        advisory_service = advisory_runtime.advisory_service
    else:
        raise Sprint3BExecutionError(f"unsupported Sprint 3B condition: {condition}")
    return ExperimentRunner(
        configuration,
        settings=v1._settings_for_frozen_model(settings, configuration),
        objective_evaluator=objective_evaluator,
        recurrence_evaluator=recurrence_evaluator,
        workspace_resolver=workspace_resolver,
        execution_runtime_resolver=execution_runtime_resolver,
        advisory_service=advisory_service,
        fail_closed_advisory=condition is ExperimentCondition.T,
        condition=condition,
        artifact_store=artifact_store,
    )


def validate_treatment_dependency_preflight(
    context: FrozenExecutionContext,
    *,
    settings: Settings | None = None,
) -> None:
    """Probe HF once and read the frozen Neo4j corpus before Kilo."""
    selected_settings = settings or get_settings()
    validate_treatment_embedding_configuration(context.protocol, selected_settings)
    try:
        vector = RecoveryPatternEmbedder(settings=selected_settings).embed_text(
            "GS-E003 Sprint 3B Kilo v2 treatment dependency preflight"
        )
    except Exception as error:
        raise Sprint3BExecutionError(
            f"Hugging Face treatment embedding preflight failed: {error}"
        ) from error
    if len(vector) != FROZEN_EMBEDDING_DIMENSION:
        raise Sprint3BExecutionError(
            f"Hugging Face treatment embedding preflight returned {len(vector)} dimensions"
        )

    runtime: Neo4jAdvisoryRuntime | None = None
    try:
        runtime = create_neo4j_advisory_runtime(
            selected_settings,
            fail_closed_advisory=True,
        )
        runtime.repository.verify_connectivity()
        for pattern_id in sorted(R13B_TREATMENT_PATTERN_IDS):
            runtime.treatment_repository.get_recovery_pattern(pattern_id)
    except Exception as error:
        raise Sprint3BExecutionError(
            f"Neo4j treatment dependency preflight failed: {error}"
        ) from error
    finally:
        if runtime is not None:
            runtime.close()


def validate_kilo_provider_preflight(context: FrozenExecutionContext) -> str:
    if context.protocol.get("provider") != EXPECTED_PROVIDER:
        raise FreezeValidationError("Kilo provider preflight requires the v2 Kilo protocol")
    settings = v1._settings_for_frozen_model(get_settings(), context.config)
    return coding_agent.preflight_kilo_provider(settings)


def _is_treatment_infrastructure_failure(
    execution: ExperimentExecution,
    objective: object,
) -> bool:
    """Fail closed if the agent turns a lookup exception into a tool result."""
    if v1._is_infrastructure_failure(execution, objective):
        return True
    return execution.artifact.condition is ExperimentCondition.T and bool(
        execution.dependencies.advisory_errors
    )


def validate_live_preflight(context: FrozenExecutionContext) -> None:
    return v1.validate_live_preflight(context)


def execute_primary_plan(*args: Any, **kwargs: Any) -> ExecutionReport:
    return v1.execute_primary_plan(*args, **kwargs)


def execute_frozen_primary_runs(
    *,
    project_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
) -> ExecutionReport:
    context = load_frozen_execution_context(
        project_root=project_root,
        config_path=config_path,
        freeze_path=freeze_path,
        require_credentials=True,
    )
    settings = get_settings()
    validate_treatment_embedding_configuration(context.protocol, settings)
    validate_live_preflight(context)
    settings = v1._settings_for_frozen_model(settings, context.config)
    instance_ids = sprint3._manifest_instance_ids(
        context.config.task_manifest_path, EXPECTED_TASK_IDS
    )
    images = sprint3._manifest_image_names(context.config.task_manifest_path, EXPECTED_TASK_IDS)
    frozen_by_instance = sprint3.load_frozen_swesmith_cases(
        tuple(instance_ids.values()), allow_network=False
    )
    frozen_cases = {
        task_id: frozen_by_instance[instance_ids[task_id].lower()] for task_id in EXPECTED_TASK_IDS
    }
    policy = sprint3.load_benchmark_environment_policy(
        context.project_root / "configs/research/benchmark_environments.toml"
    )
    environments = sprint3._prepare_task_environments(
        list(context.cases),
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
        coverage_policy_selection_version=sprint3.OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION,
    )
    recurrence = sprint3.make_recurrence_matcher(frozen_cases)
    artifact_store = ExperimentRunArtifactStore(context.artifact_root)
    runtime = create_neo4j_advisory_runtime(settings, fail_closed_advisory=True)
    try:
        runners: dict[ExperimentCondition, ExperimentRunner] = {
            ExperimentCondition.B0: build_condition_runner(
                context.config,
                condition=ExperimentCondition.B0,
                objective_evaluator=cast(ObjectiveTaskEvaluator, objective),
                recurrence_evaluator=cast(RecurrenceEvaluator, recurrence),
                workspace_resolver=sprint3._make_workspace_resolver(
                    source_root=context.config.workspace_baseline_root_path,
                    execution_root=context.config.workspace_execution_root_path,
                    frozen_cases=frozen_cases,
                    condition=ExperimentCondition.B0,
                ),
                execution_runtime_resolver=lambda task, _workspace: environments[
                    task.id
                ].agent_execution_runtime(),
                settings=settings,
                artifact_store=artifact_store,
            ),
            ExperimentCondition.T: build_condition_runner(
                context.config,
                condition=ExperimentCondition.T,
                advisory_runtime=runtime,
                objective_evaluator=cast(ObjectiveTaskEvaluator, objective),
                recurrence_evaluator=cast(RecurrenceEvaluator, recurrence),
                workspace_resolver=sprint3._make_workspace_resolver(
                    source_root=context.config.workspace_baseline_root_path,
                    execution_root=context.config.workspace_execution_root_path,
                    frozen_cases=frozen_cases,
                    condition=ExperimentCondition.T,
                ),
                execution_runtime_resolver=lambda task, _workspace: environments[
                    task.id
                ].agent_execution_runtime(),
                settings=settings,
                artifact_store=artifact_store,
            ),
        }

        def run_slot(slot: ExecutionSlot, case: BenchmarkTaskCase) -> ExperimentExecution:
            return runners[slot.condition].run_case(case)

        identities = {
            task_id: {
                "environment_fingerprint": environments[task_id].environment_fingerprint,
                "runtime_type": environments[task_id].runtime_type,
            }
            for task_id in EXPECTED_TASK_IDS
        }
        return execute_primary_plan(
            context,
            run_slot,
            artifact_store=artifact_store,
            environment_identities=identities,
            infrastructure_checker=lambda execution: _is_treatment_infrastructure_failure(
                execution, objective
            ),
        )
    finally:
        runtime.close()


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
    )
    settings = get_settings()
    validate_treatment_embedding_configuration(context.protocol, settings)
    if args.preflight_only:
        validate_treatment_dependency_preflight(context, settings=settings)
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
    "EXPECTED_TASK_IDS",
    "FROZEN_EMBEDDING_DIMENSION",
    "FROZEN_EMBEDDING_MODEL",
    "FROZEN_EMBEDDING_NORMALIZED",
    "FreezeValidationError",
    "FrozenExecutionContext",
    "Sprint3BExecutionError",
    "build_condition_runner",
    "build_execution_plan",
    "execute_frozen_primary_runs",
    "execute_primary_plan",
    "inspect_primary_slots",
    "load_frozen_execution_context",
    "validate_kilo_provider_preflight",
    "validate_live_preflight",
    "validate_treatment_dependency_preflight",
    "validate_treatment_embedding_configuration",
]
