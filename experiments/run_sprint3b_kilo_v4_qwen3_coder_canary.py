"""Execute the six-run GS-E003 Sprint 3B v4 Qwen3-Coder development canary."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml

from experiments import run_sprint3b_kilo_v3 as v3
from experiments import run_sprint3b_reduced_b0_t as v1
from experiments import sprint3
from experiments.sprint3 import IsolatedTaskEnvironment
from graph_swarm.agent import coding_agent
from graph_swarm.agent.prompts import system_prompt_sha256
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.integration.advisory_runtime import (
    create_neo4j_advisory_runtime,
)
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    ExperimentConfigurationError,
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

# This entrypoint deliberately reuses the established Sprint 3B execution
# primitives while keeping v3 validation and defaults entirely separate.
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportUnnecessaryCast=false

EXPERIMENT_ID = "GS-E003"
PROTOCOL_REVISION = "sprint3b-kilo-reduced-b0-t-v4-qwen3-coder-prompt-canary"
CONFIG_VERSION = "sprint3b-kilo-v4-qwen3-coder-prompt-canary"
EXPECTED_CODING_MODEL = "qwen/qwen3-coder"
EXPECTED_TASK_IDS = ("GS-T006", "GS-T007", "GS-T008")
EXPECTED_CONDITIONS = (ExperimentCondition.B0, ExperimentCondition.T)
EXPECTED_PLAN = tuple(
    (task_id, condition.value)
    for task_id in EXPECTED_TASK_IDS
    for condition in EXPECTED_CONDITIONS
)
EXPECTED_PRIMARY_RUNS = 6
EXPECTED_MAX_ACTIONS = 28
EXPECTED_MAX_REQUESTS = 24
EXPECTED_RECURRENCE_VERSION = "frozen_recurrence_matcher_v2_structured_pytest"
DEFAULT_CONFIG = Path("configs/experiments/sprint3b_kilo_v4_canary_qwen3_coder.yaml")
DEFAULT_FREEZE = Path(
    "research/evidence/results/GS-E003/sprint3b_kilo_v4_qwen3_coder_canary/freeze.json"
)

QWEN3_CODER_FREEZE_INPUT_PATHS = (
    "configs/experiments/sprint3b_kilo_v4_canary_qwen3_coder.yaml",
    "experiments/run_sprint3b_kilo_v4_qwen3_coder_canary.py",
    "experiments/run_sprint3b_kilo_v3.py",
    "experiments/run_sprint3b_reduced_b0_t.py",
    "experiments/sprint3.py",
    "experiments/sprint3a.py",
    "benchmark/manifests/pilot.jsonl",
    "benchmark/annotations/recurrence_validation.csv",
    "configs/models/kilo_coding_qwen3_coder.yaml",
    "src/graph_swarm/agent/prompts.py",
    "src/graph_swarm/agent/coding_agent.py",
    "src/graph_swarm/agent/advisory.py",
    "src/graph_swarm/agent/dependencies.py",
    "src/graph_swarm/agent/hooks.py",
    "src/graph_swarm/agent/model_output.py",
    "src/graph_swarm/agent/pacing.py",
    "src/graph_swarm/agent/stagnation.py",
    "src/graph_swarm/agent/tools/edit_file.py",
    "src/graph_swarm/agent/tools/read_file.py",
    "src/graph_swarm/agent/tools/run_command.py",
    "src/graph_swarm/agent/tools/run_tests.py",
    "src/graph_swarm/agent/tools/write_file.py",
    "src/graph_swarm/advisory/formatting.py",
    "src/graph_swarm/advisory/service.py",
    "src/graph_swarm/graph/_validation.py",
    "src/graph_swarm/graph/neo4j_repository.py",
    "src/graph_swarm/graph/queries.py",
    "src/graph_swarm/graph/read_models.py",
    "src/graph_swarm/graph/repository.py",
    "src/graph_swarm/research/runner.py",
    "src/graph_swarm/research/contracts.py",
    "src/graph_swarm/research/artifacts.py",
    "src/graph_swarm/research/benchmark_environments.py",
    "src/graph_swarm/research/benchmark_runtime_smoke.py",
    "src/graph_swarm/retrieval/applicability.py",
    "src/graph_swarm/retrieval/candidates.py",
    "src/graph_swarm/retrieval/query.py",
    "src/graph_swarm/retrieval/scorer.py",
    "src/graph_swarm/retrieval/service.py",
    "src/graph_swarm/integration/advisory_runtime.py",
    "src/graph_swarm/memory/recovery_evidence.py",
    "src/graph_swarm/domain/environment.py",
    "src/graph_swarm/domain/action.py",
    "src/graph_swarm/domain/actions.py",
    "src/graph_swarm/domain/advice.py",
    "src/graph_swarm/domain/behavior.py",
    "src/graph_swarm/domain/events.py",
    "src/graph_swarm/domain/failures.py",
    "src/graph_swarm/domain/recovery_patterns.py",
    "src/graph_swarm/domain/tasks.py",
    "src/graph_swarm/memory/recovery_embeddings.py",
    "src/graph_swarm/settings.py",
    "configs/research/benchmark_environments.toml",
)


class Qwen3CoderCanaryConfigurationError(ExperimentConfigurationError):
    """Raised when the six-run v4 canary package is not exact."""


def _project_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise Qwen3CoderCanaryConfigurationError(
            f"{label} differs from the v4 canary contract: {actual!r}"
        )


def _settings_for_frozen_model(
    settings: Settings | None,
    configuration: LoadedExperimentConfiguration,
) -> Settings:
    """Pin both preflight and execution to the loaded Qwen3-Coder model."""
    base = settings or get_settings()
    if configuration.model.provider != v3.EXPECTED_PROVIDER:
        return base
    return base.model_copy(
        update={
            "model_provider": v3.EXPECTED_PROVIDER,
            "kilo_coding_model": configuration.model.model,
            "kilo_api_key": v1._dotenv_value(Path.cwd(), "KILO_API_KEY")
            or base.kilo_api_key,
        }
    )


def _protocol(config_path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise Qwen3CoderCanaryConfigurationError(
            f"could not read v4 canary configuration {config_path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Qwen3CoderCanaryConfigurationError("v4 canary configuration must be a mapping")
    return cast(dict[str, Any], value)


def _validate_protocol(
    protocol: dict[str, Any], configuration: LoadedExperimentConfiguration
) -> None:
    _require_equal(protocol.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _require_equal(protocol.get("protocol_revision"), PROTOCOL_REVISION, "protocol_revision")
    _require_equal(protocol.get("config_version"), CONFIG_VERSION, "config_version")
    _require_equal(protocol.get("conditions"), ["B0", "T"], "conditions")
    _require_equal(tuple(protocol.get("task_ids", ())), EXPECTED_TASK_IDS, "task_ids")
    _require_equal(tuple(protocol.get("task_order", ())), EXPECTED_TASK_IDS, "task_order")
    raw_plan = protocol.get("execution_plan")
    if not isinstance(raw_plan, list):
        raise Qwen3CoderCanaryConfigurationError("execution_plan must be a list")
    plan = tuple(tuple(item) for item in raw_plan)
    _require_equal(plan, EXPECTED_PLAN, "execution_plan")
    _require_equal(
        protocol.get("planned_primary_runs"), EXPECTED_PRIMARY_RUNS, "planned_primary_runs"
    )
    _require_equal(protocol.get("provider"), v3.EXPECTED_PROVIDER, "provider")
    _require_equal(protocol.get("model"), EXPECTED_CODING_MODEL, "model")
    _require_equal(
        protocol.get("limits", {}).get("max_actions"), EXPECTED_MAX_ACTIONS, "max_actions"
    )
    _require_equal(
        protocol.get("limits", {}).get("max_requests"), EXPECTED_MAX_REQUESTS, "max_requests"
    )
    _require_equal(
        protocol.get("recurrence_evaluator_version"),
        EXPECTED_RECURRENCE_VERSION,
        "recurrence_evaluator_version",
    )
    _require_equal(protocol.get("system_prompt"), "R13B_SYSTEM_PROMPT", "system_prompt")
    _require_equal(
        protocol.get("system_prompt_sha256"),
        system_prompt_sha256("R13B_SYSTEM_PROMPT"),
        "system_prompt_sha256",
    )
    _require_equal(
        configuration.config.system_prompt, "R13B_SYSTEM_PROMPT", "loaded system_prompt"
    )
    _require_equal(
        configuration.config.limits.max_actions, EXPECTED_MAX_ACTIONS, "loaded max_actions"
    )
    _require_equal(
        configuration.config.limits.max_requests, EXPECTED_MAX_REQUESTS, "loaded max_requests"
    )
    _require_equal(configuration.model.provider, v3.EXPECTED_PROVIDER, "loaded provider")
    _require_equal(configuration.model.model, EXPECTED_CODING_MODEL, "loaded model")
    if protocol.get("task_workspace_network") != "disabled":
        raise Qwen3CoderCanaryConfigurationError("task workspace network must remain disabled")
    if protocol.get("memory_writes") != "forbidden":
        raise Qwen3CoderCanaryConfigurationError("memory writes must remain forbidden")
    if protocol.get("neo4j_write_policy") != "forbidden":
        raise Qwen3CoderCanaryConfigurationError("Neo4j writes must remain forbidden")
    _require_equal(
        protocol.get("tools"),
        ["read_file", "write_file", "edit_file", "run_tests", "run_command"],
        "tools",
    )


def _input_hashes(root: Path, config_path: Path) -> dict[str, str]:
    paths: list[str] = list(QWEN3_CODER_FREEZE_INPUT_PATHS)
    paths[0] = config_path.resolve().relative_to(root.resolve()).as_posix()
    hashes: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise Qwen3CoderCanaryConfigurationError(f"v4 freeze input is missing: {relative}")
        hashes[relative] = _sha256(path)
    return hashes


def _require_fresh_primary_artifact_root(protocol: dict[str, Any], root: Path) -> Path:
    """Refuse a live canary when its immutable primary root already exists."""
    artifact_root = _project_path(str(protocol["artifact_root"]), root)
    if artifact_root.exists():
        raise Qwen3CoderCanaryConfigurationError(
            f"primary artifact root already exists; refusing a second canary execution: "
            f"{artifact_root}"
        )
    return artifact_root


def _load_package(
    *, project_root: Path, config_path: Path
) -> tuple[dict[str, Any], LoadedExperimentConfiguration, tuple[BenchmarkTaskCase, ...]]:
    root = project_root.expanduser().resolve()
    resolved_config = _project_path(config_path, root)
    protocol = _protocol(resolved_config)
    configuration = load_experiment_configuration(resolved_config, project_root=root)
    _validate_protocol(protocol, configuration)
    cases = tuple(
        case
        for case in load_task_cases(
            configuration.task_manifest_path,
            problem_statements_path=configuration.task_problems_path,
        )
        if case.task.id in EXPECTED_TASK_IDS
    )
    if tuple(case.task.id for case in cases) != EXPECTED_TASK_IDS:
        raise Qwen3CoderCanaryConfigurationError(
            "v4 canary task cases are incomplete or out of order"
        )
    if tuple(case.task.chronological_index for case in cases) != (6, 7, 8):
        raise Qwen3CoderCanaryConfigurationError("v4 canary task indexes must be 6, 7, and 8")
    return protocol, configuration, cases


def create_freeze_artifact(
    *, project_root: Path, config_path: Path = DEFAULT_CONFIG, freeze_path: Path = DEFAULT_FREEZE
) -> dict[str, Any]:
    """Create the v4 package freeze without provider, model, or memory calls."""
    root = project_root.expanduser().resolve()
    resolved_config = _project_path(config_path, root)
    resolved_freeze = _project_path(freeze_path, root)
    if resolved_freeze.exists():
        raise Qwen3CoderCanaryConfigurationError(
            f"freeze artifact already exists: {resolved_freeze}"
        )
    protocol, configuration, _cases = _load_package(
        project_root=root, config_path=resolved_config
    )
    hashes = _input_hashes(root, resolved_config)
    freeze: dict[str, Any] = {
        "status": "READY",
        "freeze_type": "development_canary_package_no_results",
        "experiment_id": EXPERIMENT_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "config_version": CONFIG_VERSION,
        "config_path": resolved_config.resolve().relative_to(root).as_posix(),
        "config_sha256": _sha256(resolved_config),
        "freeze_input_hashes": hashes,
        "task_ids": list(EXPECTED_TASK_IDS),
        "task_order": list(EXPECTED_TASK_IDS),
        "conditions": [condition.value for condition in EXPECTED_CONDITIONS],
        "execution_plan": [list(item) for item in EXPECTED_PLAN],
        "planned_primary_runs": EXPECTED_PRIMARY_RUNS,
        "model": {
            "provider": configuration.model.provider,
            "name": configuration.model.model,
            "system_prompt_id": configuration.system_prompt_id,
            "system_prompt_sha256": configuration.resolved_system_prompt_sha256,
        },
        "limits": {
            "max_actions": configuration.config.limits.max_actions,
            "max_requests": configuration.config.limits.max_requests,
        },
        "recurrence_evaluator": {
            "name": protocol.get("recurrence_evaluator"),
            "version": protocol.get("recurrence_evaluator_version"),
        },
        "artifact_destination": {
            "run_artifact_root": protocol.get("artifact_root"),
            "freeze_artifact": protocol.get("freeze_artifact"),
        },
    }
    resolved_freeze.parent.mkdir(parents=True, exist_ok=True)
    resolved_freeze.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return freeze


def load_canary_context(
    *,
    project_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
    require_credentials: bool = False,
    allow_existing_artifacts: bool = True,
) -> v1.FrozenExecutionContext:
    """Load and validate v4 inputs without provider, model, or Neo4j calls."""
    root = project_root.expanduser().resolve()
    resolved_config = _project_path(config_path, root)
    resolved_freeze = _project_path(freeze_path, root)
    protocol, configuration, cases = _load_package(
        project_root=root, config_path=resolved_config
    )
    try:
        freeze = json.loads(resolved_freeze.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Qwen3CoderCanaryConfigurationError(
            f"could not load v4 canary freeze {resolved_freeze}: {error}"
        ) from error
    if not isinstance(freeze, dict):
        raise Qwen3CoderCanaryConfigurationError("v4 canary freeze must be a JSON object")
    _require_equal(freeze.get("status"), "READY", "freeze status")
    _require_equal(freeze.get("experiment_id"), EXPERIMENT_ID, "freeze experiment_id")
    _require_equal(freeze.get("protocol_revision"), PROTOCOL_REVISION, "freeze protocol_revision")
    _require_equal(freeze.get("config_version"), CONFIG_VERSION, "freeze config_version")
    _require_equal(freeze.get("config_sha256"), _sha256(resolved_config), "freeze config hash")
    _require_equal(
        freeze.get("freeze_input_hashes"),
        _input_hashes(root, resolved_config),
        "freeze input hashes",
    )
    _require_equal(
        freeze.get("execution_plan"),
        [list(item) for item in EXPECTED_PLAN],
        "freeze execution plan",
    )
    _require_equal(
        freeze.get("planned_primary_runs"),
        EXPECTED_PRIMARY_RUNS,
        "freeze planned_primary_runs",
    )
    destination = freeze.get("artifact_destination")
    if not isinstance(destination, dict):
        raise Qwen3CoderCanaryConfigurationError(
            "v4 freeze has no artifact destination metadata"
        )
    _require_equal(
        destination.get("run_artifact_root"),
        protocol.get("artifact_root"),
        "freeze artifact root",
    )
    _require_equal(
        destination.get("freeze_artifact"),
        protocol.get("freeze_artifact"),
        "freeze artifact path",
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
            raise Qwen3CoderCanaryConfigurationError(
                "required runtime credentials are missing: " + ", ".join(missing)
            )
    artifact_root = _project_path(str(protocol["artifact_root"]), root)
    if not allow_existing_artifacts:
        artifact_root = _require_fresh_primary_artifact_root(protocol, root)
    plan = tuple(
        v1.ExecutionSlot(index=index, task_id=task_id, condition=ExperimentCondition(condition))
        for index, (task_id, condition) in enumerate(EXPECTED_PLAN, start=1)
    )
    return v1.FrozenExecutionContext(
        project_root=root,
        config=configuration,
        protocol=protocol,
        freeze=freeze,
        freeze_hash=_sha256(resolved_freeze),
        execution_commit_sha=v1._git_head(root),
        plan=plan,
        cases=cases,
        artifact_root=artifact_root,
    )


def build_condition_runner(
    configuration: LoadedExperimentConfiguration,
    *,
    condition: ExperimentCondition,
    objective_evaluator: ObjectiveTaskEvaluator,
    recurrence_evaluator: RecurrenceEvaluator,
    workspace_resolver: Any,
    execution_runtime_resolver: Any,
    environment_resolver: Any | None = None,
    advisory_runtime: Any | None = None,
    settings: Settings | None = None,
    artifact_store: ExperimentRunArtifactStore | None = None,
) -> ExperimentRunner:
    """Build a condition runner while preserving the frozen Qwen3-Coder model."""
    if condition is ExperimentCondition.B0:
        if advisory_runtime is not None:
            raise v3.Sprint3BExecutionError(
                "B0 cannot be constructed with an advisory runtime"
            )
        if environment_resolver is not None:
            raise v3.Sprint3BExecutionError(
                "B0 cannot be constructed with a treatment environment resolver"
            )
        advisory_service = None
    elif condition is ExperimentCondition.T:
        if advisory_runtime is None or not isinstance(
            advisory_runtime.treatment_repository, v3.R13bTreatmentRepository
        ):
            raise v3.Sprint3BExecutionError(
                "T must use the real injected R13bTreatmentRepository"
            )
        advisory_service = advisory_runtime.advisory_service
    else:
        raise v3.Sprint3BExecutionError(f"unsupported Sprint 3B condition: {condition}")
    return ExperimentRunner(
        configuration,
        settings=_settings_for_frozen_model(settings, configuration),
        objective_evaluator=objective_evaluator,
        recurrence_evaluator=recurrence_evaluator,
        workspace_resolver=workspace_resolver,
        execution_runtime_resolver=execution_runtime_resolver,
        environment_resolver=environment_resolver,
        advisory_service=advisory_service,
        fail_closed_advisory=condition is ExperimentCondition.T,
        condition=condition,
        artifact_store=artifact_store,
    )


def make_environment_resolver(
    environments: dict[str, IsolatedTaskEnvironment],
) -> Any:
    """Build the T resolver from the prepared selected-task environments."""

    def resolve(task: Any, run_id: str) -> EnvironmentContext:
        try:
            prepared = environments[task.id]
        except KeyError as error:
            raise v3.Sprint3BExecutionError(
                f"no prepared benchmark environment for {task.id}"
            ) from error
        versions = (
            {"python": prepared.python_version}
            if prepared.python_version is not None
            else {}
        )
        return EnvironmentContext(
            id=f"{run_id}-environment",
            repository=task.repository,
            runtime=prepared.runtime_type,
            versions=versions,
            markers={},
        )

    return resolve


def validate_live_preflight(context: v1.FrozenExecutionContext) -> None:
    """Validate only the three selected benchmark environments before execution."""
    cases = list(context.cases)
    baseline = context.config.workspace_baseline_root_path
    execution = context.config.workspace_execution_root_path
    sprint3._verify_baselines(cases, baseline)
    policy = sprint3.load_benchmark_environment_policy(
        context.project_root / "configs/research/benchmark_environments.toml"
    )
    instance_ids = sprint3._manifest_instance_ids(
        context.config.task_manifest_path, EXPECTED_TASK_IDS
    )
    images = sprint3._manifest_image_names(context.config.task_manifest_path, EXPECTED_TASK_IDS)
    frozen_by_instance = sprint3.load_frozen_swesmith_cases(
        tuple(instance_ids.values()), allow_network=False
    )
    frozen_cases = {
        task_id: frozen_by_instance[instance_ids[task_id].lower()]
        for task_id in EXPECTED_TASK_IDS
    }
    sprint3._verify_frozen_patches(cases, baseline, frozen_cases)
    environments = sprint3._prepare_task_environments(
        cases,
        environment_root=execution / "sprint3-task-environments",
        source_root=baseline,
        dependency_overlays=sprint3.TASK_DEPENDENCY_OVERLAYS,
        container_images=images,
        benchmark_policy=policy,
        benchmark_manifest_path=context.config.task_manifest_path,
        mode="preflight",
    )
    sprint3._require_validated_environments(cases, environments)


def validate_kilo_provider_preflight(context: v1.FrozenExecutionContext) -> str:
    """Probe Kilo using the model loaded from the frozen Qwen3-Coder configuration."""
    if context.protocol.get("provider") != v3.EXPECTED_PROVIDER:
        raise Qwen3CoderCanaryConfigurationError(
            "Kilo provider preflight requires the Kilo protocol"
        )
    settings = _settings_for_frozen_model(get_settings(), context.config)
    return coding_agent.preflight_kilo_provider(settings)


def validate_canary_preflight(
    context: v1.FrozenExecutionContext,
    *,
    settings: Any | None = None,
) -> Any:
    """Run all provider/dependency/environment checks before primary artifacts."""
    selected_settings = settings or get_settings()
    v3.validate_treatment_embedding_configuration(context.protocol, selected_settings)
    validate_kilo_provider_preflight(context)
    v3.validate_treatment_dependency_preflight(
        cast(v3.FrozenExecutionContext, context),
        settings=selected_settings,
    )
    validate_live_preflight(context)
    return selected_settings


def execute_canary(
    *, project_root: Path, config_path: Path = DEFAULT_CONFIG, freeze_path: Path = DEFAULT_FREEZE
) -> v1.ExecutionReport:
    """Run the frozen six-slot canary after all live preflight checks."""
    context = load_canary_context(
        project_root=project_root,
        config_path=config_path,
        freeze_path=freeze_path,
        require_credentials=True,
        allow_existing_artifacts=False,
    )
    settings = validate_canary_preflight(context)
    settings = _settings_for_frozen_model(settings, context.config)
    instance_ids = sprint3._manifest_instance_ids(
        context.config.task_manifest_path, EXPECTED_TASK_IDS
    )
    images = sprint3._manifest_image_names(context.config.task_manifest_path, EXPECTED_TASK_IDS)
    frozen_by_instance = sprint3.load_frozen_swesmith_cases(
        tuple(instance_ids.values()), allow_network=False
    )
    frozen_cases = {
        task_id: frozen_by_instance[instance_ids[task_id].lower()]
        for task_id in EXPECTED_TASK_IDS
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
        runners: dict[ExperimentCondition, ExperimentRunner] = {}
        for condition in EXPECTED_CONDITIONS:
            runners[condition] = build_condition_runner(
                context.config,
                condition=condition,
                objective_evaluator=cast(ObjectiveTaskEvaluator, objective),
                recurrence_evaluator=cast(RecurrenceEvaluator, recurrence),
                workspace_resolver=sprint3._make_workspace_resolver(
                    source_root=context.config.workspace_baseline_root_path,
                    execution_root=context.config.workspace_execution_root_path,
                    frozen_cases=frozen_cases,
                    condition=condition,
                ),
                execution_runtime_resolver=lambda task, _workspace: environments[
                    task.id
                ].agent_execution_runtime(),
                environment_resolver=(
                    make_environment_resolver(environments)
                    if condition is ExperimentCondition.T
                    else None
                ),
                advisory_runtime=runtime if condition is ExperimentCondition.T else None,
                settings=settings,
                artifact_store=artifact_store,
            )

        def run_slot(slot: v1.ExecutionSlot, case: BenchmarkTaskCase) -> ExperimentExecution:
            return runners[slot.condition].run_case(case)

        identities = {
            task_id: {
                "environment_fingerprint": environments[task_id].environment_fingerprint,
                "runtime_type": environments[task_id].runtime_type,
            }
            for task_id in EXPECTED_TASK_IDS
        }
        return v1.execute_primary_plan(
            context,
            run_slot,
            artifact_store=artifact_store,
            environment_identities=identities,
            infrastructure_checker=lambda execution: v3._is_treatment_infrastructure_failure(
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
    parser.add_argument("--create-freeze", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    args = make_cli_parser().parse_args()
    if args.create_freeze:
        freeze = create_freeze_artifact(
            project_root=args.project_root,
            config_path=args.config,
            freeze_path=args.freeze,
        )
        print(json.dumps({"status": freeze["status"], "planned_slots": len(EXPECTED_PLAN)}))
        return
    context = load_canary_context(
        project_root=args.project_root,
        config_path=args.config,
        freeze_path=args.freeze,
        require_credentials=True,
    )
    if args.preflight_only:
        validate_canary_preflight(context)
        print(json.dumps({"status": "READY", "planned_slots": len(context.plan)}, sort_keys=True))
        return
    report = execute_canary(
        project_root=args.project_root,
        config_path=args.config,
        freeze_path=args.freeze,
    )
    print(json.dumps(dataclasses.asdict(report), default=str, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_VERSION",
    "DEFAULT_CONFIG",
    "DEFAULT_FREEZE",
    "EXPECTED_CODING_MODEL",
    "EXPECTED_PLAN",
    "EXPECTED_PRIMARY_RUNS",
    "EXPECTED_TASK_IDS",
    "PROTOCOL_REVISION",
    "Qwen3CoderCanaryConfigurationError",
    "create_freeze_artifact",
    "execute_canary",
    "load_canary_context",
    "make_cli_parser",
]









