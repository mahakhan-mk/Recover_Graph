"""Execute the frozen GS-T018 two-slot plan, or validate it without execution."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments import run_sprint3b_reduced_b0_t as execution_core
from experiments import sprint3
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.integration.advisory_runtime import (
    R13B_TREATMENT_PATTERN_IDS,
    Neo4jAdvisoryRuntime,
    R13bTreatmentRepository,
    create_neo4j_advisory_runtime,
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

# The frozen YAML and legacy runner boundaries are intentionally dynamic; keep
# strict checking for this module while suppressing only those boundary reports.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnnecessaryIsInstance=false, reportUnnecessaryCast=false, reportPrivateUsage=false

EXPERIMENT_ID = "GS-E003"
CONFIG_VERSION = "sprint3b-kilo-hardened-development-v3"
EXPECTED_TASK_ID = "GS-T018"
EXPECTED_PLAN = ((EXPECTED_TASK_ID, "B0"), (EXPECTED_TASK_ID, "T"))
EXPECTED_RECURRENCE_VERSION = "frozen_recurrence_matcher_v3_exact_pytest_outcome"
DEFAULT_CONFIG = Path("configs/experiments/gs_e003_hardened_development_v3.yaml")
DEFAULT_FREEZE = Path(
    "research/evidence/results/GS-E003/sprint3b_hardened_development_v3/freeze.json"
)
EXPECTED_PROVIDER = "kilo"
EXPECTED_MODEL = "qwen/qwen3-coder"
EXPECTED_PYTHON = "3.12.1"
EXPECTED_CONTAINER_PYTHON = "/opt/miniconda3/bin/python3.12"
EXPECTED_TOOLS = ("read_file", "write_file", "edit_file", "run_tests", "run_command")


class GST018RunnerConfigurationError(ValueError):
    """Raised when the prepared two-slot contract is not exact."""


@dataclass(frozen=True)
class ExecutionSlot:
    index: int
    task_id: str
    condition: ExperimentCondition

    @property
    def key(self) -> tuple[str, ExperimentCondition]:
        return self.task_id, self.condition


@dataclass(frozen=True)
class PreparedRunner:
    config_path: Path
    freeze_path: Path
    config_sha256: str
    freeze_sha256: str
    slots: tuple[ExecutionSlot, ...]
    protocol: dict[str, Any]
    freeze: dict[str, Any]
    configuration: LoadedExperimentConfiguration
    cases: tuple[BenchmarkTaskCase, ...]
    artifact_root: Path


@dataclass(frozen=True)
class PreparedEnvironment:
    task_id: str
    image: str
    python_executable: str
    python_version: str
    environment_fingerprint: str
    docker_executable: Path

    def agent_execution_runtime(self) -> sprint3.ExecutionRuntime:
        return sprint3.ExecutionRuntime(
            runtime_type="docker",
            docker_executable=self.docker_executable,
            container_image=self.image,
            container_python_executable=self.python_executable,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GST018RunnerConfigurationError("configuration must be a mapping")
    return value


def prepare_runner(
    project_root: Path = Path("."),
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
) -> PreparedRunner:
    root = project_root.resolve()
    config = (root / config_path).resolve() if not config_path.is_absolute() else config_path
    freeze_file = (root / freeze_path).resolve() if not freeze_path.is_absolute() else freeze_path
    protocol = _load_yaml(config)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise GST018RunnerConfigurationError("experiment ID is not GS-E003")
    if protocol.get("config_version") != CONFIG_VERSION:
        raise GST018RunnerConfigurationError("config version is not v3")
    if protocol.get("recurrence_evaluator_version") != EXPECTED_RECURRENCE_VERSION:
        raise GST018RunnerConfigurationError("recurrence evaluator is not v3")
    if protocol.get("provider") != EXPECTED_PROVIDER:
        raise GST018RunnerConfigurationError("provider is not Kilo")
    if protocol.get("model") != EXPECTED_MODEL:
        raise GST018RunnerConfigurationError("coding model is not qwen/qwen3-coder")
    if tuple(protocol.get("tools", ())) != EXPECTED_TOOLS:
        raise GST018RunnerConfigurationError("tools differ from the frozen v3 tool set")
    if protocol.get("conditions") != ["B0", "T"]:
        raise GST018RunnerConfigurationError("conditions are not exactly B0 then T")
    expected_limits = {
        "max_actions": 28,
        "max_requests": 24,
        "timeout_seconds": 300,
        "tool_retries": 3,
        "task_timeout_seconds": 300,
        "model_request_timeout_seconds": 300,
        "tool_timeout_seconds": {"run_command": 30, "run_tests": 120},
    }
    if protocol.get("limits") != expected_limits:
        raise GST018RunnerConfigurationError("limits differ from the frozen v3 limits")
    if protocol.get("memory_writes") != "forbidden":
        raise GST018RunnerConfigurationError("memory writes are not forbidden")
    if protocol.get("neo4j_write_policy") != "forbidden":
        raise GST018RunnerConfigurationError("Neo4j writes are not forbidden")
    if frozenset(protocol.get("treatment_pattern_ids", ())) != R13B_TREATMENT_PATTERN_IDS:
        raise GST018RunnerConfigurationError("treatment pattern allowlist differs from frozen v3")
    environment_gates = protocol.get("environment_version_gates", {})
    if (
        not isinstance(environment_gates, dict)
        or environment_gates.get("python") != EXPECTED_PYTHON
    ):
        raise GST018RunnerConfigurationError("Python version gate is not 3.12.1")
    plan = tuple(tuple(item) for item in protocol.get("execution_plan", ()))
    if plan != EXPECTED_PLAN:
        raise GST018RunnerConfigurationError(f"execution plan is not exactly {EXPECTED_PLAN!r}")
    if protocol.get("planned_primary_runs") != 2:
        raise GST018RunnerConfigurationError("planned_primary_runs must be exactly two")
    loaded = load_experiment_configuration(config, project_root=root)
    if loaded.model.provider != EXPECTED_PROVIDER or loaded.model.model != EXPECTED_MODEL:
        raise GST018RunnerConfigurationError("loaded model configuration differs from v3")
    cases = tuple(
        load_task_cases(
            loaded.task_manifest_path,
            problem_statements_path=loaded.task_problems_path,
        )
    )
    if tuple(case.task.id for case in cases) != (EXPECTED_TASK_ID,):
        raise GST018RunnerConfigurationError("only GS-T018 may be registered")
    raw_freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if raw_freeze.get("status") != "READY" or raw_freeze.get("config_sha256") != _sha256(config):
        raise GST018RunnerConfigurationError("freeze does not match the v3 configuration")
    if raw_freeze.get("recurrence_evaluator", {}).get("version") != EXPECTED_RECURRENCE_VERSION:
        raise GST018RunnerConfigurationError("freeze recurrence evaluator is not v3")
    if raw_freeze.get("model", {}).get("name") != EXPECTED_MODEL:
        raise GST018RunnerConfigurationError("freeze model is not qwen/qwen3-coder")
    if (
        protocol.get("system_prompt") != raw_freeze.get("model", {}).get("system_prompt_id")
        or protocol.get("system_prompt_sha256")
        != raw_freeze.get("model", {}).get("system_prompt_sha256")
    ):
        raise GST018RunnerConfigurationError("prompt identity differs from frozen v3")
    if loaded.system_prompt_id != "R13B_SYSTEM_PROMPT":
        raise GST018RunnerConfigurationError("system prompt is not R13B_SYSTEM_PROMPT")
    if (
        raw_freeze.get("environment_version_gates", {}).get("python") != EXPECTED_PYTHON
        or not raw_freeze.get("environment_version_gates", {}).get("prepared_image")
    ):
        raise GST018RunnerConfigurationError("freeze Python gate is not 3.12.1")
    slots = tuple(
        ExecutionSlot(index=index, task_id=task_id, condition=ExperimentCondition(condition))
        for index, (task_id, condition) in enumerate(EXPECTED_PLAN, start=1)
    )
    return PreparedRunner(
        config,
        freeze_file,
        _sha256(config),
        _sha256(freeze_file),
        slots,
        protocol,
        raw_freeze,
        loaded,
        cases,
        loaded.artifact_root_path,
    )


def _frozen_cases(prepared: PreparedRunner) -> dict[str, sprint3.FrozenSWEsmithCase]:
    instance_ids = sprint3._manifest_instance_ids(  # pyright: ignore[reportPrivateUsage]
        prepared.configuration.task_manifest_path,
        (EXPECTED_TASK_ID,),
    )
    cases_by_instance = sprint3.load_frozen_swesmith_cases(
        tuple(instance_ids.values()),
        allow_network=False,
    )
    try:
        return {
            EXPECTED_TASK_ID: cases_by_instance[instance_ids[EXPECTED_TASK_ID].lower()]
        }
    except KeyError as error:
        raise GST018RunnerConfigurationError(
            "local frozen SWE-smith objective data is missing GS-T018"
        ) from error


def _prepared_environment(
    prepared: PreparedRunner,
    *,
    policy: Any,
) -> sprint3.IsolatedTaskEnvironment:
    gates = prepared.freeze.get("environment_version_gates", {})
    image = gates.get("prepared_image") if isinstance(gates, dict) else None
    if not isinstance(image, str) or not image:
        raise GST018RunnerConfigurationError("freeze has no prepared GS-T018 image")
    docker = shutil.which("docker")
    if docker is None:
        raise GST018RunnerConfigurationError("Docker CLI is unavailable")
    inspected = subprocess.run(
        [docker, "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if inspected.returncode != 0:
        raise GST018RunnerConfigurationError(
            f"frozen prepared image is unavailable: {image}: {inspected.stderr.strip()}"
        )
    version = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            image,
            EXPECTED_CONTAINER_PYTHON,
            "-c",
            "import platform; print(platform.python_version())",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if version.returncode != 0 or version.stdout.strip() != EXPECTED_PYTHON:
        raise GST018RunnerConfigurationError(
            "prepared GS-T018 image does not expose the frozen Python 3.12.1 "
            f"runtime: {version.stdout.strip()!r} {version.stderr.strip()!r}"
        )
    case = prepared.cases[0]
    repository = (
        prepared.configuration.workspace_baseline_root_path / case.task.repository
    ).resolve()
    fingerprint = hashlib.sha256(
        f"{image}\n{EXPECTED_CONTAINER_PYTHON}\n{prepared.freeze_sha256}".encode()
    ).hexdigest()[:16]
    return sprint3.IsolatedTaskEnvironment(
        case.task.id,
        Path(docker).resolve(),
        None,
        EXPECTED_PYTHON,
        fingerprint,
        "docker",
        image,
        None,
        repository,
        EXPECTED_CONTAINER_PYTHON,
        str(gates.get("upstream_digest", "frozen-upstream")),
        policy.path,
        prepared.configuration.task_manifest_path,
    )


def _environment_context(
    task: Any,
    run_id: str,
    environment: sprint3.IsolatedTaskEnvironment,
) -> EnvironmentContext:
    return EnvironmentContext(
        id=f"{run_id}-environment",
        repository=task.repository,
        runtime=environment.runtime_type,
        versions={"python": environment.python_version}
        if environment.python_version is not None
        else {},
        markers={},
    )


def _settings_for_frozen_model(settings: Settings | None = None) -> Settings:
    base = settings or get_settings()
    return base.model_copy(
        update={
            "model_provider": EXPECTED_PROVIDER,
            "kilo_coding_model": EXPECTED_MODEL,
            "kilo_api_key": os.environ.get("KILO_API_KEY") or base.kilo_api_key,
        }
    )


def _require_execution_credentials(settings: Settings) -> None:
    required = {
        "KILO_API_KEY": settings.kilo_api_key,
        "NEO4J_URI": settings.neo4j_uri,
        "NEO4J_USERNAME": settings.neo4j_username,
        "NEO4J_PASSWORD": settings.neo4j_password,
        "NEO4J_DATABASE": settings.neo4j_database,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise GST018RunnerConfigurationError(
            "paid execution credentials are missing: " + ", ".join(missing)
        )


def build_condition_runner(
    configuration: LoadedExperimentConfiguration,
    *,
    condition: ExperimentCondition,
    objective_evaluator: ObjectiveTaskEvaluator,
    recurrence_evaluator: RecurrenceEvaluator,
    environment: sprint3.IsolatedTaskEnvironment,
    frozen_cases: dict[str, sprint3.FrozenSWEsmithCase],
    advisory_runtime: Neo4jAdvisoryRuntime | None = None,
    settings: Settings,
    artifact_store: ExperimentRunArtifactStore,
) -> ExperimentRunner:
    if condition is ExperimentCondition.B0:
        if advisory_runtime is not None:
            raise GST018RunnerConfigurationError(
                "B0 cannot be constructed with an advisory runtime"
            )
        advisory_service = None
    elif condition is ExperimentCondition.T:
        if advisory_runtime is None or not isinstance(
            advisory_runtime.treatment_repository, R13bTreatmentRepository
        ):
            raise GST018RunnerConfigurationError(
                "T must use the read-only R13bTreatmentRepository"
            )
        if set(R13B_TREATMENT_PATTERN_IDS) != {
            "recovery-pattern-ad07a6a45718b848a30ad377",
            "recovery-pattern-162e3999c4a2c66a1ff647ed",
            "recovery-pattern-f490f62ab931191c6eac6db1",
            "recovery-pattern-99a54266f940e1d4648f4698",
            "recovery-pattern-fd7b65022b22dc5f2a42816f",
        }:
            raise GST018RunnerConfigurationError(
                "T treatment corpus differs from the frozen canonical corpus"
            )
        advisory_service = advisory_runtime.advisory_service
    else:
        raise GST018RunnerConfigurationError(f"unsupported condition: {condition.value}")

    return ExperimentRunner(
        configuration,
        settings=settings,
        objective_evaluator=objective_evaluator,
        recurrence_evaluator=recurrence_evaluator,
        workspace_resolver=sprint3._make_workspace_resolver(  # pyright: ignore[reportPrivateUsage]
            source_root=configuration.workspace_baseline_root_path,
            execution_root=configuration.workspace_execution_root_path,
            frozen_cases=frozen_cases,
            condition=condition,
        ),
        execution_runtime_resolver=lambda _task, _workspace: environment.agent_execution_runtime(),
        environment_resolver=(
            None
            if condition is ExperimentCondition.B0
            else lambda task, run_id: _environment_context(task, run_id, environment)
        ),
        advisory_service=advisory_service,
        fail_closed_advisory=condition is ExperimentCondition.T,
        condition=condition,
        artifact_store=artifact_store,
    )


def _prepared_context(
    prepared: PreparedRunner,
    *,
    plan: tuple[execution_core.ExecutionSlot, ...],
) -> execution_core.FrozenExecutionContext:
    return execution_core.FrozenExecutionContext(
        project_root=prepared.configuration.project_root,
        config=prepared.configuration,
        protocol=prepared.protocol,
        freeze=prepared.freeze,
        freeze_hash=prepared.freeze_sha256,
        execution_commit_sha=str(prepared.freeze.get("code_commit_sha", "unknown")),
        plan=plan,
        cases=prepared.cases,
        artifact_root=prepared.artifact_root,
    )


def _load_prepared_inputs(
    prepared: PreparedRunner,
) -> tuple[dict[str, sprint3.FrozenSWEsmithCase], sprint3.IsolatedTaskEnvironment]:
    frozen_cases = _frozen_cases(prepared)
    policy = sprint3.load_benchmark_environment_policy(
        prepared.configuration.project_root / "configs/research/benchmark_environments.toml"
    )
    environment = _prepared_environment(prepared, policy=policy)
    sprint3._verify_baselines(  # pyright: ignore[reportPrivateUsage]
        prepared.cases,
        prepared.configuration.workspace_baseline_root_path,
    )
    sprint3._verify_frozen_patches(  # pyright: ignore[reportPrivateUsage]
        prepared.cases,
        prepared.configuration.workspace_baseline_root_path,
        frozen_cases,
    )
    return frozen_cases, environment


def validate_preflight(prepared: PreparedRunner) -> dict[str, Any]:
    """Validate frozen local inputs without provider, agent, or Neo4j calls."""
    frozen_cases, environment = _load_prepared_inputs(prepared)
    plan = tuple(
        execution_core.ExecutionSlot(slot.index, slot.task_id, slot.condition)
        for slot in prepared.slots
    )
    observations = execution_core.inspect_primary_slots(prepared.artifact_root, plan)
    recurrence = sprint3.make_recurrence_matcher_v3_exact_pytest_outcome(frozen_cases)
    assert recurrence is not None
    return {
        "status": "READY",
        "planned_slots": [f"{slot.task_id}/{slot.condition.value}" for slot in prepared.slots],
        "recurrence_evaluator": EXPECTED_RECURRENCE_VERSION,
        "model": EXPECTED_MODEL,
        "provider": EXPECTED_PROVIDER,
        "artifact_root": str(prepared.artifact_root),
        "prepared_image": environment.container_image,
        "python": environment.python_version,
        "existing_slots": [
            {
                "slot": f"{item.slot.task_id}/{item.slot.condition.value}",
                "status": item.status.value,
            }
            for item in observations
        ],
    }


def execute_frozen_two_slot_plan(
    *,
    project_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
) -> execution_core.ExecutionReport:
    """Execute only GS-T018/B0 followed by GS-T018/T."""
    prepared = prepare_runner(project_root, config_path, freeze_path)
    settings = _settings_for_frozen_model()
    _require_execution_credentials(settings)
    frozen_cases, environment = _load_prepared_inputs(prepared)
    objective = sprint3.FrozenSWEsmithObjective(
        frozen_cases,
        {EXPECTED_TASK_ID: environment},
        objective_coverage_policy="no_cov",
        coverage_policy_selection_version=sprint3.OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION,
    )
    recurrence = sprint3.make_recurrence_matcher_v3_exact_pytest_outcome(frozen_cases)
    artifact_store = ExperimentRunArtifactStore(prepared.artifact_root)
    plan = tuple(
        execution_core.ExecutionSlot(slot.index, slot.task_id, slot.condition)
        for slot in prepared.slots
    )
    context = _prepared_context(prepared, plan=plan)
    b0_runner = build_condition_runner(
        prepared.configuration,
        condition=ExperimentCondition.B0,
        objective_evaluator=objective,
        recurrence_evaluator=recurrence,
        environment=environment,
        frozen_cases=frozen_cases,
        settings=settings,
        artifact_store=artifact_store,
    )
    treatment_runtime: Neo4jAdvisoryRuntime | None = None
    treatment_runner: ExperimentRunner | None = None

    def run_slot(
        slot: execution_core.ExecutionSlot,
        case: BenchmarkTaskCase,
    ) -> ExperimentExecution:
        nonlocal treatment_runtime, treatment_runner
        if slot.condition is ExperimentCondition.B0:
            return b0_runner.run_case(case)
        if slot.condition is not ExperimentCondition.T:
            raise GST018RunnerConfigurationError(f"unexpected execution slot: {slot}")
        if treatment_runner is None:
            treatment_runtime = create_neo4j_advisory_runtime(
                settings,
                fail_closed_advisory=True,
            )
            treatment_runner = build_condition_runner(
                prepared.configuration,
                condition=ExperimentCondition.T,
                objective_evaluator=objective,
                recurrence_evaluator=recurrence,
                environment=environment,
                frozen_cases=frozen_cases,
                advisory_runtime=treatment_runtime,
                settings=settings,
                artifact_store=artifact_store,
            )
        return treatment_runner.run_case(case)

    try:
        return execution_core.execute_primary_plan(
            context,
            run_slot,
            artifact_store=artifact_store,
            environment_identities={
                EXPECTED_TASK_ID: {
                    "environment_fingerprint": environment.environment_fingerprint,
                    "runtime_type": environment.runtime_type,
                    "container_image": environment.container_image,
                    "python": environment.python_version,
                }
            },
            infrastructure_checker=lambda execution: (
                execution_core._is_infrastructure_failure(execution, objective)  # pyright: ignore[reportPrivateUsage]
                or (
                    execution.artifact.condition is ExperimentCondition.T
                    and bool(execution.dependencies.advisory_errors)
                )
            ),
        )
    finally:
        if treatment_runtime is not None:
            treatment_runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the frozen local plan without provider, agent, or Neo4j calls",
    )
    args = parser.parse_args()
    prepared = prepare_runner(args.project_root, args.config, args.freeze)
    if args.preflight_only:
        print(json.dumps(validate_preflight(prepared), sort_keys=True))
        return
    report = execute_frozen_two_slot_plan(
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
    "EXPECTED_MODEL",
    "EXPECTED_PROVIDER",
    "EXPECTED_RECURRENCE_VERSION",
    "ExecutionSlot",
    "GST018RunnerConfigurationError",
    "PreparedRunner",
    "build_condition_runner",
    "execute_frozen_two_slot_plan",
    "prepare_runner",
    "validate_preflight",
]
