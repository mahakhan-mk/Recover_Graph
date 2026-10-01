"""Execute only the two-slot GS-E003 hardened-development-v1 plan."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from experiments import run_sprint3b_kilo_v3 as v3
from experiments import run_sprint3b_kilo_v4_qwen3_coder_canary as v4
from experiments import run_sprint3b_reduced_b0_t as v1
from experiments import sprint3
from graph_swarm.integration.advisory_runtime import create_neo4j_advisory_runtime
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
from graph_swarm.settings import get_settings

# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryCast=false

EXPERIMENT_ID = "GS-E003"
PROTOCOL_REVISION = "sprint3b-kilo-hardened-development-v1"
CONFIG_VERSION = "sprint3b-kilo-hardened-development-v1"
EXPECTED_PROVIDER = "kilo"
EXPECTED_CODING_MODEL = "qwen/qwen3-coder"
EXPECTED_TASK_IDS = ("GS-T017",)
EXPECTED_CONDITIONS = (ExperimentCondition.B0, ExperimentCondition.T)
EXPECTED_PLAN = (("GS-T017", "B0"), ("GS-T017", "T"))
EXPECTED_PRIMARY_RUNS = 2
EXPECTED_MAX_ACTIONS = 28
EXPECTED_MAX_REQUESTS = 24
EXPECTED_TOOL_RETRIES = 3
EXPECTED_RECURRENCE_VERSION = "frozen_recurrence_matcher_v2_structured_pytest"
EXPECTED_TOOLS = ("read_file", "write_file", "edit_file", "run_tests", "run_command")
EXPECTED_PATTERN_IDS = frozenset(v3.EXPECTED_PATTERN_IDS)
DEFAULT_CONFIG = Path("configs/experiments/gs_e003_hardened_development_v1.yaml")
DEFAULT_FREEZE = Path(
    "research/evidence/results/GS-E003/sprint3b_hardened_development_v1/freeze.json"
)
ENVIRONMENT_POLICY = Path(
    "configs/research/gs_e003_hardened_development_v1_environments.toml"
)
MUTATION_PATCH = Path(
    "research/evidence/results/GS-E003/positive_control_missing_iteration/mutation.patch"
)


class HardenedDevelopmentConfigurationError(ExperimentConfigurationError):
    """Raised when the two-slot hardened-development contract is not exact."""


def _project_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise HardenedDevelopmentConfigurationError(
            f"{label} differs from hardened-development-v1: {actual!r}"
        )


def _protocol(config_path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise HardenedDevelopmentConfigurationError(
            f"could not read hardened-development-v1 config {config_path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise HardenedDevelopmentConfigurationError(
            "hardened-development-v1 config must be a mapping"
        )
    return cast(dict[str, Any], raw)


def _validate_protocol(
    protocol: dict[str, Any], configuration: LoadedExperimentConfiguration
) -> None:
    _require_equal(protocol.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _require_equal(protocol.get("protocol_revision"), PROTOCOL_REVISION, "protocol_revision")
    _require_equal(protocol.get("config_version"), CONFIG_VERSION, "config_version")
    _require_equal(protocol.get("conditions"), ["B0", "T"], "conditions")
    _require_equal(tuple(protocol.get("task_ids", ())), EXPECTED_TASK_IDS, "task_ids")
    _require_equal(tuple(protocol.get("task_order", ())), EXPECTED_TASK_IDS, "task_order")
    _require_equal(
        tuple(tuple(item) for item in protocol.get("execution_plan", ())), EXPECTED_PLAN,
        "execution_plan",
    )
    _require_equal(
        protocol.get("planned_primary_runs"), EXPECTED_PRIMARY_RUNS, "planned_primary_runs"
    )
    _require_equal(protocol.get("provider"), EXPECTED_PROVIDER, "provider")
    _require_equal(protocol.get("model"), EXPECTED_CODING_MODEL, "model")
    limits = protocol.get("limits", {})
    _require_equal(limits.get("max_actions"), EXPECTED_MAX_ACTIONS, "max_actions")
    _require_equal(limits.get("max_requests"), EXPECTED_MAX_REQUESTS, "max_requests")
    _require_equal(limits.get("tool_retries"), EXPECTED_TOOL_RETRIES, "tool_retries")
    _require_equal(
        protocol.get("recurrence_evaluator_version"), EXPECTED_RECURRENCE_VERSION,
        "recurrence_evaluator_version",
    )
    _require_equal(protocol.get("system_prompt"), "R13B_SYSTEM_PROMPT", "system_prompt")
    _require_equal(
        protocol.get("system_prompt_sha256"),
        "f3bf4187cc9bfabcf2551aa3988e592a2d0f2d0e61ad2984d7d90ddf13d54502",
        "system_prompt_sha256",
    )
    _require_equal(configuration.model.provider, EXPECTED_PROVIDER, "loaded provider")
    _require_equal(configuration.model.model, EXPECTED_CODING_MODEL, "loaded model")
    _require_equal(configuration.config.system_prompt, "R13B_SYSTEM_PROMPT", "loaded prompt")
    _require_equal(
        configuration.config.limits.max_actions, EXPECTED_MAX_ACTIONS, "loaded max_actions"
    )
    _require_equal(
        configuration.config.limits.max_requests, EXPECTED_MAX_REQUESTS, "loaded max_requests"
    )
    _require_equal(protocol.get("tools"), list(EXPECTED_TOOLS), "tools")
    _require_equal(protocol.get("task_workspace_network"), "disabled", "task workspace network")
    _require_equal(protocol.get("memory_writes"), "forbidden", "memory writes")
    _require_equal(protocol.get("neo4j_write_policy"), "forbidden", "Neo4j write policy")
    _require_equal(
        protocol.get("treatment_repository"), "R13bTreatmentRepository", "treatment repository"
    )
    _require_equal(
        frozenset(protocol.get("treatment_pattern_ids", ())),
        EXPECTED_PATTERN_IDS,
        "treatment pattern IDs",
    )
    _require_equal(
        protocol.get("artifact_root"),
        "research/evidence/results/GS-E003/sprint3b_hardened_development_v1/runs",
        "artifact root",
    )


def _load_package(
    *, project_root: Path, config_path: Path
) -> tuple[dict[str, Any], LoadedExperimentConfiguration, tuple[BenchmarkTaskCase, ...]]:
    root = project_root.expanduser().resolve()
    resolved_config = _project_path(config_path, root)
    protocol = _protocol(resolved_config)
    configuration = load_experiment_configuration(resolved_config, project_root=root)
    _validate_protocol(protocol, configuration)
    cases = tuple(
        load_task_cases(
            configuration.task_manifest_path,
            problem_statements_path=configuration.task_problems_path,
        )
    )
    if tuple(case.task.id for case in cases) != EXPECTED_TASK_IDS:
        raise HardenedDevelopmentConfigurationError("only GS-T017 may be in the execution manifest")
    if tuple(case.task.chronological_index for case in cases) != (17,):
        raise HardenedDevelopmentConfigurationError("GS-T017 chronological index must be 17")
    if tuple(case.occurrence_index for case in cases) != (4,):
        raise HardenedDevelopmentConfigurationError("GS-T017 occurrence index must be 4")
    return protocol, configuration, cases


def _load_freeze(root: Path, config_path: Path, freeze_path: Path) -> dict[str, Any]:
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HardenedDevelopmentConfigurationError(f"could not read freeze: {error}") from error
    if not isinstance(freeze, dict):
        raise HardenedDevelopmentConfigurationError("freeze must be a JSON object")
    _require_equal(freeze.get("status"), "READY", "freeze status")
    _require_equal(freeze.get("experiment_id"), EXPERIMENT_ID, "freeze experiment_id")
    _require_equal(freeze.get("protocol_revision"), PROTOCOL_REVISION, "freeze protocol_revision")
    _require_equal(freeze.get("config_sha256"), _sha256(config_path), "freeze config hash")
    _require_equal(freeze.get("code_commit_sha"), v1._git_head(root), "freeze code commit")
    candidate = freeze.get("candidate")
    if not isinstance(candidate, dict):
        raise HardenedDevelopmentConfigurationError("freeze candidate metadata is missing")
    _require_equal(candidate.get("task_id"), "GS-T017", "freeze candidate task")
    _require_equal(candidate.get("b0_t_runs"), 0, "freeze candidate run count")
    return freeze


def load_hardened_context(
    *,
    project_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
    allow_existing_artifacts: bool = True,
) -> v1.FrozenExecutionContext:
    """Load the frozen two-slot surface without provider or Neo4j calls."""
    root = project_root.expanduser().resolve()
    resolved_config = _project_path(config_path, root)
    resolved_freeze = _project_path(freeze_path, root)
    protocol, configuration, cases = _load_package(
        project_root=root, config_path=resolved_config
    )
    freeze = _load_freeze(root, resolved_config, resolved_freeze)
    artifact_root = _project_path(str(protocol["artifact_root"]), root)
    if not allow_existing_artifacts and artifact_root.exists():
        raise HardenedDevelopmentConfigurationError(
            f"hardened-development-v1 artifact root already exists: {artifact_root}"
        )
    slots = tuple(
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
        plan=slots,
        cases=cases,
        artifact_root=artifact_root,
    )


def build_execution_plan(protocol: dict[str, Any]) -> tuple[v1.ExecutionSlot, ...]:
    """Build exactly GS-T017/B0 followed by GS-T017/T."""
    _require_equal(
        protocol.get("execution_plan"), [list(item) for item in EXPECTED_PLAN], "execution_plan"
    )
    return tuple(
        v1.ExecutionSlot(index=index, task_id=task_id, condition=ExperimentCondition(condition))
        for index, (task_id, condition) in enumerate(EXPECTED_PLAN, start=1)
    )


def _manifest_record(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise HardenedDevelopmentConfigurationError(
            "GS-T017 manifest must contain exactly one object"
        )
    return cast(dict[str, Any], records[0])


def _frozen_case(context: v1.FrozenExecutionContext) -> sprint3.FrozenSWEsmithCase:
    record = _manifest_record(context.config.task_manifest_path)
    try:
        fail_to_pass = tuple(record["fail_to_pass"])
        patch = _project_path(MUTATION_PATCH, context.project_root).read_text(encoding="utf-8")
    except (KeyError, OSError, TypeError) as error:
        raise HardenedDevelopmentConfigurationError(
            f"GS-T017 frozen objective data is incomplete: {error}"
        ) from error
    if not fail_to_pass or not all(isinstance(test, str) and test.strip() for test in fail_to_pass):
        raise HardenedDevelopmentConfigurationError("GS-T017 FAIL_TO_PASS boundary is invalid")
    return sprint3.FrozenSWEsmithCase(
        str(record["instance_id"]),
        tuple(str(test) for test in fail_to_pass),
        patch,
    )


def _environment(context: v1.FrozenExecutionContext) -> sprint3.IsolatedTaskEnvironment:
    policy_path = _project_path(ENVIRONMENT_POLICY, context.project_root)
    policy = sprint3.load_benchmark_environment_policy(
        policy_path, expected_task_order=EXPECTED_TASK_IDS
    )
    task_policy = policy.task("GS-T017")
    if (
        task_policy.runtime != "manifest_container_required"
        or task_policy.python_constraint_assertion != "==3.12.1"
    ):
        raise HardenedDevelopmentConfigurationError(
            "GS-T017 environment gate is not Docker/Python 3.12.1"
        )
    record = _manifest_record(context.config.task_manifest_path)
    image = record.get("image_name")
    if not isinstance(image, str) or not image.strip():
        raise HardenedDevelopmentConfigurationError("GS-T017 manifest image is missing")
    return sprint3.IsolatedTaskEnvironment(
        task_id="GS-T017",
        python_executable=Path("docker"),
        python_version="3.12.1",
        environment_fingerprint=hashlib.sha256(image.encode("utf-8")).hexdigest(),
        runtime_type="docker",
        container_image=image,
        dependency_source_root=(
            context.config.workspace_baseline_root_path / "Textualize__rich.9d8f9a37"
        ),
        container_python_executable="/usr/local/bin/python",
        base_container_image=image,
        benchmark_policy_path=policy_path,
        benchmark_manifest_path=context.config.task_manifest_path,
    )


def _validate_runtime_environment(
    context: v1.FrozenExecutionContext,
    environment: sprint3.IsolatedTaskEnvironment,
) -> None:
    baseline = context.config.workspace_baseline_root_path / "Textualize__rich.9d8f9a37"
    if not baseline.is_dir():
        raise HardenedDevelopmentConfigurationError(f"GS-T017 baseline is missing: {baseline}")
    inspected = subprocess.run(
        ["docker", "image", "inspect", cast(str, environment.container_image)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if inspected.returncode != 0:
        raise HardenedDevelopmentConfigurationError(
            f"GS-T017 Docker image is unavailable: {inspected.stderr.strip()}"
        )


def execute_hardened_runs(
    *, project_root: Path, config_path: Path = DEFAULT_CONFIG, freeze_path: Path = DEFAULT_FREEZE
) -> v1.ExecutionReport:
    """Execute only GS-T017/B0 and GS-T017/T after provider-backed preflight."""
    context = load_hardened_context(
        project_root=project_root,
        config_path=config_path,
        freeze_path=freeze_path,
        allow_existing_artifacts=True,
    )
    environment = _environment(context)
    _validate_runtime_environment(context, environment)
    frozen_case = _frozen_case(context)
    frozen_cases = {"GS-T017": frozen_case}
    environments = {"GS-T017": environment}
    objective = sprint3.FrozenSWEsmithObjective(
        frozen_cases,
        environments,
        objective_coverage_policy="no_cov",
        coverage_policy_selection_version=sprint3.OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION,
    )
    recurrence = sprint3.make_recurrence_matcher(frozen_cases)
    settings = v4._settings_for_frozen_model(get_settings(), context.config)
    artifact_store = ExperimentRunArtifactStore(context.artifact_root)
    runtime = create_neo4j_advisory_runtime(settings, fail_closed_advisory=True)
    try:
        runners: dict[ExperimentCondition, ExperimentRunner] = {}
        for condition in EXPECTED_CONDITIONS:
            def execution_runtime_resolver(_task: Any, _workspace: Path) -> Any:
                return environment.agent_execution_runtime()

            runners[condition] = v4.build_condition_runner(
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
                execution_runtime_resolver=execution_runtime_resolver,
                environment_resolver=(v4.make_environment_resolver(environments) if condition is ExperimentCondition.T else None),
                advisory_runtime=runtime if condition is ExperimentCondition.T else None,
                settings=settings,
                artifact_store=artifact_store,
            )

        def run_slot(slot: v1.ExecutionSlot, case: BenchmarkTaskCase) -> ExperimentExecution:
            return runners[slot.condition].run_case(case)

        return v1.execute_primary_plan(
            context,
            run_slot,
            artifact_store=artifact_store,
            environment_identities={
                "GS-T017": {
                    "environment_fingerprint": environment.environment_fingerprint,
                    "runtime_type": environment.runtime_type,
                }
            },
            infrastructure_checker=lambda execution: v1._is_infrastructure_failure(
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
    context = load_hardened_context(
        project_root=args.project_root,
        config_path=args.config,
        freeze_path=args.freeze,
        allow_existing_artifacts=True,
    )
    if args.preflight_only:
        _validate_runtime_environment(context, _environment(context))
        print(json.dumps({"status": "READY", "planned_slots": len(context.plan)}, sort_keys=True))
        return
    report = execute_hardened_runs(
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
    "EXPECTED_MAX_ACTIONS",
    "EXPECTED_MAX_REQUESTS",
    "EXPECTED_PLAN",
    "EXPECTED_PRIMARY_RUNS",
    "EXPECTED_TASK_IDS",
    "HardenedDevelopmentConfigurationError",
    "build_execution_plan",
    "execute_hardened_runs",
    "load_hardened_context",
    "make_cli_parser",
]
