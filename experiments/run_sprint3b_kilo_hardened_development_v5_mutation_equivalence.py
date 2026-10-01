"""Prepare or execute the frozen GS-T018/T mutation-equivalence revision."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from experiments import run_gs_e003_hardened_development_v3 as v3
from experiments import sprint3
from graph_swarm.integration.advisory_runtime import create_neo4j_advisory_runtime
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    ExperimentExecution,
    ExperimentRunArtifactStore,
    load_experiment_configuration,
    load_task_cases,
)

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportPrivateUsage=false

EXPERIMENT_ID = "GS-E003"
CONFIG_VERSION = "sprint3b-kilo-hardened-development-v5-mutation-equivalence"
EXPECTED_TASK_ID = "GS-T018"
EXPECTED_PLAN = ((EXPECTED_TASK_ID, "T"),)
EXPECTED_RECURRENCE_VERSION = "frozen_recurrence_matcher_v3_exact_pytest_outcome"
EXPECTED_ADVISORY_REVISION = "generic_pre_mutation_recovery_action_equivalence_v1"
DEFAULT_CONFIG = Path(
    "configs/experiments/sprint3b_kilo_hardened_development_v5_mutation_equivalence.yaml"
)
DEFAULT_FREEZE = Path(
    "research/evidence/results/GS-E003/"
    "sprint3b_hardened_development_v5_mutation_equivalence/freeze.json"
)


class GST018V5RunnerConfigurationError(ValueError):
    """Raised when the frozen T-only contract is not exact."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GST018V5RunnerConfigurationError("configuration must be a mapping")
    return value


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def create_freeze_artifact(
    project_root: Path = Path("."),
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
) -> dict[str, object]:
    """Create the provider-free v5 freeze; this does not run a task slot."""
    root = project_root.resolve()
    config = (root / config_path).resolve() if not config_path.is_absolute() else config_path
    freeze_file = (root / freeze_path).resolve() if not freeze_path.is_absolute() else freeze_path
    protocol = _load_yaml(config)
    freeze: dict[str, object] = {
        "status": "READY",
        "freeze_type": "development_hardened_pre_mutation_recovery_action_equivalence_no_results",
        "experiment_id": EXPERIMENT_ID,
        "protocol_revision": protocol["protocol_revision"],
        "config_version": protocol["config_version"],
        "config_path": _display_path(config, root),
        "config_sha256": _sha256(config),
        "code_commit_sha": _git_head(root),
        "advisory_revision": {
            "name": EXPECTED_ADVISORY_REVISION,
            "implementation": "src/graph_swarm/retrieval/applicability.py",
            "service": "src/graph_swarm/advisory/service.py",
            "retrieval": "src/graph_swarm/retrieval/service.py",
            "agent_boundary": "src/graph_swarm/agent/advisory.py",
            "real_planned_action_preserved": True,
            "completed_prefix_evidence_only": True,
            "current_edit_contents_excluded": True,
            "test_boundary_equivalence_disabled": True,
        },
        "model": {
            "future_paid_runs": True,
            "provider": v3.EXPECTED_PROVIDER,
            "name": v3.EXPECTED_MODEL,
            "system_prompt_id": "R13B_SYSTEM_PROMPT",
            "system_prompt_sha256": protocol["system_prompt_sha256"],
        },
        "tools": protocol["tools"],
        "limits": protocol["limits"],
        "recurrence_evaluator": {
            "name": protocol["recurrence_evaluator"],
            "version": EXPECTED_RECURRENCE_VERSION,
        },
        "objective_evaluator": {
            "name": protocol["objective_evaluator"],
            "version": protocol["objective_evaluator_version"],
        },
        "recovery_memory": {
            "policy": protocol["treatment_memory_policy"],
            "repository": protocol["treatment_repository"],
            "memory_writes": protocol["memory_writes"],
            "online_learning": protocol["online_learning"],
            "neo4j_write_policy": protocol["neo4j_write_policy"],
        },
        "candidate": {
            "task_id": EXPECTED_TASK_ID,
            "b0_t_runs_in_this_revision": 0,
            "model_provider_calls_in_this_revision": 0,
            "agent_runs_in_this_revision": 0,
            "neo4j_writes_in_this_revision": 0,
        },
        "artifact_destination": {
            "freeze_artifact": _display_path(freeze_file, root),
            "run_artifact_root": protocol["artifact_root"],
        },
    }
    freeze_file.parent.mkdir(parents=True, exist_ok=True)
    freeze_file.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return freeze


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare_runner(
    project_root: Path = Path("."),
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
) -> v3.PreparedRunner:
    root = project_root.resolve()
    config = (root / config_path).resolve() if not config_path.is_absolute() else config_path
    freeze_file = (root / freeze_path).resolve() if not freeze_path.is_absolute() else freeze_path
    protocol = _load_yaml(config)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise GST018V5RunnerConfigurationError("experiment ID is not GS-E003")
    if protocol.get("config_version") != CONFIG_VERSION:
        raise GST018V5RunnerConfigurationError("configuration is not v5 mutation equivalence")
    if (
        protocol.get("provider") != v3.EXPECTED_PROVIDER
        or protocol.get("model") != v3.EXPECTED_MODEL
    ):
        raise GST018V5RunnerConfigurationError("frozen model/provider differ")
    if protocol.get("recurrence_evaluator_version") != EXPECTED_RECURRENCE_VERSION:
        raise GST018V5RunnerConfigurationError("recurrence evaluator is not v3 exact pytest")
    if protocol.get("conditions") != ["T"]:
        raise GST018V5RunnerConfigurationError("only the treatment condition is allowed")
    if tuple(tuple(item) for item in protocol.get("execution_plan", ())) != EXPECTED_PLAN:
        raise GST018V5RunnerConfigurationError("execution plan is not exactly GS-T018/T")
    if protocol.get("planned_primary_runs") != 1:
        raise GST018V5RunnerConfigurationError("planned_primary_runs must be exactly one")
    limits = protocol.get("limits")
    if limits != {
        "max_actions": 28,
        "max_requests": 24,
        "timeout_seconds": 300,
        "tool_retries": 3,
        "task_timeout_seconds": 300,
        "model_request_timeout_seconds": 300,
        "tool_timeout_seconds": {"run_command": 30, "run_tests": 120},
    }:
        raise GST018V5RunnerConfigurationError("limits differ from the frozen contract")
    if (
        protocol.get("memory_writes") != "forbidden"
        or protocol.get("neo4j_write_policy") != "forbidden"
    ):
        raise GST018V5RunnerConfigurationError("memory writes must remain forbidden")
    advisory = protocol.get("pre_mutation_advisory")
    if not isinstance(advisory, dict) or advisory.get("revision") != EXPECTED_ADVISORY_REVISION:
        raise GST018V5RunnerConfigurationError("mutation-equivalence advisory contract differs")
    if (
        advisory.get("real_planned_action_preserved") is not True
        or advisory.get("completed_prefix_evidence_only") is not True
        or advisory.get("current_edit_contents_excluded") is not True
        or advisory.get("test_boundary_equivalence_disabled") is not True
    ):
        raise GST018V5RunnerConfigurationError("planned-action or boundary contract differs")
    loaded = load_experiment_configuration(config, project_root=root)
    if loaded.model.provider != v3.EXPECTED_PROVIDER or loaded.model.model != v3.EXPECTED_MODEL:
        raise GST018V5RunnerConfigurationError("loaded model configuration differs")
    cases = tuple(
        load_task_cases(
            loaded.task_manifest_path,
            problem_statements_path=loaded.task_problems_path,
        )
    )
    if tuple(case.task.id for case in cases) != (EXPECTED_TASK_ID,):
        raise GST018V5RunnerConfigurationError("only GS-T018 may be registered")
    raw_freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if raw_freeze.get("status") != "READY" or raw_freeze.get("config_sha256") != _sha256(config):
        raise GST018V5RunnerConfigurationError("freeze does not match the v5 configuration")
    if raw_freeze.get("recurrence_evaluator", {}).get("version") != EXPECTED_RECURRENCE_VERSION:
        raise GST018V5RunnerConfigurationError("freeze recurrence evaluator is not v3")
    if raw_freeze.get("model", {}).get("name") != v3.EXPECTED_MODEL:
        raise GST018V5RunnerConfigurationError("freeze model is not qwen/qwen3-coder")
    if raw_freeze.get("advisory_revision", {}).get("name") != EXPECTED_ADVISORY_REVISION:
        raise GST018V5RunnerConfigurationError("freeze advisory revision differs")
    return v3.PreparedRunner(
        config_path=config,
        freeze_path=freeze_file,
        config_sha256=_sha256(config),
        freeze_sha256=_sha256(freeze_file),
        slots=(v3.ExecutionSlot(1, EXPECTED_TASK_ID, ExperimentCondition.T),),
        protocol=protocol,
        freeze=raw_freeze,
        configuration=loaded,
        cases=cases,
        artifact_root=loaded.artifact_root_path,
    )


def validate_preflight(prepared: v3.PreparedRunner) -> dict[str, object]:
    """Validate frozen local inputs without provider, agent, or Neo4j calls."""
    frozen_cases, environment = v3._load_prepared_inputs(prepared)  # pyright: ignore[reportPrivateUsage]
    recurrence = sprint3.make_recurrence_matcher_v3_exact_pytest_outcome(frozen_cases)
    assert recurrence is not None
    return {
        "status": "READY",
        "planned_slots": ["GS-T018/T"],
        "recurrence_evaluator": EXPECTED_RECURRENCE_VERSION,
        "model": v3.EXPECTED_MODEL,
        "provider": v3.EXPECTED_PROVIDER,
        "artifact_root": str(prepared.artifact_root),
        "prepared_image": environment.container_image,
        "python": environment.python_version,
        "advisory": EXPECTED_ADVISORY_REVISION,
    }


def execute_t_only(
    *,
    project_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    freeze_path: Path = DEFAULT_FREEZE,
) -> ExperimentExecution:
    """Execute exactly GS-T018/T; this function is never called by preflight."""
    prepared = prepare_runner(project_root, config_path, freeze_path)
    settings = v3._settings_for_frozen_model()  # pyright: ignore[reportPrivateUsage]
    v3._require_execution_credentials(settings)  # pyright: ignore[reportPrivateUsage]
    frozen_cases, environment = v3._load_prepared_inputs(prepared)  # pyright: ignore[reportPrivateUsage]
    objective = sprint3.FrozenSWEsmithObjective(
        frozen_cases,
        {EXPECTED_TASK_ID: environment},
        objective_coverage_policy="no_cov",
        coverage_policy_selection_version=sprint3.OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION,
    )
    recurrence = sprint3.make_recurrence_matcher_v3_exact_pytest_outcome(frozen_cases)
    artifact_store = ExperimentRunArtifactStore(prepared.artifact_root)
    runtime = create_neo4j_advisory_runtime(settings, fail_closed_advisory=True)
    try:
        runner = v3.build_condition_runner(
            prepared.configuration,
            condition=ExperimentCondition.T,
            objective_evaluator=objective,
            recurrence_evaluator=recurrence,
            environment=environment,
            frozen_cases=frozen_cases,
            advisory_runtime=runtime,
            settings=settings,
            artifact_store=artifact_store,
        )
        return runner.run_case(prepared.cases[0])
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--create-freeze", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.create_freeze:
        if not args.preflight_only:
            parser.error("--create-freeze requires --preflight-only")
        create_freeze_artifact(args.project_root, args.config, args.freeze)
    prepared = prepare_runner(args.project_root, args.config, args.freeze)
    if args.preflight_only:
        print(json.dumps(validate_preflight(prepared), sort_keys=True))
        return
    execution = execute_t_only(
        project_root=args.project_root,
        config_path=args.config,
        freeze_path=args.freeze,
    )
    print(json.dumps(dataclasses.asdict(execution), default=str, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_VERSION",
    "DEFAULT_CONFIG",
    "DEFAULT_FREEZE",
    "EXPECTED_ADVISORY_REVISION",
    "EXPECTED_PLAN",
    "EXPECTED_RECURRENCE_VERSION",
    "GST018V5RunnerConfigurationError",
    "create_freeze_artifact",
    "execute_t_only",
    "prepare_runner",
    "validate_preflight",
]
