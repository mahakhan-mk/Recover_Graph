"""Freeze and preflight the reduced Track B B0-versus-T protocol.

This module is deliberately a protocol/preflight boundary.  It does not add a
runner or execute an agent, provider, benchmark, or Neo4j operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.tasks import Task
from graph_swarm.research.runner import load_task_cases
from graph_swarm.retrieval.query import recovery_retrieval_query_text

SPRINT3A_EXPERIMENT_ID = "GS-E003"
SPRINT3A_PROTOCOL_REVISION = "sprint3a-reduced-b0-t-v1"
SPRINT3A_TASK_IDS: tuple[str, ...] = tuple(f"GS-T{i:03d}" for i in range(6, 16))
SPRINT3A_CONDITIONS: tuple[str, ...] = ("B0", "T")
SPRINT3A_PATTERN_IDS: tuple[str, ...] = (
    "recovery-pattern-ad07a6a45718b848a30ad377",
    "recovery-pattern-f490f62ab931191c6eac6db1",
    "recovery-pattern-162e3999c4a2c66a1ff647ed",
    "recovery-pattern-fd7b65022b22dc5f2a42816f",
    "recovery-pattern-99a54266f940e1d4648f4698",
)
INTEGRATION_POINT_1_SHA = "169cf089b84f7a2df2e513ff393dbc629d77523b"
R13B_FREEZE_SHA = "857bd073de420d0129435e3bb47e96db3741ad4f"
TREATMENT_MEMORY_POLICY = "fixed frozen acquisition corpus / read-only treatment memory"
DEFAULT_CONFIG = Path("configs/experiments/sprint3a_reduced_b0_t_v1.yaml")
DEFAULT_FREEZE = Path("research/evidence/results/GS-E003/sprint3a_reduced_b0_t/freeze.json")
EXPECTED_CODING_MODEL = "nex-agi/nex-n2.5-pro:free"
EXPECTED_ABSTRACTION_MODEL = "cohere/north-mini-code:free"
FREEZE_INPUT_PATHS: tuple[str, ...] = (
    "configs/experiments/sprint3a_reduced_b0_t_v1.yaml",
    "experiments/sprint3a.py",
    "benchmark/manifests/pilot.jsonl",
    "benchmark/annotations/recurrence_validation.csv",
    "configs/models/openrouter.yaml",
    "configs/models/openrouter_coding_r13b.yaml",
    "src/graph_swarm/agent/prompts.py",
    "configs/research/gate_b1_environments.json",
)


class Sprint3AProtocolError(ValueError):
    """Raised when the frozen reduced protocol is incomplete or unsafe."""


@dataclass(frozen=True)
class PreflightReport:
    """Offline validation result suitable for CLI and unit-test inspection."""

    status: str
    config_path: str
    artifact_root: str
    checks: Mapping[str, str]


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise Sprint3AProtocolError(f"could not read YAML {path}: {error}") from error
    if not isinstance(raw, dict):
        raise Sprint3AProtocolError(f"protocol YAML must contain a mapping: {path}")
    return cast(dict[str, Any], raw)


def load_protocol(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the dedicated protocol document without loading a provider."""
    return _read_yaml(config_path.expanduser().resolve())


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise Sprint3AProtocolError(f"{name} must be {expected!r}; got {actual!r}")


def build_execution_plan(protocol: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return the frozen paired task-major plan."""
    task_ids = tuple(protocol.get("task_order", ()))
    conditions = tuple(protocol.get("conditions", ()))
    _require_equal(task_ids, SPRINT3A_TASK_IDS, "task_order")
    _require_equal(conditions, SPRINT3A_CONDITIONS, "conditions")
    raw_plan = protocol.get("execution_plan")
    if not isinstance(raw_plan, list):
        raise Sprint3AProtocolError("execution_plan must be a list")
    plan = tuple(tuple(item) for item in raw_plan if isinstance(item, list) and len(item) == 2)
    expected = tuple((task_id, condition) for task_id in task_ids for condition in conditions)
    _require_equal(plan, expected, "execution_plan")
    _require_equal(protocol.get("planned_primary_runs"), 20, "planned_primary_runs")
    return plan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_input_hashes(project_root: Path) -> dict[str, str]:
    """Hash every source input that defines the pre-execution protocol."""
    hashes: dict[str, str] = {}
    for relative_path in FREEZE_INPUT_PATHS:
        path = project_root / relative_path
        if not path.is_file():
            raise Sprint3AProtocolError(f"freeze input is missing: {relative_path}")
        hashes[relative_path] = _sha256(path)
    return hashes


def freeze_inputs_hash(hashes: Mapping[str, str]) -> str:
    """Return a deterministic aggregate hash of sorted freeze input hashes."""
    return _canonical_hash(sorted(hashes.items()))


def _configured_role_model(project_root: Path, variable: str) -> str | None:
    """Resolve one role-specific model without loading a provider or calling it."""
    configured = os.environ.get(variable)
    if configured is not None:
        return configured.strip()
    env_path = project_root / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{variable}="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return None


def validate_coding_model(protocol: Mapping[str, Any], project_root: Path) -> str:
    """Enforce the frozen coding model before any future execution."""
    _require_equal(protocol.get("model"), EXPECTED_CODING_MODEL, "coding model")
    resolved = _configured_role_model(project_root, "OPENROUTER_CODING_MODEL")
    if resolved != EXPECTED_CODING_MODEL:
        raise Sprint3AProtocolError(
            "resolved OPENROUTER_CODING_MODEL must equal the frozen coding model "
            f"{EXPECTED_CODING_MODEL!r}; got {resolved!r}"
        )
    return resolved


def _leakage_check() -> None:
    task = Task(
        id="current-task",
        problem_statement="Repair the current failure without using hidden metadata.",
        family_id="FORBIDDEN-FAMILY-ID",
        repository="current/repository",
        chronological_index=8,
    )
    action = PlannedAction(
        id="current-action",
        run_id="current-run",
        task_id=task.id,
        tool="run_tests",
        operation="pytest",
        arguments={"visible": "current action"},
        planned_at=datetime.now(UTC),
    )
    environment = EnvironmentContext(
        id="current-environment",
        repository=task.repository,
        runtime="python",
    )
    query = recovery_retrieval_query_text(task, action, environment)
    forbidden = (
        task.family_id,
        "family_label",
        "mutation_family",
        "expected recovery",
        "gold patch",
        "benchmark label",
        "expected solution",
        "recurrence ground truth",
        "future task metadata",
        "evaluator annotation",
    )
    if any(value in query for value in forbidden):
        raise Sprint3AProtocolError("forbidden evaluation metadata reached treatment retrieval")


def _common_comparison_fields(protocol: Mapping[str, Any]) -> None:
    common = (
        "provider",
        "model",
        "model_settings",
        "prompt_version",
        "system_prompt",
        "tools",
        "tool_permissions",
        "limits",
        "pacing",
        "objective_evaluator",
        "objective_evaluator_version",
        "recurrence_evaluator",
        "recurrence_evaluator_version",
        "task_order",
    )
    overrides = protocol.get("condition_runtime")
    if not isinstance(overrides, dict) or set(overrides) != set(SPRINT3A_CONDITIONS):
        raise Sprint3AProtocolError("condition_runtime must define exactly B0 and T")
    b0 = overrides["B0"]
    treatment = overrides["T"]
    if not isinstance(b0, dict) or not isinstance(treatment, dict):
        raise Sprint3AProtocolError("condition_runtime entries must be mappings")
    for field in common:
        _require_equal(
            b0.get(field, protocol.get(field)),
            treatment.get(field, protocol.get(field)),
            field,
        )
    _require_equal(b0.get("advisory_service"), "none", "B0 advisory_service")
    _require_equal(treatment.get("advisory_service"), "AdvisoryService", "T advisory_service")


def _validate_protocol(
    protocol: Mapping[str, Any],
    project_root: Path,
    *,
    config_path: Path,
) -> Path:
    _require_equal(protocol.get("experiment_id"), SPRINT3A_EXPERIMENT_ID, "experiment_id")
    _require_equal(
        protocol.get("protocol_revision"), SPRINT3A_PROTOCOL_REVISION, "protocol_revision"
    )
    _require_equal(tuple(protocol.get("conditions", ())), SPRINT3A_CONDITIONS, "conditions")
    _require_equal(tuple(protocol.get("task_ids", ())), SPRINT3A_TASK_IDS, "task_ids")
    _require_equal(
        protocol.get("treatment_memory_policy"),
        TREATMENT_MEMORY_POLICY,
        "treatment_memory_policy",
    )
    _require_equal(
        protocol.get("treatment_repository"),
        "R13bTreatmentRepository",
        "treatment_repository",
    )
    _require_equal(
        tuple(protocol.get("treatment_pattern_ids", ())),
        SPRINT3A_PATTERN_IDS,
        "treatment_pattern_ids",
    )
    _require_equal(protocol.get("memory_writes"), "forbidden", "memory_writes")
    _require_equal(protocol.get("online_learning"), False, "online_learning")
    _require_equal(protocol.get("neo4j_write_policy"), "forbidden", "neo4j_write_policy")
    _require_equal(protocol.get("task_workspace_network"), "disabled", "task_workspace_network")
    external_network = protocol.get("external_runtime_network")
    _require_equal(
        external_network,
        {"openrouter": "required", "huggingface_inference": "required"},
        "external_runtime_network",
    )
    _require_equal(protocol.get("repetition_count"), 1, "repetition_count")
    _require_equal(
        protocol.get("abstraction_model"),
        {
            "provider": "openrouter",
            "name": EXPECTED_ABSTRACTION_MODEL,
            "resolution_variable": "OPENROUTER_ABSTRACTION_MODEL",
            "phase": "acquisition_only",
            "used_during_transfer_execution": False,
        },
        "abstraction_model",
    )
    model_config = _read_yaml(_project_path(str(protocol["model_config"]), project_root))
    _require_equal(model_config.get("provider"), "openrouter", "coding model provider")
    _require_equal(
        model_config.get("model"),
        "runtime-selected-from-OPENROUTER_CODING_MODEL",
        "coding model config model",
    )
    _require_equal(model_config.get("settings", {}).get("temperature"), 0, "coding temperature")
    validate_coding_model(protocol, project_root)
    freeze_input_hashes(project_root)
    build_execution_plan(protocol)
    _common_comparison_fields(protocol)

    manifest = _project_path(str(protocol["task_manifest"]), project_root)
    problems = _project_path(str(protocol["task_problems"]), project_root)
    cases = load_task_cases(manifest, problem_statements_path=problems)
    selected = tuple(case for case in cases if case.task.id in SPRINT3A_TASK_IDS)
    if tuple(case.task.id for case in selected) != SPRINT3A_TASK_IDS:
        raise Sprint3AProtocolError(
            "task manifest does not provide the frozen chronological task order"
        )
    if tuple(case.task.chronological_index for case in selected) != tuple(range(6, 16)):
        raise Sprint3AProtocolError(
            "frozen transfer tasks must have chronological indexes 6 through 15"
        )
    if len(selected) != 10:
        raise Sprint3AProtocolError("exactly ten transfer tasks are required")

    artifact_root = _project_path(str(protocol["artifact_root"]), project_root)
    if artifact_root.exists():
        raise Sprint3AProtocolError(
            f"artifact root already exists; refusing collision: {artifact_root}"
        )
    freeze_path = _project_path(str(protocol["freeze_artifact"]), project_root)
    if freeze_path.exists():
        try:
            frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise Sprint3AProtocolError(
                f"existing freeze artifact is unreadable: {freeze_path}"
            ) from error
        if not isinstance(frozen, dict) or frozen.get("status") != "READY":
            raise Sprint3AProtocolError(
                f"existing freeze artifact is not a READY immutable freeze: {freeze_path}"
            )
        if frozen.get("config_sha256") != _sha256(config_path):
            raise Sprint3AProtocolError(
                f"existing freeze artifact does not match this protocol: {freeze_path}"
            )
    _leakage_check()
    return artifact_root


def preflight(
    config_path: Path = DEFAULT_CONFIG,
    *,
    project_root: Path | None = None,
) -> PreflightReport:
    """Validate the protocol offline; no provider, Neo4j, or benchmark is called."""
    root = (project_root or Path.cwd()).resolve()
    resolved_config = _project_path(config_path, root)
    protocol = load_protocol(resolved_config)
    artifact_root = _validate_protocol(protocol, root, config_path=resolved_config)
    checks = {
        "config_parses": "PASS",
        "tasks_and_order": "PASS",
        "conditions": "PASS",
        "b0_memory_free": "PASS",
        "t_r13b_repository": "PASS",
        "five_pattern_allowlist": "PASS",
        "common_comparison_settings": "PASS",
        "new_artifact_paths": "PASS",
        "retrieval_leakage": "PASS",
        "neo4j_read_only": "PASS",
        "memory_writes_disabled": "PASS",
        "coding_model_exact_match": "PASS",
        "abstraction_execution": "NOT CALLED",
        "task_workspace_network_disabled": "PASS",
        "external_runtime_network_explicit": "PASS",
        "provider_calls": "NOT CALLED",
        "benchmark_execution": "NOT CALLED",
    }
    return PreflightReport("READY", str(resolved_config), str(artifact_root), checks)


def _existing_configuration_snapshot(project_root: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in ("gate_b1.yaml", "rollout_3a_pilot.yaml"):
        path = project_root / "configs" / "experiments" / name
        snapshot[name] = _read_yaml(path)
    model_path = project_root / "configs" / "models" / "openrouter.yaml"
    snapshot["openrouter.yaml"] = _read_yaml(model_path)
    return snapshot


def freeze_protocol(
    config_path: Path = DEFAULT_CONFIG,
    *,
    project_root: Path | None = None,
    output_path: Path = DEFAULT_FREEZE,
) -> Path:
    """Write one immutable protocol artifact after a successful offline preflight."""
    root = (project_root or Path.cwd()).resolve()
    resolved_config = _project_path(config_path, root)
    protocol = load_protocol(resolved_config)
    report = preflight(resolved_config, project_root=root)
    input_hashes = freeze_input_hashes(root)
    manifest = _project_path(str(protocol["task_manifest"]), root)
    model_config = _project_path(str(protocol["model_config"]), root)
    prompt_source = root / "src" / "graph_swarm" / "agent" / "prompts.py"
    environment_manifest = root / "configs" / "research" / "gate_b1_environments.json"

    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    artifact = {
        "status": report.status,
        "freeze_type": "protocol_only_no_experimental_results",
        "experiment_id": protocol["experiment_id"],
        "protocol_revision": protocol["protocol_revision"],
        "integration_point_1_sha": INTEGRATION_POINT_1_SHA,
        "r13b_freeze_sha": R13B_FREEZE_SHA,
        "base_repository_sha": INTEGRATION_POINT_1_SHA,
        "freeze_input_hashes": input_hashes,
        "freeze_inputs_sha256": freeze_inputs_hash(input_hashes),
        "config_path": display_path(resolved_config),
        "config_sha256": _sha256(resolved_config),
        "task_manifest_path": display_path(manifest),
        "task_manifest_sha256": _sha256(manifest),
        "task_order_hash": _canonical_hash(protocol["task_order"]),
        "task_ids": protocol["task_ids"],
        "task_order": protocol["task_order"],
        "execution_order": protocol["execution_order"],
        "execution_plan": protocol["execution_plan"],
        "conditions": protocol["conditions"],
        "planned_primary_runs": protocol["planned_primary_runs"],
        "model": {
            "provider": protocol["provider"],
            "name": protocol["model"],
            "settings": protocol["model_settings"],
            "source": protocol["model_source"],
            "source_evidence": protocol["model_source_evidence"],
            "resolution_variable": protocol["model_resolution_variable"],
            "validation": protocol["model_validation"],
            "resolved_model": validate_coding_model(protocol, root),
            "prompt_version": protocol["prompt_version"],
            "system_prompt": protocol["system_prompt"],
            "system_prompt_source": protocol["system_prompt_source"],
            "system_prompt_source_sha256": _sha256(prompt_source),
            "model_config_path": display_path(model_config),
            "model_config_sha256": _sha256(model_config),
        },
        "abstraction_model": protocol["abstraction_model"],
        "retrieval_embedding": protocol["embedding"],
        "tools": protocol["tools"],
        "tool_permissions": protocol["tool_permissions"],
        "limits": protocol["limits"],
        "pacing": protocol["pacing"],
        "objective_evaluator": {
            "name": protocol["objective_evaluator"],
            "version": protocol["objective_evaluator_version"],
        },
        "recurrence_evaluator": {
            "name": protocol["recurrence_evaluator"],
            "version": protocol["recurrence_evaluator_version"],
            "ground_truth_visibility": protocol["recurrence_ground_truth_visibility"],
        },
        "treatment_memory": {
            "policy": protocol["treatment_memory_policy"],
            "repository": protocol["treatment_repository"],
            "pattern_ids": protocol["treatment_pattern_ids"],
            "embedding": protocol["embedding"],
            "memory_writes": protocol["memory_writes"],
            "online_learning": protocol["online_learning"],
        },
        "metrics": protocol["metrics"],
        "rerun_policy": protocol["rerun_policy"],
        "leakage_safeguards": protocol["leakage_safeguards"],
        "neo4j_write_policy": protocol["neo4j_write_policy"],
        "network_policy": {
            "task_workspace_network": protocol["task_workspace_network"],
            "external_runtime_network": protocol["external_runtime_network"],
        },
        "runtime_environment_manifest": {
            "path": display_path(environment_manifest),
            "sha256": _sha256(environment_manifest),
        },
        "artifact_destination": {
            "freeze_artifact": display_path(_project_path(output_path, root)),
            "run_artifact_root": protocol["artifact_root"],
        },
        "existing_configuration_snapshot": _existing_configuration_snapshot(root),
        "preflight_checks": dict(report.checks),
    }
    output = _project_path(output_path, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_FREEZE)
    return parser


def main() -> None:
    args = make_cli_parser().parse_args()
    try:
        if args.freeze:
            output = freeze_protocol(
                args.config,
                project_root=args.project_root,
                output_path=args.output,
            )
            print(json.dumps({"status": "READY", "freeze_artifact": str(output)}, sort_keys=True))
        else:
            report = preflight(args.config, project_root=args.project_root)
            print(json.dumps({"status": report.status, "checks": report.checks}, sort_keys=True))
    except Sprint3AProtocolError as error:
        print(json.dumps({"status": "NOT READY", "error": str(error)}, sort_keys=True))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_FREEZE",
    "EXPECTED_ABSTRACTION_MODEL",
    "EXPECTED_CODING_MODEL",
    "FREEZE_INPUT_PATHS",
    "INTEGRATION_POINT_1_SHA",
    "R13B_FREEZE_SHA",
    "PreflightReport",
    "SPRINT3A_CONDITIONS",
    "SPRINT3A_EXPERIMENT_ID",
    "SPRINT3A_PATTERN_IDS",
    "SPRINT3A_TASK_IDS",
    "Sprint3AProtocolError",
    "TREATMENT_MEMORY_POLICY",
    "build_execution_plan",
    "freeze_input_hashes",
    "freeze_inputs_hash",
    "freeze_protocol",
    "load_protocol",
    "preflight",
    "validate_coding_model",
]
