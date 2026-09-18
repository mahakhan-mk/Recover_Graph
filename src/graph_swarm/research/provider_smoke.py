"""Provider-backed B0/O1 development smoke for Track B.

This module is intentionally a thin orchestration layer around the existing
Sprint 3 runner, frozen Oracle boundary, Docker environment loader, and
SWE-smith objective.  It does not prepare environments, change validation
markers, use Graph Swarm memory, or calculate Gate B1.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from experiments.oracle import FrozenOracleResolver
from experiments.sprint3 import (
    BenchmarkPreflightError,
    IsolatedTaskEnvironment,
    prepare_provider_smoke_context,
)
from graph_swarm.agent.pacing import ProviderRequestPacing
from graph_swarm.domain.advice import HistoricalRecoveryAdvice
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    ExperimentExecution,
    ExperimentRunArtifactStore,
    ExperimentRunner,
    LoadedExperimentConfiguration,
)

PROVIDER_SMOKE_TASK_ID = "GS-T007"
PROVIDER_SMOKE_REVIEW_ID = "GS-R014"
PROVIDER_SMOKE_STATUS = "PROVIDER_B0_O1_SMOKE_COMPLETE"
PROVIDER_SMOKE_FAILED = "PROVIDER_B0_O1_SMOKE_FAILED"
RUNTIME_SMOKE_STATUS = "AGENT_RUNTIME_SMOKE_READY"


@dataclass(frozen=True)
class RuntimeSmokeArtifact:
    """Narrowed readiness evidence required before provider execution."""

    path: Path
    task_id: str
    status: str
    provider_calls: int
    b0_launched: bool
    o1_launched: bool
    environment_fingerprint: str


class ProviderSmokeError(RuntimeError):
    """Raised when the provider smoke contract cannot be established."""


class ProviderSmokePrerequisiteError(ProviderSmokeError):
    """Raised before any provider execution when a readiness barrier is absent."""


def write_provider_smoke_summary(
    artifact_root: Path,
    payload: Mapping[str, Any],
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Write one exclusive, timestamped, append-only provider-smoke summary."""
    result_root = artifact_root.expanduser().resolve() / "GS-E003" / "sprint3b"
    result_root.mkdir(parents=True, exist_ok=True)
    moment = timestamp or datetime.now(UTC)
    path = result_root / (
        "provider-smoke-"
        + moment.strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid4().hex
        + ".json"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def run_provider_smoke(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    artifact_root: Path,
    config_path: Path | None = None,
    task_id: str = PROVIDER_SMOKE_TASK_ID,
) -> Path:
    """Run exactly one provider-backed B0 execution and one O1 execution.

    The readiness checks intentionally happen before constructing either
    provider-backed agent.  Provider/model failures are retained by the
    canonical runner and represented in the append-only summary; this
    function never retries or replaces a failed run.
    """
    root = project_root.expanduser().resolve()
    baseline = baseline_root.expanduser().resolve()
    execution = execution_root.expanduser().resolve()
    artifacts = artifact_root.expanduser().resolve()
    if task_id != PROVIDER_SMOKE_TASK_ID:
        raise ProviderSmokeError(
            f"provider smoke task is frozen to {PROVIDER_SMOKE_TASK_ID}: {task_id}"
        )

    selected_config_path = config_path or root / "configs/experiments/rollout_3a_pilot.yaml"
    runtime_artifact = find_runtime_smoke_artifact(artifacts, task_id=task_id)
    try:
        prepared = prepare_provider_smoke_context(
            project_root=root,
            baseline_root=baseline,
            execution_root=execution,
            config_path=selected_config_path,
            task_id=task_id,
        )
    except BenchmarkPreflightError as error:
        raise ProviderSmokePrerequisiteError(str(error)) from error

    b0_configuration = prepared.configuration
    o1_configuration = _condition_configuration(
        b0_configuration,
        ExperimentCondition.O1,
    )
    case = prepared.case

    environment = prepared.environment
    if runtime_artifact.environment_fingerprint != environment.environment_fingerprint:
        raise ProviderSmokePrerequisiteError(
            "runtime-smoke artifact does not match the selected environment fingerprint"
        )
    if not environment.container_image or not environment.container_python_executable:
        raise ProviderSmokePrerequisiteError("prepared Docker runtime metadata is incomplete")

    try:
        oracle = FrozenOracleResolver.from_frozen_files(
            o1_configuration.task_manifest_path,
            o1_configuration.task_problems_path,
        )
    except Exception as error:  # noqa: BLE001 - Oracle evidence is a pre-provider barrier
        raise ProviderSmokePrerequisiteError(str(error)) from error
    if oracle.review_id_for(task_id) != PROVIDER_SMOKE_REVIEW_ID:
        raise ProviderSmokePrerequisiteError(
            f"provider smoke Oracle mapping for {task_id} does not match frozen review "
            f"{PROVIDER_SMOKE_REVIEW_ID}"
        )
    expected_recovery_pattern = oracle.recovery_pattern_for(task_id)
    if not expected_recovery_pattern:
        raise ProviderSmokePrerequisiteError(
            f"provider smoke Oracle mapping for {task_id} has no frozen recovery pattern"
        )
    executions: list[ExperimentExecution] = []
    request_pacing = ProviderRequestPacing()

    # Each runner owns its run ID, workspace copy, conversation ID, and SQLite
    # step database.  The same task/model/runtime/objective contract is shared
    # while the B0 and O1 agent instances remain completely separate.
    for configuration, condition, resolver in (
        (b0_configuration, ExperimentCondition.B0, None),
        (o1_configuration, ExperimentCondition.O1, oracle),
    ):
        runner = ExperimentRunner(
            configuration,
            objective_evaluator=prepared.objective,
            recurrence_evaluator=prepared.recurrence_evaluator,
            oracle_resolver=resolver,
            condition=condition,
            execution_runtime_resolver=lambda _task, _workspace: (
                environment.agent_execution_runtime()
            ),
            workspace_resolver=prepared.workspace_resolver_for(condition),
            artifact_store=ExperimentRunArtifactStore(artifacts),
            request_pacing=request_pacing,
        )
        executions.append(runner.run_case(case))

    summary = build_provider_smoke_summary(
        executions,
        objective_observations=prepared.objective.observations,
        runtime_artifact=runtime_artifact,
        configuration=b0_configuration,
        environment=environment,
        task=case,
        expected_oracle_review_id=PROVIDER_SMOKE_REVIEW_ID,
        expected_recovery_pattern=expected_recovery_pattern,
    )
    summary_path = write_provider_smoke_summary(artifacts, summary)
    if summary["status"] != PROVIDER_SMOKE_STATUS:
        raise ProviderSmokeError(
            f"provider B0/O1 smoke failed; failure summary: {summary_path}"
        )
    return summary_path


def build_provider_smoke_summary(
    executions: Sequence[ExperimentExecution],
    *,
    objective_observations: Sequence[Any],
    runtime_artifact: RuntimeSmokeArtifact,
    configuration: LoadedExperimentConfiguration,
    environment: IsolatedTaskEnvironment,
    task: BenchmarkTaskCase,
    expected_oracle_review_id: str,
    expected_recovery_pattern: str,
) -> dict[str, Any]:
    """Validate and serialize the two immutable run results."""
    by_condition = {
        execution.artifact.condition.value: execution
        for execution in executions
    }
    failures: list[str] = []
    if len(executions) != 2 or set(by_condition) != {"B0", "O1"}:
        failures.append("exactly one B0 and one O1 execution were not produced")
    if any(execution.artifact.task_id != task.task.id for execution in executions):
        failures.append("run artifact task identity differs from the fixed smoke task")
    if any(execution.artifact.condition.value == "T" for execution in executions):
        failures.append("T execution was attempted")

    b0 = by_condition.get("B0")
    o1 = by_condition.get("O1")
    if b0 is None or o1 is None:
        # Keep a stable, inspectable failure payload even if a caller supplies
        # an incomplete fake execution in unit tests.
        return {
            "status": PROVIDER_SMOKE_FAILED,
            "smoke_type": "development_provider_smoke",
            "not_gate_b1": True,
            "gate_b1_evaluated": False,
            "b0_runs": sum(item.artifact.condition.value == "B0" for item in executions),
            "o1_runs": sum(item.artifact.condition.value == "O1" for item in executions),
            "t_runs": sum(item.artifact.condition.value == "T" for item in executions),
            "smoke_task_id": task.task.id,
            "failures": failures,
        }

    if b0.artifact.advice_received is not None or b0.dependencies.advice_events:
        failures.append("B0 received Oracle advice")
    o1_result = o1.artifact.advice_received
    delivered_advice: HistoricalRecoveryAdvice | None = None
    actual_oracle_review_id: str | None = None
    if o1_result is None or not isinstance(o1_result.advice, HistoricalRecoveryAdvice):
        failures.append("O1 did not receive the expected frozen Oracle intervention")
    else:
        delivered_advice = o1_result.advice
        actual_oracle_review_id = o1.artifact.advice_review_id or expected_oracle_review_id
        if actual_oracle_review_id != expected_oracle_review_id:
            failures.append("O1 intervention provenance does not match GS-R014")
        if delivered_advice.recovery_summary != expected_recovery_pattern:
            failures.append(
                "O1 delivered recovery pattern does not match frozen Oracle recovery pattern"
            )
    if len(o1.dependencies.advice_events) != 1:
        failures.append("O1 did not receive exactly one Oracle intervention")
    if o1.artifact.advice_intervention_boundary != "task_start":
        failures.append("O1 intervention did not occur at task_start")
    if o1.artifact.advice_delivery_timing != "pre_first_model_request":
        failures.append("O1 intervention was not delivered before the first model request")
    for condition, execution in (("B0", b0), ("O1", o1)):
        if not execution.artifact_path.is_file():
            failures.append(f"{condition} canonical artifact is missing")
        if not execution.raw_evidence_path.is_file():
            failures.append(f"{condition} raw evidence artifact is missing")

    for field in (
        "task_id",
        "family_id",
        "chronological_index",
        "model",
        "model_settings",
        "prompt_version",
        "provider_request_pacing",
    ):
        if getattr(b0.artifact, field) != getattr(o1.artifact, field):
            failures.append(f"B0/O1 contract field differs: {field}")
    if b0.artifact.run_id == o1.artifact.run_id:
        failures.append("B0/O1 run IDs are not fresh")
    if b0.dependencies.workspace_root == o1.dependencies.workspace_root:
        failures.append("B0/O1 workspaces are not fresh")
    if b0.step_database_path == o1.step_database_path:
        failures.append("B0/O1 SQLite step stores are not isolated")

    behavior_evidence = [item.model_dump(mode="json") for item in o1.dependencies.behavior_evidence]
    objective_by_condition = {
        str(getattr(observation, "condition", "unknown")): {
            "status": str(getattr(observation, "status", "unknown")),
            "return_code": getattr(observation, "return_code", None),
            "duration_seconds": getattr(observation, "duration_seconds", None),
        }
        for observation in objective_observations
    }
    provider_or_agent_errors = {
        condition: _execution_error(execution)
        for condition, execution in (("B0", b0), ("O1", o1))
    }
    if any(value is not None for value in provider_or_agent_errors.values()):
        # The distinction is retained in the payload; an error is never hidden
        # behind a replacement run.
        failures.extend(
            f"{condition} error classified as {value['classification']}"
            for condition, value in provider_or_agent_errors.items()
            if value is not None and value["classification"] != "bounded_termination"
        )

    summary = {
        "status": PROVIDER_SMOKE_STATUS if not failures else PROVIDER_SMOKE_FAILED,
        "smoke_type": "development_provider_smoke",
        "not_gate_b1": True,
        "gate_b1_evaluated": False,
        "b0_runs": 1,
        "o1_runs": 1,
        "t_runs": 0,
        "smoke_task_id": task.task.id,
        "b0_run_id": b0.artifact.run_id,
        "o1_run_id": o1.artifact.run_id,
        "smoke_task": {
            "id": task.task.id,
            "review_id": expected_oracle_review_id,
            "selection": "fixed frozen transfer task; never selected by model outcome",
            "family_id": task.task.family_id,
            "repository": task.task.repository,
            "occurrence_index": task.occurrence_index,
            "rationale": (
                "GS-T007 is the fixed Sprint 3C-B runtime-smoke task and its frozen "
                "GS-R014 mapping is validated, transferable, and has a non-empty recovery pattern."
            ),
        },
        "provider": configuration.model.provider,
        "model": configuration.model.model,
        "effective_model": _effective_model(configuration),
        "model_settings": dict(configuration.model.settings),
        "temperature": configuration.model.settings.get("temperature"),
        "timeout_contract": b0.artifact.timeout_contract.model_dump(mode="json"),
        "timeout_contract_by_condition": {
            "B0": b0.artifact.timeout_contract.model_dump(mode="json"),
            "O1": o1.artifact.timeout_contract.model_dump(mode="json"),
        },
        "provider_request_pacing": b0.artifact.provider_request_pacing.model_dump(mode="json"),
        "provider_request_pacing_by_condition": {
            "B0": b0.artifact.provider_request_pacing.model_dump(mode="json"),
            "O1": o1.artifact.provider_request_pacing.model_dump(mode="json"),
        },
        "provider_pacing_wait_seconds": {
            "B0": b0.artifact.provider_pacing_wait_seconds,
            "O1": o1.artifact.provider_pacing_wait_seconds,
        },
        "prompt_config_identity": {
            "config_path": str(configuration.config_path),
            "model_config_path": configuration.config.model_config_path,
            "config_version": configuration.config.config_version,
            "prompt_version": configuration.model.prompt_version,
            "limits": configuration.config.limits.model_dump(mode="json"),
        },
        "environment": {
            "runtime_type": environment.runtime_type,
            "environment_fingerprint": environment.environment_fingerprint,
            "prepared_image": environment.container_image,
            "container_python": environment.container_python_executable,
            "runtime_smoke_artifact": _runtime_artifact_reference(runtime_artifact),
        },
        "environment_fingerprint": environment.environment_fingerprint,
        "prepared_image": environment.container_image,
        "container_python": environment.container_python_executable,
        "runtime_contract": {
            "runtime_type": environment.runtime_type,
            "network": "none",
            "workdir": "/workspace",
            "pythonpath": "/workspace/src:/workspace",
            "objective_evaluator": "FrozenSWEsmithObjective",
            "persistent_memory": "disabled",
        },
        "objective": {
            "b0": b0.artifact.task_success,
            "o1": o1.artifact.task_success,
            "by_condition": objective_by_condition,
        },
        "b0_objective_result": b0.artifact.task_success,
        "o1_objective_result": o1.artifact.task_success,
        "oracle_advice_count": {
            "b0": len(b0.dependencies.advice_events),
            "o1": len(o1.dependencies.advice_events),
        },
        "b0_oracle_advice_count": len(b0.dependencies.advice_events),
        "o1_oracle_advice_count": len(o1.dependencies.advice_events),
        "o1_oracle_review_id": actual_oracle_review_id,
        "frozen_recovery_pattern": expected_recovery_pattern,
        "o1_intervention_timing": {
            "boundary": o1.artifact.advice_intervention_boundary,
            "delivery": o1.artifact.advice_delivery_timing,
        },
        "o1_oracle_intervention": (
            None
            if delivered_advice is None
            else delivered_advice.model_dump(mode="json")
        ),
        "o1_behavior_change_evidence": behavior_evidence,
        "o1_behavior_changed": any(
            item.behavior_changed is True for item in o1.dependencies.behavior_evidence
        ),
        "tool_call_counts": {"b0": b0.artifact.tool_calls, "o1": o1.artifact.tool_calls},
        "b0_tool_calls": b0.artifact.tool_calls,
        "o1_tool_calls": o1.artifact.tool_calls,
        "retries": {"b0": b0.artifact.retries, "o1": o1.artifact.retries},
        "b0_retries": b0.artifact.retries,
        "o1_retries": o1.artifact.retries,
        "provider_usage": {
            "b0": _usage(b0),
            "o1": _usage(o1),
        },
        "latency_ms": {"b0": b0.artifact.latency_ms, "o1": o1.artifact.latency_ms},
        "error_classification": provider_or_agent_errors,
        "termination": {
            "b0": (
                None
                if b0.artifact.termination is None
                else b0.artifact.termination.model_dump(mode="json")
            ),
            "o1": (
                None
                if o1.artifact.termination is None
                else o1.artifact.termination.model_dump(mode="json")
            ),
        },
        "budget_exhausted": {
            "b0": b0.artifact.termination is not None,
            "o1": o1.artifact.termination is not None,
        },
        "artifact_paths": {
            "b0_artifact": str(b0.artifact_path),
            "b0_raw_evidence": str(b0.raw_evidence_path),
            "o1_artifact": str(o1.artifact_path),
            "o1_raw_evidence": str(o1.raw_evidence_path),
        },
        "failures": failures,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the development-only provider-backed Track B B0/O1 smoke"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-root", type=Path, default=Path("benchmark/workspaces"))
    parser.add_argument(
        "--execution-root",
        type=Path,
        default=Path("research/evidence/workspaces"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("research/evidence/results"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/rollout_3a_pilot.yaml"),
        help="B0 experiment config; O1 is derived from this exact model/runtime contract",
    )
    parser.add_argument("--task-id", default=PROVIDER_SMOKE_TASK_ID)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    try:
        path = run_provider_smoke(
            project_root=project_root,
            baseline_root=_resolve_cli_path(args.baseline_root, project_root),
            execution_root=_resolve_cli_path(args.execution_root, project_root),
            artifact_root=_resolve_cli_path(args.artifact_root, project_root),
            config_path=_resolve_cli_path(args.config, project_root),
            task_id=args.task_id,
        )
    except ProviderSmokeError as error:
        parser.error(str(error))
    print(f"{PROVIDER_SMOKE_STATUS} {path}")
    return 0


def _condition_configuration(
    configuration: LoadedExperimentConfiguration,
    condition: ExperimentCondition,
) -> LoadedExperimentConfiguration:
    return replace(
        configuration,
        config=configuration.config.model_copy(update={"conditions": (condition,)}),
    )


def find_runtime_smoke_artifact(
    artifact_root: Path,
    *,
    task_id: str,
) -> RuntimeSmokeArtifact:
    """Find and validate one successful Sprint 3C-B runtime-smoke artifact."""
    candidates = sorted(
        (artifact_root / "GS-E003" / "sprint3b").glob("runtime-smoke-*.json"),
        reverse=True,
    )
    for path in candidates:
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        artifact = _parse_runtime_smoke_artifact(raw, path)
        if artifact is not None and artifact.task_id == task_id:
            return artifact
    raise ProviderSmokePrerequisiteError(
        f"no successful Sprint 3C-B runtime-smoke artifact exists for {task_id}"
    )


def _parse_runtime_smoke_artifact(
    raw: object,
    path: Path,
) -> RuntimeSmokeArtifact | None:
    if not isinstance(raw, dict):
        return None
    record = cast(dict[object, object], raw)
    task_id = _runtime_text(record.get("task_id"))
    status = _runtime_text(record.get("status"))
    fingerprint = _runtime_text(record.get("environment_fingerprint"))
    provider_calls = record.get("provider_calls")
    b0_launched = record.get("b0_launched")
    o1_launched = record.get("o1_launched")
    if task_id is None or status != RUNTIME_SMOKE_STATUS or fingerprint is None:
        return None
    if (
        not isinstance(provider_calls, int)
        or isinstance(provider_calls, bool)
        or provider_calls != 0
        or b0_launched is not False
        or o1_launched is not False
    ):
        return None
    return RuntimeSmokeArtifact(
        path=path,
        task_id=task_id,
        status=status,
        provider_calls=provider_calls,
        b0_launched=False,
        o1_launched=False,
        environment_fingerprint=fingerprint,
    )


def _runtime_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _effective_model(configuration: LoadedExperimentConfiguration) -> str:
    if configuration.model.provider == "openrouter":
        return os.environ.get("OPENROUTER_MODEL") or configuration.model.model
    if configuration.model.provider == "groq":
        return os.environ.get("GROQ_MODEL") or configuration.model.model
    return configuration.model.model


def _runtime_artifact_reference(runtime_artifact: RuntimeSmokeArtifact) -> dict[str, object]:
    return {
        "path": str(runtime_artifact.path),
        "status": runtime_artifact.status,
        "task_id": runtime_artifact.task_id,
        "provider_calls": runtime_artifact.provider_calls,
        "b0_launched": runtime_artifact.b0_launched,
        "o1_launched": runtime_artifact.o1_launched,
        "environment_fingerprint": runtime_artifact.environment_fingerprint,
    }


def _usage(execution: ExperimentExecution) -> dict[str, int | None]:
    return {
        "input_tokens": execution.artifact.input_tokens,
        "output_tokens": execution.artifact.output_tokens,
    }


def _execution_error(execution: ExperimentExecution) -> dict[str, object] | None:
    error = execution.error
    if error is None:
        return None
    name = str(getattr(error, "error_type", type(error).__name__))
    if execution.artifact.termination is not None:
        classification = "bounded_termination"
    elif name.endswith("HTTPError") or name.endswith("APIError"):
        classification = "provider"
    elif name in {"UnexpectedModelBehavior", "UsageLimitExceeded"}:
        classification = "agent_control"
    else:
        classification = "infrastructure"
    return {
        "type": name,
        "classification": classification,
        "message": str(error),
        "timeout_layer": getattr(error, "timeout_layer", None),
        "timeout_seconds": getattr(error, "timeout_seconds", None),
    }


def _resolve_cli_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
