"""Policy and validation for reproducible frozen benchmark environments."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from packaging.specifiers import SpecifierSet

FROZEN_TASK_ORDER = tuple(f"GS-T{i:03d}" for i in range(6, 16))


class BenchmarkEnvironmentConfigurationError(ValueError):
    """Raised when benchmark environment policy is missing or incompatible."""


@dataclass(frozen=True)
class TaskEnvironmentPolicy:
    """Reproducibility policy for one frozen task."""

    task_id: str
    runtime: str
    python_source: str
    dependency_source: str
    network_during_execution: bool
    require_validated_environment: bool
    rationale: str
    python_constraint_assertion: str | None
    required_extras: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkEnvironmentPolicy:
    """Validated benchmark-wide and per-task environment policy."""

    path: Path
    task_order: tuple[str, ...]
    network_during_execution: bool
    container_image_source: str
    tasks: dict[str, TaskEnvironmentPolicy]
    docker_build_timeout_seconds: int = 7200

    def task(self, task_id: str) -> TaskEnvironmentPolicy:
        try:
            return self.tasks[task_id]
        except KeyError as error:
            raise BenchmarkEnvironmentConfigurationError(
                f"benchmark environment policy has no entry for {task_id}"
            ) from error


def _as_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkEnvironmentConfigurationError(f"{field} must be a boolean")
    return value


def _as_mapping(value: Any) -> dict[str, Any]:
    """Narrow a TOML table to a mapping with string keys."""
    if not isinstance(value, dict):
        raise BenchmarkEnvironmentConfigurationError("TOML tables must be mappings")
    raw_mapping = cast(dict[Any, Any], value)
    if not all(isinstance(key, str) for key in raw_mapping):
        raise BenchmarkEnvironmentConfigurationError("TOML table keys must be strings")
    return cast(dict[str, Any], value)


def load_benchmark_environment_policy(path: Path) -> BenchmarkEnvironmentPolicy:
    """Load and validate policy without duplicating manifest image names."""
    try:
        raw: Any = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise BenchmarkEnvironmentConfigurationError(
            f"could not read benchmark environment policy {path}: {error}"
        ) from error

    raw = _as_mapping(raw)
    unknown_sections = set(raw) - {"benchmark", "tasks"}
    if unknown_sections:
        raise BenchmarkEnvironmentConfigurationError(
            f"benchmark environment policy contains unknown sections: {sorted(unknown_sections)}"
        )
    benchmark = raw.get("benchmark")
    tasks = raw.get("tasks")
    if not isinstance(benchmark, dict) or not isinstance(tasks, dict):
        raise BenchmarkEnvironmentConfigurationError(
            "benchmark environment policy requires [benchmark] and [tasks]"
        )
    benchmark = _as_mapping(benchmark)
    tasks = _as_mapping(tasks)
    task_order = tuple(benchmark.get("task_order", ()))
    if task_order != FROZEN_TASK_ORDER:
        raise BenchmarkEnvironmentConfigurationError(
            f"benchmark environment task order must be {FROZEN_TASK_ORDER!r}"
        )
    allowed_benchmark_fields = {
        "name",
        "task_order",
        "network_during_execution",
        "container_image_source",
        "docker_build_timeout_seconds",
    }
    unknown_benchmark_fields = set(benchmark) - allowed_benchmark_fields
    if unknown_benchmark_fields:
        raise BenchmarkEnvironmentConfigurationError(
            f"benchmark contains unknown fields: {sorted(unknown_benchmark_fields)}"
        )
    if not isinstance(benchmark.get("name"), str) or not benchmark["name"].strip():
        raise BenchmarkEnvironmentConfigurationError(
            "benchmark.name must be a non-empty string"
        )
    image_source = benchmark.get("container_image_source")
    if image_source != "frozen_task_manifest":
        raise BenchmarkEnvironmentConfigurationError(
            "benchmark.container_image_source must be frozen_task_manifest"
        )
    network = _as_bool(
        benchmark.get("network_during_execution"),
        field="benchmark.network_during_execution",
    )
    if network:
        raise BenchmarkEnvironmentConfigurationError(
            "benchmark execution must disable network access"
        )
    docker_build_timeout_seconds = benchmark.get("docker_build_timeout_seconds")
    if isinstance(docker_build_timeout_seconds, bool) or not isinstance(
        docker_build_timeout_seconds, int
    ):
        raise BenchmarkEnvironmentConfigurationError(
            "benchmark.docker_build_timeout_seconds must be an integer"
        )
    if docker_build_timeout_seconds <= 0:
        raise BenchmarkEnvironmentConfigurationError(
            "benchmark.docker_build_timeout_seconds must be greater than zero"
        )

    validated: dict[str, TaskEnvironmentPolicy] = {}
    for task_id in FROZEN_TASK_ORDER:
        raw_task = tasks.get(task_id)
        if not isinstance(raw_task, dict):
            raise BenchmarkEnvironmentConfigurationError(
                f"benchmark environment policy is missing {task_id}"
            )
        raw_task = _as_mapping(raw_task)
        allowed_task_fields = {
            "runtime",
            "python_source",
            "dependency_source",
            "network_during_execution",
            "require_validated_environment",
            "rationale",
            "python_constraint_assertion",
            "required_extras",
        }
        unknown_task_fields = set(raw_task) - allowed_task_fields
        if unknown_task_fields:
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_id} contains unknown fields: {sorted(unknown_task_fields)}"
            )
        if "image_name" in raw_task or "container_image" in raw_task:
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_id} must source its Docker image from the frozen manifest"
            )
        runtime = raw_task.get("runtime")
        if runtime not in {"local_preferred_manifest_container", "manifest_container_required"}:
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_id}.runtime is not a supported runtime policy"
            )
        python_source = raw_task.get("python_source")
        dependency_source = raw_task.get("dependency_source")
        if python_source != "repository" or dependency_source != "repository":
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_id} must use repository-authoritative Python and dependency metadata"
            )
        task_network = _as_bool(
            raw_task.get("network_during_execution"),
            field=f"tasks.{task_id}.network_during_execution",
        )
        if task_network:
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_id} cannot enable network during execution"
            )
        rationale = raw_task.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_id}.rationale must be a non-empty string"
            )
        assertion = raw_task.get("python_constraint_assertion")
        if assertion is not None:
            if not isinstance(assertion, str) or not assertion.strip():
                raise BenchmarkEnvironmentConfigurationError(
                    f"{task_id}.python_constraint_assertion must be a non-empty string"
                )
            try:
                SpecifierSet(assertion)
            except ValueError as error:
                raise BenchmarkEnvironmentConfigurationError(
                    f"{task_id}.python_constraint_assertion is invalid"
                ) from error
        raw_extras_value = raw_task.get("required_extras", [])
        if not isinstance(raw_extras_value, list):
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_id}.required_extras must be a list of non-empty strings"
            )
        raw_extras = cast(list[Any], raw_extras_value)
        if not all(isinstance(extra, str) and extra.strip() for extra in raw_extras):
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_id}.required_extras must be a list of non-empty strings"
            )
        extras = [extra.strip() for extra in raw_extras if isinstance(extra, str)]
        required_extras: tuple[str, ...] = tuple(dict.fromkeys(extras))
        validated[task_id] = TaskEnvironmentPolicy(
            task_id=task_id,
            runtime=runtime,
            python_source=python_source,
            dependency_source=dependency_source,
            network_during_execution=task_network,
            require_validated_environment=_as_bool(
                raw_task.get("require_validated_environment"),
                field=f"tasks.{task_id}.require_validated_environment",
            ),
            rationale=rationale,
            python_constraint_assertion=assertion,
            required_extras=required_extras,
        )

    unknown_tasks = set(tasks) - set(FROZEN_TASK_ORDER)
    if unknown_tasks:
        raise BenchmarkEnvironmentConfigurationError(
            f"benchmark environment policy contains unknown tasks: {sorted(unknown_tasks)}"
        )
    return BenchmarkEnvironmentPolicy(
        path,
        task_order,
        network,
        image_source,
        validated,
        docker_build_timeout_seconds,
    )


def reconcile_task_policy(
    policy: BenchmarkEnvironmentPolicy,
    task_id: str,
    constraints: SpecifierSet,
) -> TaskEnvironmentPolicy:
    """Ensure policy leaves Python selection authoritative to the repository."""
    task_policy = policy.task(task_id)
    effective_python_constraints(task_policy, constraints)
    return task_policy


def effective_python_constraints(
    task_policy: TaskEnvironmentPolicy,
    repository_constraints: SpecifierSet,
) -> SpecifierSet:
    """Return the intersection of repository and policy Python constraints."""
    if not str(repository_constraints) and task_policy.python_source != "repository":
        raise BenchmarkEnvironmentConfigurationError(
            f"{task_policy.task_id} has no repository Python metadata and policy is not "
            "repository-authoritative"
        )
    if task_policy.python_constraint_assertion:
        assertion = SpecifierSet(task_policy.python_constraint_assertion)
        candidates = [
            f"3.{minor}.0"
            for minor in range(8, 15)
        ]
        if not any(
            candidate in repository_constraints and candidate in assertion
            for candidate in candidates
        ):
            raise BenchmarkEnvironmentConfigurationError(
                f"{task_policy.task_id}.python_constraint_assertion is incompatible "
                "with repository metadata"
            )
        if str(repository_constraints):
            return SpecifierSet(f"{repository_constraints},{assertion}")
        return assertion
    return repository_constraints


def policy_fingerprint_payload(policy: BenchmarkEnvironmentPolicy) -> bytes:
    """Return canonical policy content for environment fingerprints."""
    payload = {
        "task_order": policy.task_order,
        "network_during_execution": policy.network_during_execution,
        "container_image_source": policy.container_image_source,
        "docker_build_timeout_seconds": policy.docker_build_timeout_seconds,
        "tasks": {
            task_id: {
                "runtime": task.runtime,
                "python_source": task.python_source,
                "dependency_source": task.dependency_source,
                "network_during_execution": task.network_during_execution,
                "require_validated_environment": task.require_validated_environment,
                "rationale": task.rationale,
                "python_constraint_assertion": task.python_constraint_assertion,
                "required_extras": task.required_extras,
            }
            for task_id, task in sorted(policy.tasks.items())
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
