"""Prepare and smoke-test the Track B Sprint 3A B0/O1 runtime.

The module is an experiment harness, not a second Track B runner.  It supplies
the frozen SWE-smith objective boundary and a frozen-test recurrence matcher to
the approved :class:`ExperimentRunner`.
"""

from __future__ import annotations

import configparser
import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.agent.tools.run_command import run_command, workspace_process_environment
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.tasks import Task
from graph_swarm.memory.recovery_evidence import is_test_execution
from graph_swarm.research.benchmark_environments import (
    BenchmarkEnvironmentConfigurationError,
    BenchmarkEnvironmentPolicy,
    effective_python_constraints,
    load_benchmark_environment_policy,
    policy_fingerprint_payload,
    reconcile_task_policy,
)
from graph_swarm.research.benchmark_runtime_smoke import (
    RuntimeSmokeValidationError,
    validate_mutated_test_failure,
    validate_run_command_evidence,
    write_runtime_smoke_evidence,
)
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    LoadedExperimentConfiguration,
    RecurrenceEvaluationRequired,
    RecurrenceEvaluator,
    RecurrenceEventStream,
    load_experiment_configuration,
    load_task_cases,
)

TRANSFER_TASKS = tuple(f"GS-T{i:03d}" for i in range(6, 16))
OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION = "pytest_help_option_discovery_v1"


@dataclass(frozen=True)
class FrozenSWEsmithCase:
    """Research-only objective data; never included in the agent prompt."""

    instance_id: str
    fail_to_pass: tuple[str, ...]
    patch: str


@dataclass(frozen=True)
class ObjectiveObservation:
    task_id: str
    condition: str
    command: tuple[str, ...]
    return_code: int | None
    status: str
    duration_seconds: float
    stdout: str
    stderr: str


class BenchmarkPreflightError(RuntimeError):
    """Raised before model execution when the isolated benchmark is unusable."""


# SunPy's frozen image omits its repository-declared build/runtime
# ``setuptools_scm`` dependency, which prevents pytest collection.  Keep the
# repair task-scoped so the other frozen environments remain unchanged.
# Stackprinter's frozen repository has no declared dependency metadata, but
# its frozen objective imports numpy through tests/source.py.  Pin the
# validated resolver result for reproducibility because the repository has no
# dependency version constraint for this objective-only requirement.
TASK_DEPENDENCY_OVERLAYS: Mapping[str, Sequence[str]] = {
    "GS-T003": ("setuptools_scm[toml]>=8.0.1",),
    "GS-T005": ("numpy==2.5.3",),
}


@dataclass(frozen=True)
class IsolatedTaskEnvironment:
    """Executable environment assigned to one benchmark task."""

    task_id: str
    python_executable: Path
    validation_marker: Path | None = None
    python_version: str | None = None
    environment_fingerprint: str | None = None
    runtime_type: str = "local_venv"
    container_image: str | None = None
    dependency_plan: DependencyInstallPlan | None = None
    dependency_source_root: Path | None = None
    container_python_executable: str | None = None
    base_container_image: str | None = None
    benchmark_policy_path: Path | None = None
    benchmark_manifest_path: Path | None = None

    def agent_execution_runtime(self) -> ExecutionRuntime:
        """Describe the runtime available to Track B process tools."""
        if self.runtime_type == "docker":
            if self.container_image is None or self.container_python_executable is None:
                raise BenchmarkPreflightError(
                    f"container execution metadata is incomplete for {self.task_id}"
                )
            return ExecutionRuntime(
                runtime_type="docker",
                docker_executable=self.python_executable,
                container_image=self.container_image,
                container_python_executable=self.container_python_executable,
            )
        return ExecutionRuntime(
            runtime_type="local",
            python_executable=self.python_executable,
        )

    def mark_validated(self) -> None:
        if self.validation_marker is None:
            return
        metadata: dict[str, Any] = {}
        if self.validation_marker.is_file():
            try:
                raw: Any = json.loads(self.validation_marker.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, dict):
                metadata = cast(dict[str, Any], raw)
        metadata.update(
            {
                "task_id": self.task_id,
                "validated": True,
                "python_executable": str(self.python_executable),
                "python_version": self.python_version,
                "environment_fingerprint": self.environment_fingerprint,
                "runtime_type": self.runtime_type,
                "container_image": self.container_image,
                "container_python_executable": self.container_python_executable,
                "base_container_image": self.base_container_image,
                "benchmark_policy_path": str(self.benchmark_policy_path)
                if self.benchmark_policy_path is not None
                else None,
                "benchmark_manifest_path": str(self.benchmark_manifest_path)
                if self.benchmark_manifest_path is not None
                else None,
            }
        )
        self.validation_marker.write_text(
            json.dumps(metadata, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class RecordedExecutionError(Exception):
    """Preserve the original runner error category during analysis reload."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class FrozenSWEsmithObjective:
    """Run the frozen FAIL_TO_PASS tests as the objective benchmark boundary."""

    def __init__(
        self,
        cases: Mapping[str, FrozenSWEsmithCase],
        environments: Mapping[str, IsolatedTaskEnvironment],
        *,
        objective_coverage_policy: str | None = None,
        coverage_policy_selection_version: str | None = None,
        effective_coverage_policies: Mapping[str, str] | None = None,
        objective_timeout_seconds: float = 900,
    ) -> None:
        self.cases = cases
        self.environments = environments
        self._validate_coverage = objective_coverage_policy is not None
        self.objective_coverage_policy = objective_coverage_policy or "no_cov"
        self.coverage_policy_selection_version = coverage_policy_selection_version
        self._coverage_policy_by_task: dict[str, str] = dict(
            effective_coverage_policies or {}
        )
        invalid_policies = {
            policy
            for policy in self._coverage_policy_by_task.values()
            if policy not in {"no_cov", "plain_pytest", "no_cov_addopts"}
        }
        if invalid_policies:
            raise BenchmarkPreflightError(
                "unsupported effective objective coverage policy: "
                + ", ".join(sorted(invalid_policies))
            )
        if objective_timeout_seconds <= 0:
            raise ValueError("objective_timeout_seconds must be positive")
        self.objective_timeout_seconds = objective_timeout_seconds
        self.observations: list[ObjectiveObservation] = []

    def __call__(self, task: Any, workspace: Path) -> bool:
        environment = self.environments.get(task.id)
        if environment is None:
            raise BenchmarkPreflightError(
                f"no isolated SWE-smith executable was selected for {task.id}"
            )
        if self._validate_coverage and task.id not in self._coverage_policy_by_task:
            self._select_coverage_policy(task.id, workspace)
        command = self._objective_command(task.id, workspace, environment)
        condition = _condition_from_workspace(workspace)
        started = datetime.now(UTC)
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=workspace_process_environment(workspace),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.objective_timeout_seconds,
            )
            output = f"{completed.stdout}\n{completed.stderr}"
            status = "passed" if completed.returncode == 0 else "test_failure"
            if (
                completed.returncode in (2, 3, 4, 5)
                or "__GS_DEPENDENCY_SETUP_FAILED__" in output
                or _looks_like_collection_failure(output)
            ):
                status = "objective_infrastructure_failure"
            self.observations.append(
                ObjectiveObservation(
                    task_id=task.id,
                    condition=condition,
                    command=command,
                    return_code=completed.returncode,
                    status=status,
                    duration_seconds=(datetime.now(UTC) - started).total_seconds(),
                    stdout=completed.stdout[-12000:],
                    stderr=completed.stderr[-12000:],
                )
            )
            return completed.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as error:
            self.observations.append(
                ObjectiveObservation(
                    task_id=task.id,
                    condition=condition,
                    command=command,
                    return_code=None,
                    status="objective_infrastructure_failure",
                    duration_seconds=(datetime.now(UTC) - started).total_seconds(),
                    stdout="",
                    stderr=str(error),
                )
            )
            return False

    def preflight(
        self,
        task: Any,
        workspace: Path,
    ) -> ObjectiveObservation:
        """Verify the mutated task can collect and execute before model calls."""
        if self._validate_coverage and task.id not in self._coverage_policy_by_task:
            self._select_coverage_policy(task.id, workspace)
        before = len(self.observations)
        self(task, workspace)
        observation = self.observations[-1]
        if observation.status == "objective_infrastructure_failure":
            details = "\n".join(
                value for value in (observation.stdout, observation.stderr) if value
            )
            raise BenchmarkPreflightError(
                f"SWE-smith preflight failed for {task.id}: {details[-4000:]}"
            )
        if observation.status != "test_failure" or observation.return_code == 0:
            raise BenchmarkPreflightError(
                f"SWE-smith preflight for {task.id} did not produce the expected "
                "mutated test failure:\n"
                f"return_code={observation.return_code}\n"
                f"status={observation.status}\n"
                f"command={observation.command!r}\n"
                f"stdout:\n{observation.stdout[-4000:] or '<empty>'}\n"
                f"stderr:\n{observation.stderr[-4000:] or '<empty>'}"
            )
        self.environments[task.id].mark_validated()
        if len(self.observations) != before + 1:  # pragma: no cover - defensive invariant
            raise BenchmarkPreflightError(
                f"SWE-smith preflight produced invalid evidence for {task.id}"
            )
        return observation

    def _pytest_arguments(self, task_id: str) -> tuple[str, ...]:
        case = self.cases[task_id]
        coverage_policy = self._coverage_policy_by_task.get(
            task_id,
            self.objective_coverage_policy,
        )
        if coverage_policy == "no_cov":
            coverage_arguments = ("--no-cov",)
        elif coverage_policy == "plain_pytest":
            coverage_arguments = ()
        elif coverage_policy == "no_cov_addopts":
            coverage_arguments = ("-o", "addopts=")
        else:
            raise BenchmarkPreflightError(
                f"unsupported objective coverage policy: {coverage_policy}"
            )
        return (
            "-m",
            "pytest",
            *coverage_arguments,
            *(_pytest_target(test_id) for test_id in case.fail_to_pass),
            "-q",
        )

    def _objective_command(
        self,
        task_id: str,
        workspace: Path,
        environment: IsolatedTaskEnvironment,
        *,
        pytest_arguments: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        arguments = pytest_arguments or self._pytest_arguments(task_id)
        if environment.runtime_type == "docker":
            if not environment.container_image:
                raise BenchmarkPreflightError(
                    f"container image is missing for {task_id}"
                )
            container_python = environment.container_python_executable or "python"
            return (
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--mount",
                f"type=bind,source={workspace.resolve()},target=/workspace",
                "--workdir",
                "/workspace",
                "--env",
                "PYTHONPATH=/workspace/src:/workspace",
                environment.container_image,
                container_python,
                *arguments,
            )
        return (str(environment.python_executable), *arguments)

    def effective_coverage_policy(self, task_id: str) -> str:
        """Return the resolved policy after the task environment was probed."""
        return self._coverage_policy_by_task.get(task_id, self.objective_coverage_policy)

    def _select_coverage_policy(self, task_id: str, workspace: Path) -> None:
        if self.coverage_policy_selection_version is not None:
            self._resolve_r7_coverage_policy(task_id, workspace)
            return
        self._validate_coverage_policy(task_id, workspace)

    def _validate_coverage_policy(self, task_id: str, workspace: Path) -> None:
        """Preserve the historical R6 probe behavior."""
        environment = self.environments[task_id]
        probe_arguments = ("-m", "pytest", "--no-cov", "--help")
        command = self._objective_command(
            task_id,
            workspace,
            environment,
            pytest_arguments=probe_arguments,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=workspace_process_environment(workspace),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BenchmarkPreflightError(
                f"objective coverage policy probe failed for {task_id}: {error}"
            ) from error
        if completed.returncode == 0:
            self._coverage_policy_by_task[task_id] = "no_cov"
            return
        self._coverage_policy_by_task[task_id] = "no_cov_addopts"

    def _resolve_r7_coverage_policy(self, task_id: str, workspace: Path) -> None:
        """Select coverage-neutral invocation from ordinary pytest help output."""
        environment = self.environments[task_id]
        command = self._objective_command(
            task_id,
            workspace,
            environment,
            pytest_arguments=("-m", "pytest", "--help"),
        )
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=workspace_process_environment(workspace),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BenchmarkPreflightError(
                f"R7 objective coverage policy discovery failed for {task_id}: {error}"
            ) from error
        help_output = f"{completed.stdout}\n{completed.stderr}"
        if _pytest_help_supports_no_cov(help_output):
            self._coverage_policy_by_task[task_id] = "no_cov"
            return
        if completed.returncode != 0 and not _is_no_cov_argument_error(help_output):
            raise BenchmarkPreflightError(
                f"R7 pytest help discovery failed for {task_id}: "
                f"return_code={completed.returncode}\n{help_output[-4000:]}"
            )
        if _repository_coverage_enforcement(workspace):
            self._coverage_policy_by_task[task_id] = "no_cov_addopts"
            return
        self._coverage_policy_by_task[task_id] = "plain_pytest"


def load_frozen_swesmith_cases(
    instance_ids: Sequence[str],
    *,
    allow_network: bool = True,
) -> dict[str, FrozenSWEsmithCase]:
    """Load only frozen objective test IDs from the published benchmark data."""
    target_ids = {instance_id.lower() for instance_id in instance_ids}
    cases: dict[str, FrozenSWEsmithCase] = {}
    parquet_root = (
        Path.home() / ".cache" / "huggingface" / "hub" / "datasets--SWE-bench--SWE-smith-py"
    )
    parquet_files = sorted(parquet_root.rglob("train-*.parquet"))
    if parquet_files:
        parquet: Any = importlib.import_module("pyarrow.parquet")
        for parquet_file in parquet_files:
            table: Any = parquet.read_table(
                str(parquet_file),
                columns=["instance_id", "FAIL_TO_PASS", "patch"],
            )
            for raw_row in table.to_pylist():
                row = cast(dict[str, Any], raw_row)
                instance_id = str(row["instance_id"])
                if instance_id.lower() not in target_ids:
                    continue
                tests = _test_ids(row.get("FAIL_TO_PASS"))
                patch = row.get("patch")
                if tests and isinstance(patch, str) and patch.strip():
                    cases[instance_id.lower()] = FrozenSWEsmithCase(
                        instance_id,
                        tests,
                        patch,
                    )
        missing = sorted(target_ids - set(cases))
        if missing:
            raise BenchmarkPreflightError(
                "frozen SWE-smith objective data is incomplete: " + ", ".join(missing)
            )
        return cases

    if not allow_network:
        raise BenchmarkPreflightError(
            "frozen SWE-smith objective data is not available locally; "
            "acquire the frozen benchmark data before benchmark_preflight"
        )
    datasets_module: Any = importlib.import_module("datasets")
    load_dataset: Any = datasets_module.load_dataset
    dataset: Any = load_dataset(
        "SWE-bench/SWE-smith-py",
        split="train",
        streaming=True,
    )
    for raw_row in dataset:
        row = cast(dict[str, Any], raw_row)
        instance_id = str(row["instance_id"])
        if instance_id.lower() not in target_ids:
            continue
        tests = _test_ids(row.get("FAIL_TO_PASS"))
        patch = row.get("patch")
        if tests and isinstance(patch, str) and patch.strip():
            cases[instance_id.lower()] = FrozenSWEsmithCase(instance_id, tests, patch)
    missing = sorted(target_ids - set(cases))
    if missing:
        raise BenchmarkPreflightError(
            "frozen SWE-smith objective data is incomplete: " + ", ".join(missing)
        )
    return cases


def make_recurrence_matcher(
    frozen_cases: Mapping[str, FrozenSWEsmithCase],
) -> RecurrenceEvaluator:
    """Match observed failed test output against frozen FAIL_TO_PASS IDs."""

    def determine(
        case: BenchmarkTaskCase,
        events: Sequence[AgentEvent],
        _result: Any,
        _workspace: Path,
    ) -> bool:
        if case.occurrence_index == 1:
            return False
        benchmark = frozen_cases.get(case.task.id)
        if benchmark is None:
            raise RecurrenceEvaluationRequired(f"no frozen recurrence matcher for {case.task.id}")
        for event in events:
            result = event.result
            if result.success:
                continue
            is_test_event = result.tool_name == "run_tests"
            if result.tool_name == "run_command" and isinstance(
                events, RecurrenceEventStream
            ):
                planned_action = events.planned_action_for(event.action_id)
                is_test_event = (
                    planned_action is not None and is_test_execution(planned_action)
                )
            if not is_test_event:
                continue
            observed = "\n".join(value for value in (result.output, result.error) if value)
            if any(
                signature in observed
                for test_id in benchmark.fail_to_pass
                for signature in (test_id, _pytest_target(test_id))
            ):
                return True
        return False

    return determine


def make_recurrence_matcher_v3_exact_pytest_outcome(
    frozen_cases: Mapping[str, FrozenSWEsmithCase],
) -> RecurrenceEvaluator:
    """Match only an exact frozen target with a failed pytest outcome.

    The v2 matcher intentionally remains unchanged for historical artifacts.  This
    matcher does not infer recurrence from a non-zero aggregate command or from a
    target merely appearing in traceback text: it requires a pytest report line
    whose outcome is ``FAILED`` or ``ERROR`` for the exact frozen target.
    """

    def determine(
        case: BenchmarkTaskCase,
        events: Sequence[AgentEvent],
        _result: Any,
        _workspace: Path,
    ) -> bool:
        if case.occurrence_index == 1:
            return False
        benchmark = frozen_cases.get(case.task.id)
        if benchmark is None:
            raise RecurrenceEvaluationRequired(
                f"no frozen recurrence matcher for {case.task.id}"
            )
        targets = tuple(
            dict.fromkeys(
                target
                for test_id in benchmark.fail_to_pass
                for target in (test_id, _pytest_target(test_id))
                if target
            )
        )
        for event in events:
            result = event.result
            if result.success:
                continue
            is_test_event = result.tool_name == "run_tests"
            if result.tool_name == "run_command" and isinstance(
                events, RecurrenceEventStream
            ):
                planned_action = events.planned_action_for(event.action_id)
                is_test_event = (
                    planned_action is not None and is_test_execution(planned_action)
                )
            if not is_test_event:
                continue
            observed = "\n".join(
                value for value in (result.output, result.error) if value
            )
            if any(
                _pytest_target_has_failed_outcome(observed, target)
                for target in targets
            ):
                return True
        return False

    return determine


def _pytest_target_has_failed_outcome(output: str, target: str) -> bool:
    """Return whether a pytest report line marks ``target`` failed or errored."""
    clean_target = _strip_ansi(target).strip()
    if not clean_target:
        return False
    for raw_line in output.splitlines():
        line = _strip_ansi(raw_line).strip()
        if not line:
            continue
        for outcome in ("FAILED", "ERROR"):
            prefixes = (f"{outcome} ",)
            suffixes = (f" {outcome}",)
            if any(line.startswith(prefix) for prefix in prefixes):
                reported = line[len(outcome) + 1 :].strip()
                if reported == clean_target or reported.startswith(clean_target + " -"):
                    return True
            for suffix in suffixes:
                if not line.endswith(suffix):
                    continue
                reported = line[: -len(suffix)].strip()
                if reported == clean_target:
                    return True
    return False


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)


def _configured_runtime(
    configuration: LoadedExperimentConfiguration,
    baseline_root: Path,
    execution_root: Path,
) -> LoadedExperimentConfiguration:
    config = configuration.config.model_copy(
        update={
            "workspace_baseline_root": str(baseline_root.resolve()),
            "workspace_execution_root": str(execution_root.resolve()),
        }
    )
    return replace(configuration, config=config)


def _verify_baselines(cases: Sequence[BenchmarkTaskCase], baseline_root: Path) -> None:
    missing: list[str] = []
    dirty: list[str] = []
    for case in cases:
        path = (baseline_root / case.task.repository).resolve()
        if not path.is_dir():
            missing.append(str(path))
            continue
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        if head.returncode != 0 or status.returncode != 0 or status.stdout.strip():
            dirty.append(str(path))
    if missing:
        raise RuntimeError("missing clean benchmark baselines: " + ", ".join(missing))
    if dirty:
        raise RuntimeError("benchmark baselines are not clean Git workspaces: " + ", ".join(dirty))


def _looks_like_collection_failure(output: str) -> bool:
    markers = (
        "ERROR collecting",
        "ImportError while loading conftest",
        "ModuleNotFoundError:",
        "No module named",
    )
    return any(marker in output for marker in markers)


def _is_no_cov_argument_error(output: str) -> bool:
    lowered = output.lower()
    return (
        "unrecognized arguments: --no-cov" in lowered
        or "no such option: --no-cov" in lowered
    )


def _pytest_help_supports_no_cov(output: str) -> bool:
    """Detect the real pytest-cov option, rather than trusting probe status."""
    if _is_no_cov_argument_error(output):
        return False
    return any(
        re.search(r"(?<![\w-])--no-cov(?:\s|,|=|$)", line) is not None
        for line in output.splitlines()
    )


def _repository_coverage_enforcement(workspace: Path) -> bool:
    """Detect configured coverage enforcement without discarding pytest options."""
    option_pattern = re.compile(r"(?:^|\s)--cov(?:[-=\s]|$)")
    configured_addopts: list[str] = []
    pyproject = workspace / "pyproject.toml"
    if pyproject.is_file():
        try:
            raw: Any = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            addopts = raw.get("tool", {}).get("pytest", {}).get("ini_options", {}).get(
                "addopts"
            )
            if isinstance(addopts, str):
                configured_addopts.append(addopts)
            elif isinstance(addopts, list):
                configured_addopts.extend(
                    value for value in addopts if isinstance(value, str)
                )
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            pass

    for filename in ("pytest.ini", "tox.ini", "setup.cfg"):
        path = workspace / filename
        try:
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(path, encoding="utf-8")
        except (OSError, configparser.Error):
            continue
        for section in ("pytest", "tool:pytest"):
            if parser.has_option(section, "addopts"):
                configured_addopts.append(parser.get(section, "addopts"))
    return any(option_pattern.search(value) for value in configured_addopts)


def _pytest_target(test_id: str) -> str:
    """Convert SWE-bench's display selector to an executable pytest node ID."""
    if " (" not in test_id or not test_id.endswith(")"):
        return test_id
    method, qualified_class = test_id[:-1].split(" (", maxsplit=1)
    parts = qualified_class.split(".")
    if len(parts) < 2 or not method.strip():
        return test_id
    module = "/".join(parts[:-1]) + ".py"
    return f"{module}::{parts[-1]}::{method.strip()}"


@dataclass(frozen=True)
class PythonInterpreter:
    """A concrete interpreter selected for one frozen benchmark repository."""

    executable: Path
    version: str


@dataclass(frozen=True)
class DependencyInstallPlan:
    """Repository-declared dependency sources plus explicit test overlays."""

    requirement_files: tuple[Path, ...]
    package_extras: tuple[str, ...]
    overlays: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()


_PYTHON_CONSTRAINT_PATTERN = re.compile(
    r"(?:python_requires|requires-python)\s*\+?=\s*[\"'](?P<constraint>[^\"']+)[\"']",
    re.IGNORECASE,
)
_PIP_MANIFEST_DIRECTIVE = re.compile(
    r"^\s*(?:-r|--requirement|-c|--constraint)\s+(?P<reference>\S+)(?:\s+#.*)?$"
)


def _combine_python_constraint_fragments(fragments: Sequence[str]) -> str:
    """Combine setup.py assignments into a valid comma-separated specifier."""
    normalized = [fragment.strip().strip(",") for fragment in fragments if fragment.strip()]
    return ",".join(fragment for fragment in normalized if fragment)


def _python_constraints(repository: Path) -> SpecifierSet:
    """Read the repository's declared Python constraints without modifying it."""
    constraints: list[str] = []
    pyproject = repository / "pyproject.toml"
    if pyproject.is_file():
        try:
            raw: Any = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            requires_python = raw.get("project", {}).get("requires-python")
            if isinstance(requires_python, str) and requires_python.strip():
                constraints.append(requires_python.strip())
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise BenchmarkPreflightError(
                f"could not read Python constraints from {pyproject}: {error}"
            ) from error

    for filename in ("setup.cfg", "setup.py"):
        path = repository / filename
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise BenchmarkPreflightError(
                f"could not read Python constraints from {path}: {error}"
            ) from error
        matches = [
            match.group("constraint").strip()
            for match in _PYTHON_CONSTRAINT_PATTERN.finditer(text)
        ]
        if filename == "setup.py" and matches:
            constraints.append(_combine_python_constraint_fragments(matches))
        else:
            constraints.extend(matches)
        if filename == "setup.cfg":
            parser = configparser.ConfigParser()
            try:
                parser.read_string(text)
            except configparser.Error:
                parser = configparser.ConfigParser()
            if parser.has_option("options", "python_requires"):
                constraints.append(parser.get("options", "python_requires").strip())

    if not constraints:
        return SpecifierSet()
    try:
        return SpecifierSet(",".join(constraints))
    except ValueError as error:
        raise BenchmarkPreflightError(
            f"invalid Python constraint for {repository}: {constraints!r}"
        ) from error


def _interpreter_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "-c", "import platform; print(platform.python_version())"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise BenchmarkPreflightError(
            f"could not inspect Python interpreter {executable}: {completed.stderr.strip()}"
        )
    version = completed.stdout.strip()
    try:
        Version(version)
    except ValueError as error:
        raise BenchmarkPreflightError(
            f"Python interpreter {executable} returned an invalid version {version!r}"
        ) from error
    return version


def _discover_python_interpreters() -> tuple[PythonInterpreter, ...]:
    """Discover installed interpreters; never treat the host environment as a fallback."""
    candidates: list[Path] = []
    if sys.executable:
        candidates.append(Path(sys.executable).resolve())
    launcher = shutil.which("py")
    if launcher is not None:
        listed = subprocess.run(
            [launcher, "-0p"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in listed.stdout.splitlines():
            match = re.search(r"(?P<path>[A-Za-z]:\\[^\r\n]+python(?:\.exe)?)$", line.strip())
            if match:
                candidates.append(Path(match.group("path")).resolve())
    interpreters: list[PythonInterpreter] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            interpreters.append(PythonInterpreter(candidate, _interpreter_version(candidate)))
        except (BenchmarkPreflightError, OSError, subprocess.SubprocessError):
            continue
    return tuple(interpreters)


def _select_python_interpreter(repository: Path, task_id: str) -> PythonInterpreter:
    constraints = _python_constraints(repository)
    available = _discover_python_interpreters()
    return _select_compatible_interpreter(available, constraints, task_id)


def _select_compatible_interpreter(
    available: Sequence[PythonInterpreter],
    constraints: SpecifierSet,
    task_id: str,
    *,
    constraint_description: str | None = None,
) -> PythonInterpreter:
    """Select the highest available interpreter satisfying all constraints."""
    compatible = [item for item in available if Version(item.version) in constraints]
    if not compatible:
        available_text = ", ".join(
            f"{item.executable} ({item.version})" for item in available
        ) or "none"
        constraint_text = constraint_description or str(constraints) or "any Python version"
        raise BenchmarkPreflightError(
            f"no compatible Python interpreter for {task_id}: requires {constraint_text}; "
            f"available: {available_text}"
        )
    compatible.sort(key=lambda item: Version(item.version), reverse=True)
    return compatible[0]


def _dependency_install_plan(
    repository: Path,
    *,
    task_id: str = "",
    overlays: Mapping[str, Sequence[str]] | None = None,
    required_extras: Sequence[str] = (),
) -> DependencyInstallPlan:
    """Collect repository-declared test dependencies and explicit task overlays."""
    requirement_files: set[Path] = {
        path for path in repository.glob("requirements*.txt") if path.is_file()
    }
    extras: set[str] = {extra.strip() for extra in required_extras if extra.strip()}
    environment: dict[str, str] = {}
    tox_path = repository / "tox.ini"
    if tox_path.is_file():
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(tox_path, encoding="utf-8")
        except (OSError, configparser.Error):
            parser = configparser.ConfigParser(interpolation=None)
        for section in parser.sections():
            if section != "testenv":
                continue
            deps = parser.get(section, "deps", fallback="")
            for line in deps.splitlines():
                reference = _local_dependency_manifest_reference(line)
                if reference is not None:
                    requirement_files.add(
                        _resolve_dependency_manifest(repository, tox_path, reference)
                    )
                match = re.search(r"\.\[([^\]]+)\]", line.strip())
                if match:
                    extras.update(part.strip() for part in match.group(1).split(","))
            declared_extras = parser.get(section, "extras", fallback="")
            extras.update(
                part.strip()
                for part in re.split(r"[,\r\n]+", declared_extras)
                if part.strip()
            )
            setenv = parser.get(section, "setenv", fallback="")
            for line in setenv.splitlines():
                if "=" not in line:
                    continue
                key, value = (part.strip() for part in line.split("=", 1))
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    environment[key] = value
    requirement_files = _dependency_manifest_closure(repository, requirement_files)
    overlay_values = tuple(
        str(item).strip()
        for item in (overlays or {}).get(task_id, ())
        if str(item).strip()
    )
    return DependencyInstallPlan(
        tuple(sorted(requirement_files)),
        tuple(sorted(extras)),
        overlay_values,
        tuple(sorted(environment.items())),
    )


def _local_dependency_manifest_reference(line: str) -> str | None:
    """Return a local pip manifest reference, ignoring packages and URLs."""
    match = _PIP_MANIFEST_DIRECTIVE.match(line)
    if match is None:
        return None
    reference = match.group("reference")
    if "://" in reference:
        return None
    return reference


def _resolve_dependency_manifest(
    repository: Path,
    source_manifest: Path,
    reference: str,
) -> Path:
    """Resolve one pip include and enforce the frozen repository boundary."""
    repository_root = repository.resolve()
    referenced = (source_manifest.parent / reference).resolve()
    try:
        referenced.relative_to(repository_root)
    except ValueError as error:
        raise BenchmarkPreflightError(
            "dependency manifest reference escapes the frozen repository: "
            f"source={source_manifest} referenced={reference} resolved={referenced}"
        ) from error
    if not referenced.is_file():
        raise BenchmarkPreflightError(
            "referenced dependency manifest is missing: "
            f"source={source_manifest} referenced={reference} resolved={referenced}"
        )
    return referenced


def _dependency_manifest_closure(
    repository: Path,
    seeds: set[Path],
) -> set[Path]:
    """Expand local -r/--requirement/-c/--constraint references recursively."""
    repository = repository.resolve()
    resolved_seeds: set[Path] = set()
    for path in seeds:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repository)
        except ValueError as error:
            raise BenchmarkPreflightError(
                "dependency manifest reference escapes the frozen repository: "
                f"source={repository} referenced={path} resolved={resolved}"
            ) from error
        if not resolved.is_file():
            raise BenchmarkPreflightError(
                "referenced dependency manifest is missing: "
                f"source={repository} referenced={relative} resolved={resolved}"
            )
        resolved_seeds.add(resolved)
    visited: set[Path] = set()
    pending = list(resolved_seeds)
    while pending:
        manifest = pending.pop()
        if manifest in visited:
            continue
        visited.add(manifest)
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise BenchmarkPreflightError(
                f"could not read dependency manifest {manifest}: {error}"
            ) from error
        for line in lines:
            reference = _local_dependency_manifest_reference(line)
            if reference is None:
                continue
            nested = _resolve_dependency_manifest(repository, manifest, reference)
            if nested not in visited:
                pending.append(nested)
    return visited


def _dependency_fingerprint(
    repository: Path,
    interpreter: PythonInterpreter,
    plan: DependencyInstallPlan,
    *,
    runtime_type: str = "local_venv",
    base_image_digest: str | None = None,
    benchmark_policy: BenchmarkEnvironmentPolicy | None = None,
    benchmark_manifest_path: Path | None = None,
) -> str:
    digest = hashlib.sha256()
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        raise BenchmarkPreflightError(
            f"could not identify the clean benchmark baseline for {repository}: "
            f"{head.stderr.strip()}"
        )
    digest.update(head.stdout.strip().encode("utf-8"))
    digest.update(str(interpreter.executable).encode("utf-8"))
    digest.update(interpreter.version.encode("ascii"))
    digest.update(runtime_type.encode("ascii"))
    if base_image_digest is not None:
        digest.update(base_image_digest.encode("utf-8"))
    if benchmark_policy is not None:
        digest.update(policy_fingerprint_payload(benchmark_policy))
    if benchmark_manifest_path is not None:
        digest.update(benchmark_manifest_path.read_bytes())
    selected_manifests = {path.resolve() for path in plan.requirement_files}
    for path in sorted(
        path
        for path in repository.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and (
            path.name.startswith("requirements")
            and path.suffix == ".txt"
            or path.resolve() in selected_manifests
            or path.name in {"pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "tox.ini"}
        )
    ):
        digest.update(str(path.relative_to(repository)).encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(json.dumps(plan.overlays, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(plan.package_extras, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(plan.environment, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def _run_install(
    python_executable: Path,
    arguments: Sequence[str],
    task_id: str,
    *,
    cwd: Path | None = None,
) -> None:
    completed = subprocess.run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=cwd,
    )
    if completed.returncode != 0:
        raise BenchmarkPreflightError(
            f"could not prepare dependencies for {task_id}: {completed.stderr[-4000:]}"
        )


def _docker_engine_status() -> tuple[bool, str]:
    """Return Docker Linux-engine readiness without starting or installing anything."""
    docker = shutil.which("docker")
    if docker is None:
        return False, "Docker CLI is not installed"
    try:
        completed = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"Docker engine is unavailable: {error}"
    if completed.returncode != 0:
        return False, completed.stderr.strip() or "Docker engine is unavailable"
    return True, completed.stdout.strip() or "Docker engine is available"


def _docker_python_interpreters(
    image: str,
    task_id: str,
) -> tuple[PythonInterpreter, ...]:
    docker = shutil.which("docker") or "docker"
    completed = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            image,
            "sh",
            "-lc",
            (
                "for candidate in python python3 python3.12 python3.11 python3.10 "
                "python3.9 python3.8; do "
                "if command -v \"$candidate\" >/dev/null 2>&1; then "
                "printf '%s\\t' \"$(command -v \"$candidate\")\"; "
                "\"$candidate\" -c 'import platform; print(platform.python_version())'; "
                "fi; done"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise BenchmarkPreflightError(
            f"could not inspect container Python for {task_id} ({image}): "
            f"{completed.stderr.strip()}"
        )
    interpreters: list[PythonInterpreter] = []
    for line in completed.stdout.splitlines():
        executable, _, version = line.partition("\t")
        if not executable or not version:
            continue
        try:
            Version(version)
        except ValueError:
            continue
        interpreter = PythonInterpreter(Path(executable), version)
        if interpreter not in interpreters:
            interpreters.append(interpreter)
    if not interpreters:
        raise BenchmarkPreflightError(
            f"container {image} exposed no usable Python interpreter for {task_id}"
        )
    return tuple(interpreters)


def _docker_base_image_digest(image: str, task_id: str) -> str:
    docker = shutil.which("docker") or "docker"
    completed = subprocess.run(
        [docker, "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if completed.returncode != 0:
        raise BenchmarkPreflightError(
            f"required SWE-smith container image is unavailable for {task_id}: "
            f"{image}: {completed.stderr.strip()}"
        )
    try:
        repo_digests = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        repo_digests = []
    if isinstance(repo_digests, list) and repo_digests:
        return str(repo_digests[0])
    image_id = subprocess.run(
        [docker, "image", "inspect", image, "--format", "{{.Id}}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if image_id.returncode != 0 or not image_id.stdout.strip():
        raise BenchmarkPreflightError(
            f"container image {image} has no usable immutable digest for {task_id}"
        )
    return f"{image}@{image_id.stdout.strip()}"


def _prepared_dockerfile(
    *,
    base_image_digest: str,
    python_executable: str,
    repository: Path,
    plan: DependencyInstallPlan,
) -> str:
    base_image, _, digest = base_image_digest.partition("@")
    if not digest:
        raise BenchmarkPreflightError(
            f"base container image digest is invalid for {repository.name}: {base_image_digest}"
        )
    python = shlex.quote(python_executable)
    pip = f"{python} -m pip install --disable-pip-version-check"
    pip_cache = "--mount=type=cache,target=/root/.cache/pip"
    lines = [
        "# syntax=docker/dockerfile:1.7",
        f"FROM {base_image}@{digest}",
        *[f"ENV {key}={json.dumps(value)}" for key, value in plan.environment],
        "WORKDIR /workspace",
        # Keep the test runner independent from repository dependency changes.
        f"RUN {pip_cache} {pip} pytest",
    ]

    requirement_relatives = [
        requirement_file.resolve().relative_to(repository.resolve()).as_posix()
        for requirement_file in plan.requirement_files
    ]
    for relative in requirement_relatives:
        # Copy requirement manifests before the full source tree so a later
        # package-install failure does not invalidate successful requirements.
        lines.append(f"COPY {shlex.quote(relative)} /workspace/{relative}")
    if requirement_relatives:
        requirement_commands = " && ".join(
            f"{pip} --requirement {shlex.quote('/workspace/' + relative)}"
            for relative in requirement_relatives
        )
        lines.append(f"RUN {pip_cache} {requirement_commands}")
    else:
        lines.append("RUN true")

    # The package layer is intentionally after the repository requirements.
    # Source edits therefore invalidate only this layer and later overlays.
    lines.append("COPY . /workspace")
    extras = ",".join(plan.package_extras)
    package_spec = f"/workspace[{extras}]" if extras else "/workspace"
    lines.append(f"RUN {pip_cache} {pip} {shlex.quote(package_spec)}")
    if plan.overlays:
        overlay_commands = " && ".join(
            f"{pip} {shlex.quote(overlay)}" for overlay in plan.overlays
        )
        lines.append(f"RUN {pip_cache} {overlay_commands}")
    else:
        lines.append("RUN true")
    return "\n".join(lines) + "\n"


def _prepare_docker_image(
    *,
    environment_root: Path,
    task_id: str,
    image_tag: str,
    dockerfile: str,
    repository: Path,
    docker_build_timeout_seconds: int,
) -> None:
    dockerfile_path = environment_root / task_id / image_tag.rsplit(":", 1)[-1] / "Dockerfile"
    dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
    dockerfile_path.write_text(dockerfile, encoding="utf-8")
    docker = shutil.which("docker") or "docker"
    built = subprocess.run(
        [
            docker,
            "build",
            "--pull=false",
            "--tag",
            image_tag,
            "--file",
            str(dockerfile_path),
            str(repository),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=docker_build_timeout_seconds,
    )
    if built.returncode != 0:
        raise BenchmarkPreflightError(
            f"could not prepare immutable container environment for {task_id}: "
            f"{built.stderr[-4000:]}"
        )


def _container_environment(
    environment_root: Path,
    task_id: str,
    repository: Path,
    image: str,
    plan: DependencyInstallPlan,
    benchmark_policy: BenchmarkEnvironmentPolicy,
    benchmark_manifest_path: Path | None = None,
    *,
    prepare: bool = True,
) -> IsolatedTaskEnvironment:
    available, detail = _docker_engine_status()
    if not available:
        raise BenchmarkPreflightError(
            f"container runtime unavailable for {task_id}: {detail}; "
            "no host-Python fallback is permitted"
        )
    docker = shutil.which("docker") or "docker"
    if not prepare:
        return _load_prepared_container_environment(
            environment_root=environment_root,
            task_id=task_id,
            repository=repository,
            image=image,
            plan=plan,
            benchmark_policy=benchmark_policy,
            benchmark_manifest_path=benchmark_manifest_path,
            docker=docker,
        )
    base_image_digest = _docker_base_image_digest(image, task_id)
    repository_constraints = _python_constraints(repository)
    try:
        task_policy = benchmark_policy.task(task_id)
        constraints = effective_python_constraints(task_policy, repository_constraints)
    except BenchmarkEnvironmentConfigurationError as error:
        raise BenchmarkPreflightError(str(error)) from error
    container_interpreters = _docker_python_interpreters(image, task_id)
    container_interpreter = _select_compatible_interpreter(
        container_interpreters,
        constraints,
        task_id,
        constraint_description=(
            f"repository {repository_constraints} and policy "
            f"{task_policy.python_constraint_assertion or 'any Python version'}"
        ),
    )
    container_python_executable = str(container_interpreter.executable).replace("\\", "/")
    version = container_interpreter.version
    interpreter = PythonInterpreter(
        Path(f"docker://{image}{container_python_executable}"),
        version,
    )
    fingerprint = _dependency_fingerprint(
        repository,
        interpreter,
        plan,
        runtime_type="docker",
        base_image_digest=base_image_digest,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=benchmark_manifest_path,
    )
    prepared_image_tag = f"graph-swarm/sprint3b-{task_id.lower()}:{fingerprint}"
    task_environment_root = environment_root / task_id / fingerprint
    task_environment_root.mkdir(parents=True, exist_ok=True)
    marker = task_environment_root / "environment.json"
    docker_path = Path(docker).resolve()
    if marker.is_file():
        try:
            metadata: Any = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            metadata = None
        if (
            isinstance(metadata, dict)
            and cast(dict[str, Any], metadata).get("validated") in (False, True)
            and cast(dict[str, Any], metadata).get("environment_fingerprint") == fingerprint
            and cast(dict[str, Any], metadata).get("python_version") == version
            and cast(dict[str, Any], metadata).get("runtime_type") == "docker"
            and cast(dict[str, Any], metadata).get("container_image") == prepared_image_tag
            and cast(dict[str, Any], metadata).get("container_python_executable")
            == container_python_executable
            and subprocess.run(
                [docker, "image", "inspect", prepared_image_tag],
                check=False,
                capture_output=True,
                timeout=60,
            ).returncode
            == 0
        ):
            return IsolatedTaskEnvironment(
                task_id,
                docker_path,
                marker,
                version,
                fingerprint,
                "docker",
                prepared_image_tag,
                plan,
                repository,
                container_python_executable,
                image,
                benchmark_policy.path,
                benchmark_manifest_path,
            )
    _prepare_docker_image(
        environment_root=environment_root,
        task_id=task_id,
        image_tag=prepared_image_tag,
        dockerfile=_prepared_dockerfile(
            base_image_digest=base_image_digest,
            python_executable=container_python_executable,
            repository=repository,
            plan=plan,
        ),
        repository=repository,
        docker_build_timeout_seconds=benchmark_policy.docker_build_timeout_seconds,
    )
    marker.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "validated": False,
                "python_executable": str(docker_path),
                "python_version": version,
                "environment_fingerprint": fingerprint,
                "runtime_type": "docker",
                "container_image": prepared_image_tag,
                "base_container_image": image,
                "base_image_digest": base_image_digest,
                "container_python_executable": container_python_executable,
                "requirement_files": [str(path) for path in plan.requirement_files],
                "package_extras": list(plan.package_extras),
                "overlays": list(plan.overlays),
                "environment": dict(plan.environment),
                "benchmark_policy_path": str(benchmark_policy.path),
                "benchmark_manifest_path": str(benchmark_manifest_path)
                if benchmark_manifest_path is not None
                else None,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return IsolatedTaskEnvironment(
        task_id,
        docker_path,
        marker,
        version,
        fingerprint,
        "docker",
        prepared_image_tag,
        plan,
        repository,
        container_python_executable,
        image,
        benchmark_policy.path,
        benchmark_manifest_path,
    )


def _load_prepared_container_environment(
    *,
    environment_root: Path,
    task_id: str,
    repository: Path,
    image: str,
    plan: DependencyInstallPlan,
    benchmark_policy: BenchmarkEnvironmentPolicy,
    benchmark_manifest_path: Path | None,
    docker: str,
) -> IsolatedTaskEnvironment:
    """Load an existing prepared image without base-image execution or builds."""
    for marker in sorted((environment_root / task_id).glob("*/environment.json")):
        try:
            raw: Any = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        metadata = cast(dict[str, Any], raw)
        if (
            metadata.get("task_id") != task_id
            or metadata.get("runtime_type") != "docker"
            or metadata.get("validated") not in (False, True)
            or metadata.get("base_container_image") != image
            or not isinstance(metadata.get("base_image_digest"), str)
            or not isinstance(metadata.get("container_python_executable"), str)
            or not isinstance(metadata.get("python_version"), str)
        ):
            continue
        version = str(metadata["python_version"])
        try:
            repository_constraints = _python_constraints(repository)
            task_policy = benchmark_policy.task(task_id)
            constraints = effective_python_constraints(task_policy, repository_constraints)
            if Version(version) not in constraints:
                continue
        except (BenchmarkEnvironmentConfigurationError, BenchmarkPreflightError):
            continue
        interpreter = PythonInterpreter(
            Path(
                "docker://"
                + image
                + str(metadata["container_python_executable"])
            ),
            version,
        )
        try:
            fingerprint = _dependency_fingerprint(
                repository,
                interpreter,
                plan,
                runtime_type="docker",
                base_image_digest=str(metadata["base_image_digest"]),
                benchmark_policy=benchmark_policy,
                benchmark_manifest_path=benchmark_manifest_path,
            )
        except BenchmarkPreflightError:
            continue
        prepared_image_tag = f"graph-swarm/sprint3b-{task_id.lower()}:{fingerprint}"
        if (
            metadata.get("environment_fingerprint") != fingerprint
            or metadata.get("container_image") != prepared_image_tag
            or marker.parent.name != fingerprint
        ):
            continue
        inspected = subprocess.run(
            [docker, "image", "inspect", prepared_image_tag],
            check=False,
            capture_output=True,
            timeout=60,
        )
        if inspected.returncode != 0:
            continue
        return IsolatedTaskEnvironment(
            task_id,
            Path(docker).resolve(),
            marker,
            version,
            fingerprint,
            "docker",
            prepared_image_tag,
            plan,
            repository,
            str(metadata["container_python_executable"]),
            image,
            benchmark_policy.path,
            benchmark_manifest_path,
        )
    raise BenchmarkPreflightError(
        f"{task_id} prepared benchmark environment is missing.\n"
        "Run benchmark_prepare before benchmark_preflight."
    )


def _isolated_python(
    environment_root: Path,
    task_id: str,
    repository: Path | None = None,
    *,
    dependency_overlays: Mapping[str, Sequence[str]] | None = None,
    benchmark_policy: BenchmarkEnvironmentPolicy | None = None,
    benchmark_manifest_path: Path | None = None,
    required_extras: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Create/cache an environment from the selected repository metadata."""
    if repository is None:
        raise BenchmarkPreflightError(
            f"repository metadata is required to prepare the isolated environment for {task_id}"
        )
    repository = repository.expanduser().resolve()
    if not repository.is_dir():
        raise BenchmarkPreflightError(
            f"benchmark repository is missing for {task_id}: {repository}"
        )
    interpreter = _select_python_interpreter(repository, task_id)
    plan = _dependency_install_plan(
        repository,
        task_id=task_id,
        overlays=dependency_overlays,
        required_extras=required_extras,
    )
    fingerprint = _dependency_fingerprint(
        repository,
        interpreter,
        plan,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=benchmark_manifest_path,
    )
    task_environment_root = environment_root / task_id / fingerprint
    environment_dir = task_environment_root / "venv"
    executable = environment_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    validation_marker = task_environment_root / "environment.json"
    if executable.is_file() and validation_marker.is_file():
        try:
            raw_metadata: Any = json.loads(validation_marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raw_metadata = None
        if (
            isinstance(raw_metadata, dict)
            and cast(dict[str, Any], raw_metadata).get("validated") in (False, True)
            and cast(dict[str, Any], raw_metadata).get("environment_fingerprint") == fingerprint
            and cast(dict[str, Any], raw_metadata).get("python_version") == interpreter.version
            and cast(dict[str, Any], raw_metadata).get("python_executable") == str(executable)
            and cast(dict[str, Any], raw_metadata).get("runtime_type") == "local_venv"
        ):
            return executable, validation_marker

    if not executable.is_file():
        task_environment_root.mkdir(parents=True, exist_ok=True)
        created = subprocess.run(
            [str(interpreter.executable), "-m", "venv", str(environment_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            raise BenchmarkPreflightError(
                f"could not create isolated SWE-smith environment for {task_id}: "
                f"{created.stderr.strip()}"
            )
    if not executable.is_file():
        raise BenchmarkPreflightError(f"isolated Python was not created for {task_id}")

    _run_install(executable, ["pytest"], task_id)
    for requirement_file in plan.requirement_files:
        _run_install(executable, ["--requirement", str(requirement_file)], task_id)
    if any((repository / name).is_file() for name in ("pyproject.toml", "setup.py", "setup.cfg")):
        # Install the distribution for declared runtime dependencies.  The
        # objective later places the copied task workspace first on PYTHONPATH,
        # so this baseline package copy cannot hide the mutation under test.
        package_spec = "."
        if plan.package_extras:
            package_spec += "[" + ",".join(plan.package_extras) + "]"
        _run_install(executable, [package_spec], task_id, cwd=repository)
    for overlay in plan.overlays:
        _run_install(executable, [overlay], task_id)
    validation_marker.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "validated": False,
                "python_executable": str(executable),
                "base_python_executable": str(interpreter.executable),
                "python_version": interpreter.version,
                "environment_fingerprint": fingerprint,
                "runtime_type": "local_venv",
                "container_image": None,
                "requirement_files": [str(path) for path in plan.requirement_files],
                "package_extras": list(plan.package_extras),
                "overlays": list(plan.overlays),
                "environment": dict(plan.environment),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return executable, validation_marker


def _prepare_task_environments(
    cases: Sequence[BenchmarkTaskCase],
    *,
    environment_root: Path,
    source_root: Path,
    dependency_overlays: Mapping[str, Sequence[str]] | None = None,
    container_images: Mapping[str, str] | None = None,
    benchmark_policy: BenchmarkEnvironmentPolicy | None = None,
    benchmark_manifest_path: Path | None = None,
    mode: Literal["prepare", "preflight"] = "prepare",
) -> dict[str, IsolatedTaskEnvironment]:
    if benchmark_policy is None:
        raise BenchmarkPreflightError("validated benchmark environment policy is required")
    environments: dict[str, IsolatedTaskEnvironment] = {}
    for case in cases:
        repository = (source_root / case.task.repository).resolve()
        constraints = _python_constraints(repository)
        task_policy = reconcile_task_policy(benchmark_policy, case.task.id, constraints)
        plan = _dependency_install_plan(
            repository,
            task_id=case.task.id,
            overlays=dependency_overlays,
            required_extras=task_policy.required_extras,
        )
        image = (container_images or {}).get(case.task.id)
        if task_policy.runtime == "manifest_container_required":
            if not image:
                raise BenchmarkPreflightError(
                    f"frozen manifest container image is missing for {case.task.id}"
                )
            environments[case.task.id] = _container_environment(
                environment_root,
                case.task.id,
                repository,
                image,
                plan,
                benchmark_policy,
                benchmark_manifest_path,
                prepare=mode == "prepare",
            )
            continue

        if mode == "preflight":
            raise BenchmarkPreflightError(
                f"{case.task.id} has no prepared manifest container environment"
            )
        interpreter = _select_python_interpreter(repository, case.task.id)
        executable, marker = _isolated_python(
            environment_root,
            case.task.id,
            repository,
            dependency_overlays=dependency_overlays,
            benchmark_policy=benchmark_policy,
            benchmark_manifest_path=benchmark_manifest_path,
            required_extras=task_policy.required_extras,
        )
        environments[case.task.id] = IsolatedTaskEnvironment(
            case.task.id,
            executable,
            marker,
            interpreter.version,
            _dependency_fingerprint(
                repository,
                interpreter,
                plan,
                benchmark_policy=benchmark_policy,
                benchmark_manifest_path=benchmark_manifest_path,
            ),
        )
    return environments


def _materialize_workspace(
    *,
    source_root: Path,
    execution_root: Path,
    frozen_cases: Mapping[str, FrozenSWEsmithCase],
    condition: str,
    task: Any,
    workspace_line_ending_policy: str = "git_aware_line_endings_v1",
) -> Path:
    source = (source_root / task.repository).resolve()
    frozen = frozen_cases[task.id]
    destination = execution_root / "GS-E003" / condition / task.id / uuid.uuid4().hex / "workspace"
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, destination, dirs_exist_ok=False)
    materialized = _materialize_git_worktree(destination)
    if materialized.returncode != 0:
        raise RuntimeError(
            f"could not materialize benchmark Git worktree for {task.id}: "
            f"stdout: {materialized.stdout.strip()!r}; "
            f"stderr: {materialized.stderr.strip()!r}"
        )
    reset = subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "reset",
            "--hard",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if reset.returncode != 0:
        raise RuntimeError(
            f"could not canonicalize copied benchmark Git worktree for {task.id}: "
            f"stdout: {reset.stdout.strip()!r}; stderr: {reset.stderr.strip()!r}"
        )
    baseline_status = subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "status",
            "--porcelain",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if baseline_status.returncode != 0:
        raise RuntimeError(
            f"could not verify canonical benchmark baseline for {task.id}: "
            f"stdout: {baseline_status.stdout.strip()!r}; "
            f"stderr: {baseline_status.stderr.strip()!r}"
        )
    if baseline_status.stdout:
        raise RuntimeError(
            f"copied benchmark baseline is dirty for {task.id}: "
            f"paths: {baseline_status.stdout.strip()!r}"
        )
    applied = _apply_patch(destination, frozen.patch)
    if applied.returncode != 0:
        raise RuntimeError(
            f"could not materialize frozen SWE-smith task {task.id}: {applied.stderr.strip()}"
        )
    index_before_cleanup = _index_snapshot(destination)
    if workspace_line_ending_policy == "git_index_consistent_line_endings_v2":
        _remove_eol_only_worktree_noise(destination)
    _assert_materialization_postcondition(
        destination,
        expected_index_snapshot=index_before_cleanup,
        task_id=task.id,
    )
    return destination


def _materialize_git_worktree(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Materialize the untouched tracked baseline through Git attributes."""
    materialized = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "checkout",
            "--force",
            "--",
            ".",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if materialized.returncode == 0:
        _apply_explicit_git_eol_attributes(workspace)
    return materialized


def _apply_explicit_git_eol_attributes(workspace: Path) -> None:
    """Apply only explicit tracked ``eol`` attributes after Git checkout."""
    tracked = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if tracked.returncode != 0:
        raise RuntimeError(f"could not inspect Git tracked files: {tracked.stderr!r}")
    candidates: list[str] = []
    for raw_path in tracked.stdout.split(b"\0"):
        if not raw_path:
            continue
        candidates.append(raw_path.decode("utf-8"))

    for offset in range(0, len(candidates), 200):
        batch = candidates[offset : offset + 200]
        attributes = subprocess.run(
            ["git", "-C", str(workspace), "check-attr", "-z", "eol", "--", *batch],
            check=False,
            capture_output=True,
        )
        if attributes.returncode != 0:
            raise RuntimeError(f"could not inspect Git eol attributes: {attributes.stderr!r}")
        fields = attributes.stdout.split(b"\0")
        for index in range(0, len(fields) - 2, 3):
            relative_path = fields[index].decode("utf-8")
            value = fields[index + 2].decode("utf-8")
            if value not in {"crlf", "lf"}:
                continue
            path = workspace / Path(relative_path)
            content = path.read_bytes()
            normalized = content.replace(b"\r\n", b"\n")
            if value == "crlf":
                normalized = normalized.replace(b"\n", b"\r\n")
            if normalized == content:
                continue
            metadata = path.stat()
            path.write_bytes(normalized)
            os.chmod(path, metadata.st_mode)
            os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))


def _remove_eol_only_worktree_noise(workspace: Path) -> None:
    """Restore only unstaged changes that are byte-level EOL noise.

    The index is authoritative for this correction.  A path is restored only
    when Git's ignore-space-at-EOL comparison says the worktree has no
    substantive change, so benchmark mutations and intentional edits remain
    untouched.  The correction writes only the worktree bytes and uses
    ``update-index --refresh`` solely to refresh Git's worktree comparison;
    it never writes a new index blob.
    """
    subprocess.run(
        ["git", "-C", str(workspace), "update-index", "--really-refresh"],
        check=False,
        capture_output=True,
        text=True,
    )
    changed = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain=v1", "-z"],
        check=False,
        capture_output=True,
        text=True,
    )
    if changed.returncode != 0:
        raise RuntimeError(
            "could not inspect materialized worktree diff: "
            f"{changed.stderr.strip()}"
        )
    status_records = [record for record in changed.stdout.split("\0") if record]
    for record in status_records:
        if len(record) < 4 or record[1] != "M":
            continue
        relative_path = record[3:]
        index_content = subprocess.run(
            ["git", "-C", str(workspace), "show", f":{relative_path}"],
            check=False,
            capture_output=True,
        )
        if index_content.returncode != 0:
            raise RuntimeError(
                f"could not read index content for {relative_path}: "
                f"{index_content.stderr!r}"
            )
        target = workspace / Path(relative_path)
        worktree_bytes = target.read_bytes()
        index_bytes = index_content.stdout
        if b"\0" in worktree_bytes or b"\0" in index_bytes:
            continue
        diff_summary = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--numstat", "--", relative_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if diff_summary.returncode != 0 or diff_summary.stdout.startswith("-\t-\t"):
            continue
        if worktree_bytes.replace(b"\r\n", b"\n") != index_bytes.replace(b"\r\n", b"\n"):
            continue
        attributes = subprocess.run(
            ["git", "-C", str(workspace), "check-attr", "-z", "eol", "--", relative_path],
            check=False,
            capture_output=True,
        )
        if attributes.returncode != 0:
            raise RuntimeError(
                f"could not inspect EOL attributes for {relative_path}: "
                f"{attributes.stderr!r}"
            )
        fields = attributes.stdout.split(b"\0")
        eol = fields[2].decode("utf-8") if len(fields) >= 3 else "unspecified"
        restored_bytes = index_bytes
        if eol == "crlf":
            restored_bytes = index_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        elif eol == "lf":
            restored_bytes = index_bytes.replace(b"\r\n", b"\n")
        metadata = target.stat()
        target.write_bytes(restored_bytes)
        os.chmod(target, metadata.st_mode)
        refreshed = subprocess.run(
            ["git", "-C", str(workspace), "update-index", "--refresh", "--", relative_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if refreshed.returncode not in (0, 1):
            raise RuntimeError(
                f"could not refresh restored worktree path {relative_path}: "
                f"{refreshed.stderr.strip()}"
            )


def _index_snapshot(workspace: Path) -> bytes:
    """Return the index entries without including mutable worktree metadata."""
    result = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "-s", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not snapshot materialized workspace index: {result.stderr!r}"
        )
    return result.stdout


def _worktree_eol_only_paths(workspace: Path) -> list[str]:
    """Find unstaged paths whose only worktree difference is CRLF versus LF."""
    changed = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain=v1", "-z"],
        check=False,
        capture_output=True,
        text=True,
    )
    if changed.returncode != 0:
        raise RuntimeError(
            "could not inspect materialized worktree diff: "
            f"{changed.stderr.strip()}"
        )
    eol_only: list[str] = []
    for record in (record for record in changed.stdout.split("\0") if record):
        if len(record) < 4 or record[1] != "M":
            continue
        relative_path = record[3:]
        index_content = subprocess.run(
            ["git", "-C", str(workspace), "show", f":{relative_path}"],
            check=False,
            capture_output=True,
        )
        if index_content.returncode != 0:
            raise RuntimeError(
                f"could not read index content for {relative_path}: "
                f"{index_content.stderr!r}"
            )
        target = workspace / Path(relative_path)
        if not target.is_file():
            continue
        worktree_bytes = target.read_bytes()
        index_bytes = index_content.stdout
        if b"\0" in worktree_bytes or b"\0" in index_bytes:
            continue
        diff_summary = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--numstat", "--", relative_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if diff_summary.returncode != 0 or diff_summary.stdout.startswith("-\t-\t"):
            continue
        if worktree_bytes.replace(b"\r\n", b"\n") == index_bytes.replace(b"\r\n", b"\n"):
            eol_only.append(relative_path)
    return eol_only


def _assert_materialization_postcondition(
    workspace: Path,
    *,
    expected_index_snapshot: bytes,
    task_id: str,
) -> None:
    """Ensure materialization leaves no EOL-only unstaged worktree delta."""
    actual_index_snapshot = _index_snapshot(workspace)
    if actual_index_snapshot != expected_index_snapshot:
        raise RuntimeError(
            f"materialization changed the Git index for {task_id}; "
            "EOL cleanup must modify worktree bytes only"
        )
    remaining = _worktree_eol_only_paths(workspace)
    if remaining:
        raise RuntimeError(
            f"materialization left EOL-only unstaged changes for {task_id}: "
            + ", ".join(remaining)
        )


def _make_workspace_resolver(
    *,
    source_root: Path,
    execution_root: Path,
    frozen_cases: Mapping[str, FrozenSWEsmithCase],
    condition: ExperimentCondition,
) -> Any:
    def resolve(task: Any) -> Path:
        return _materialize_workspace(
            source_root=source_root,
            execution_root=execution_root,
            frozen_cases=frozen_cases,
            condition=condition.value,
            task=task,
        )

    return resolve


def _verify_frozen_patches(
    cases: Sequence[BenchmarkTaskCase],
    source_root: Path,
    frozen_cases: Mapping[str, FrozenSWEsmithCase],
) -> None:
    for case in cases:
        source = (source_root / case.task.repository).resolve()
        check = _apply_patch(
            source,
            frozen_cases[case.task.id].patch,
            check_only=True,
        )
        if check.returncode != 0:
            raise RuntimeError(
                f"frozen patch cannot be applied for {case.task.id}: {check.stderr.strip()}"
            )


def _apply_patch(
    repository: Path,
    patch: str,
    *,
    check_only: bool = False,
) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        suffix=".patch",
        delete=False,
    ) as handle:
        handle.write(patch)
        patch_path = Path(handle.name)
    try:
        arguments = [
            "git",
            "-C",
            str(repository),
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "apply",
        ]
        if check_only:
            arguments.append("--check")
        arguments.extend(("--3way", "--whitespace=nowarn", str(patch_path)))
        return subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        patch_path.unlink(missing_ok=True)


def _require_validated_environments(
    cases: Sequence[BenchmarkTaskCase],
    environments: Mapping[str, IsolatedTaskEnvironment],
) -> None:
    """Enforce the prepared-and-validated barrier before B0/O1 can run."""
    for case in cases:
        environment = environments.get(case.task.id)
        metadata: Any = None
        if environment is not None and environment.validation_marker is not None:
            try:
                metadata = json.loads(
                    environment.validation_marker.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                metadata = None
        prepared = (
            environment is not None
            and environment.runtime_type == "docker"
            and isinstance(metadata, dict)
            and metadata.get("validated") in (False, True)
            and metadata.get("environment_fingerprint")
            == environment.environment_fingerprint
        )
        if not (
            prepared
            and isinstance(metadata, dict)
            and metadata.get("validated") is True
        ):
            instruction = (
                "Run benchmark_preflight."
                if prepared and isinstance(metadata, dict) and metadata.get("validated") is False
                else "Run benchmark_prepare before benchmark_preflight."
            )
            raise BenchmarkPreflightError(
                f"{case.task.id} benchmark environment is missing or unvalidated.\n"
                + instruction
            )


@dataclass(frozen=True)
class PreparedProviderSmokeContext:
    """One fixed transfer task with its already-prepared execution boundary."""

    configuration: LoadedExperimentConfiguration
    case: BenchmarkTaskCase
    frozen_cases: Mapping[str, FrozenSWEsmithCase]
    environment: IsolatedTaskEnvironment
    objective: FrozenSWEsmithObjective
    recurrence_evaluator: RecurrenceEvaluator
    baseline_root: Path
    execution_root: Path

    def workspace_resolver_for(self, condition: ExperimentCondition) -> Callable[[Task], Path]:
        """Create the existing fresh patched-workspace resolver for a condition."""
        return cast(
            Callable[[Task], Path],
            _make_workspace_resolver(
                source_root=self.baseline_root,
                execution_root=self.execution_root,
                frozen_cases=self.frozen_cases,
                condition=condition,
            ),
        )


def prepare_provider_smoke_context(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    config_path: Path | None = None,
    task_id: str = "GS-T007",
) -> PreparedProviderSmokeContext:
    """Load one validated transfer task without preparing or mutating its runtime."""
    root = project_root.expanduser().resolve()
    baseline = baseline_root.expanduser().resolve()
    execution = execution_root.expanduser().resolve()
    selected_config_path = config_path or root / "configs/experiments/rollout_3a_pilot.yaml"
    configuration = _configured_runtime(
        load_experiment_configuration(selected_config_path, project_root=root),
        baseline,
        execution,
    )
    if configuration.config.conditions != (ExperimentCondition.B0,):
        raise BenchmarkPreflightError(
            "provider smoke config must enable exactly B0; O1 is derived from the same contract"
        )
    try:
        benchmark_policy = load_benchmark_environment_policy(
            root / "configs/research/benchmark_environments.toml"
        )
    except BenchmarkEnvironmentConfigurationError as error:
        raise BenchmarkPreflightError(str(error)) from error

    cases = [
        case
        for case in load_task_cases(
            configuration.task_manifest_path,
            problem_statements_path=configuration.task_problems_path,
        )
        if case.task.id == task_id
    ]
    if len(cases) != 1:
        raise BenchmarkPreflightError(
            f"frozen provider smoke task {task_id} is missing or not unique"
        )
    case = cases[0]
    if case.occurrence_index <= 1:
        raise BenchmarkPreflightError(
            f"provider smoke task {task_id} is not a frozen transfer opportunity"
        )

    _verify_baselines([case], baseline)
    instance_ids = _manifest_instance_ids(configuration.task_manifest_path, [task_id])
    container_images = _manifest_image_names(configuration.task_manifest_path, [task_id])
    frozen_by_instance = load_frozen_swesmith_cases(
        tuple(instance_ids.values()),
        allow_network=False,
    )
    frozen = frozen_by_instance.get(instance_ids[task_id].lower())
    if frozen is None:
        raise BenchmarkPreflightError(f"frozen SWE-smith data is incomplete for {task_id}")
    frozen_cases = {task_id: frozen}
    _verify_frozen_patches([case], baseline, frozen_cases)
    environments = _prepare_task_environments(
        [case],
        environment_root=execution / "sprint3-task-environments",
        source_root=baseline,
        dependency_overlays=TASK_DEPENDENCY_OVERLAYS,
        container_images=container_images,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=configuration.task_manifest_path,
        mode="preflight",
    )
    _require_validated_environments([case], environments)
    environment = environments[task_id]
    if environment.runtime_type != "docker":
        raise BenchmarkPreflightError("provider smoke requires the prepared Docker runtime")
    if not environment.container_image or not environment.container_python_executable:
        raise BenchmarkPreflightError("prepared Docker runtime metadata is incomplete")

    return PreparedProviderSmokeContext(
        configuration=configuration,
        case=case,
        frozen_cases=frozen_cases,
        environment=environment,
        objective=FrozenSWEsmithObjective(frozen_cases, {task_id: environment}),
        recurrence_evaluator=make_recurrence_matcher(frozen_cases),
        baseline_root=baseline,
        execution_root=execution,
    )


def run_prepare(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    artifact_root: Path,
    task_ids: Sequence[str] = TRANSFER_TASKS,
    policy_path: Path | None = None,
    environment_subdirectory: str = "sprint3-task-environments",
    result_subdirectory: str = "sprint3b",
) -> Path:
    """Build or reuse prepared environments without objective work.

    The defaults preserve the frozen Track B preparation contract. Narrow
    development gates may provide their own task order, policy, and isolated
    environment directory while reusing the same SWE-smith machinery.
    """
    requested_task_ids = tuple(task_ids)
    b0_config = _configured_runtime(
        load_experiment_configuration(
            project_root / "configs/experiments/rollout_3a_pilot.yaml",
            project_root=project_root,
        ),
        baseline_root,
        execution_root,
    )
    try:
        selected_policy_path = policy_path or (
            project_root / "configs/research/benchmark_environments.toml"
        )
        if requested_task_ids == TRANSFER_TASKS:
            benchmark_policy = load_benchmark_environment_policy(selected_policy_path)
        else:
            benchmark_policy = load_benchmark_environment_policy(
                selected_policy_path,
                expected_task_order=requested_task_ids,
            )
    except BenchmarkEnvironmentConfigurationError as error:
        raise BenchmarkPreflightError(str(error)) from error
    all_cases = load_task_cases(
        b0_config.task_manifest_path,
        problem_statements_path=b0_config.task_problems_path,
    )
    cases = [case for case in all_cases if case.task.id in requested_task_ids]
    if tuple(case.task.id for case in cases) != requested_task_ids:
        raise BenchmarkPreflightError(
            f"requested benchmark cases are incomplete or out of order: {requested_task_ids!r}"
        )
    _verify_baselines(cases, baseline_root)
    instance_ids = _manifest_instance_ids(
        b0_config.task_manifest_path,
        [case.task.id for case in cases],
    )
    container_images = _manifest_image_names(
        b0_config.task_manifest_path,
        [case.task.id for case in cases],
    )
    # The selectors and mutation patches remain researcher-only. Loading the
    # frozen rows here makes preflight validation independent of network/data
    # downloads while still keeping preparation free of objective execution.
    load_frozen_swesmith_cases(tuple(instance_ids.values()))
    environments = _prepare_task_environments(
        cases,
        environment_root=execution_root / environment_subdirectory,
        source_root=baseline_root,
        dependency_overlays=TASK_DEPENDENCY_OVERLAYS,
        container_images=container_images,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=b0_config.task_manifest_path,
        mode="prepare",
    )
    prepared: list[dict[str, Any]] = []
    for case in cases:
        environment = environments[case.task.id]
        validated = False
        if environment.validation_marker is not None:
            try:
                metadata: Any = json.loads(
                    environment.validation_marker.read_text(encoding="utf-8")
                )
                validated = isinstance(metadata, dict) and metadata.get("validated") is True
            except (OSError, UnicodeError, json.JSONDecodeError):
                validated = False
        prepared.append(
            {
                "task_id": case.task.id,
                "runtime_type": environment.runtime_type,
                "python_version": environment.python_version,
                "python_executable": str(environment.python_executable),
                "container_python_executable": environment.container_python_executable,
                "container_image": environment.container_image,
                "base_container_image": environment.base_container_image,
                "environment_fingerprint": environment.environment_fingerprint,
                "validation_marker": (
                    str(environment.validation_marker)
                    if environment.validation_marker is not None
                    else None
                ),
                "base_image_digest": _marker_value(
                    environment.validation_marker, "base_image_digest"
                ),
                "validated": validated,
            }
        )
    result_root = artifact_root / "GS-E003" / result_subdirectory
    result_root.mkdir(parents=True, exist_ok=True)
    result_path = result_root / (
        "preparation-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    result_path.write_text(
        json.dumps(
            {
                "status": "BENCHMARK_ENVIRONMENTS_PREPARED",
                "provider_calls": 0,
                "b0_launched": False,
                "o1_launched": False,
                "objective_calls": 0,
                "tasks": prepared,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result_path


def run_preflight(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    artifact_root: Path,
) -> Path:
    """Validate prepared T006-T015 environments without installing anything."""
    b0_config = _configured_runtime(
        load_experiment_configuration(
            project_root / "configs/experiments/rollout_3a_pilot.yaml",
            project_root=project_root,
        ),
        baseline_root,
        execution_root,
    )
    try:
        benchmark_policy = load_benchmark_environment_policy(
            project_root / "configs/research/benchmark_environments.toml"
        )
    except BenchmarkEnvironmentConfigurationError as error:
        raise BenchmarkPreflightError(str(error)) from error
    all_cases = load_task_cases(
        b0_config.task_manifest_path,
        problem_statements_path=b0_config.task_problems_path,
    )
    cases = [case for case in all_cases if case.task.id in TRANSFER_TASKS]
    if tuple(case.task.id for case in cases) != TRANSFER_TASKS:
        raise BenchmarkPreflightError("frozen T006-T015 cases are incomplete or out of order")
    _verify_baselines(cases, baseline_root)
    instance_ids = _manifest_instance_ids(
        b0_config.task_manifest_path,
        [case.task.id for case in cases],
    )
    container_images = _manifest_image_names(
        b0_config.task_manifest_path,
        [case.task.id for case in cases],
    )
    environments = _prepare_task_environments(
        cases,
        environment_root=execution_root / "sprint3-task-environments",
        source_root=baseline_root,
        dependency_overlays=TASK_DEPENDENCY_OVERLAYS,
        container_images=container_images,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=b0_config.task_manifest_path,
        mode="preflight",
    )
    frozen_by_instance = load_frozen_swesmith_cases(
        tuple(instance_ids.values()),
        allow_network=False,
    )
    frozen_cases = {
        case.task.id: frozen_by_instance[instance_ids[case.task.id].lower()]
        for case in cases
    }
    _verify_frozen_patches(cases, baseline_root, frozen_cases)
    objective = FrozenSWEsmithObjective(
        frozen_cases,
        environments,
        # Sprint 3B must resolve coverage compatibility from the prepared
        # task environment.  The historical R6 default of ``no_cov`` is not
        # safe when pytest-cov is absent.
        objective_coverage_policy="no_cov",
        coverage_policy_selection_version=OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION,
    )
    observations: list[dict[str, Any]] = []
    for case in cases:
        workspace = _materialize_workspace(
            source_root=baseline_root,
            execution_root=execution_root,
            frozen_cases=frozen_cases,
            condition="preflight",
            task=case.task,
        )
        observation = objective.preflight(case.task, workspace)
        observations.append(
            {
                "task_id": case.task.id,
                "runtime_type": environments[case.task.id].runtime_type,
                "python_version": environments[case.task.id].python_version,
                "python_executable": str(environments[case.task.id].python_executable),
                "container_python_executable": (
                    environments[case.task.id].container_python_executable
                ),
                "environment_fingerprint": environments[case.task.id].environment_fingerprint,
                "status": observation.status,
                "return_code": observation.return_code,
            }
        )
    result_root = artifact_root / "GS-E003" / "sprint3b"
    result_root.mkdir(parents=True, exist_ok=True)
    result_path = result_root / (
        "preflight-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    result_path.write_text(
        json.dumps(
            {
                "status": "READY_FOR_B0_O1",
                "provider_calls": 0,
                "b0_launched": False,
                "o1_launched": False,
                "tasks": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result_path


def run_runtime_smoke(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    artifact_root: Path,
    task_id: str = "GS-T007",
) -> Path:
    """Exercise Track B process tools in one already validated container."""
    if task_id != "GS-T007":
        raise BenchmarkPreflightError(
            f"Sprint 3C-B runtime smoke is frozen to GS-T007: {task_id}"
        )
    b0_config = _configured_runtime(
        load_experiment_configuration(
            project_root / "configs/experiments/rollout_3a_pilot.yaml",
            project_root=project_root,
        ),
        baseline_root,
        execution_root,
    )
    try:
        benchmark_policy = load_benchmark_environment_policy(
            project_root / "configs/research/benchmark_environments.toml"
        )
    except BenchmarkEnvironmentConfigurationError as error:
        raise BenchmarkPreflightError(str(error)) from error

    cases = [
        case
        for case in load_task_cases(
            b0_config.task_manifest_path,
            problem_statements_path=b0_config.task_problems_path,
        )
        if case.task.id == task_id
    ]
    if len(cases) != 1:
        raise BenchmarkPreflightError(f"runtime smoke task {task_id} is missing or not unique")
    case = cases[0]
    _verify_baselines(cases, baseline_root)
    instance_ids = _manifest_instance_ids(b0_config.task_manifest_path, [task_id])
    container_images = _manifest_image_names(b0_config.task_manifest_path, [task_id])
    frozen_by_instance = load_frozen_swesmith_cases(
        tuple(instance_ids.values()),
        allow_network=False,
    )
    frozen = frozen_by_instance.get(instance_ids[task_id].lower())
    if frozen is None:
        raise BenchmarkPreflightError(f"frozen SWE-smith data is incomplete for {task_id}")
    frozen_cases = {task_id: frozen}
    _verify_frozen_patches(cases, baseline_root, frozen_cases)

    # ``mode=preflight`` only loads and inspects an existing image; it cannot
    # enter the preparation/build path. The validation barrier stays ahead of
    # workspace materialization and every agent tool call.
    environments = _prepare_task_environments(
        cases,
        environment_root=execution_root / "sprint3-task-environments",
        source_root=baseline_root,
        dependency_overlays=TASK_DEPENDENCY_OVERLAYS,
        container_images=container_images,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=b0_config.task_manifest_path,
        mode="preflight",
    )
    _require_validated_environments(cases, environments)
    environment = environments[task_id]
    runtime = environment.agent_execution_runtime()
    if runtime.runtime_type != "docker":
        raise BenchmarkPreflightError(f"runtime smoke requires Docker for {task_id}")
    if runtime.container_image != environment.container_image:
        raise BenchmarkPreflightError(f"runtime smoke image contract mismatch for {task_id}")
    container_python = runtime.container_python_executable
    if container_python is None:
        raise BenchmarkPreflightError(f"runtime smoke Python is missing for {task_id}")

    workspace = _materialize_workspace(
        source_root=baseline_root,
        execution_root=execution_root,
        frozen_cases=frozen_cases,
        condition="runtime-smoke",
        task=case.task,
    )
    sentinel = "AGENT_RUNTIME_SMOKE_SENTINEL"
    (workspace / ".agent-runtime-smoke-sentinel").write_text(
        sentinel + "\n",
        encoding="utf-8",
    )
    dependencies = AgentDependencies(
        workspace,
        "GS-E003-runtime-smoke-" + uuid.uuid4().hex,
        task_id,
        task=case.task,
        execution_runtime=runtime,
    )
    runtime_contract = {
        "docker_executable": str(runtime.docker_executable),
        "prepared_image": runtime.container_image,
        "network": "none",
        "mount": f"type=bind,source={workspace.resolve()},target=/workspace",
        "workdir": "/workspace",
        "pythonpath": "/workspace/src:/workspace",
    }
    command_result = run_command(
        dependencies,
        [
            container_python,
            "-c",
            (
                "import os\n"
                "from pathlib import Path\n"
                "import jinja2\n"
                "assert Path.cwd() == Path('/workspace')\n"
                "assert os.environ['PYTHONPATH'] == '/workspace/src:/workspace'\n"
                "print(Path('/workspace/.agent-runtime-smoke-sentinel').read_text())\n"
                "print('JINJA2_SOURCE=' + str(Path(jinja2.__file__).resolve()))\n"
            ),
        ],
        timeout_seconds=120,
    )
    try:
        source_evidence = validate_run_command_evidence(
            command_result,
            sentinel=sentinel,
        )
    except RuntimeSmokeValidationError as error:
        raise BenchmarkPreflightError(str(error)) from error

    expected_selectors = tuple(
        dict.fromkeys(
            (*frozen.fail_to_pass, *(_pytest_target(test_id) for test_id in frozen.fail_to_pass))
        )
    )
    tests_result = run_tests(dependencies, timeout_seconds=900)
    try:
        tests_evidence = validate_mutated_test_failure(tests_result, expected_selectors)
    except RuntimeSmokeValidationError as error:
        raise BenchmarkPreflightError(str(error)) from error

    evidence = {
        "status": "AGENT_RUNTIME_SMOKE_READY",
        "task_id": task_id,
        "runtime_type": environment.runtime_type,
        "prepared_image": runtime.container_image,
        "container_image": runtime.container_image,
        "container_python": container_python,
        "container_python_executable": container_python,
        "environment_fingerprint": environment.environment_fingerprint,
        "workspace": str(workspace),
        "runtime_contract": runtime_contract,
        "run_command": command_result.model_dump(mode="json"),
        "source_import": source_evidence,
        "run_tests": {
            **tests_evidence,
            "result": tests_result.model_dump(mode="json"),
            "exit_code": tests_result.exit_code,
        },
        "provider_calls": 0,
        "b0_launched": False,
        "o1_launched": False,
    }
    return write_runtime_smoke_evidence(artifact_root, evidence)


def _manifest_instance_ids(manifest_path: Path, task_ids: Sequence[str]) -> dict[str, str]:
    target_ids = set(task_ids)
    found: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = cast(dict[str, Any], json.loads(line))
            task_id = record.get("task_id")
            instance_id = record.get("instance_id")
            if task_id in target_ids and isinstance(instance_id, str) and instance_id.strip():
                found[task_id] = instance_id
    if set(found) != target_ids:
        raise RuntimeError("frozen manifest instance IDs are incomplete for T006-T015")
    return found


def _manifest_image_names(manifest_path: Path, task_ids: Sequence[str]) -> dict[str, str]:
    """Load frozen SWE-smith container image names without exposing them to the agent."""
    target_ids = set(task_ids)
    found: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = cast(dict[str, Any], json.loads(line))
            task_id = record.get("task_id")
            image_name = record.get("image_name")
            if task_id in target_ids and isinstance(image_name, str) and image_name.strip():
                found[task_id] = image_name.strip()
    if set(found) != target_ids:
        raise BenchmarkPreflightError(
            "frozen manifest container image names are incomplete for T006-T015"
        )
    return found


def _marker_value(marker: Path | None, key: str) -> Any:
    """Read one optional preparation value without changing marker semantics."""
    if marker is None or not marker.is_file():
        return None
    try:
        raw: Any = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return raw.get(key) if isinstance(raw, dict) else None


def _test_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [line.strip() for line in value.splitlines() if line.strip()]
    if not isinstance(value, list):
        return ()
    items = cast(list[Any], value)
    return tuple(item for item in items if isinstance(item, str) and item.strip())


def _condition_from_workspace(workspace: Path) -> str:
    for condition in ("B0", "O1"):
        if condition in workspace.parts:
            return condition
    return "unknown"


def _write_exclusive(path: Path, content: Any) -> None:
    serialized = (
        content if isinstance(content, str) else json.dumps(content, indent=2, sort_keys=True)
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        if not serialized.endswith("\n"):
            handle.write("\n")
