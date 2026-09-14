"""Prepare and smoke-test the Track B Sprint 3A B0/O1 runtime.

The module is an experiment harness, not a second Track B runner.  It supplies
the frozen SWE-smith objective boundary and a frozen-test recurrence matcher to
the approved :class:`ExperimentRunner`.
"""

from __future__ import annotations

import argparse
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from experiments.oracle import FrozenOracleResolver
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.run_command import workspace_process_environment
from graph_swarm.domain.behavior import BehaviorChangeEvidence
from graph_swarm.domain.events import AgentEvent
from graph_swarm.research.benchmark_environments import (
    BenchmarkEnvironmentConfigurationError,
    BenchmarkEnvironmentPolicy,
    effective_python_constraints,
    load_benchmark_environment_policy,
    policy_fingerprint_payload,
    reconcile_task_policy,
)
from graph_swarm.research.contracts import ExperimentCondition, ExperimentRunArtifact
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    ExperimentExecution,
    ExperimentRunner,
    LoadedExperimentConfiguration,
    RecurrenceEvaluationRequired,
    load_experiment_configuration,
    load_task_cases,
)

TRANSFER_TASKS = tuple(f"GS-T{i:03d}" for i in range(6, 16))


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


TASK_DEPENDENCY_OVERLAYS: Mapping[str, Sequence[str]] = {}


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
    ) -> None:
        self.cases = cases
        self.environments = environments
        self.observations: list[ObjectiveObservation] = []

    def __call__(self, task: Any, workspace: Path) -> bool:
        case = self.cases[task.id]
        environment = self.environments.get(task.id)
        if environment is None:
            raise BenchmarkPreflightError(
                f"no isolated SWE-smith executable was selected for {task.id}"
            )
        pytest_arguments = (
            "-m",
            "pytest",
            *(_pytest_target(test_id) for test_id in case.fail_to_pass),
            "-q",
        )
        if environment.runtime_type == "docker":
            if not environment.container_image:
                raise BenchmarkPreflightError(
                    f"container image is missing for {task.id}"
                )
            container_python = environment.container_python_executable or "python"
            command = (
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--mount",
                f"type=bind,source={workspace.resolve()},target=/workspace",
                "--workdir",
                "/workspace",
                environment.container_image,
                container_python,
                *pytest_arguments,
            )
        else:
            command = (str(environment.python_executable), *pytest_arguments)
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
                timeout=900,
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
                "mutated test failure"
            )
        self.environments[task.id].mark_validated()
        if len(self.observations) != before + 1:  # pragma: no cover - defensive invariant
            raise BenchmarkPreflightError(
                f"SWE-smith preflight produced invalid evidence for {task.id}"
            )
        return observation


def load_frozen_swesmith_cases(
    instance_ids: Sequence[str],
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
        return cases

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
    return cases


def make_recurrence_matcher(
    frozen_cases: Mapping[str, FrozenSWEsmithCase],
):
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
            if result.tool_name != "run_tests" or result.success:
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
                requirement_match = re.search(r"(?:-r|--requirement)\s+([^\s]+)", line.strip())
                if requirement_match:
                    requirement_file = (repository / requirement_match.group(1)).resolve()
                    if requirement_file.is_file():
                        requirement_files.add(requirement_file)
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
    for path in sorted(
        path
        for path in repository.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and (
            path.name.startswith("requirements")
            and path.suffix == ".txt"
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
    commands = ["set -eu"]
    for key, value in plan.environment:
        commands.append(f"export {key}={shlex.quote(value)}")
    python = shlex.quote(python_executable)
    commands.append(
        f"{python} -m pip install --disable-pip-version-check pytest"
    )
    for requirement_file in plan.requirement_files:
        relative = requirement_file.resolve().relative_to(repository.resolve()).as_posix()
        commands.append(
            f"{python} -m pip install --disable-pip-version-check "
            f"--requirement {shlex.quote('/workspace/' + relative)}"
        )
    extras = ",".join(plan.package_extras)
    package_spec = f"/workspace[{extras}]" if extras else "/workspace"
    commands.append(
        f"{python} -m pip install --disable-pip-version-check {shlex.quote(package_spec)}"
    )
    for overlay in plan.overlays:
        commands.append(
            f"{python} -m pip install --disable-pip-version-check {shlex.quote(overlay)}"
        )
    return "\n".join(
        [
            f"FROM {base_image}@{digest}",
            *[f"ENV {key}={json.dumps(value)}" for key, value in plan.environment],
            "WORKDIR /workspace",
            "COPY . /workspace",
            "RUN " + " && ".join(commands),
        ]
    ) + "\n"


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
) -> IsolatedTaskEnvironment:
    available, detail = _docker_engine_status()
    if not available:
        raise BenchmarkPreflightError(
            f"container runtime unavailable for {task_id}: {detail}; "
            "no host-Python fallback is permitted"
        )
    docker = shutil.which("docker") or "docker"
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
    compatible = (container_interpreter,)
    if not compatible:
        raise BenchmarkPreflightError(
            f"no compatible Python interpreter in container for {task_id}: "
            f"requires {constraints or 'any Python version'}; available: "
            + ", ".join(
                f"{item.executable} ({item.version})" for item in container_interpreters
            )
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
            and cast(dict[str, Any], metadata).get("validated") is True
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
            and cast(dict[str, Any], raw_metadata).get("validated") is True
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
) -> dict[str, IsolatedTaskEnvironment]:
    if benchmark_policy is None:
        raise BenchmarkPreflightError("validated benchmark environment policy is required")
    environments: dict[str, IsolatedTaskEnvironment] = {}
    for case in cases:
        repository = (source_root / case.task.repository).resolve()
        try:
            constraints = _python_constraints(repository)
            task_policy = reconcile_task_policy(benchmark_policy, case.task.id, constraints)
            plan = _dependency_install_plan(
                repository,
                task_id=case.task.id,
                overlays=dependency_overlays,
                required_extras=task_policy.required_extras,
            )
            if task_policy.runtime == "manifest_container_required":
                raise BenchmarkPreflightError(
                    f"benchmark policy requires the manifest container for {case.task.id}"
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
        except BenchmarkPreflightError as local_error:
            image = (container_images or {}).get(case.task.id)
            if not image:
                raise
            try:
                environments[case.task.id] = _container_environment(
                    environment_root,
                    case.task.id,
                    repository,
                    image,
                    plan,
                    benchmark_policy,
                    benchmark_manifest_path,
                )
            except BenchmarkPreflightError as container_error:
                raise BenchmarkPreflightError(
                    f"{local_error}; approved container path also failed: {container_error}"
                ) from container_error
    return environments


def _materialize_workspace(
    *,
    source_root: Path,
    execution_root: Path,
    frozen_cases: Mapping[str, FrozenSWEsmithCase],
    condition: str,
    task: Any,
) -> Path:
    source = (source_root / task.repository).resolve()
    frozen = frozen_cases[task.id]
    destination = execution_root / "GS-E003" / condition / task.id / uuid.uuid4().hex / "workspace"
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, destination, dirs_exist_ok=False)
    refreshed = subprocess.run(
        ["git", "-C", str(destination), "update-index", "--refresh"],
        check=False,
        capture_output=True,
        text=True,
    )
    if refreshed.returncode != 0:
        raise RuntimeError(
            f"could not refresh benchmark workspace index for {task.id}: {refreshed.stderr.strip()}"
        )
    applied = _apply_patch(destination, frozen.patch)
    if applied.returncode != 0:
        raise RuntimeError(
            f"could not materialize frozen SWE-smith task {task.id}: {applied.stderr.strip()}"
        )
    return destination


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
        arguments = ["git", "-C", str(repository), "apply"]
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


def run_preflight(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    artifact_root: Path,
) -> Path:
    """Prepare and preflight T006-T015 without invoking any experiment runner."""
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
    frozen_by_instance = load_frozen_swesmith_cases(tuple(instance_ids.values()))
    frozen_cases = {
        case.task.id: frozen_by_instance[instance_ids[case.task.id].lower()]
        for case in cases
    }
    _verify_frozen_patches(cases, baseline_root, frozen_cases)
    environments = _prepare_task_environments(
        cases,
        environment_root=execution_root / "sprint3-task-environments",
        source_root=baseline_root,
        dependency_overlays=TASK_DEPENDENCY_OVERLAYS,
        container_images=container_images,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=b0_config.task_manifest_path,
    )
    objective = FrozenSWEsmithObjective(frozen_cases, environments)
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


def run_diagnostic(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    artifact_root: Path,
) -> Path:
    """Run chronological B0 then O1 and write one append-only summary."""
    b0_config = _configured_runtime(
        load_experiment_configuration(
            project_root / "configs/experiments/rollout_3a_pilot.yaml",
            project_root=project_root,
        ),
        baseline_root,
        execution_root,
    )
    o1_config = _configured_runtime(
        load_experiment_configuration(
            project_root / "configs/experiments/rollout_3a_o1.yaml",
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
        raise RuntimeError(str(error)) from error
    all_cases = load_task_cases(
        b0_config.task_manifest_path,
        problem_statements_path=b0_config.task_problems_path,
    )
    cases = [case for case in all_cases if case.task.id in TRANSFER_TASKS]
    if tuple(case.task.id for case in cases) != TRANSFER_TASKS:
        raise RuntimeError("frozen T006-T015 cases are incomplete or out of order")
    _verify_baselines(cases, baseline_root)

    instance_ids = _manifest_instance_ids(
        b0_config.task_manifest_path,
        [case.task.id for case in cases],
    )
    container_images = _manifest_image_names(
        b0_config.task_manifest_path,
        [case.task.id for case in cases],
    )
    frozen_cases_by_instance = load_frozen_swesmith_cases(tuple(instance_ids.values()))
    frozen_cases = {
        case.task.id: frozen_cases_by_instance[instance_ids[case.task.id].lower()] for case in cases
    }
    if len(frozen_cases) != len(cases):
        raise RuntimeError("objective SWE-smith data is incomplete for T006-T015")
    _verify_frozen_patches(cases, baseline_root, frozen_cases)
    environments = _prepare_task_environments(
        cases,
        environment_root=execution_root / "sprint3-task-environments",
        source_root=baseline_root,
        dependency_overlays=TASK_DEPENDENCY_OVERLAYS,
        container_images=container_images,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=b0_config.task_manifest_path,
    )
    objective = FrozenSWEsmithObjective(frozen_cases, environments)
    for case in cases:
        preflight_workspace = _materialize_workspace(
            source_root=baseline_root,
            execution_root=execution_root,
            frozen_cases=frozen_cases,
            condition="preflight",
            task=case.task,
        )
        objective.preflight(case.task, preflight_workspace)
    recurrence = make_recurrence_matcher(frozen_cases)
    oracle = FrozenOracleResolver.from_frozen_files(
        o1_config.task_manifest_path,
        o1_config.task_problems_path,
    )

    executions: list[ExperimentExecution] = []
    for configuration, condition, resolver in (
        (b0_config, ExperimentCondition.B0, None),
        (o1_config, ExperimentCondition.O1, oracle),
    ):
        runner = ExperimentRunner(
            configuration,
            objective_evaluator=objective,
            recurrence_evaluator=recurrence,
            oracle_resolver=resolver,
            condition=condition,
            python_executable_resolver=lambda task, _workspace: (
                environments[task.id].python_executable
            ),
            workspace_resolver=_make_workspace_resolver(
                source_root=baseline_root,
                execution_root=execution_root,
                frozen_cases=frozen_cases,
                condition=condition,
            ),
        )
        for case in cases:
            executions.append(runner.run_case(case))

    group = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    summary_root = artifact_root / "GS-E003" / "sprint3" / group
    summary_root.mkdir(parents=True, exist_ok=False)
    summary = build_summary(executions, objective.observations)
    _write_exclusive(summary_root / "summary.json", summary)
    _write_exclusive(
        summary_root / "paired_results.jsonl",
        "\n".join(json.dumps(row, sort_keys=True) for row in summary["paired_tasks"]) + "\n",
    )
    _write_exclusive(
        summary_root / "objective_evaluations.jsonl",
        "\n".join(
            json.dumps(observation.__dict__, default=str, sort_keys=True)
            for observation in objective.observations
        )
        + "\n",
    )
    return summary_root / "summary.json"


def run_smoke(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    artifact_root: Path,
    task_id: str = "GS-T006",
) -> Path:
    """Run exactly one B0/O1 pair after isolated benchmark preflight."""
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
        raise RuntimeError(str(error)) from error
    o1_config = _configured_runtime(
        load_experiment_configuration(
            project_root / "configs/experiments/rollout_3a_o1.yaml",
            project_root=project_root,
        ),
        baseline_root,
        execution_root,
    )
    cases = [
        case
        for case in load_task_cases(
            b0_config.task_manifest_path,
            problem_statements_path=b0_config.task_problems_path,
        )
        if case.task.id == task_id
    ]
    if len(cases) != 1:
        raise RuntimeError(f"smoke task {task_id} is missing or not unique")
    case = cases[0]
    _verify_baselines([case], baseline_root)
    instance_ids = _manifest_instance_ids(b0_config.task_manifest_path, [task_id])
    container_images = _manifest_image_names(b0_config.task_manifest_path, [task_id])
    frozen_by_instance = load_frozen_swesmith_cases(tuple(instance_ids.values()))
    frozen = frozen_by_instance.get(instance_ids[task_id].lower())
    if frozen is None:
        raise RuntimeError(f"objective SWE-smith data is incomplete for {task_id}")
    frozen_cases = {task_id: frozen}
    _verify_frozen_patches([case], baseline_root, frozen_cases)
    environments = _prepare_task_environments(
        [case],
        environment_root=execution_root / "sprint3a-task-environments",
        source_root=baseline_root,
        dependency_overlays=TASK_DEPENDENCY_OVERLAYS,
        container_images=container_images,
        benchmark_policy=benchmark_policy,
        benchmark_manifest_path=b0_config.task_manifest_path,
    )
    objective = FrozenSWEsmithObjective(frozen_cases, environments)
    preflight_workspace = _materialize_workspace(
        source_root=baseline_root,
        execution_root=execution_root,
        frozen_cases=frozen_cases,
        condition="preflight",
        task=case.task,
    )
    preflight = objective.preflight(case.task, preflight_workspace)
    oracle = FrozenOracleResolver.from_frozen_files(
        o1_config.task_manifest_path,
        o1_config.task_problems_path,
    )

    executions: list[ExperimentExecution] = []
    for configuration, condition, resolver in (
        (b0_config, ExperimentCondition.B0, None),
        (o1_config, ExperimentCondition.O1, oracle),
    ):
        runner = ExperimentRunner(
            configuration,
            objective_evaluator=objective,
            recurrence_evaluator=make_recurrence_matcher(frozen_cases),
            oracle_resolver=resolver,
            condition=condition,
            python_executable_resolver=lambda task, _workspace: (
                environments[task.id].python_executable
            ),
            workspace_resolver=_make_workspace_resolver(
                source_root=baseline_root,
                execution_root=execution_root,
                frozen_cases=frozen_cases,
                condition=condition,
            ),
        )
        executions.append(runner.run_case(case))

    group = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    summary_root = artifact_root / "GS-E003" / "sprint3a" / group
    summary_root.mkdir(parents=True, exist_ok=False)
    summary = build_summary(executions, objective.observations)
    summary.update(
        {
            "sprint": "3A",
            "smoke_task": task_id,
            "preflight": preflight.__dict__,
            "gate_b1_evaluation": "deferred_to_sprint3b",
        }
    )
    _write_summary_bundle(summary_root, summary, objective.observations)
    return summary_root / "summary.json"


def _write_summary_bundle(
    summary_root: Path,
    summary: Mapping[str, Any],
    observations: Sequence[ObjectiveObservation],
) -> None:
    _write_exclusive(summary_root / "summary.json", summary)
    _write_exclusive(
        summary_root / "paired_results.jsonl",
        "\n".join(json.dumps(row, sort_keys=True) for row in summary["paired_tasks"]) + "\n",
    )
    _write_exclusive(
        summary_root / "objective_evaluations.jsonl",
        "\n".join(
            json.dumps(observation.__dict__, default=str, sort_keys=True)
            for observation in observations
        )
        + "\n",
    )


def build_summary(
    executions: Sequence[ExperimentExecution],
    observations: Sequence[ObjectiveObservation],
) -> dict[str, Any]:
    by_condition = {
        condition: [
            execution for execution in executions if execution.artifact.condition.value == condition
        ]
        for condition in ("B0", "O1")
    }
    opportunities = sum(
        execution.artifact.chronological_index >= 6 for execution in by_condition["B0"]
    )
    conditions_summary: dict[str, Any] = {}
    for condition, condition_runs in by_condition.items():
        repeated = sum(run.artifact.known_failure_repeated for run in condition_runs)
        condition_observations = [
            observation for observation in observations if observation.condition == condition
        ]
        conditions_summary[condition] = {
            "recurrence_opportunities": opportunities,
            "repeated_failures": repeated,
            "rfr": repeated / opportunities if opportunities else None,
            "task_success_count": sum(run.artifact.task_success for run in condition_runs),
            "task_success_rate": (
                sum(run.artifact.task_success for run in condition_runs) / len(condition_runs)
                if condition_runs
                else None
            ),
            "advice_received": sum(
                run.artifact.advice_received is not None for run in condition_runs
            ),
            "behavior_changed": sum(
                any(item.behavior_changed for item in run.dependencies.behavior_evidence)
                for run in condition_runs
            ),
            "repeated_failure_despite_advice": sum(
                run.artifact.advice_received is not None and run.artifact.known_failure_repeated
                for run in condition_runs
            ),
            "tool_calls": sum(run.artifact.tool_calls for run in condition_runs),
            "retries": sum(run.artifact.retries for run in condition_runs),
            "input_tokens": sum(run.artifact.input_tokens or 0 for run in condition_runs),
            "output_tokens": sum(run.artifact.output_tokens or 0 for run in condition_runs),
            "latency_ms": sum(run.artifact.latency_ms or 0 for run in condition_runs),
            "model_or_infrastructure_failures": sum(
                run.error is not None for run in condition_runs
            ),
            "model_failures": sum(_error_category(run.error) == "model" for run in condition_runs),
            "agent_control_failures": sum(
                _error_category(run.error) == "agent_control" for run in condition_runs
            ),
            "runner_infrastructure_failures": sum(
                _error_category(run.error) == "infrastructure" for run in condition_runs
            ),
            "objective_test_failures": sum(
                observation.status == "test_failure" for observation in condition_observations
            ),
            "objective_infrastructure_failures": sum(
                observation.status == "objective_infrastructure_failure"
                for observation in condition_observations
            ),
        }

    paired: list[dict[str, Any]] = []
    by_task: dict[str, dict[str, ExperimentExecution]] = {}
    for execution in executions:
        by_task.setdefault(execution.artifact.task_id, {})[execution.artifact.condition.value] = (
            execution
        )
    for task_id, pair in sorted(by_task.items()):
        b0 = pair["B0"]
        o1 = pair["O1"]
        advice = o1.artifact.advice_received
        changed = any(item.behavior_changed for item in o1.dependencies.behavior_evidence)
        paired.append(
            {
                "task_id": task_id,
                "b0_run_id": b0.artifact.run_id,
                "o1_run_id": o1.artifact.run_id,
                "b0_repeated": b0.artifact.known_failure_repeated,
                "o1_repeated": o1.artifact.known_failure_repeated,
                "b0_success": b0.artifact.task_success,
                "o1_success": o1.artifact.task_success,
                "o1_advice_delivered": advice is not None,
                "o1_advice": None if advice is None else advice.recovery_summary,
                "o1_behavior_changed": changed,
                "b0_tool_calls": b0.artifact.tool_calls,
                "o1_tool_calls": o1.artifact.tool_calls,
                "b0_retries": b0.artifact.retries,
                "o1_retries": o1.artifact.retries,
                "b0_input_tokens": b0.artifact.input_tokens,
                "o1_input_tokens": o1.artifact.input_tokens,
                "b0_output_tokens": b0.artifact.output_tokens,
                "o1_output_tokens": o1.artifact.output_tokens,
                "b0_latency_ms": b0.artifact.latency_ms,
                "o1_latency_ms": o1.artifact.latency_ms,
                "b0_error": _error_name(b0.error),
                "o1_error": _error_name(o1.error),
            }
        )
    objective_statuses = {
        status: sum(observation.status == status for observation in observations)
        for status in sorted({observation.status for observation in observations})
    }
    return {
        "experiment_id": "GS-E003",
        "sprint": "3",
        "conditions": conditions_summary,
        "objective_evaluation_statuses": objective_statuses,
        "paired_tasks": paired,
    }


def write_corrected_summary(summary_group: Path) -> Path:
    """Rebuild analysis from immutable run evidence without rerunning the model."""
    executions = _load_recorded_executions(summary_group.parents[2])
    objective_observations = [
        ObjectiveObservation(
            task_id=str(record["task_id"]),
            condition=str(record["condition"]),
            command=tuple(record["command"]),
            return_code=record["return_code"],
            status=str(record["status"]),
            duration_seconds=float(record["duration_seconds"]),
            stdout=str(record["stdout"]),
            stderr=str(record["stderr"]),
        )
        for line in (summary_group / "objective_evaluations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if (record := json.loads(line))
    ]
    corrected_group = summary_group.parent / (summary_group.name + "-corrected-" + uuid.uuid4().hex)
    corrected_group.mkdir(parents=True, exist_ok=False)
    summary = build_summary(executions, objective_observations)
    _write_exclusive(corrected_group / "summary.json", summary)
    _write_exclusive(
        corrected_group / "paired_results.jsonl",
        "\n".join(json.dumps(row, sort_keys=True) for row in summary["paired_tasks"]) + "\n",
    )
    _write_exclusive(
        corrected_group / "objective_evaluations.jsonl",
        "\n".join(
            json.dumps(observation.__dict__, default=str, sort_keys=True)
            for observation in objective_observations
        )
        + "\n",
    )
    return corrected_group / "summary.json"


def _load_recorded_executions(results_root: Path) -> list[ExperimentExecution]:
    executions: list[ExperimentExecution] = []
    for artifact_path in sorted(results_root.glob("GS-E003/*/*/*/artifact.json")):
        artifact = ExperimentRunArtifact.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        raw_path = artifact_path.with_name("raw_evidence.json")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        dependencies = AgentDependencies(
            Path("recorded-workspace"),
            artifact.run_id,
            artifact.task_id,
        )
        dependencies.behavior_evidence = [
            BehaviorChangeEvidence.model_validate(item) for item in raw.get("behavior_evidence", [])
        ]
        error = (
            None
            if raw.get("error") is None
            else RecordedExecutionError(
                str(raw["error"]),
                str(raw.get("error_message") or raw["error"]),
            )
        )
        executions.append(
            ExperimentExecution(
                artifact=artifact,
                artifact_path=artifact_path,
                raw_evidence_path=raw_path,
                dependencies=dependencies,
                agent_result=None,
                error=error,
                step_database_path=Path(raw["step_database"]),
            )
        )
    return executions


def _error_name(error: Exception | None) -> str | None:
    if error is None:
        return None
    return str(getattr(error, "error_type", type(error).__name__))


def _error_category(error: Exception | None) -> str | None:
    name = _error_name(error)
    if name is None:
        return None
    if name.endswith("HTTPError") or name.endswith("APIError"):
        return "model"
    if name in {"UnexpectedModelBehavior", "UsageLimitExceeded"}:
        return "agent_control"
    return "infrastructure"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("research/evidence/results"),
    )
    parser.add_argument("--task-id", default="GS-T006")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    summary_path = run_smoke(
        project_root=project_root,
        baseline_root=args.baseline_root,
        execution_root=args.execution_root,
        artifact_root=(
            args.artifact_root
            if args.artifact_root.is_absolute()
            else project_root / args.artifact_root
        ),
        task_id=args.task_id,
    )
    print(summary_path)


if __name__ == "__main__":
    main()
