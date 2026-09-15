from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest

import experiments.sprint3 as sprint3
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.tasks import Task
from graph_swarm.research.benchmark_environments import BenchmarkEnvironmentPolicy
from graph_swarm.research.benchmark_runtime_smoke import (
    RuntimeSmokeValidationError,
    validate_mutated_test_failure,
    validate_run_command_evidence,
    write_runtime_smoke_evidence,
)
from graph_swarm.research.runner import LoadedExperimentConfiguration

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def make_result(
    *,
    success: bool,
    exit_code: int | None,
    output: str | None,
    error: str | None = None,
    tool_name: str = "run_tests",
) -> ActionResult:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return ActionResult(
        action_id="action-001",
        tool_name=tool_name,
        success=success,
        exit_code=exit_code,
        output=output,
        error=error,
        started_at=timestamp,
        completed_at=timestamp,
    )


def test_run_command_evidence_accepts_mounted_sentinel_and_source_import() -> None:
    result = make_result(
        success=True,
        exit_code=0,
        output=(
            "AGENT_RUNTIME_SMOKE_SENTINEL\n"
            "JINJA2_SOURCE=/workspace/src/jinja2/__init__.py\n"
        ),
        tool_name="run_command",
    )

    evidence = validate_run_command_evidence(
        result,
        sentinel="AGENT_RUNTIME_SMOKE_SENTINEL",
    )

    assert evidence["sentinel_visible"] is True
    assert evidence["source_first"] is True
    assert evidence["import_location"] == "/workspace/src/jinja2/__init__.py"


def test_run_command_evidence_rejects_site_packages_import() -> None:
    result = make_result(
        success=True,
        exit_code=0,
        output=(
            "AGENT_RUNTIME_SMOKE_SENTINEL\n"
            "JINJA2_SOURCE=/opt/venv/lib/python3.10/site-packages/jinja2/__init__.py\n"
        ),
        tool_name="run_command",
    )

    with pytest.raises(RuntimeSmokeValidationError, match="workspace source"):
        validate_run_command_evidence(
            result,
            sentinel="AGENT_RUNTIME_SMOKE_SENTINEL",
        )


def test_mutated_run_tests_failure_is_accepted_for_frozen_selector() -> None:
    result = make_result(
        success=False,
        exit_code=1,
        output="FAILED tests/test_api.py::test_frozen_failure",
    )

    evidence = validate_mutated_test_failure(
        result,
        ("tests/test_api.py::test_frozen_failure",),
    )

    assert evidence["expected_failure"] is True
    assert evidence["matched_selectors"] == ("tests/test_api.py::test_frozen_failure",)


@pytest.mark.parametrize(
    ("exit_code", "output"),
    [
        (2, "collected 0 items"),
        (1, "ERROR collecting tests/test_api.py\nModuleNotFoundError: no module named x"),
    ],
)
def test_infrastructure_or_collection_failure_is_rejected(
    exit_code: int,
    output: str,
) -> None:
    result = make_result(success=False, exit_code=exit_code, output=output)

    with pytest.raises(RuntimeSmokeValidationError, match="infrastructure"):
        validate_mutated_test_failure(result, ("tests/test_api.py::test_frozen_failure",))


def test_unvalidated_environment_fails_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "environment.json"
    marker.write_text(
        '{"environment_fingerprint":"fingerprint","validated":false}\n',
        encoding="utf-8",
    )
    environment = sprint3.IsolatedTaskEnvironment(
        "GS-T007",
        Path("docker.exe"),
        validation_marker=marker,
        environment_fingerprint="fingerprint",
        runtime_type="docker",
        container_image="prepared:image",
        container_python_executable="/usr/bin/python3.10",
    )
    materialize_calls = 0
    command_calls = 0
    tests_calls = 0
    prepared_environments = {"GS-T007": environment}

    def verify_baselines(_cases: Sequence[sprint3.BenchmarkTaskCase], _root: Path) -> None:
        return None

    def manifest_instance_ids(_path: Path, _task_ids: Sequence[str]) -> dict[str, str]:
        return {"GS-T007": "instance"}

    def manifest_image_names(_path: Path, _task_ids: Sequence[str]) -> dict[str, str]:
        return {"GS-T007": "image"}

    def load_frozen(
        _instance_ids: Sequence[str],
        *,
        allow_network: bool = True,
    ) -> dict[str, sprint3.FrozenSWEsmithCase]:
        assert allow_network is False
        return {
            "instance": sprint3.FrozenSWEsmithCase(
                "instance",
                ("tests/test_api.py::test_frozen_failure",),
                "patch",
            )
        }

    def verify_frozen_patches(
        _cases: Sequence[sprint3.BenchmarkTaskCase],
        _source_root: Path,
        _frozen_cases: Mapping[str, sprint3.FrozenSWEsmithCase],
    ) -> None:
        return None

    def prepare_environments(
        _cases: Sequence[sprint3.BenchmarkTaskCase],
        *,
        environment_root: Path,
        source_root: Path,
        dependency_overlays: Mapping[str, Sequence[str]] | None = None,
        container_images: Mapping[str, str] | None = None,
        benchmark_policy: BenchmarkEnvironmentPolicy | None = None,
        benchmark_manifest_path: Path | None = None,
        mode: Literal["prepare", "preflight"] = "prepare",
    ) -> dict[str, sprint3.IsolatedTaskEnvironment]:
        del (
            environment_root,
            source_root,
            dependency_overlays,
            container_images,
            benchmark_policy,
            benchmark_manifest_path,
            mode,
        )
        return prepared_environments

    def materialize_workspace(
        *,
        source_root: Path,
        execution_root: Path,
        frozen_cases: Mapping[str, sprint3.FrozenSWEsmithCase],
        condition: str,
        task: Task,
    ) -> Path:
        nonlocal materialize_calls
        del source_root, execution_root, frozen_cases, condition, task
        materialize_calls += 1
        return tmp_path / "workspace"

    def run_command_before_validation(
        _dependencies: AgentDependencies,
        _command: list[str],
        _timeout_seconds: float | int,
        *,
        action_id: str | None = None,
    ) -> ActionResult:
        nonlocal command_calls
        del action_id
        command_calls += 1
        return make_result(success=True, exit_code=0, output="", tool_name="run_command")

    def run_tests_before_validation(
        _dependencies: AgentDependencies,
        _timeout_seconds: float | int = 120,
        *,
        action_id: str | None = None,
    ) -> ActionResult:
        nonlocal tests_calls
        del action_id
        tests_calls += 1
        return make_result(success=True, exit_code=0, output="")

    monkeypatch.setattr(sprint3, "_verify_baselines", verify_baselines)
    monkeypatch.setattr(sprint3, "_manifest_instance_ids", manifest_instance_ids)
    monkeypatch.setattr(sprint3, "_manifest_image_names", manifest_image_names)
    monkeypatch.setattr(sprint3, "load_frozen_swesmith_cases", load_frozen)
    monkeypatch.setattr(sprint3, "_verify_frozen_patches", verify_frozen_patches)
    monkeypatch.setattr(sprint3, "_prepare_task_environments", prepare_environments)
    monkeypatch.setattr(sprint3, "_materialize_workspace", materialize_workspace)
    monkeypatch.setattr(sprint3, "run_command", run_command_before_validation)
    monkeypatch.setattr(sprint3, "run_tests", run_tests_before_validation)

    with pytest.raises(sprint3.BenchmarkPreflightError, match="Run benchmark_preflight"):
        sprint3.run_runtime_smoke(
            project_root=PROJECT_ROOT,
            baseline_root=tmp_path,
            execution_root=tmp_path,
            artifact_root=tmp_path,
        )

    prepared_environments.clear()
    with pytest.raises(sprint3.BenchmarkPreflightError, match="missing or unvalidated"):
        sprint3.run_runtime_smoke(
            project_root=PROJECT_ROOT,
            baseline_root=tmp_path,
            execution_root=tmp_path,
            artifact_root=tmp_path,
        )

    assert materialize_calls == 0
    assert command_calls == 0
    assert tests_calls == 0


def test_runtime_smoke_rejects_non_jinja_transfer_task_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_environment_loads(
        config_path: Path,
        *,
        project_root: Path | None = None,
    ) -> LoadedExperimentConfiguration:
        del config_path, project_root
        raise AssertionError("environment loading must not occur for GS-T008")

    monkeypatch.setattr(sprint3, "load_experiment_configuration", fail_if_environment_loads)

    with pytest.raises(sprint3.BenchmarkPreflightError, match="frozen to GS-T007"):
        sprint3.run_runtime_smoke(
            project_root=tmp_path,
            baseline_root=tmp_path,
            execution_root=tmp_path,
            artifact_root=tmp_path,
            task_id="GS-T008",
        )


def test_runtime_smoke_evidence_is_append_only_and_keeps_zero_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixed_uuid() -> UUID:
        return UUID(int=0)

    monkeypatch.setattr(
        "graph_swarm.research.benchmark_runtime_smoke.uuid4",
        fixed_uuid,
    )
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    payload = {
        "status": "AGENT_RUNTIME_SMOKE_READY",
        "provider_calls": 0,
        "b0_launched": False,
        "o1_launched": False,
    }

    first = write_runtime_smoke_evidence(tmp_path, payload, timestamp=timestamp)

    assert first.name.startswith("runtime-smoke-20260101T000000Z-")
    assert first.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        write_runtime_smoke_evidence(tmp_path, payload, timestamp=timestamp)


def test_runtime_smoke_uses_validated_runtime_without_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "environment.json"
    marker.write_text(
        '{"environment_fingerprint":"fingerprint","validated":true}\n',
        encoding="utf-8",
    )
    environment = sprint3.IsolatedTaskEnvironment(
        "GS-T007",
        Path("docker.exe"),
        validation_marker=marker,
        environment_fingerprint="fingerprint",
        runtime_type="docker",
        container_image="prepared:image",
        container_python_executable="/usr/bin/python3.10",
    )
    prepared_calls: list[dict[str, object]] = []
    captured: dict[str, Any] = {}
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def verify_baselines(_cases: Sequence[sprint3.BenchmarkTaskCase], _root: Path) -> None:
        return None

    def manifest_instance_ids(_path: Path, _task_ids: Sequence[str]) -> dict[str, str]:
        return {"GS-T007": "instance"}

    def manifest_image_names(_path: Path, _task_ids: Sequence[str]) -> dict[str, str]:
        return {"GS-T007": "image"}

    def load_frozen(
        _instance_ids: Sequence[str],
        *,
        allow_network: bool = True,
    ) -> dict[str, sprint3.FrozenSWEsmithCase]:
        assert allow_network is False
        return {
            "instance": sprint3.FrozenSWEsmithCase(
                "instance",
                ("tests/test_api.py::test_frozen_failure",),
                "patch",
            )
        }

    def verify_frozen_patches(
        _cases: Sequence[sprint3.BenchmarkTaskCase],
        _source_root: Path,
        _frozen_cases: Mapping[str, sprint3.FrozenSWEsmithCase],
    ) -> None:
        return None

    def fake_prepare(
        _cases: Sequence[sprint3.BenchmarkTaskCase],
        *,
        environment_root: Path,
        source_root: Path,
        dependency_overlays: Mapping[str, Sequence[str]] | None = None,
        container_images: Mapping[str, str] | None = None,
        benchmark_policy: BenchmarkEnvironmentPolicy | None = None,
        benchmark_manifest_path: Path | None = None,
        mode: Literal["prepare", "preflight"] = "prepare",
    ) -> dict[str, sprint3.IsolatedTaskEnvironment]:
        prepared_calls.append(
            {
                "environment_root": environment_root,
                "source_root": source_root,
                "dependency_overlays": dependency_overlays,
                "container_images": container_images,
                "benchmark_policy": benchmark_policy,
                "benchmark_manifest_path": benchmark_manifest_path,
                "mode": mode,
            }
        )
        return {"GS-T007": environment}

    def fail_if_builds(
        *,
        environment_root: Path,
        task_id: str,
        image_tag: str,
        dockerfile: str,
        repository: Path,
        docker_build_timeout_seconds: int,
    ) -> None:
        del (
            environment_root,
            task_id,
            image_tag,
            dockerfile,
            repository,
            docker_build_timeout_seconds,
        )
        raise AssertionError("runtime smoke must not prepare an image")

    def materialize_workspace(
        *,
        source_root: Path,
        execution_root: Path,
        frozen_cases: Mapping[str, sprint3.FrozenSWEsmithCase],
        condition: str,
        task: Task,
    ) -> Path:
        del source_root, execution_root, frozen_cases, condition, task
        return workspace

    monkeypatch.setattr(sprint3, "_prepare_task_environments", fake_prepare)
    monkeypatch.setattr(sprint3, "_verify_baselines", verify_baselines)
    monkeypatch.setattr(sprint3, "_manifest_instance_ids", manifest_instance_ids)
    monkeypatch.setattr(sprint3, "_manifest_image_names", manifest_image_names)
    monkeypatch.setattr(sprint3, "load_frozen_swesmith_cases", load_frozen)
    monkeypatch.setattr(sprint3, "_verify_frozen_patches", verify_frozen_patches)
    monkeypatch.setattr(sprint3, "_prepare_docker_image", fail_if_builds)
    monkeypatch.setattr(sprint3, "_materialize_workspace", materialize_workspace)

    runtime = environment.agent_execution_runtime()

    def fake_run_command(
        dependencies: AgentDependencies,
        _command: list[str],
        timeout_seconds: float | int,
        *,
        action_id: str | None = None,
    ) -> ActionResult:
        del action_id
        assert timeout_seconds == 120
        assert dependencies.execution_runtime == runtime
        return make_result(
            success=True,
            exit_code=0,
            output=(
                "AGENT_RUNTIME_SMOKE_SENTINEL\n"
                "JINJA2_SOURCE=/workspace/src/jinja2/__init__.py\n"
            ),
            tool_name="run_command",
        )

    def fake_run_tests(
        dependencies: AgentDependencies,
        timeout_seconds: float | int = 120,
        *,
        action_id: str | None = None,
    ) -> ActionResult:
        del action_id
        assert timeout_seconds == 900
        assert dependencies.execution_runtime == runtime
        return make_result(
            success=False,
            exit_code=1,
            output="FAILED tests/test_api.py::test_frozen_failure",
        )

    monkeypatch.setattr(sprint3, "run_command", fake_run_command)
    monkeypatch.setattr(sprint3, "run_tests", fake_run_tests)

    def write_evidence(
        artifact_root: Path,
        payload: dict[str, Any],
        *,
        timestamp: datetime | None = None,
    ) -> Path:
        del artifact_root, timestamp
        captured.update(payload)
        return tmp_path / "evidence.json"

    monkeypatch.setattr(sprint3, "write_runtime_smoke_evidence", write_evidence)

    result_path = sprint3.run_runtime_smoke(
        project_root=PROJECT_ROOT,
        baseline_root=tmp_path,
        execution_root=tmp_path,
        artifact_root=tmp_path,
    )

    assert result_path == tmp_path / "evidence.json"
    assert prepared_calls[0]["mode"] == "preflight"
    assert captured["task_id"] == "GS-T007"
    assert captured["provider_calls"] == 0
    assert captured["b0_launched"] is False
    assert captured["o1_launched"] is False
