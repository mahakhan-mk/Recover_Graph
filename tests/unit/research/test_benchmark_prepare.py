from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, NoReturn, cast

import pytest

import experiments.sprint3 as sprint3
from experiments.sprint3 import (
    BenchmarkPreflightError,
    DependencyInstallPlan,
    FrozenSWEsmithCase,
    FrozenSWEsmithObjective,
    IsolatedTaskEnvironment,
)
from graph_swarm.domain.tasks import Task
from graph_swarm.research.benchmark_environments import (
    BenchmarkEnvironmentPolicy,
    TaskEnvironmentPolicy,
)
from graph_swarm.research.runner import BenchmarkTaskCase, LoadedExperimentConfiguration


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        problem_statement="Fix the benchmark issue.",
        family_id="GS-F001",
        repository="swesmith/example__repo.abc12345",
        chronological_index=int(task_id.removeprefix("GS-T")),
    )


def _case(task_id: str) -> BenchmarkTaskCase:
    return BenchmarkTaskCase(task=_task(task_id), occurrence_index=1)


def _policy(tmp_path: Path, task_id: str = "GS-T013") -> BenchmarkEnvironmentPolicy:
    task = TaskEnvironmentPolicy(
        task_id,
        "manifest_container_required",
        "repository",
        "repository",
        False,
        True,
        "Unit-test policy.",
        None,
        (),
    )
    return BenchmarkEnvironmentPolicy(
        tmp_path / "policy.toml",
        (task_id,),
        False,
        "frozen_task_manifest",
        {task_id: task},
    )


def _prepared_marker(
    root: Path,
    *,
    task_id: str = "GS-T013",
    fingerprint: str = "fingerprint",
    validated: bool = False,
) -> Path:
    marker = root / task_id / fingerprint / "environment.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "validated": validated,
                "environment_fingerprint": fingerprint,
                "runtime_type": "docker",
                "container_image": f"graph-swarm/sprint3b-{task_id.lower()}:{fingerprint}",
                "base_container_image": "frozen:image",
                "base_image_digest": "frozen:image@sha256:base",
                "container_python_executable": "/usr/bin/python3.10",
                "python_version": "3.10.12",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return marker


def test_prepared_dockerfile_has_independent_install_boundaries(tmp_path: Path) -> None:
    dockerfile = sprint3._prepared_dockerfile(  # pyright: ignore[reportPrivateUsage]
        base_image_digest="frozen:image@sha256:base",
        python_executable="/usr/bin/python3.10",
        repository=tmp_path,
        plan=DependencyInstallPlan((tmp_path / "requirements.txt",), (), ("trio",)),
    )

    assert dockerfile.count("RUN") >= 4
    assert "pytest" in dockerfile
    assert "--requirement /workspace/requirements.txt" in dockerfile
    assert "RUN --mount=type=cache,target=/root/.cache/pip" in dockerfile


def test_run_prepare_only_builds_environments_and_writes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = tuple(f"GS-T{i:03d}" for i in range(6, 16))
    cases = [_case(task_id) for task_id in task_ids]
    configuration = cast(
        LoadedExperimentConfiguration,
        SimpleNamespace(
            task_manifest_path=tmp_path / "pilot.jsonl",
            task_problems_path=tmp_path / "problems.csv",
        ),
    )
    modes: list[str] = []

    def fake_load_configuration(
        *_args: object,
        **_kwargs: object,
    ) -> LoadedExperimentConfiguration:
        return configuration

    def fake_configured_runtime(
        loaded: LoadedExperimentConfiguration,
        _baseline_root: Path,
        _execution_root: Path,
    ) -> LoadedExperimentConfiguration:
        return loaded

    def fake_load_policy(_path: Path) -> BenchmarkEnvironmentPolicy:
        return _policy(tmp_path)

    def fake_load_cases(
        _manifest_path: Path,
        *,
        problem_statements_path: Path | None = None,
    ) -> list[BenchmarkTaskCase]:
        del problem_statements_path
        return cases

    def fake_verify_baselines(
        _cases: Sequence[BenchmarkTaskCase],
        _baseline_root: Path,
    ) -> None:
        return None

    def fake_manifest_instance_ids(
        _manifest_path: Path,
        _task_ids: Sequence[str],
    ) -> dict[str, str]:
        return {task_id: f"instance-{task_id}" for task_id in task_ids}

    def fake_manifest_images(
        _manifest_path: Path,
        _task_ids: Sequence[str],
    ) -> dict[str, str]:
        return {task_id: "frozen:image" for task_id in task_ids}

    def fake_load_frozen_cases(
        _instance_ids: Sequence[str],
        *,
        allow_network: bool = True,
    ) -> dict[str, FrozenSWEsmithCase]:
        del allow_network
        return {}

    monkeypatch.setattr(
        sprint3,
        "load_experiment_configuration",
        fake_load_configuration,
    )
    monkeypatch.setattr(sprint3, "_configured_runtime", fake_configured_runtime)
    monkeypatch.setattr(
        sprint3,
        "load_benchmark_environment_policy",
        fake_load_policy,
    )
    monkeypatch.setattr(sprint3, "load_task_cases", fake_load_cases)
    monkeypatch.setattr(sprint3, "_verify_baselines", fake_verify_baselines)
    monkeypatch.setattr(sprint3, "_manifest_instance_ids", fake_manifest_instance_ids)
    monkeypatch.setattr(
        sprint3,
        "_manifest_image_names",
        fake_manifest_images,
    )
    monkeypatch.setattr(sprint3, "load_frozen_swesmith_cases", fake_load_frozen_cases)

    def fake_prepare(
        _cases: Sequence[BenchmarkTaskCase],
        *,
        environment_root: Path,
        source_root: Path,
        dependency_overlays: Mapping[str, Sequence[str]] | None = None,
        container_images: Mapping[str, str] | None = None,
        benchmark_policy: BenchmarkEnvironmentPolicy | None = None,
        benchmark_manifest_path: Path | None = None,
        mode: Literal["prepare", "preflight"] = "prepare",
    ) -> dict[str, IsolatedTaskEnvironment]:
        del (
            environment_root,
            source_root,
            dependency_overlays,
            container_images,
            benchmark_policy,
            benchmark_manifest_path,
        )
        modes.append(mode)
        return {
            task_id: IsolatedTaskEnvironment(
                task_id,
                Path("docker"),
                runtime_type="docker",
                container_image="prepared:image",
                environment_fingerprint="fingerprint",
            )
            for task_id in task_ids
        }

    monkeypatch.setattr(sprint3, "_prepare_task_environments", fake_prepare)

    def fail_objective(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("preparation invoked the objective")

    monkeypatch.setattr(
        sprint3,
        "FrozenSWEsmithObjective",
        fail_objective,
    )

    evidence = sprint3.run_prepare(
        project_root=tmp_path,
        baseline_root=tmp_path / "baselines",
        execution_root=tmp_path / "workspaces",
        artifact_root=tmp_path / "results",
    )

    assert modes == ["prepare"]
    assert json.loads(evidence.read_text(encoding="utf-8"))["objective_calls"] == 0


def test_preflight_reuses_unvalidated_prepared_image_without_build_or_base_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "GS-T013"
    marker = _prepared_marker(tmp_path / "environments", task_id=task_id)
    policy = _policy(tmp_path, task_id)
    fingerprint_calls: list[str] = []

    def fake_engine_status() -> tuple[bool, str]:
        return True, "28.5.1"

    def fail_base_image(_image: str, _task_id: str) -> NoReturn:
        pytest.fail("preflight inspected the base image")

    def fail_container_python(_image: str, _task_id: str) -> NoReturn:
        pytest.fail("preflight executed the base image")

    def fake_fingerprint(*_args: object, **_kwargs: object) -> str:
        fingerprint_calls.append("called")
        return "fingerprint"

    def fail_prepare(**_kwargs: object) -> NoReturn:
        pytest.fail("preflight built an image")

    def fake_which(_name: str) -> str:
        return "docker"

    def fake_inspect_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sprint3, "_docker_engine_status", fake_engine_status)
    monkeypatch.setattr(
        sprint3,
        "_docker_base_image_digest",
        fail_base_image,
    )
    monkeypatch.setattr(
        sprint3,
        "_docker_python_interpreters",
        fail_container_python,
    )
    monkeypatch.setattr(
        sprint3,
        "_dependency_fingerprint",
        fake_fingerprint,
    )
    monkeypatch.setattr(
        sprint3,
        "_prepare_docker_image",
        fail_prepare,
    )
    monkeypatch.setattr(sprint3.shutil, "which", fake_which)
    monkeypatch.setattr(sprint3.subprocess, "run", fake_inspect_run)

    environment = sprint3._container_environment(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "environments",
        task_id,
        tmp_path,
        "frozen:image",
        DependencyInstallPlan((), ()),
        policy,
        prepare=False,
    )

    assert environment.validation_marker == marker
    assert fingerprint_calls == ["called"]


def test_preflight_rejects_missing_prepared_image_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_engine_status() -> tuple[bool, str]:
        return True, "28.5.1"

    def fake_which(_name: str) -> str:
        return "docker"

    def fake_fingerprint(*_args: object, **_kwargs: object) -> str:
        return "fingerprint"

    monkeypatch.setattr(sprint3, "_docker_engine_status", fake_engine_status)
    monkeypatch.setattr(sprint3.shutil, "which", fake_which)
    monkeypatch.setattr(
        sprint3,
        "_dependency_fingerprint",
        fake_fingerprint,
    )

    with pytest.raises(BenchmarkPreflightError, match="Run benchmark_prepare"):
        sprint3._container_environment(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "environments",
            "GS-T013",
            tmp_path,
            "frozen:image",
            DependencyInstallPlan((), ()),
            _policy(tmp_path),
            prepare=False,
        )


def test_successful_objective_preflight_marks_prepared_environment_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "environment.json"
    marker.write_text(
        json.dumps(
            {
                "task_id": "GS-T006",
                "validated": False,
                "environment_fingerprint": "fingerprint",
            }
        ),
        encoding="utf-8",
    )
    environment = IsolatedTaskEnvironment(
        "GS-T006",
        Path("python"),
        validation_marker=marker,
        environment_fingerprint="fingerprint",
    )
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_missing.py",), "")},
        {"GS-T006": environment},
    )

    def fake_failed_test_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=1,
            stdout="FAILED test_missing.py",
            stderr="",
        )

    monkeypatch.setattr(
        sprint3.subprocess,
        "run",
        fake_failed_test_run,
    )

    objective.preflight(_task("GS-T006"), tmp_path)

    assert json.loads(marker.read_text(encoding="utf-8"))["validated"] is True


def test_b0_o1_barrier_rejects_unvalidated_environment_before_execution(
    tmp_path: Path,
) -> None:
    marker = _prepared_marker(tmp_path / "environments", validated=False)
    environment = IsolatedTaskEnvironment(
        "GS-T013",
        Path("docker"),
        validation_marker=marker,
        environment_fingerprint="fingerprint",
        runtime_type="docker",
        container_image="graph-swarm/sprint3b-gs-t013:fingerprint",
    )
    case = _case("GS-T013")

    with pytest.raises(BenchmarkPreflightError, match="Run benchmark_preflight"):
        sprint3._require_validated_environments(  # pyright: ignore[reportPrivateUsage]
            [case], {"GS-T013": environment}
        )


def test_b0_o1_barrier_accepts_validated_environment(tmp_path: Path) -> None:
    marker = _prepared_marker(tmp_path / "environments", validated=True)
    environment = IsolatedTaskEnvironment(
        "GS-T013",
        Path("docker"),
        validation_marker=marker,
        environment_fingerprint="fingerprint",
        runtime_type="docker",
    )
    case = _case("GS-T013")

    sprint3._require_validated_environments(  # pyright: ignore[reportPrivateUsage]
        [case], {"GS-T013": environment}
    )
