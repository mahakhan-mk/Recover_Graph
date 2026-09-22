from __future__ import annotations

import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import experiments.sprint3 as sprint3
from experiments.sprint3 import (
    BenchmarkPreflightError,
    FrozenSWEsmithCase,
    FrozenSWEsmithObjective,
    IsolatedTaskEnvironment,
    make_recurrence_matcher,
)
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.tasks import Task
from graph_swarm.research.benchmark_environments import (
    BenchmarkEnvironmentConfigurationError,
    BenchmarkEnvironmentPolicy,
    TaskEnvironmentPolicy,
    load_benchmark_environment_policy,
)
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    RecurrenceEvaluationRequired,
)


def test_prepare_docker_image_uses_policy_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeouts: list[object] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        observed_timeouts.append(kwargs["timeout"])
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(sprint3.subprocess, "run", fake_run)

    sprint3._prepare_docker_image(  # pyright: ignore[reportPrivateUsage]
        environment_root=tmp_path / "environments",
        task_id="GS-T014",
        image_tag="example:image",
        dockerfile="FROM scratch\n",
        repository=tmp_path,
        docker_build_timeout_seconds=7200,
    )

    assert observed_timeouts == [7200]


def _case(occurrence_index: int) -> BenchmarkTaskCase:
    return BenchmarkTaskCase(
        task=Task(
            id="GS-T006",
            problem_statement="Fix the benchmark issue.",
            family_id="GS-F001",
            repository="swesmith/example__repo.abc12345",
            chronological_index=6,
        ),
        occurrence_index=occurrence_index,
    )


def _test_event(output: str | None, *, success: bool = False) -> AgentEvent:
    started = datetime.now(UTC)
    result = ActionResult(
        action_id="action-1",
        tool_name="run_tests",
        success=success,
        exit_code=0 if success else 1,
        output=output,
        started_at=started,
        completed_at=started,
    )
    return AgentEvent(
        event_id="event-1",
        run_id="run-1",
        task_id="GS-T006",
        action_id="action-1",
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=started,
    )


def test_sprint3_recurrence_requires_the_frozen_failure_signature() -> None:
    frozen = {
        "GS-T006": FrozenSWEsmithCase(
            "example__repo.abc12345",
            ("test_historical (example.test_recurrence.TestCase)",),
            "",
        )
    }
    matcher = make_recurrence_matcher(frozen)

    assert (
        matcher(
            _case(2),
            [_test_event("FAILED example/test_recurrence.py::TestCase::test_historical")],
            None,
            Path("workspace"),
        )
        is True
    )
    assert (
        matcher(
            _case(2),
            [_test_event("FAILED example/test_recurrence.py::test_unrelated")],
            None,
            Path("workspace"),
        )
        is False
    )
    assert matcher(_case(2), [_test_event(None)], None, Path("workspace")) is False
    assert (
        matcher(
            _case(1),
            [_test_event("FAILED example/test_recurrence.py::TestCase::test_historical")],
            None,
            Path("workspace"),
        )
        is False
    )


def test_sprint3_recurrence_missing_frozen_evidence_fails_explicitly() -> None:
    matcher = make_recurrence_matcher({})

    with pytest.raises(RecurrenceEvaluationRequired):
        matcher(_case(2), [], None, Path("workspace"))


def test_benchmark_preflight_requires_a_task_isolated_executable(tmp_path: Path) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_missing.py",), "")},
        {},
    )

    with pytest.raises(BenchmarkPreflightError, match="isolated SWE-smith"):
        objective.preflight(task_case.task, tmp_path)


def test_objective_collection_failure_is_infrastructure_evidence(tmp_path: Path) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_missing.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path(sys.executable))},
    )

    assert objective(task_case.task, tmp_path) is False
    assert objective.observations[-1].status == "objective_infrastructure_failure"


def test_missing_repository_dependency_cannot_pass_preflight(tmp_path: Path) -> None:
    (tmp_path / "test_dependency.py").write_text(
        "import package_that_is_not_declared_or_installed\n\n"
        "def test_dependency():\n    assert True\n",
        encoding="utf-8",
    )
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_dependency.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path(sys.executable))},
    )

    with pytest.raises(BenchmarkPreflightError, match="preflight failed"):
        objective.preflight(task_case.task, tmp_path)


def test_python_selection_uses_repository_constraint_not_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.11,<3.12"\n',
        encoding="utf-8",
    )
    compatible = sprint3.PythonInterpreter(Path("python311"), "3.11.9")
    host = sprint3.PythonInterpreter(Path("python313"), "3.13.7")
    monkeypatch.setattr(sprint3, "_discover_python_interpreters", lambda: (host, compatible))

    selected = sprint3._select_python_interpreter(tmp_path, "GS-T013")  # pyright: ignore[reportPrivateUsage]

    assert selected == compatible


def test_python_selection_reports_missing_compatible_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.9,<3.13"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sprint3,
        "_discover_python_interpreters",
        lambda: (sprint3.PythonInterpreter(Path("python313"), "3.13.7"),),
    )

    with pytest.raises(BenchmarkPreflightError, match="no compatible Python interpreter"):
        sprint3._select_python_interpreter(tmp_path, "GS-T012")  # pyright: ignore[reportPrivateUsage]


def test_setup_py_constraints_are_combined_as_valid_specifiers(tmp_path: Path) -> None:
    (tmp_path / "setup.py").write_text(
        'python_requires = ">=3.8.1"\n'
        'python_requires += "<3.12"\n',
        encoding="utf-8",
    )

    constraints = sprint3._python_constraints(tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert "3.11" in constraints
    assert "3.12" not in constraints


def test_container_runtime_failure_does_not_fall_back_to_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_docker(_name: str) -> None:
        return None

    monkeypatch.setattr(sprint3.shutil, "which", missing_docker)

    available, detail = sprint3._docker_engine_status()  # pyright: ignore[reportPrivateUsage]

    assert available is False
    assert "not installed" in detail


def test_dependency_plan_reads_nested_requirements_and_tox_extras(tmp_path: Path) -> None:
    (tmp_path / "requirements").mkdir()
    requirement = tmp_path / "requirements" / "tests.txt"
    requirement.write_text("trio\n", encoding="utf-8")
    (tmp_path / "tox.ini").write_text(
        "[testenv]\n"
        "deps =\n"
        "    -r requirements/tests.txt\n"
        "    .[test]\n"
        "extras = test\n"
        "setenv =\n"
        "    TOX_RED = 1\n",
        encoding="utf-8",
    )

    plan = sprint3._dependency_install_plan(tmp_path, task_id="GS-T007")  # pyright: ignore[reportPrivateUsage]

    assert plan.requirement_files == (requirement,)
    assert plan.package_extras == ("test",)
    assert plan.environment == (("TOX_RED", "1"),)


def test_dependency_plan_recursively_closes_requirement_and_constraint_manifests(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements"
    nested = requirements / "nested"
    nested.mkdir(parents=True)
    root = requirements / "requirements-tests.txt"
    first = requirements / "requirements.txt"
    second = nested / "second.txt"
    root.write_text(
        "--requirement requirements.txt\n-c nested/constraints-one.txt\n",
        encoding="utf-8",
    )
    first.write_text(
        "pytest\n--requirement nested/second.txt\n--constraint nested/constraints-two.txt\n",
        encoding="utf-8",
    )
    second.write_text("trio\n", encoding="utf-8")
    (nested / "constraints-one.txt").write_text(
        "--constraint constraints-two.txt\n", encoding="utf-8"
    )
    (nested / "constraints-two.txt").write_text("urllib3<3\n", encoding="utf-8")
    (tmp_path / "tox.ini").write_text(
        "[testenv]\ndeps = -r requirements/requirements-tests.txt\n",
        encoding="utf-8",
    )

    plan = sprint3._dependency_install_plan(tmp_path, task_id="GS-T001")  # pyright: ignore[reportPrivateUsage]

    assert plan.requirement_files == (
        nested / "constraints-one.txt",
        nested / "constraints-two.txt",
        second,
        root,
        first,
    )


def test_dependency_manifest_closure_deduplicates_and_terminates_cycles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "requirements.txt"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    root.write_text("-r first.txt\n-r first.txt\n", encoding="utf-8")
    first.write_text("-r second.txt\n", encoding="utf-8")
    second.write_text("-r first.txt\n", encoding="utf-8")

    plan = sprint3._dependency_install_plan(tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert plan.requirement_files == (first, root, second)


@pytest.mark.parametrize("directive", ("-r", "--requirement", "-c", "--constraint"))
def test_dependency_manifest_missing_reference_fails_before_docker(
    tmp_path: Path,
    directive: str,
) -> None:
    source = tmp_path / "requirements.txt"
    source.write_text(f"{directive} missing.txt\n", encoding="utf-8")

    with pytest.raises(BenchmarkPreflightError, match="source=.*requirements.txt.*missing.txt"):
        sprint3._dependency_install_plan(tmp_path)  # pyright: ignore[reportPrivateUsage]


def test_dependency_manifest_reference_cannot_escape_repository(tmp_path: Path) -> None:
    source = tmp_path / "requirements.txt"
    source.write_text("-r ../outside.txt\n", encoding="utf-8")

    with pytest.raises(BenchmarkPreflightError, match="escapes the frozen repository"):
        sprint3._dependency_install_plan(tmp_path)  # pyright: ignore[reportPrivateUsage]


def test_prepared_dockerfile_copies_recursive_manifests_before_pip_install(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    tests_file = requirements / "requirements-tests.txt"
    base_file = requirements / "requirements.txt"
    tests_file.write_text("-r requirements.txt\n", encoding="utf-8")
    base_file.write_text("pytest\n", encoding="utf-8")
    (tmp_path / "tox.ini").write_text(
        "[testenv]\ndeps = -r requirements/requirements-tests.txt\n",
        encoding="utf-8",
    )
    plan = sprint3._dependency_install_plan(tmp_path)  # pyright: ignore[reportPrivateUsage]

    dockerfile = sprint3._prepared_dockerfile(  # pyright: ignore[reportPrivateUsage]
        base_image_digest="frozen:image@sha256:base",
        python_executable="/usr/bin/python3.12",
        repository=tmp_path,
        plan=plan,
    )

    tests_copy = (
        "COPY requirements/requirements-tests.txt /workspace/requirements/requirements-tests.txt"
    )
    base_copy = "COPY requirements/requirements.txt /workspace/requirements/requirements.txt"
    pip_install = "--requirement /workspace/requirements/requirements-tests.txt"
    assert tests_copy in dockerfile
    assert base_copy in dockerfile
    assert dockerfile.index(tests_copy) < dockerfile.index(pip_install)
    assert dockerfile.index(base_copy) < dockerfile.index(pip_install)


def test_nested_dependency_manifest_changes_fingerprint(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    nested = tmp_path / "requirements" / "requirements.txt"
    nested.parent.mkdir()
    root = tmp_path / "requirements" / "requirements-tests.txt"
    root.write_text("-r requirements.txt\n", encoding="utf-8")
    nested.write_text("pytest==8.0.0\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "baseline"],
        check=True,
        capture_output=True,
    )
    plan = sprint3._dependency_install_plan(tmp_path)  # pyright: ignore[reportPrivateUsage]
    interpreter = sprint3.PythonInterpreter(Path("python312"), "3.12.8")
    first = sprint3._dependency_fingerprint(  # pyright: ignore[reportPrivateUsage]
        tmp_path, interpreter, plan
    )
    nested.write_text("pytest==8.1.0\n", encoding="utf-8")
    second = sprint3._dependency_fingerprint(  # pyright: ignore[reportPrivateUsage]
        tmp_path, interpreter, plan
    )
    assert first != second


def test_dependency_overlay_is_task_scoped(tmp_path: Path) -> None:
    overlays = {"GS-T007": ("trio",)}

    selected = sprint3._dependency_install_plan(  # pyright: ignore[reportPrivateUsage]
        tmp_path,
        task_id="GS-T007",
        overlays=overlays,
    )
    other = sprint3._dependency_install_plan(  # pyright: ignore[reportPrivateUsage]
        tmp_path,
        task_id="GS-T006",
        overlays=overlays,
    )

    assert selected.overlays == ("trio",)
    assert other.overlays == ()


def test_dependency_fingerprint_includes_interpreter_and_overlay(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "baseline"],
        check=True,
        capture_output=True,
    )
    plan = sprint3.DependencyInstallPlan((), (), ("trio",))

    first = sprint3._dependency_fingerprint(  # pyright: ignore[reportPrivateUsage]
        tmp_path,
        sprint3.PythonInterpreter(Path("python312"), "3.12.8"),
        plan,
    )
    second = sprint3._dependency_fingerprint(  # pyright: ignore[reportPrivateUsage]
        tmp_path,
        sprint3.PythonInterpreter(Path("python311"), "3.11.9"),
        plan,
    )

    assert first != second


def test_benchmark_environment_manifest_validates_frozen_tasks() -> None:
    policy = load_benchmark_environment_policy(
        Path("configs/research/benchmark_environments.toml")
    )

    assert policy.task_order == tuple(f"GS-T{i:03d}" for i in range(6, 16))
    assert policy.task("GS-T012").runtime == "manifest_container_required"
    assert all(not task.network_during_execution for task in policy.tasks.values())


def test_benchmark_environment_manifest_rejects_non_repository_python_policy(
    tmp_path: Path,
) -> None:
    source = Path("configs/research/benchmark_environments.toml").read_text(encoding="utf-8")
    incompatible = source.replace('python_source = "repository"', 'python_source = "3.13"', 1)
    policy_path = tmp_path / "benchmark_environments.toml"
    policy_path.write_text(incompatible, encoding="utf-8")

    with pytest.raises(BenchmarkEnvironmentConfigurationError, match="repository-authoritative"):
        load_benchmark_environment_policy(policy_path)


def test_dependency_fingerprint_includes_container_digest_and_policy(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "baseline"],
        check=True,
        capture_output=True,
    )
    policy = load_benchmark_environment_policy(
        Path("configs/research/benchmark_environments.toml")
    )
    plan = sprint3.DependencyInstallPlan((), ())

    first = sprint3._dependency_fingerprint(  # pyright: ignore[reportPrivateUsage]
        tmp_path,
        sprint3.PythonInterpreter(Path("docker://base/usr/bin/python3.10"), "3.10.12"),
        plan,
        runtime_type="docker",
        base_image_digest="sha256:one",
        benchmark_policy=policy,
    )
    second = sprint3._dependency_fingerprint(  # pyright: ignore[reportPrivateUsage]
        tmp_path,
        sprint3.PythonInterpreter(Path("docker://base/usr/bin/python3.10"), "3.10.12"),
        plan,
        runtime_type="docker",
        base_image_digest="sha256:two",
        benchmark_policy=policy,
    )

    assert first != second


def test_prepared_container_environment_reuses_validated_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "GS-T013"
    task_policy = TaskEnvironmentPolicy(
        task_id,
        "manifest_container_required",
        "repository",
        "repository",
        False,
        True,
        "Prepared manifest container is the authoritative runtime.",
        None,
        (),
    )
    policy = BenchmarkEnvironmentPolicy(
        tmp_path / "policy.toml",
        (task_id,),
        False,
        "frozen_task_manifest",
        {task_id: task_policy},
    )
    plan = sprint3.DependencyInstallPlan((), ())

    def fake_base_image_digest(image: str, task: str) -> str:
        return "image@sha256:base"

    def fake_python_interpreters(
        image: str, task: str
    ) -> tuple[sprint3.PythonInterpreter, ...]:
        return (sprint3.PythonInterpreter(Path("/usr/bin/python3.10"), "3.10.12"),)

    def fake_dependency_fingerprint(*args: object, **kwargs: object) -> str:
        return "fingerprint"

    def fake_prepare_docker_image(**kwargs: object) -> None:
        builds.append(str(kwargs["image_tag"]))

    def fake_which(name: str) -> str:
        return "docker"

    def fake_subprocess_run(*args: object, **kwargs: object) -> object:
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(sprint3, "_docker_engine_status", lambda: (True, "28.5.1"))
    builds: list[str] = []
    monkeypatch.setattr(sprint3, "_docker_base_image_digest", fake_base_image_digest)
    monkeypatch.setattr(sprint3, "_docker_python_interpreters", fake_python_interpreters)
    monkeypatch.setattr(sprint3, "_dependency_fingerprint", fake_dependency_fingerprint)
    monkeypatch.setattr(sprint3, "_prepare_docker_image", fake_prepare_docker_image)
    monkeypatch.setattr(sprint3.shutil, "which", fake_which)
    monkeypatch.setattr(sprint3.subprocess, "run", fake_subprocess_run)

    first = sprint3._container_environment(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "environments", task_id, tmp_path, "frozen:image", plan, policy
    )
    first.mark_validated()
    second = sprint3._container_environment(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "environments", task_id, tmp_path, "frozen:image", plan, policy
    )

    assert builds == ["graph-swarm/sprint3b-gs-t013:fingerprint"]
    assert second.container_image == first.container_image
    assert first.container_python_executable == "/usr/bin/python3.10"


def test_container_objective_uses_source_first_workspace_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_missing.py",), "")},
        {
            "GS-T006": IsolatedTaskEnvironment(
                "GS-T006",
                Path("docker"),
                runtime_type="docker",
                container_image="prepared:image",
                container_python_executable="/usr/bin/python3.10",
            )
        },
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        commands.append(command)
        return type(
            "Completed", (), {"returncode": 1, "stdout": "FAILED test_missing.py", "stderr": ""}
        )()

    monkeypatch.setattr(sprint3.subprocess, "run", fake_run)

    objective(task_case.task, tmp_path)

    assert commands
    command = commands[0]
    assert "pip" not in " ".join(command)
    assert command[3:5] == ("--network", "none")
    assert "--mount" in command
    assert command[command.index("--workdir") + 1] == "/workspace"
    environment_index = command.index("--env")
    image_index = command.index("prepared:image")
    assert command[environment_index + 1] == "PYTHONPATH=/workspace/src:/workspace"
    assert ":" in command[environment_index + 1]
    assert environment_index < image_index
    assert command[image_index] == "prepared:image"
    assert command[image_index + 1] == "/usr/bin/python3.10"


def test_container_agent_runtime_separates_docker_cli_and_python() -> None:
    environment = IsolatedTaskEnvironment(
        "GS-T006",
        Path("docker.exe"),
        runtime_type="docker",
        container_image="prepared:image",
        container_python_executable="/opt/miniconda3/bin/python",
    )

    runtime = environment.agent_execution_runtime()

    assert runtime.runtime_type == "docker"
    assert runtime.docker_executable == Path("docker.exe")
    assert runtime.container_image == "prepared:image"
    assert runtime.container_python_executable == "/opt/miniconda3/bin/python"
    assert runtime.python_executable is None


def test_local_venv_objective_command_disables_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_missing.py",), "")},
        {
            "GS-T006": IsolatedTaskEnvironment(
                "GS-T006",
                Path("venv-python"),
                runtime_type="local_venv",
            )
        },
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        commands.append(command)
        return type(
            "Completed", (), {"returncode": 1, "stdout": "FAILED test_missing.py", "stderr": ""}
        )()

    monkeypatch.setattr(sprint3.subprocess, "run", fake_run)

    objective(task_case.task, tmp_path)

    assert commands == [
        ("venv-python", "-m", "pytest", "--no-cov", "test_missing.py", "-q")
    ]


def test_objective_command_disables_coverage_thresholds_only_at_objective_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_target.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path("venv-python"))},
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        commands.append(command)
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "1 passed", "stderr": "coverage fail-under ignored"},
        )()

    monkeypatch.setattr(sprint3.subprocess, "run", fake_run)

    assert objective(task_case.task, tmp_path) is True
    assert "--no-cov" in commands[0]
    assert objective.observations[-1].status == "passed"


def test_objective_preflight_falls_back_when_no_cov_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_target.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path("venv-python"))},
        objective_coverage_policy="no_cov",
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        commands.append(command)
        if len(commands) == 1:
            return type(
                "Completed",
                (),
                {"returncode": 2, "stdout": "", "stderr": "unknown option"},
            )()
        return type(
            "Completed", (), {"returncode": 1, "stdout": "FAILED test_target.py", "stderr": ""}
        )()

    monkeypatch.setattr(sprint3.subprocess, "run", fake_run)

    observation = objective.preflight(task_case.task, tmp_path)

    assert observation.status == "test_failure"
    assert commands[0][0:4] == ("venv-python", "-m", "pytest", "--no-cov")
    assert commands[0][-1] == "--help"
    assert "-o" in commands[1]
    assert "addopts=" in commands[1]


def test_r7_help_option_discovery_selects_no_cov_without_using_probe_status() -> None:
    assert sprint3._pytest_help_supports_no_cov(  # pyright: ignore[reportPrivateUsage]
        "pytest options:\n  --no-cov  disable coverage"
    )
    assert not sprint3._pytest_help_supports_no_cov(  # pyright: ignore[reportPrivateUsage]
        "pytest: error: unrecognized arguments: --no-cov"
    )


def test_r7_plain_pytest_preserves_unrelated_options_and_objective_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_target.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path("venv-python"))},
        objective_coverage_policy="no_cov",
        coverage_policy_selection_version=sprint3.OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION,
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        commands.append(command)
        if len(commands) == 1:
            return type("Completed", (), {"returncode": 0, "stdout": "-q", "stderr": ""})()
        return type(
            "Completed",
            (),
            {"returncode": 1, "stdout": "FAILED test_target.py", "stderr": ""},
        )()

    monkeypatch.setattr(sprint3.subprocess, "run", fake_run)

    observation = objective.preflight(task_case.task, tmp_path)

    assert observation.status == "test_failure"
    assert objective.effective_coverage_policy("GS-T006") == "plain_pytest"
    assert commands == [
        ("venv-python", "-m", "pytest", "--help"),
        ("venv-python", "-m", "pytest", "test_target.py", "-q"),
    ]


def test_r7_pytest_cov_help_selects_no_cov_for_the_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_case = _case(2)
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_target.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path("venv-python"))},
        objective_coverage_policy="no_cov",
        coverage_policy_selection_version=sprint3.OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION,
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        commands.append(command)
        if len(commands) == 1:
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "--no-cov  disable coverage", "stderr": ""},
            )()
        return type("Completed", (), {"returncode": 0, "stdout": "1 passed", "stderr": ""})()

    monkeypatch.setattr(sprint3.subprocess, "run", fake_run)

    assert objective(task_case.task, tmp_path) is True
    assert objective.effective_coverage_policy("GS-T006") == "no_cov"
    assert commands == [
        ("venv-python", "-m", "pytest", "--help"),
        ("venv-python", "-m", "pytest", "--no-cov", "test_target.py", "-q"),
    ]


def test_r7_coverage_enforcement_without_no_cov_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = --cov --cov-fail-under=80\n",
        encoding="utf-8",
    )
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_target.py",), "")},
        {"GS-T006": IsolatedTaskEnvironment("GS-T006", Path("venv-python"))},
        objective_coverage_policy="no_cov",
        coverage_policy_selection_version=sprint3.OBJECTIVE_COVERAGE_POLICY_SELECTION_VERSION,
    )

    def fake_run(*_args: object, **_kwargs: object) -> object:
        return type("Completed", (), {"returncode": 0, "stdout": "-q", "stderr": ""})()

    monkeypatch.setattr(sprint3.subprocess, "run", fake_run)

    with pytest.raises(BenchmarkPreflightError, match="active coverage enforcement"):
        objective.preflight(_case(2).task, tmp_path)


def test_git_worktree_materialization_preserves_attributes_mutation_binary_and_modes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "-C", str(workspace), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.name", "Test"],
        check=True,
    )
    normal_file = workspace / "normal.py"
    sensitive_file = workspace / "sensitive.txt"
    binary_file = workspace / "image.bin"
    attributes_file = workspace / ".gitattributes"
    normal_file.write_bytes(b"print('baseline')\n")
    sensitive_file.write_bytes(b"line one\nline two\n")
    binary = b"\x89PNG\r\n\x00\xff\r\n"
    binary_file.write_bytes(binary)
    attributes_file.write_text("sensitive.txt text eol=crlf\n", encoding="utf-8")
    normal_file.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    subprocess.run(
        ["git", "-C", str(workspace), "add", "."],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "update-index", "--chmod=+x", "normal.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-qm", "baseline"],
        check=True,
    )
    index_blob = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", ":normal.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    normal_file.write_bytes(b"print('baseline')\r\n")
    mode_before = stat.S_IMODE(normal_file.stat().st_mode)

    prepared = sprint3._materialize_git_worktree(workspace)  # pyright: ignore[reportPrivateUsage]

    assert prepared.returncode == 0
    assert normal_file.read_bytes() == b"print('baseline')\n"
    assert sensitive_file.read_bytes() == b"line one\r\nline two\r\n"
    assert binary_file.read_bytes() == binary
    assert stat.S_IMODE(normal_file.stat().st_mode) == mode_before
    assert (
        subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", ":normal.py"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == index_blob
    )
    assert subprocess.run(
        ["git", "-C", str(workspace), "diff", "--cached", "--quiet"],
        check=False,
    ).returncode == 0

    normal_file.write_bytes(b"print('GS-T001 mutation')\n")
    assert b"GS-T001 mutation" in normal_file.read_bytes()


def test_preflight_does_not_validate_unexpectedly_passing_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_case = _case(2)
    marker = tmp_path / "environment.json"
    environment = IsolatedTaskEnvironment(
        "GS-T006",
        Path(sys.executable),
        validation_marker=marker,
        runtime_type="local_venv",
    )
    objective = FrozenSWEsmithObjective(
        {"GS-T006": FrozenSWEsmithCase("example", ("test_missing.py",), "")},
        {"GS-T006": environment},
    )

    def fake_passing_run(*args: object, **kwargs: object) -> object:
        return type("Completed", (), {"returncode": 0, "stdout": "1 passed", "stderr": ""})()

    monkeypatch.setattr(
        sprint3.subprocess,
        "run",
        fake_passing_run,
    )

    with pytest.raises(BenchmarkPreflightError, match="expected mutated test failure") as error:
        objective.preflight(task_case.task, tmp_path)
    message = str(error.value)
    assert "return_code=0" in message
    assert "status=passed" in message
    assert repr(str(sys.executable)) in message
    assert "1 passed" in message
    assert "stderr:\n<empty>" in message
    assert not marker.exists()


def test_tox_setenv_is_persisted_in_prepared_container_environment(
    tmp_path: Path,
) -> None:
    plan = sprint3.DependencyInstallPlan((), (), environment=(("TOX_RED", "1"),))

    dockerfile = sprint3._prepared_dockerfile(  # pyright: ignore[reportPrivateUsage]
        base_image_digest="frozen:image@sha256:base",
        python_executable="/usr/bin/python3.10",
        repository=tmp_path,
        plan=plan,
    )

    assert 'ENV TOX_RED="1"' in dockerfile


def test_unrelated_optional_extra_is_not_selected(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'example'\n"
        "[project.optional-dependencies]\n"
        "dev = ['unrelated-dev-package']\n"
        "test = ['pytest']\n",
        encoding="utf-8",
    )
    (tmp_path / "tox.ini").write_text(
        "[testenv]\n"
        "extras =\n"
        "    test\n",
        encoding="utf-8",
    )

    plan = sprint3._dependency_install_plan(tmp_path, task_id="GS-T006")  # pyright: ignore[reportPrivateUsage]

    assert plan.package_extras == ("test",)


def test_missing_container_image_is_reported_without_host_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_benchmark_environment_policy(
        Path("configs/research/benchmark_environments.toml")
    )
    monkeypatch.setattr(sprint3, "_docker_engine_status", lambda: (True, "28.5.1"))

    def missing_base_image_digest(image: str, task: str) -> str:
        raise BenchmarkPreflightError("required SWE-smith container image is unavailable")

    monkeypatch.setattr(
        sprint3,
        "_docker_base_image_digest",
        missing_base_image_digest,
    )

    with pytest.raises(BenchmarkPreflightError, match="container image is unavailable"):
        sprint3._container_environment(  # pyright: ignore[reportPrivateUsage]
            tmp_path,
            "GS-T013",
            tmp_path,
            "missing:image",
            sprint3.DependencyInstallPlan((), ()),
            policy,
        )


def test_container_interpreter_selection_uses_repository_constraint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.9,<3.12"\n',
        encoding="utf-8",
    )

    def fake_container_interpreters(
        image: str, task: str
    ) -> tuple[sprint3.PythonInterpreter, ...]:
        return (
            sprint3.PythonInterpreter(Path("/usr/bin/python3.12"), "3.12.1"),
            sprint3.PythonInterpreter(Path("/usr/bin/python3.10"), "3.10.12"),
        )

    monkeypatch.setattr(
        sprint3,
        "_docker_python_interpreters",
        fake_container_interpreters,
    )
    selected = [
        interpreter
        for interpreter in sprint3._docker_python_interpreters("image", "GS-T013")  # pyright: ignore[reportPrivateUsage]
        if sprint3.Version(interpreter.version) in sprint3._python_constraints(tmp_path)  # pyright: ignore[reportPrivateUsage]
    ]

    assert [str(interpreter.executable).replace("\\", "/") for interpreter in selected] == [
        "/usr/bin/python3.10"
    ]


def test_gs_t014_selects_python310_when_python312_violates_policy() -> None:
    policy = load_benchmark_environment_policy(
        Path("configs/research/benchmark_environments.toml")
    )
    task_policy = policy.task("GS-T014")
    repository_constraints = sprint3.SpecifierSet(">=3.7")
    effective_constraints = sprint3.effective_python_constraints(
        task_policy, repository_constraints
    )
    selected = sprint3._select_compatible_interpreter(  # pyright: ignore[reportPrivateUsage]
        (
            sprint3.PythonInterpreter(Path("/usr/bin/python3.12"), "3.12.1"),
            sprint3.PythonInterpreter(Path("/usr/bin/python3.10"), "3.10.12"),
        ),
        effective_constraints,
        "GS-T014",
    )

    assert selected.version == "3.10.12"


def test_interpreter_violating_policy_assertion_is_rejected() -> None:
    policy = load_benchmark_environment_policy(
        Path("configs/research/benchmark_environments.toml")
    )
    effective_constraints = sprint3.effective_python_constraints(
        policy.task("GS-T014"), sprint3.SpecifierSet(">=3.7")
    )

    with pytest.raises(BenchmarkPreflightError, match="no compatible Python interpreter"):
        sprint3._select_compatible_interpreter(  # pyright: ignore[reportPrivateUsage]
            (sprint3.PythonInterpreter(Path("/usr/bin/python3.12"), "3.12.1"),),
            effective_constraints,
            "GS-T014",
        )


def test_incompatible_policy_assertion_fails_clearly() -> None:
    task_policy = TaskEnvironmentPolicy(
        "GS-T014",
        "manifest_container_required",
        "repository",
        "repository",
        False,
        True,
        "Incompatible test policy.",
        ">=3.12",
        (),
    )

    with pytest.raises(
        BenchmarkEnvironmentConfigurationError,
        match="incompatible with repository metadata",
    ):
        sprint3.effective_python_constraints(
            task_policy, sprint3.SpecifierSet(">=3.7,<3.12")
        )


def test_string2string_faiss_pin_remains_repository_authoritative() -> None:
    repository = Path(
        "benchmark/workspaces/swesmith/stanfordnlp__string2string.c4a72f59"
    )
    metadata_files = tuple(
        path
        for path in repository.rglob("*")
        if path.is_file() and path.name in {"pyproject.toml", "setup.py", "setup.cfg"}
    )
    authoritative_declarations = tuple(
        path
        for path in metadata_files
        if "faiss-cpu==1.7.3" in path.read_text(encoding="utf-8")
    )

    assert authoritative_declarations
