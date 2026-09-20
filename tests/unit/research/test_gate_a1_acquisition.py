import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.sprint3 import IsolatedTaskEnvironment
from graph_swarm.domain.tasks import Task
from graph_swarm.research import gate_a1
from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research.runner import BenchmarkTaskCase


def _fake_ready_statuses() -> tuple[Any, ...]:
    return tuple(
        SimpleNamespace(task_id=task_id, ready=True, blockers=())
        for task_id in acquisition.ACQUISITION_TASK_IDS
    )


def _contract_project(tmp_path: Path, source_root: Path) -> Path:
    for relative in (
        gate_a1.DEVELOPMENT_SUBSET_PATH,
        gate_a1.BASELINE_MANIFEST_PATH,
        Path("benchmark/manifests/pilot.jsonl"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_root / relative, destination)

    contract = json.loads(
        (source_root / gate_a1.ACQUISITION_CONTRACT).read_text(encoding="utf-8")
    )
    for record in contract["environments"]:
        task_id = record["task_id"]
        fingerprint = record["environment_fingerprint"]
        marker = tmp_path / "env" / task_id / fingerprint / "environment.json"
        workspace = tmp_path / "env" / task_id / "workspace"
        marker.parent.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "validated": True,
                    "runtime_type": "docker",
                    "environment_fingerprint": fingerprint,
                    "container_image": record["prepared_image"],
                    "base_container_image": record["upstream_image"],
                    "base_image_digest": record["upstream_digest"],
                    "python_executable": "docker",
                    "python_version": "3.12.1",
                    "container_python_executable": "/opt/miniconda3/bin/python",
                }
            ),
            encoding="utf-8",
        )
        record["validation_marker"] = marker.relative_to(tmp_path).as_posix()
        record["workspace"] = workspace.relative_to(tmp_path).as_posix()
    contract_path = tmp_path / gate_a1.ACQUISITION_CONTRACT
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return tmp_path


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _patch_contract_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_assess(*_args: object, **_kwargs: object) -> tuple[Any, ...]:
        return _fake_ready_statuses()

    def fake_image_status(_image: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(acquisition, "assess_acquisition_environments", fake_assess)
    monkeypatch.setattr(acquisition, "_docker_image_status", fake_image_status)


def test_resolver_uses_current_five_task_contract_and_accepts_repaired_t003(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_root = _contract_project(tmp_path, project_root)
    _patch_contract_checks(monkeypatch)

    environments = acquisition.resolve_gate_a1_environments(test_root)

    assert tuple(environments) == acquisition.ACQUISITION_TASK_IDS
    assert environments["GS-T003"].environment_fingerprint == "5de87654e89917a4"
    assert "1459499a6c71c40b" not in {
        environment.environment_fingerprint for environment in environments.values()
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prepared_image", "missing/image"),
        ("environment_fingerprint", "wrong-fingerprint"),
        ("upstream_digest", "wrong-digest"),
    ],
)
def test_resolver_fails_closed_for_contract_mismatches(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    test_root = _contract_project(tmp_path, project_root)
    contract_path = test_root / gate_a1.ACQUISITION_CONTRACT
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    record = next(
        record for record in contract["environments"] if record["task_id"] == "GS-T003"
    )
    record[field] = value
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    _patch_contract_checks(monkeypatch)

    with pytest.raises(acquisition.GateA1AcquisitionPreflightError):
        acquisition.resolve_gate_a1_environments(test_root)


def test_resolver_fails_closed_for_missing_or_invalid_marker(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_root = _contract_project(tmp_path, project_root)
    contract_path = test_root / gate_a1.ACQUISITION_CONTRACT
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    record = next(record for record in contract["environments"] if record["task_id"] == "GS-T003")
    marker = test_root / record["validation_marker"]
    marker.unlink()
    _patch_contract_checks(monkeypatch)
    with pytest.raises(acquisition.GateA1AcquisitionPreflightError):
        acquisition.resolve_gate_a1_environments(test_root)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(acquisition.GateA1AcquisitionPreflightError):
        acquisition.resolve_gate_a1_environments(test_root)


def test_resolver_rejects_non_acquisition_scope(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_root = _contract_project(tmp_path, project_root)
    _patch_contract_checks(monkeypatch)

    with pytest.raises(acquisition.GateA1AcquisitionPreflightError):
        acquisition.resolve_gate_a1_environments(test_root, task_ids=("GS-T006",))


def test_preflight_has_zero_attempts_and_zero_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [
        Task(
            id=task_id,
            problem_statement="Fix the benchmark issue.",
            family_id="evaluator-only",
            repository="repo",
            chronological_index=index,
        )
        for index, task_id in enumerate(acquisition.ACQUISITION_TASK_IDS, start=1)
    ]
    configuration = SimpleNamespace(
        model=SimpleNamespace(provider="openrouter", settings={}, prompt_version="v1"),
        config=SimpleNamespace(),
    )
    environment = IsolatedTaskEnvironment(
        task_id="GS-T001",
        python_executable=Path("docker"),
        runtime_type="docker",
        container_image="image:t001",
        container_python_executable="python",
        environment_fingerprint="fingerprint",
    )
    calls: list[str] = []

    class FakeRepository:
        def verify_connectivity(self) -> None:
            calls.append("neo4j")

        def ensure_recovery_pattern_vector_index(self) -> None:
            calls.append("index")

        def close(self) -> None:
            calls.append("close")

    def fake_configured_runtime(_loaded: object, *_args: object) -> Any:
        return configuration

    def fake_load_configuration(*_args: object, **_kwargs: object) -> Any:
        return configuration

    def fake_task_cases(*_args: object, **_kwargs: object) -> list[BenchmarkTaskCase]:
        return [BenchmarkTaskCase(task=task, occurrence_index=1) for task in tasks]

    def fake_environments(*_args: object, **_kwargs: object) -> dict[str, IsolatedTaskEnvironment]:
        return {task_id: environment for task_id in acquisition.ACQUISITION_TASK_IDS}

    monkeypatch.setattr(acquisition, "_configured_runtime", fake_configured_runtime)
    monkeypatch.setattr(
        acquisition, "load_experiment_configuration", fake_load_configuration
    )
    monkeypatch.setattr(
        acquisition,
        "_task_cases",
        fake_task_cases,
    )
    monkeypatch.setattr(acquisition, "resolve_gate_a1_environments", fake_environments)

    def fake_settings() -> Any:
        return SimpleNamespace(
            model_provider="openrouter",
            openrouter_api_key="key",
            openrouter_model="model",
            neo4j_uri="neo4j://test",
            neo4j_username="user",
            neo4j_password="password",
            neo4j_database="database",
        )

    def fake_repository(**_kwargs: object) -> FakeRepository:
        return FakeRepository()

    def fail_agent(*_args: object, **_kwargs: object) -> Any:
        pytest.fail("provider agent constructed")

    monkeypatch.setattr(acquisition, "_settings_for_agent", fake_settings)
    monkeypatch.setattr(acquisition, "Neo4jRepository", fake_repository)
    monkeypatch.setattr(acquisition, "create_coding_agent", fail_agent)

    status, artifact_root = acquisition.run_gate_a1_acquisition_preflight(tmp_path)

    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    assert status == "READY_FOR_GATE_A1_ACQUISITION_EXECUTION"
    assert manifest["tasks_attempted"] == 0
    assert manifest["provider_calls"] == 0
    assert calls == ["neo4j", "index", "close"]
