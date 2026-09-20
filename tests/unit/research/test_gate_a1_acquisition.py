# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from experiments.sprint3 import IsolatedTaskEnvironment
from graph_swarm.domain.tasks import Task
from graph_swarm.research import gate_a1
from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research.runner import BenchmarkTaskCase, load_experiment_configuration
from graph_swarm.settings import Settings


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
                openrouter_model=acquisition.FROZEN_CODING_MODEL,
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


def test_coding_agent_and_recovery_abstraction_models_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = Settings(
        neo4j_uri="neo4j://test",
        neo4j_username="user",
        neo4j_password="password",
        neo4j_database="database",
        openrouter_model=acquisition.FROZEN_RECOVERY_MODEL,
    )
    monkeypatch.setattr(acquisition, "get_settings", lambda: base)
    monkeypatch.setenv("OPENROUTER_MODEL", acquisition.FROZEN_RECOVERY_MODEL)
    monkeypatch.setenv("OPENROUTER_CODING_MODEL", acquisition.FROZEN_CODING_MODEL)

    settings = acquisition._settings_for_agent()

    assert settings.openrouter_model == acquisition.FROZEN_CODING_MODEL
    assert settings.openrouter_model != acquisition.FROZEN_RECOVERY_MODEL
    assert acquisition.FROZEN_RECOVERY_MODEL == "cohere/north-mini-code:free"


def test_r2_uses_the_frozen_track_b_resource_budget(project_root: Path) -> None:
    r2 = load_experiment_configuration(
        project_root / acquisition.ACQUISITION_R2_CONFIG,
        project_root=project_root,
    )
    pilot = load_experiment_configuration(
        project_root / "configs/experiments/rollout_3a_pilot.yaml",
        project_root=project_root,
    )

    assert r2.config.limits == pilot.config.limits
    assert r2.config.limits.max_actions == 20
    assert r2.config.limits.max_requests == 24
    assert r2.config.limits.timeout_seconds == 300


def test_r2_provider_blocked_evidence_remains_immutable(project_root: Path) -> None:
    evidence_root = (
        project_root
        / "research/evidence/results/GS-E003/gate_a1/acquisition-r2-20260920T164708Z"
    )
    manifest_path = evidence_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "0f5892f12b2fc25df83226a42a0f7455dcc9d6b488205f4176ee46a4364410d7"
    )
    assert manifest["run_revision"] == "R2"
    assert manifest["namespace"] == acquisition.ACQUISITION_R2_NAMESPACE
    assert manifest["coding_model"] == acquisition.FROZEN_CODING_MODEL
    assert manifest["configuration_hash"] == (
        "17d04cfbac1736f0138f41d24477b7e13e07651fdc698c56c06c8d8c89ca5299"
    )
    assert [task["task_id"] for task in manifest["tasks"]] == ["GS-T001"]
    assert not (evidence_root / "tasks/GS-T002/started.json").exists()


def test_r3_resolves_distinct_coding_model_and_frozen_abstraction(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r3 = load_experiment_configuration(
        project_root / acquisition.ACQUISITION_R3_CONFIG,
        project_root=project_root,
    )
    monkeypatch.setenv("OPENROUTER_MODEL", acquisition.FROZEN_RECOVERY_MODEL)

    settings = acquisition._settings_for_agent(coding_model=r3.model.model)

    assert r3.model.model == "cohere/north-mini-code:free"
    assert settings.openrouter_model == "cohere/north-mini-code:free"
    assert acquisition.FROZEN_R3_CODING_MODEL == "cohere/north-mini-code:free"
    assert acquisition.FROZEN_RECOVERY_MODEL == "cohere/north-mini-code:free"
    assert r3.config.model_config_path == "configs/models/openrouter_coding_r3.yaml"


def test_r3_preserves_b0_limits_prompt_chronology_and_transfer_contract(
    project_root: Path,
) -> None:
    r2 = load_experiment_configuration(
        project_root / acquisition.ACQUISITION_R2_CONFIG,
        project_root=project_root,
    )
    r3 = load_experiment_configuration(
        project_root / acquisition.ACQUISITION_R3_CONFIG,
        project_root=project_root,
    )

    assert r3.config.conditions == r2.config.conditions
    assert r3.config.limits == r2.config.limits
    assert r3.model.prompt_version == r2.model.prompt_version == "v1"
    assert acquisition.ACQUISITION_TASK_IDS == (
        "GS-T001",
        "GS-T002",
        "GS-T003",
        "GS-T004",
        "GS-T005",
    )
    assert r3.config.conditions[0].value == "B0"


def test_r3_configuration_hash_differs_from_r2_for_same_runtime_inputs() -> None:
    configuration = SimpleNamespace(
        model=SimpleNamespace(settings={"temperature": 0}, prompt_version="v1"),
        config=SimpleNamespace(
            limits=SimpleNamespace(max_actions=20, max_requests=24, timeout_seconds=300)
        ),
    )
    environments = {
        task_id: IsolatedTaskEnvironment(
            task_id=task_id,
            python_executable=Path("docker"),
            runtime_type="docker",
            container_image=f"image:{task_id.lower()}",
            container_python_executable="python",
            environment_fingerprint=f"fingerprint-{task_id.lower()}",
        )
        for task_id in acquisition.ACQUISITION_TASK_IDS
    }

    r2_hash = acquisition._r2_configuration_hash(
        configuration,
        SimpleNamespace(openrouter_model=acquisition.FROZEN_CODING_MODEL),
        environments,
    )
    r3_hash = acquisition._r2_configuration_hash(
        configuration,
        SimpleNamespace(openrouter_model=acquisition.FROZEN_R3_CODING_MODEL),
        environments,
    )

    assert r3_hash != r2_hash
    assert r3_hash == (
        "df6403270e7f0374a689ff0a310f1bf6b05236c3bbc5214d45dab654581b1f05"
    )


def test_r3_starts_with_fresh_t001_plan_and_rejects_r2_resume_root(
    tmp_path: Path,
    project_root: Path,
) -> None:
    r3_root = tmp_path / "acquisition-r3-20260920T000000Z"
    r3_root.mkdir()
    assert acquisition._r2_task_plan(r3_root, max_new_tasks=1) == (
        ("GS-T001", "start"),
    )

    r2_root = (
        project_root
        / "research/evidence/results/GS-E003/gate_a1/acquisition-r2-20260920T164708Z"
    )
    with pytest.raises(acquisition.GateA1AcquisitionPreflightError):
        acquisition.run_gate_a1_acquisition_r3(
            project_root,
            resume_root=r2_root,
            max_new_tasks=1,
        )


def test_r2_resume_skips_started_and_completed_tasks_without_rerun(tmp_path: Path) -> None:
    completed = acquisition._r2_task_marker(tmp_path, "GS-T001", "completed.json")
    completed.parent.mkdir(parents=True, exist_ok=True)
    completed.write_text("{}", encoding="utf-8")
    started = acquisition._r2_task_marker(tmp_path, "GS-T002", "started.json")
    started.parent.mkdir(parents=True, exist_ok=True)
    started.write_text("{}", encoding="utf-8")

    assert acquisition._r2_resume_action(tmp_path, "GS-T001") == "skipped_completed"
    assert (
        acquisition._r2_resume_action(tmp_path, "GS-T002")
        == "skipped_started_no_rerun"
    )
    assert acquisition._r2_resume_action(tmp_path, "GS-T003") is None


def test_r2_max_new_tasks_one_starts_only_first_never_started_task(
    tmp_path: Path,
) -> None:
    assert acquisition._r2_task_plan(tmp_path, max_new_tasks=1) == (
        ("GS-T001", "start"),
    )


def test_r2_max_new_tasks_resume_starts_only_next_never_started_task(
    tmp_path: Path,
) -> None:
    completed = acquisition._r2_task_marker(tmp_path, "GS-T001", "completed.json")
    completed.parent.mkdir(parents=True, exist_ok=True)
    completed.write_text("{}", encoding="utf-8")

    assert acquisition._r2_task_plan(tmp_path, max_new_tasks=1) == (
        ("GS-T001", "skipped_completed"),
        ("GS-T002", "start"),
    )


def test_r2_started_incomplete_task_is_skipped_without_rerun(tmp_path: Path) -> None:
    started = acquisition._r2_task_marker(tmp_path, "GS-T001", "started.json")
    started.parent.mkdir(parents=True, exist_ok=True)
    started.write_text("{}", encoding="utf-8")

    assert acquisition._r2_task_plan(tmp_path, max_new_tasks=1) == (
        ("GS-T001", "skipped_started_no_rerun"),
        ("GS-T002", "start"),
    )


def test_r2_task_plan_preserves_canonical_order_without_boundary(tmp_path: Path) -> None:
    assert acquisition._r2_task_plan(tmp_path, max_new_tasks=None) == tuple(
        (task_id, "start") for task_id in acquisition.ACQUISITION_TASK_IDS
    )


def test_boundary_reporting_clears_stale_next_task_after_all_markers_complete(
    tmp_path: Path,
) -> None:
    for task_id in acquisition.ACQUISITION_TASK_IDS:
        for marker_name in ("started.json", "completed.json"):
            marker = acquisition._r2_task_marker(tmp_path, task_id, marker_name)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}", encoding="utf-8")
    manifest: dict[str, object] = {"next_task_id": "GS-T005"}

    assert (
        acquisition._reconcile_execution_boundary(
            tmp_path,
            manifest,
            boundary_hit=True,
        )
        is False
    )
    assert "next_task_id" not in manifest
    assert acquisition._completed_task_count(tmp_path) == 5


def test_boundary_reporting_keeps_first_never_started_task(tmp_path: Path) -> None:
    for task_id in acquisition.ACQUISITION_TASK_IDS[:4]:
        marker = acquisition._r2_task_marker(tmp_path, task_id, "started.json")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}", encoding="utf-8")
    manifest: dict[str, object] = {}

    assert (
        acquisition._reconcile_execution_boundary(
            tmp_path,
            manifest,
            boundary_hit=True,
        )
        is True
    )
    assert manifest["next_task_id"] == "GS-T005"


def test_r2_boundary_does_not_change_task_configuration_hash() -> None:
    configuration = SimpleNamespace(
        model=SimpleNamespace(settings={"temperature": 0}, prompt_version="v1"),
        config=SimpleNamespace(
            limits=SimpleNamespace(max_actions=20, max_requests=24, timeout_seconds=300)
        ),
    )
    settings = SimpleNamespace(openrouter_model=acquisition.FROZEN_CODING_MODEL)
    environments = {
        task_id: IsolatedTaskEnvironment(
            task_id=task_id,
            python_executable=Path("docker"),
            runtime_type="docker",
            container_image=f"image:{task_id.lower()}",
            container_python_executable="python",
            environment_fingerprint=f"fingerprint-{task_id.lower()}",
        )
        for task_id in acquisition.ACQUISITION_TASK_IDS
    }

    unrestricted = acquisition._r2_configuration_hash(
        configuration,
        settings,
        environments,
        max_new_tasks=None,
    )
    bounded = acquisition._r2_configuration_hash(
        configuration,
        settings,
        environments,
        max_new_tasks=1,
    )

    assert bounded == unrestricted


def test_r2_manifest_requires_fresh_namespace_and_matching_configuration() -> None:
    manifest = {
        "run_revision": "R2",
        "namespace": acquisition.ACQUISITION_R2_NAMESPACE,
        "task_ids": list(acquisition.ACQUISITION_TASK_IDS),
        "configuration_hash": "hash",
    }

    acquisition._validate_r2_manifest(manifest, configuration_hash="hash")
    with pytest.raises(acquisition.GateA1AcquisitionPreflightError):
        acquisition._validate_r2_manifest(
            {**manifest, "namespace": acquisition.ACQUISITION_NAMESPACE},
            configuration_hash="hash",
        )
    with pytest.raises(acquisition.GateA1AcquisitionPreflightError):
        acquisition._validate_r2_manifest(manifest, configuration_hash="changed")


def test_objective_is_evaluated_after_provider_error_before_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Task(
        id="GS-T003",
        problem_statement="Fix the benchmark issue.",
        family_id="evaluator-only",
        repository="repo",
        chronological_index=3,
    )
    case = BenchmarkTaskCase(task=task, occurrence_index=1)
    environment = IsolatedTaskEnvironment(
        task_id="GS-T003",
        python_executable=Path("docker"),
        runtime_type="docker",
        container_image="image:t003",
        container_python_executable="python",
        environment_fingerprint="fingerprint",
        python_version="3.12.1",
    )
    configuration = SimpleNamespace(
        model=SimpleNamespace(settings={}, prompt_version="v1"),
        config=SimpleNamespace(
            limits=SimpleNamespace(max_actions=20, max_requests=24, timeout_seconds=300)
        ),
    )
    settings = SimpleNamespace(
        openrouter_model=acquisition.FROZEN_CODING_MODEL,
        openrouter_api_key="offline",
    )
    order: list[str] = []

    class FakeObjective:
        cases: dict[str, object] = {}
        observations: list[object] = []

        def __call__(self, _task: Task, _workspace: Path) -> bool:
            order.append("objective")
            return True

    class FakeRepository:
        def save_task(self, _task: Task) -> None:
            return None

        def save_run(self, _run: object) -> None:
            return None

        def save_environment(self, _environment: object) -> None:
            return None

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(acquisition, "_materialize_workspace", lambda **_: workspace)
    monkeypatch.setattr(acquisition, "create_coding_agent", lambda *_args, **_kwargs: object())

    def fail_provider(*_args: object, **_kwargs: object) -> object:
        order.append("agent")
        raise RuntimeError("provider quota")

    monkeypatch.setattr(acquisition, "run_coding_agent", fail_provider)

    def persist(*_args: object, **_kwargs: object) -> None:
        order.append("persistence")
        return None

    monkeypatch.setattr(acquisition, "persist_agent_event_stream", persist)
    artifact, _embedder, _acquired = acquisition._run_task(
        baseline_root=tmp_path,
        execution_root=tmp_path,
        artifact_root=tmp_path / "run",
        configuration=configuration,
        case=case,
        environment=environment,
        objective=cast(Any, FakeObjective()),
        repository=cast(Any, FakeRepository()),
        settings=settings,
        pacing=acquisition.ProviderRequestPacing(),
        embedder=None,
    )

    assert order == ["agent", "objective", "persistence"]
    assert artifact["agent_error"] == "RuntimeError"
    assert cast(dict[str, object], artifact["objective"])["task_success"] is True
