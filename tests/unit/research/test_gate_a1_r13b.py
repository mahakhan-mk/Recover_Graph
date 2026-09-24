import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from graph_swarm.agent.prompts import R13B_RUNTIME_GUIDANCE, R13B_SYSTEM_PROMPT
from graph_swarm.research import gate_a1
from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research import gate_a1_r13 as r13
from graph_swarm.research.gate_a1_r8 import objective_final_check_policy
from graph_swarm.research.runner import load_experiment_configuration

ROOT = Path(__file__).resolve().parents[3]


def test_r13b_configuration_records_new_revision_and_frozen_runtime_limits() -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )

    assert configuration.config.run_revision == "R13b"
    assert configuration.config.rollout == "track_a_gate_a1_acquisition_r13b"
    assert configuration.config.config_version == "gate-a1-r13b-v2"
    assert configuration.config.revision_reason == acquisition.R13B_REVISION_REASON
    assert configuration.model.prompt_version == acquisition.R13B_PROMPT_VERSION
    assert configuration.config.limits.timeout_seconds == acquisition.R13B_TIMEOUT_SECONDS
    assert configuration.config.agent_timeout_seconds == acquisition.R13B_AGENT_TIMEOUT_SECONDS
    assert configuration.config.objective_timeout_seconds == 900
    assert configuration.config.recovery_event_semantics == (
        "objective_anchor_trusted_mutation_objective_success_v1"
    )
    assert acquisition.R13B_EXPECTED_CODING_MODEL == "nex-agi/nex-n2.5-pro:free"


def test_historical_r13b_v1_configuration_remains_reconstructable() -> None:
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/gate_a1_acquisition_r13b.yaml",
        project_root=ROOT,
    )

    assert configuration.config.config_version == "gate-a1-r13b-v1"
    assert configuration.config.agent_timeout_seconds == 600
    assert configuration.config.limits.timeout_seconds == 600


def test_r13b_uses_r13_objective_lifecycle_and_new_prompt_only() -> None:
    assert acquisition.R13B_EXPECTED_CODING_MODEL == acquisition.R13_EXPECTED_CODING_MODEL
    assert acquisition.R13B_EXPECTED_ABSTRACTION_MODEL == acquisition.R13_EXPECTED_ABSTRACTION_MODEL
    assert objective_final_check_policy(
        "R13b",
        None,
        has_repository_mutation=False,
    ) == "skipped_r13b_requires_mutation_bound_objective"
    assert R13B_RUNTIME_GUIDANCE in R13B_SYSTEM_PROMPT
    assert "GS-T001" not in R13B_SYSTEM_PROMPT


def test_r13b_configuration_validator_rejects_r13_revision() -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )
    invalid = SimpleNamespace(
        config=configuration.config.model_copy(update={"run_revision": "R13"})
    )

    try:
        acquisition._validate_r13b_harness_configuration(invalid)  # pyright: ignore[reportPrivateUsage]
    except acquisition.GateA1AcquisitionPreflightError:
        return
    raise AssertionError("R13b validator accepted an R13 configuration")


def test_r13b_configuration_validator_rejects_wrong_prompt_version() -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )
    invalid = SimpleNamespace(
        config=configuration.config,
        model=configuration.model.model_copy(update={"prompt_version": "v1"}),
    )

    try:
        acquisition._validate_r13b_harness_configuration(invalid)  # pyright: ignore[reportPrivateUsage]
    except acquisition.GateA1AcquisitionPreflightError:
        return
    raise AssertionError("R13b validator accepted the wrong prompt version")


def test_r13_runtime_timeout_defaults_preserve_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )

    resolutions = acquisition.r13_runtime_timeout_resolutions(configuration)

    assert resolutions["agent"].configured_seconds == 900
    assert resolutions["agent"].effective_seconds == 900
    assert resolutions["agent"].override_applied is False
    assert resolutions["objective"].configured_seconds == 900
    assert resolutions["objective"].effective_seconds == 900
    assert resolutions["objective"].override_applied is False


@pytest.mark.parametrize(
    ("environment_variable", "value", "expected_agent", "expected_objective"),
    [
        (acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, "3600", 3600, 900),
        (acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, "1800", 900, 1800),
        ("both", "3600", 3600, 3600),
    ],
)
def test_r13_runtime_timeout_overrides_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    environment_variable: str,
    value: str,
    expected_agent: int,
    expected_objective: int,
) -> None:
    monkeypatch.delenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    if environment_variable == "both":
        monkeypatch.setenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, value)
        monkeypatch.setenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, value)
    else:
        monkeypatch.setenv(environment_variable, value)
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )

    resolutions = acquisition.r13_runtime_timeout_resolutions(configuration)

    assert resolutions["agent"].effective_seconds == expected_agent
    assert resolutions["objective"].effective_seconds == expected_objective
    assert resolutions["agent"].override_applied is (expected_agent != 900)
    assert resolutions["objective"].override_applied is (expected_objective != 900)


@pytest.mark.parametrize("value", ["0", "-1", "abc", "NaN", "Infinity", ""])
def test_invalid_r13_runtime_timeout_override_fails_before_agent_startup(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, value)
    agent_startup_attempted = False

    def fail_if_settings_are_resolved() -> object:
        nonlocal agent_startup_attempted
        agent_startup_attempted = True
        raise AssertionError("agent settings were resolved before invalid timeout failed")

    monkeypatch.setattr(
        "graph_swarm.research.gate_a1_r13._settings_for_agent",
        fail_if_settings_are_resolved,
    )

    with pytest.raises(ValueError, match=acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE):
        acquisition.run_gate_a1_acquisition_r13b(ROOT)

    assert agent_startup_attempted is False


def test_r13_runtime_timeout_provenance_records_configured_effective_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, "3600")
    monkeypatch.delenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )

    provenance = acquisition.runtime_timeout_provenance(
        acquisition.r13_runtime_timeout_resolutions(configuration)
    )

    assert provenance == {
        "agent": {
            "environment_variable": "GRAPH_SWARM_AGENT_TIMEOUT_SECONDS",
            "configured_seconds": 900,
            "effective_seconds": 3600,
            "override_applied": True,
        },
        "objective": {
            "environment_variable": "GRAPH_SWARM_OBJECTIVE_TIMEOUT_SECONDS",
            "configured_seconds": 900,
            "effective_seconds": 900,
            "override_applied": False,
        },
    }


def _retry_fixture(root: Path, task_id: str, *, acquisition_success: bool) -> Path:
    completed = root / "tasks" / task_id / "completed.json"
    completed.parent.mkdir(parents=True, exist_ok=True)
    completed.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "run_id": f"initial-{task_id}",
                "acquisition_success": acquisition_success,
            }
        ),
        encoding="utf-8",
    )
    return completed


def test_r13b_retry_eligibility_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "acquisition-r13b-test"
    failed_completed = _retry_fixture(root, "GS-T002", acquisition_success=False)
    _retry_fixture(root, "GS-T001", acquisition_success=True)
    before = hashlib.sha256(failed_completed.read_bytes()).digest()

    initial = r13._r13b_retry_eligibility(root, "GS-T002")  # pyright: ignore[reportPrivateUsage]

    assert initial["run_id"] == "initial-GS-T002"
    assert hashlib.sha256(failed_completed.read_bytes()).digest() == before
    with pytest.raises(ValueError, match="unsuccessful"):
        r13._r13b_retry_eligibility(  # pyright: ignore[reportPrivateUsage]
            root,
            "GS-T001",
        )
    with pytest.raises(ValueError, match="completed initial"):
        r13._r13b_retry_eligibility(  # pyright: ignore[reportPrivateUsage]
            root,
            "GS-T003",
        )
    with pytest.raises(ValueError, match="not a frozen"):
        r13._r13b_retry_eligibility(  # pyright: ignore[reportPrivateUsage]
            root,
            "GS-T999",
        )

    retry_root = r13._r13b_retry_root(root, "GS-T002")  # pyright: ignore[reportPrivateUsage]
    retry_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="already consumed"):
        r13._r13b_retry_eligibility(root, "GS-T002")  # pyright: ignore[reportPrivateUsage]


def test_r13b_retry_readiness_uses_selected_logical_attempt() -> None:
    task_ids = acquisition.ACQUISITION_TASK_IDS
    initial_records = [
        {"task_id": "GS-T001", "acquisition_success": True},
        {"task_id": "GS-T002", "acquisition_success": False},
    ]
    retry_success = [
        {"task_id": "GS-T001", "acquisition_success": True},
        {"task_id": "GS-T002", "acquisition_success": True},
    ]
    retry_failure = [
        {"task_id": "GS-T001", "acquisition_success": True},
        {"task_id": "GS-T002", "acquisition_success": False},
    ]

    assert r13.r8_acquisition_readiness(initial_records, task_ids[:2], task_ids)[2] == 1
    assert r13.r8_acquisition_readiness(retry_success, task_ids[:2], task_ids)[2] == 2
    assert r13.r8_acquisition_readiness(retry_failure, task_ids[:2], task_ids)[2] == 1


def test_r13b_controlled_retry_preserves_initial_attempt_and_uses_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )
    root = tmp_path / "acquisition-r13b-20260924T000000Z"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"run_revision": "R13b", "configuration_hash": "test-hash"}),
        encoding="utf-8",
    )
    initial_started = root / "tasks" / "GS-T002" / "started.json"
    initial_started.parent.mkdir(parents=True)
    initial_started.write_text(
        json.dumps({"task_id": "GS-T002", "run_id": "initial-run"}),
        encoding="utf-8",
    )
    initial_completed = _retry_fixture(root, "GS-T002", acquisition_success=False)
    initial_started_before = hashlib.sha256(initial_started.read_bytes()).digest()
    initial_completed_before = hashlib.sha256(initial_completed.read_bytes()).digest()
    _retry_fixture(root, "GS-T001", acquisition_success=True)

    cases = tuple(
        SimpleNamespace(task=SimpleNamespace(id=task_id))
        for task_id in acquisition.ACQUISITION_TASK_IDS
    )
    settings = SimpleNamespace(
        openrouter_coding_model=acquisition.R13B_EXPECTED_CODING_MODEL,
        openrouter_abstraction_model=acquisition.R13B_EXPECTED_ABSTRACTION_MODEL,
    )
    calls: list[dict[str, object]] = []

    class FakeMemory:
        def __init__(self, _factory: object) -> None:
            pass

        def verify_connectivity(self) -> None:
            pass

        def ensure_recovery_pattern_vector_index(self) -> None:
            pass

    class FakeObjective:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    def fake_run(**kwargs: object) -> tuple[dict[str, object], None]:
        calls.append(kwargs)
        case = cast(SimpleNamespace, kwargs["case"])
        run_id = cast(str, kwargs["run_id"])
        return {
            "task_id": case.task.id,
            "run_id": run_id,
            "acquisition_success": True,
            "runtime_timeout_overrides": kwargs["runtime_timeout_overrides"],
        }, None

    def fake_task_cases(*_args: Any) -> tuple[SimpleNamespace, ...]:
        return cases

    def fake_settings(**_kwargs: Any) -> SimpleNamespace:
        return settings

    def fake_validate(*_args: Any, **_kwargs: Any) -> None:
        return None

    def fake_preflight(**_kwargs: Any) -> tuple[dict[str, SimpleNamespace], tuple[()]]:
        return (
            {task_id: SimpleNamespace() for task_id in acquisition.ACQUISITION_TASK_IDS},
            (),
        )

    def fake_hash(*_args: Any, **_kwargs: Any) -> str:
        return "test-hash"

    monkeypatch.setenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, "3600")
    monkeypatch.delenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(r13, "_task_cases", fake_task_cases)
    monkeypatch.setattr(r13, "_settings_for_agent", fake_settings)
    monkeypatch.setattr(r13, "_validate_preflight_configuration", fake_validate)
    monkeypatch.setattr(r13, "_preflight", fake_preflight)
    monkeypatch.setattr(r13, "_r8_configuration_hash", fake_hash)
    monkeypatch.setattr(r13, "ShortLivedNeo4jRepository", FakeMemory)
    monkeypatch.setattr(r13, "_run_r8_task", fake_run)
    monkeypatch.setattr("experiments.sprint3.FrozenSWEsmithObjective", FakeObjective)

    status, artifact_root = r13.run_gate_a1_acquisition_r13(
        tmp_path,
        resume_root=root,
        _configuration=configuration,
        _revision="R13b",
        _run_prefix="acquisition-r13b",
        _namespace=acquisition.R13B_NAMESPACE,
        _condition="acquisition-r13b",
        retry_task_id="GS-T002",
    )

    assert status == "READY_TO_RESUME_GATE_A1_ACQUISITION_R13B"
    assert artifact_root == root
    assert len(calls) == 1
    assert calls[0]["condition"] == "acquisition-r13b-retry-001"
    assert cast(str, calls[0]["run_id"]) != "initial-run"
    assert cast(float, calls[0]["agent_timeout_seconds"]) == 3600
    assert cast(float, calls[0]["objective_timeout_seconds"]) == 900
    retry_root = root / "tasks" / "GS-T002" / "retries" / "attempt-001"
    assert (retry_root / "started.json").is_file()
    assert (retry_root / "completed.json").is_file()
    assert hashlib.sha256(initial_started.read_bytes()).digest() == initial_started_before
    assert hashlib.sha256(initial_completed.read_bytes()).digest() == initial_completed_before
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["retry"]["policy"] == "single_explicit_failed_task_retry_v1"
    assert manifest["retry"]["selected_attempt"] == 2
    assert [attempt["kind"] for attempt in manifest["retry"]["attempts"]] == [
        "initial",
        "controlled_retry",
    ]
    assert manifest["eligible_acquisition_tasks"] == 2
    assert manifest["tasks"][1]["runtime_timeout_overrides"]["agent"]["effective_seconds"] == 3600


def test_r13b_cli_rejects_retry_without_resume_or_with_max_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["gate_a1", "--retry-r13b-task", "GS-T002"],
    )
    with pytest.raises(SystemExit):
        gate_a1.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate_a1",
            "--resume-acquisition-r13b",
            "run-root",
            "--retry-r13b-task",
            "GS-T002",
            "--max-new-tasks",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        gate_a1.main()
