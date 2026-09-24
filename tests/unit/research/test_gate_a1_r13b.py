# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

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
    assert r13.r8_acquisition_readiness(
        retry_success + [{"task_id": "GS-T002", "acquisition_success": True}],
        task_ids[:2],
        task_ids,
    )[2] == 2


def test_r13b_effective_records_select_each_task_local_retry_once(tmp_path: Path) -> None:
    root = tmp_path / "acquisition-r13b-20260924T020000Z"
    manifest: dict[str, object] = {
        "run_revision": "R13b",
        "config_version": "gate-a1-r13b-v2",
        "retry": {
            "task_id": "GS-T004",
            "selected_attempt": 2,
        },
        "pattern_retry": {
            "task_id": "GS-T003",
            "selected_attempt": 1,
        },
    }

    def write(path: Path, record: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    for task_id in ("GS-T001", "GS-T002", "GS-T003", "GS-T004"):
        write(
            root / "tasks" / task_id / "completed.json",
            {"task_id": task_id, "acquisition_success": task_id == "GS-T001"},
        )
    write(
        root / "tasks" / "GS-T002" / "retries" / "attempt-001" / "completed.json",
        {
            "task_id": "GS-T002",
            "attempt": 2,
            "kind": "controlled_retry",
            "acquisition_success": True,
        },
    )
    write(
        root / "tasks" / "GS-T003" / "pattern-retries" / "attempt-001" / "completed.json",
        {
            "selected_task_record": {
                "task_id": "GS-T003",
                "acquisition_success": True,
            }
        },
    )
    write(
        root / "tasks" / "GS-T004" / "retries" / "attempt-001" / "completed.json",
        {
            "task_id": "GS-T004",
            "attempt": 2,
            "kind": "controlled_retry",
            "acquisition_success": True,
        },
    )

    records = r13._r13b_effective_task_records(root, manifest)  # pyright: ignore[reportPrivateUsage]

    assert [record["task_id"] for record in records] == [
        "GS-T001",
        "GS-T002",
        "GS-T003",
        "GS-T004",
    ]
    assert all(record["acquisition_success"] is True for record in records)
    assert r13.r8_acquisition_readiness(
        records,
        ("GS-T001", "GS-T002", "GS-T003", "GS-T004"),
        acquisition.ACQUISITION_TASK_IDS,
    ) == ("READY_TO_RESUME_GATE_A1_ACQUISITION_R8", False, 4)


def test_r13b_manifest_refresh_is_bookkeeping_only_and_preserves_task_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "acquisition-r13b-20260924T030000Z"
    manifest: dict[str, object] = {
        "run_revision": "R13b",
        "config_version": "gate-a1-r13b-v2",
        "tasks": [],
        "completed_tasks": 3,
        "eligible_acquisition_tasks": 3,
        "acquisition_corpus_ready": False,
        "pattern_retry": {"task_id": "GS-T003", "selected_attempt": 1},
    }
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def write_task(task_id: str, record: dict[str, object]) -> Path:
        completed = root / "tasks" / task_id / "completed.json"
        completed.parent.mkdir(parents=True, exist_ok=True)
        completed.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        return completed

    task_files = [
        write_task(
            task_id,
            {"task_id": task_id, "acquisition_success": task_id == "GS-T001"},
        )
        for task_id in ("GS-T001", "GS-T002", "GS-T003", "GS-T004")
    ]
    retry = root / "tasks" / "GS-T002" / "retries" / "attempt-001" / "completed.json"
    retry.parent.mkdir(parents=True, exist_ok=True)
    retry.write_text(
        json.dumps(
            {
                "task_id": "GS-T002",
                "attempt": 2,
                "kind": "controlled_retry",
                "acquisition_success": True,
            }
        ),
        encoding="utf-8",
    )
    pattern_retry = (
        root
        / "tasks"
        / "GS-T003"
        / "pattern-retries"
        / "attempt-001"
        / "completed.json"
    )
    pattern_retry.parent.mkdir(parents=True, exist_ok=True)
    pattern_retry.write_text(
        json.dumps(
            {
                "selected_task_record": {
                    "task_id": "GS-T003",
                    "acquisition_success": True,
                }
            }
        ),
        encoding="utf-8",
    )
    controlled_retry = (
        root / "tasks" / "GS-T004" / "retries" / "attempt-001" / "completed.json"
    )
    controlled_retry.parent.mkdir(parents=True, exist_ok=True)
    controlled_retry.write_text(
        json.dumps(
            {
                "task_id": "GS-T004",
                "attempt": 2,
                "kind": "controlled_retry",
                "acquisition_success": True,
            }
        ),
        encoding="utf-8",
    )
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in task_files}

    marker, refreshed_root = r13.refresh_r13b_manifest(root)

    refreshed = json.loads((refreshed_root / "manifest.json").read_text(encoding="utf-8"))
    assert marker == r13.R13B_MANIFEST_REFRESH_MARKER
    assert refreshed["completed_tasks"] == 4
    assert refreshed["completed_task_ids"] == [
        "GS-T001",
        "GS-T002",
        "GS-T003",
        "GS-T004",
    ]
    assert refreshed["eligible_acquisition_tasks"] == 4
    assert refreshed["acquisition_corpus_ready"] is False
    assert refreshed["status"] == "READY_TO_RESUME_GATE_A1_ACQUISITION_R13B"
    assert {
        record["task_id"]: record["acquisition_success"]
        for record in refreshed["tasks"]
    } == {
        "GS-T001": True,
        "GS-T002": True,
        "GS-T003": True,
        "GS-T004": True,
    }
    assert all(
        hashlib.sha256(path.read_bytes()).digest() == digest
        for path, digest in before.items()
    )


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
    guard = cast(Any, calls[0]["pre_mutation_guard"])
    assert guard.config.effective_nudge_seconds == 1200
    assert guard.config.effective_abort_seconds == 1800
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


def _pattern_retry_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    records: dict[str, object] = {}
    for task_id in acquisition.ACQUISITION_TASK_IDS:
        record: dict[str, object] = {
            "task_id": task_id,
            "run_id": f"initial-{task_id}",
            "task_success": task_id == "GS-T003",
            "acquisition_success": task_id in {"GS-T001", "GS-T002"},
        }
        if task_id == "GS-T003":
            record.update(
                {
                    "complete_trusted_lineage": True,
                    "counts": {"recoveries": 1, "patterns": 0},
                    "recovery_evidence_source": (
                        "objective_anchored_v1"
                    ),
                    "objective_success_trigger_action_id": "action-003",
                    "recovery_lineage": {
                        "failure_id": "failure-003",
                        "resolution_id": "resolution-003",
                        "outcome_id": "outcome-003",
                    },
                    "pattern_created": False,
                    "pattern_persisted": False,
                    "pattern_embedded": False,
                    "acquisition_reason": "recovery_pattern_not_created",
                }
            )
        records[task_id] = record
        completed = root / "tasks" / task_id / "completed.json"
        completed.parent.mkdir(parents=True, exist_ok=True)
        completed.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    manifest = {
        "run_revision": "R13b",
        "config_version": "gate-a1-r13b-v2",
        "tasks": list(records.values()),
        "completed_task_ids": list(acquisition.ACQUISITION_TASK_IDS),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, records


class _PatternRetryLineage:
    def __init__(self) -> None:
        self.failure = SimpleNamespace(id="failure-003")
        self.resolution = SimpleNamespace(id="resolution-003")
        self.outcome = SimpleNamespace(id="outcome-003")
        self.recovery_action = SimpleNamespace(
            planned_action=SimpleNamespace(
                id="action-003",
                tool="git",
                operation="apply_patch",
            )
        )

    def model_copy(self, *, update: dict[str, object]) -> "_PatternRetryLineage":
        del update
        return self


def test_r13b_pattern_retry_reuses_canonical_lineage_and_preserves_original(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )
    manifest_path, _records = _pattern_retry_fixture(
        tmp_path / "acquisition-r13b-20260924T010000Z"
    )
    original = manifest_path.parent / "tasks" / "GS-T003" / "completed.json"
    original_digest = hashlib.sha256(original.read_bytes()).digest()
    lineage = _PatternRetryLineage()
    calls: dict[str, object] = {"abstract": 0, "embed": 0, "update": 0}

    class FakeMemory:
        def __init__(self, _factory: object) -> None:
            pass

        def verify_connectivity(self) -> None:
            pass

        def get_recovery_evidence(self, failure_id: str) -> _PatternRetryLineage:
            assert failure_id == "failure-003"
            return lineage

        def get_recovery_pattern(self, _pattern_id: str) -> object:
            raise r13.EntityNotFoundError("not found")

        def update_recovery_pattern_embedding(
            self,
            pattern_id: str,
            embedding: list[float],
        ) -> None:
            assert pattern_id == "pattern-003"
            assert embedding == [0.1, 0.2]
            calls["update"] = cast(int, calls["update"]) + 1

        def close(self) -> None:
            pass

    def fake_settings(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            openrouter_abstraction_model=acquisition.R13B_EXPECTED_ABSTRACTION_MODEL,
            neo4j_uri="neo4j://unused",
            neo4j_username="unused",
            neo4j_password="unused",
            neo4j_database="neo4j",
        )

    def fake_evidence(_lineage: object) -> object:
        return SimpleNamespace(source_failure_id="failure-003")

    def fake_pattern_id(_evidence: object) -> str:
        return "pattern-003"

    def fake_abstract(*args: object) -> object:
        assert args[0] is lineage
        calls["abstract"] = cast(int, calls["abstract"]) + 1
        return SimpleNamespace(id="pattern-003", embedding=None)

    class FakeEmbedder:
        def embed_pattern(self, pattern: object) -> object:
            assert cast(Any, pattern).id == "pattern-003"
            calls["embed"] = cast(int, calls["embed"]) + 1
            return SimpleNamespace(id="pattern-003", embedding=[0.1, 0.2])

    monkeypatch.setattr(r13, "_settings_for_agent", fake_settings)
    monkeypatch.setattr(r13, "ShortLivedNeo4jRepository", FakeMemory)
    monkeypatch.setattr(r13, "build_recovery_evidence_package", fake_evidence)
    monkeypatch.setattr(r13, "deterministic_recovery_pattern_id", fake_pattern_id)
    monkeypatch.setattr(r13, "abstract_and_persist_recovery_pattern", fake_abstract)
    monkeypatch.setattr(r13, "RecoveryPatternEmbedder", FakeEmbedder)

    status, artifact_root = r13.run_gate_a1_pattern_retry_r13b(
        tmp_path,
        resume_root=manifest_path.parent,
        task_id="GS-T003",
        _configuration=configuration,
    )

    assert status == "READY_FOR_GATE_A1_RETRIEVAL_EVALUATION"
    assert artifact_root == manifest_path.parent
    assert hashlib.sha256(original.read_bytes()).digest() == original_digest
    retry_root = artifact_root / "tasks" / "GS-T003" / "pattern-retries" / "attempt-001"
    started = json.loads((retry_root / "started.json").read_text(encoding="utf-8"))
    completed = json.loads((retry_root / "completed.json").read_text(encoding="utf-8"))
    assert started["coding_agent_rerun"] is False
    assert completed["original_run_id"] == "initial-GS-T003"
    assert completed["recovery_evidence_reused"] is True
    assert completed["source_failure_id"] == "failure-003"
    assert completed["trusted_recovery_action_id"] == "action-003"
    assert completed["abstraction_model"] == acquisition.R13B_EXPECTED_ABSTRACTION_MODEL
    assert completed["acquisition_success_after_pattern_retry"] is True
    assert completed["selected_task_record"]["acquisition_success"] is True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pattern_retry"]["policy"] == "single_explicit_pattern_retry_v1"
    assert manifest["pattern_retry"]["selected_attempt"] == 1
    assert manifest["eligible_acquisition_tasks"] == 3
    assert calls == {"abstract": 1, "embed": 1, "update": 1}


@pytest.mark.parametrize("failure_stage", ["abstraction", "embedding"])
def test_r13b_pattern_retry_failure_keeps_logical_corpus_at_two(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )
    manifest_path, _records = _pattern_retry_fixture(
        tmp_path / f"acquisition-r13b-20260924T01{failure_stage}Z"
    )
    lineage = _PatternRetryLineage()

    class FakeMemory:
        def __init__(self, _factory: object) -> None:
            pass

        def verify_connectivity(self) -> None:
            pass

        def get_recovery_evidence(self, _failure_id: str) -> _PatternRetryLineage:
            return lineage

        def get_recovery_pattern(self, _pattern_id: str) -> object:
            raise r13.EntityNotFoundError("not found")

        def update_recovery_pattern_embedding(
            self,
            _pattern_id: str,
            _embedding: list[float],
        ) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        r13,
        "_settings_for_agent",
        lambda **_kwargs: SimpleNamespace(
            openrouter_abstraction_model=acquisition.R13B_EXPECTED_ABSTRACTION_MODEL,
            neo4j_uri="neo4j://unused",
            neo4j_username="unused",
            neo4j_password="unused",
            neo4j_database="neo4j",
        ),
    )
    monkeypatch.setattr(r13, "ShortLivedNeo4jRepository", FakeMemory)
    monkeypatch.setattr(
        r13,
        "build_recovery_evidence_package",
        lambda _lineage: SimpleNamespace(source_failure_id="failure-003"),
    )
    monkeypatch.setattr(
        r13,
        "deterministic_recovery_pattern_id",
        lambda _evidence: "pattern-003",
    )
    if failure_stage == "abstraction":
        monkeypatch.setattr(
            r13,
            "abstract_and_persist_recovery_pattern",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("structured output")),
        )
    else:
        monkeypatch.setattr(
            r13,
            "abstract_and_persist_recovery_pattern",
            lambda *_args: SimpleNamespace(id="pattern-003", embedding=None),
        )
        class FailingEmbedder:
            def embed_pattern(self, _pattern: object) -> object:
                raise RuntimeError("embedding")

        monkeypatch.setattr(r13, "RecoveryPatternEmbedder", FailingEmbedder)

    status, _artifact_root = r13.run_gate_a1_pattern_retry_r13b(
        tmp_path,
        resume_root=manifest_path.parent,
        task_id="GS-T003",
        _configuration=configuration,
    )

    assert status == "READY_FOR_GATE_A1_RETRIEVAL_EVALUATION"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eligible_acquisition_tasks"] == 2
    completed = json.loads(
        (
            manifest_path.parent
            / "tasks"
            / "GS-T003"
            / "pattern-retries"
            / "attempt-001"
            / "completed.json"
        ).read_text(encoding="utf-8")
    )
    assert completed["acquisition_success_after_pattern_retry"] is False
    assert completed["persistence_error"] is not None


def test_r13b_pattern_retry_reuses_fully_persisted_pattern_without_duplicate_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )
    manifest_path, _records = _pattern_retry_fixture(
        tmp_path / "acquisition-r13b-20260924T010003Z"
    )
    lineage = _PatternRetryLineage()

    class FakeMemory:
        def __init__(self, _factory: object) -> None:
            pass

        def verify_connectivity(self) -> None:
            pass

        def get_recovery_evidence(self, _failure_id: str) -> _PatternRetryLineage:
            return lineage

        def get_recovery_pattern(self, _pattern_id: str) -> object:
            return SimpleNamespace(
                pattern=SimpleNamespace(id="pattern-003", embedding=[0.1, 0.2])
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        r13,
        "_settings_for_agent",
        lambda **_kwargs: SimpleNamespace(
            openrouter_abstraction_model=acquisition.R13B_EXPECTED_ABSTRACTION_MODEL,
            neo4j_uri="neo4j://unused",
            neo4j_username="unused",
            neo4j_password="unused",
            neo4j_database="neo4j",
        ),
    )
    monkeypatch.setattr(r13, "ShortLivedNeo4jRepository", FakeMemory)
    monkeypatch.setattr(r13, "build_recovery_evidence_package", lambda _lineage: object())
    monkeypatch.setattr(r13, "deterministic_recovery_pattern_id", lambda _evidence: "pattern-003")
    monkeypatch.setattr(
        r13,
        "abstract_and_persist_recovery_pattern",
        lambda *_args: pytest.fail("existing pattern must not be recreated"),
    )
    monkeypatch.setattr(
        r13,
        "RecoveryPatternEmbedder",
        lambda: pytest.fail("existing embedding must not be regenerated"),
    )

    status, _artifact_root = r13.run_gate_a1_pattern_retry_r13b(
        tmp_path,
        resume_root=manifest_path.parent,
        task_id="GS-T003",
        _configuration=configuration,
    )

    assert status == "READY_FOR_GATE_A1_RETRIEVAL_EVALUATION"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eligible_acquisition_tasks"] == 3


def test_r13b_pattern_retry_cli_rejects_invalid_combinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    for argv in (
        ["gate_a1", "--retry-r13b-pattern", "GS-T003"],
        [
            "gate_a1",
            "--resume-acquisition-r13b",
            "run-root",
            "--retry-r13b-pattern",
            "GS-T003",
            "--max-new-tasks",
            "1",
        ],
        [
            "gate_a1",
            "--resume-acquisition-r13b",
            "run-root",
            "--retry-r13b-pattern",
            "GS-T003",
            "--retry-r13b-task",
            "GS-T003",
        ],
        ["gate_a1", "--acquire-r13b", "--retry-r13b-pattern", "GS-T003"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            gate_a1.main()
