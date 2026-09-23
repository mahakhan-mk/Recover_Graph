from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from graph_swarm.agent.coding_agent import AgentWallClockTimeoutError
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.research import gate_a1
from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research.gate_a1_r8 import objective_final_check_policy
from graph_swarm.research.gate_a1_r10 import R10ObjectiveController
from graph_swarm.research.gate_a1_r11 import R11ObjectiveController
from graph_swarm.research.gate_a1_r12 import R12ObjectiveController
from graph_swarm.research.runner import load_experiment_configuration

ROOT = Path(__file__).resolve().parents[3]


def _configuration() -> Any:
    return load_experiment_configuration(
        ROOT / "configs/experiments/gate_a1_acquisition_r11.yaml",
        project_root=ROOT,
    )


def _environments() -> dict[str, SimpleNamespace]:
    return {
        task_id: SimpleNamespace(
            environment_fingerprint=f"environment-{task_id}",
            container_image="benchmark:image",
        )
        for task_id in acquisition.ACQUISITION_TASK_IDS
    }


def test_r11_restores_effective_track_a_model_without_qwen() -> None:
    settings = acquisition._settings_for_agent(  # pyright: ignore[reportPrivateUsage]
        coding_model=acquisition.R11_EXPECTED_CODING_MODEL,
        abstraction_model=acquisition.R11_EXPECTED_ABSTRACTION_MODEL,
    )

    assert settings.openrouter_coding_model == "nex-agi/nex-n2.5-pro:free"
    assert settings.openrouter_coding_model != "qwen/qwen3-coder:free"
    assert settings.openrouter_abstraction_model == "cohere/north-mini-code:free"


def test_r11_configuration_retains_r10_runtime_boundaries() -> None:
    configuration = _configuration()

    assert configuration.config.run_revision == "R11"
    assert acquisition.R11_NAMESPACE == "GS-E003/Gate-A1/acquisition-r11"
    assert configuration.config.revision_reason == (
        "restore_track_a_frozen_nex_model_after_invalid_r10_qwen_provider_revision"
    )
    assert configuration.config.command_argv_policy == (
        "structured_argv_canonicalization_v2"
    )
    assert configuration.config.agent_timeout_seconds == 600
    assert configuration.config.objective_timeout_seconds == 900
    assert configuration.config.limits.max_actions is None
    assert configuration.config.limits.max_requests is None
    assert configuration.model.settings["temperature"] == 0
    assert configuration.config.recovery_event_semantics == (
        "trusted_action_semantics_v2"
    )
    assert configuration.config.repository_mutation_evidence_policy == (
        "git_worktree_content_fingerprint_v2"
    )


def test_r11_hash_includes_restored_model_identity_and_r10_policy() -> None:
    configuration = _configuration()
    environments = cast(Any, _environments())
    nex = SimpleNamespace(
        openrouter_coding_model="nex-agi/nex-n2.5-pro:free",
        openrouter_abstraction_model="cohere/north-mini-code:free",
    )
    qwen = SimpleNamespace(
        openrouter_coding_model="qwen/qwen3-coder:free",
        openrouter_abstraction_model="cohere/north-mini-code:free",
    )

    nex_hash = acquisition._r8_configuration_hash(configuration, nex, environments)  # pyright: ignore[reportPrivateUsage]
    qwen_hash = acquisition._r8_configuration_hash(configuration, qwen, environments)  # pyright: ignore[reportPrivateUsage]

    assert nex_hash != qwen_hash


def test_r11_retains_timeout_final_objective_guard() -> None:
    timeout = AgentWallClockTimeoutError(600)

    assert objective_final_check_policy(
        "R11", timeout, has_repository_mutation=False
    ) == "skipped_no_repository_mutation_after_agent_timeout"
    assert objective_final_check_policy(
        "R11", timeout, has_repository_mutation=True
    ) == "performed"


def test_r11_retains_r10_and_r9_recovery_semantics() -> None:
    assert issubclass(R11ObjectiveController, R10ObjectiveController)
    assert R11ObjectiveController.mutation_fingerprint is not None
    assert acquisition.R9_RECOVERY_EVENT_SEMANTICS == "trusted_action_semantics_v2"
    assert acquisition.R9_REPOSITORY_MUTATION_EVIDENCE_POLICY == (
        "git_worktree_content_fingerprint_v2"
    )


def test_r11_retains_structured_argv_canonicalization() -> None:
    from graph_swarm.agent.tools.run_command import canonicalize_command

    assert canonicalize_command(["python -m pytest -q"]) == [
        "python",
        "-m",
        "pytest",
        "-q",
    ]


def test_r11_cli_dispatch_isolated_from_r10_and_r9(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path | None, int | None]] = []

    def fake_r11(
        project_root: Path,
        *,
        resume_root: Path | None = None,
        max_new_tasks: int | None = None,
    ) -> tuple[str, Path]:
        calls.append((project_root, resume_root, max_new_tasks))
        return "READY_TO_RESUME_GATE_A1_ACQUISITION_R11", tmp_path

    monkeypatch.setattr(acquisition, "run_gate_a1_acquisition_r11", fake_r11)
    def fail_r10(**_kwargs: object) -> tuple[str, Path]:
        raise AssertionError("R10 dispatched")

    def fail_r9(**_kwargs: object) -> tuple[str, Path]:
        raise AssertionError("R9 dispatched")

    monkeypatch.setattr(acquisition, "run_gate_a1_acquisition_r10", fail_r10)
    monkeypatch.setattr(acquisition, "run_gate_a1_acquisition_r9", fail_r9)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate-a1", "--acquire-r11", "--max-new-tasks", "1", "--project-root", str(tmp_path)],
    )

    assert gate_a1.main() == 0
    assert calls == [(tmp_path.resolve(), None, 1)]


def test_r11_rejects_historical_resume_roots_without_provider_work(
    tmp_path: Path,
) -> None:
    from graph_swarm.research.gate_a1_r11 import run_gate_a1_acquisition_r11

    for revision in ("r8", "r9", "r10"):
        root = tmp_path / f"acquisition-{revision}-historical"
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps({"run_revision": revision.upper()}) + "\n",
            encoding="utf-8",
        )
        try:
            run_gate_a1_acquisition_r11(ROOT, resume_root=root)
        except ValueError as error:
            assert "acquisition-r11" in str(error) or "rejects resume" in str(error)
        else:
            raise AssertionError(f"accepted historical {revision} resume root")


def test_r11_historical_configuration_files_are_not_rewritten() -> None:
    paths = [
        ROOT / "configs/experiments/gate_a1_acquisition_r8.yaml",
        ROOT / "configs/experiments/gate_a1_acquisition_r9.yaml",
        ROOT / "configs/experiments/gate_a1_acquisition_r10.yaml",
    ]
    before = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
    assert all(path.is_file() for path in paths)
    assert [hashlib.sha256(path.read_bytes()).digest() for path in paths] == before


def test_r12_configuration_anchors_acquisition_to_objective_and_mutation() -> None:
    root = Path(__file__).resolve().parents[3]
    configuration = load_experiment_configuration(
        root / "configs/experiments/gate_a1_acquisition_r12.yaml",
        project_root=root,
    )
    acquisition._validate_r12_harness_configuration(configuration)  # pyright: ignore[reportPrivateUsage]
    assert configuration.config.run_revision == "R12"
    assert configuration.config.recovery_event_semantics == (
        "objective_anchor_trusted_mutation_objective_success_v1"
    )
    assert configuration.config.stopping_policy == (
        "objective_anchored_recovery_or_timeout_v1"
    )
    assert configuration.config.objective_mutation_check_policy == (
        "pre_agent_failure_then_post_mutation_success_v1"
    )
    assert acquisition.R12_EXPECTED_CODING_MODEL == acquisition.R11_EXPECTED_CODING_MODEL
    assert acquisition.R12_EXPECTED_ABSTRACTION_MODEL == acquisition.R11_EXPECTED_ABSTRACTION_MODEL


def test_r12_hash_includes_each_objective_anchoring_policy() -> None:
    root = Path(__file__).resolve().parents[3]
    configuration = load_experiment_configuration(
        root / "configs/experiments/gate_a1_acquisition_r12.yaml",
        project_root=root,
    )
    settings = SimpleNamespace(
        openrouter_coding_model=acquisition.R12_EXPECTED_CODING_MODEL,
        openrouter_abstraction_model=acquisition.R12_EXPECTED_ABSTRACTION_MODEL,
    )
    environments = cast(Any, _environments())
    baseline = acquisition._r8_configuration_hash(configuration, settings, environments)  # pyright: ignore[reportPrivateUsage]
    for field in (
        "recovery_event_semantics",
        "stopping_policy",
        "objective_mutation_check_policy",
    ):
        changed = replace(
            configuration,
            config=configuration.config.model_copy(update={field: "changed"}),
        )
        assert (
            acquisition._r8_configuration_hash(  # pyright: ignore[reportPrivateUsage]
                changed,
                settings,
                environments,
            )
            != baseline
        )


def test_r12_controller_stops_after_first_successful_mutation_without_test_event(
    tmp_path: Path,
) -> None:
    dependencies = AgentDependencies(tmp_path, "run-r12", "GS-T001")
    objective_results = iter((False, True))

    @dataclass(frozen=True)
    class Observation:
        status: str
        return_code: int

    class Objective:
        observations: list[Observation]

        def __init__(self) -> None:
            self.observations = []

        def __call__(self, _task: Any, _workspace: Path) -> bool:
            passed = next(objective_results)
            self.observations.append(
                Observation("passed" if passed else "test_failure", 0 if passed else 1)
            )
            return passed

    objective = Objective()

    controller = R12ObjectiveController(
        task=SimpleNamespace(id="GS-T001"),
        workspace=tmp_path,
        objective=objective,
        dependencies=dependencies,
    )
    assert controller.check_pre_agent_objective() is False
    action = PlannedAction(
        id="real-mutation",
        run_id="run-r12",
        task_id="GS-T001",
        tool="write_file",
        operation="write_file",
        arguments={"path": "source.py", "content": "fixed"},
        planned_at=datetime.now(UTC),
    )
    dependencies.record_planned_action(action)
    fingerprints = iter(("before", "after"))
    controller.mutation_fingerprint = lambda _workspace: next(fingerprints)  # type: ignore[method-assign]
    controller.before_action(action)
    event = AgentEvent(
        event_id="mutation-event",
        run_id=action.run_id,
        task_id=action.task_id,
        action_id=action.id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=ActionResult(
            action_id=action.id,
            tool_name="write_file",
            success=True,
            exit_code=0,
            started_at=action.planned_at,
            completed_at=action.planned_at,
        ),
        occurred_at=action.planned_at,
    )
    try:
        controller.after_event(event)
    except Exception as error:
        from graph_swarm.research.gate_a1_r8 import ObjectiveSatisfied

        assert isinstance(error, ObjectiveSatisfied)
    else:
        raise AssertionError("R12 did not stop after trusted objective success")
    assert controller.objective_success_trigger_action_id == action.id
    assert len(dependencies.events) == 0


@dataclass(frozen=True)
class _R12Observation:
    status: str
    return_code: int | None


class _R12ScriptedObjective:
    def __init__(
        self,
        results: list[tuple[bool, str, int | None, bool]],
        observations: list[_R12Observation] | None = None,
    ) -> None:
        self.results = iter(results)
        self.observations = list(observations or [])

    def __call__(self, _task: Any, _workspace: Path) -> bool:
        passed, status, return_code, append = next(self.results)
        if append:
            self.observations.append(_R12Observation(status, return_code))
        return passed


def _r12_controller(tmp_path: Path, objective: Any) -> R12ObjectiveController:
    return R12ObjectiveController(
        task=SimpleNamespace(id="GS-T001"),
        workspace=tmp_path,
        objective=objective,
        dependencies=AgentDependencies(tmp_path, "run-r12-evidence", "GS-T001"),
    )


def test_r12_rejects_infrastructure_false_and_stale_pre_agent_observations(
    tmp_path: Path,
) -> None:
    infrastructure = _r12_controller(
        tmp_path,
        _R12ScriptedObjective([(False, "objective_infrastructure_failure", 2, True)]),
    )
    assert infrastructure.check_pre_agent_objective() is False
    assert infrastructure.pre_agent_objective_error is not None
    assert infrastructure.termination_reason == "pre_agent_objective_not_task_failure"

    stale = _r12_controller(
        tmp_path,
        _R12ScriptedObjective(
            [(False, "test_failure", 1, False)],
            [_R12Observation("test_failure", 1)],
        ),
    )
    assert stale.check_pre_agent_objective() is False
    assert stale.pre_agent_objective_error is not None
    assert stale.pre_agent_objective_observation is None


def test_r12_requires_fresh_valid_post_mutation_evidence(tmp_path: Path) -> None:
    objective = _R12ScriptedObjective(
        [
            (False, "test_failure", 1, True),
            (False, "objective_infrastructure_failure", 2, True),
        ]
    )
    controller = _r12_controller(tmp_path, objective)
    assert controller.check_pre_agent_objective() is False
    assert controller.check_objective(SimpleNamespace(action_id="mutation")) is False
    assert controller.objective_success is False
    assert controller.objective_evidence_error is not None

    stale_success = _R12ScriptedObjective(
        [(False, "test_failure", 1, True), (True, "passed", 0, False)],
        [_R12Observation("passed", 0)],
    )
    stale_controller = _r12_controller(tmp_path, stale_success)
    assert stale_controller.check_pre_agent_objective() is False
    assert stale_controller.check_objective(SimpleNamespace(action_id="mutation")) is False
    assert stale_controller.objective_success is False
    assert stale_controller.objective_evidence_error is not None


def test_r12_final_check_policy_is_always_mutation_bound() -> None:
    assert objective_final_check_policy(
        "R12",
        None,
        has_repository_mutation=False,
    ) == "skipped_r12_requires_mutation_bound_objective"
