from __future__ import annotations

import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from graph_swarm.agent.coding_agent import AgentWallClockTimeoutError
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.research import gate_a1
from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research.gate_a1_r8 import objective_final_check_policy
from graph_swarm.research.runner import load_experiment_configuration


def _configuration() -> Any:
    root = Path(__file__).resolve().parents[3]
    return load_experiment_configuration(
        root / "configs/experiments/gate_a1_acquisition_r10.yaml",
        project_root=root,
    )


def test_r10_configuration_freezes_final_model_and_r9_runtime_policies() -> None:
    configuration = _configuration()

    assert configuration.config.run_revision == "R10"
    assert configuration.config.revision_reason == (
        "agent_runtime_and_structured_argv_stabilization_after_r9"
    )
    assert configuration.config.command_argv_policy == (
        "structured_argv_canonicalization_v2"
    )
    assert configuration.config.agent_timeout_seconds == 600
    assert configuration.config.objective_timeout_seconds == 900
    assert configuration.model.settings["temperature"] == 0
    assert configuration.config.task_manifest.endswith("benchmark/manifests/pilot.jsonl")


def test_r10_cli_dispatch_is_distinct_from_r9(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, Path | None, int | None]] = []

    def fake_r10(
        project_root: Path,
        *,
        resume_root: Path | None = None,
        max_new_tasks: int | None = None,
    ) -> tuple[str, Path]:
        calls.append((project_root, resume_root, max_new_tasks))
        return "READY_TO_RESUME_GATE_A1_ACQUISITION_R10", tmp_path

    monkeypatch.setattr(acquisition, "run_gate_a1_acquisition_r10", fake_r10)
    def fail_r9(**_kwargs: object) -> tuple[str, Path]:
        raise AssertionError("R9 dispatched")

    monkeypatch.setattr(acquisition, "run_gate_a1_acquisition_r9", fail_r9)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate-a1", "--acquire-r10", "--max-new-tasks", "1", "--project-root", str(tmp_path)],
    )

    assert gate_a1.main() == 0
    assert calls == [(tmp_path.resolve(), None, 1)]


def test_r10_configuration_hash_includes_model_identity_and_argv_policy() -> None:
    configuration = _configuration()
    settings = SimpleNamespace(
        openrouter_coding_model="qwen/qwen3-coder:free",
        openrouter_abstraction_model="cohere/north-mini-code:free",
    )
    environments = {
        task_id: SimpleNamespace(
            environment_fingerprint=f"environment-{task_id}",
            container_image="benchmark:image",
        )
        for task_id in acquisition.ACQUISITION_TASK_IDS
    }
    baseline = acquisition._r8_configuration_hash(  # pyright: ignore[reportPrivateUsage]
        configuration, settings, cast(Any, environments)
    )
    changed_model = SimpleNamespace(
        openrouter_coding_model="qwen/other-model:free",
        openrouter_abstraction_model="cohere/north-mini-code:free",
    )
    changed_policy = replace(
        configuration,
        config=configuration.config.model_copy(
            update={"command_argv_policy": "other_policy"}
        ),
    )

    assert (
        acquisition._r8_configuration_hash(  # pyright: ignore[reportPrivateUsage]
            configuration, changed_model, cast(Any, environments)
        )
        != baseline
    )
    assert (
        acquisition._r8_configuration_hash(  # pyright: ignore[reportPrivateUsage]
            changed_policy, settings, cast(Any, environments)
        )
        != baseline
    )


def test_r10_timeout_guard_only_skips_zero_mutation_wall_clock_timeout() -> None:
    timeout = AgentWallClockTimeoutError(600)

    assert objective_final_check_policy(
        "R10", timeout, has_repository_mutation=False
    ) == "skipped_no_repository_mutation_after_agent_timeout"
    assert objective_final_check_policy(
        "R10", timeout, has_repository_mutation=True
    ) == "performed"
    assert objective_final_check_policy(
        "R10", RuntimeError("ordinary failure"), has_repository_mutation=False
    ) == "performed"
    assert objective_final_check_policy(
        "R9", timeout, has_repository_mutation=False
    ) == "performed"


def test_canonical_pytest_argv_retains_test_failure_semantics() -> None:
    action = PlannedAction(
        id="pytest",
        run_id="run-1",
        task_id="GS-T001",
        tool="run_command",
        operation="run_command",
        arguments={
            "command": [
                "python",
                "-m",
                "pytest",
                "tests/test_locales.py::TestIcelandicLocale::test_format_timeframe",
                "-q",
            ]
        },
        planned_at=datetime.now(UTC),
    )
    event = AgentEvent(
        event_id="event-1",
        run_id=action.run_id,
        task_id=action.task_id,
        action_id=action.id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=ActionResult(
            action_id=action.id,
            tool_name="run_command",
            success=False,
            exit_code=1,
            output="failed",
            started_at=action.planned_at,
            completed_at=action.planned_at,
        ),
        occurred_at=action.planned_at,
    )

    from graph_swarm.detection.failure_detector import detect_failure
    from graph_swarm.memory.recovery_evidence import is_test_execution

    failure = detect_failure(event, action)
    assert is_test_execution(action)
    assert failure is not None
    assert failure.failure_type.value == "test_failure"
