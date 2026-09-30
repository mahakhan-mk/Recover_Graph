from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

from experiments import run_sprint3b_kilo_v4_canary as v4
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.prompts import (
    R13B_RUNTIME_GUIDANCE,
    R13B_SYSTEM_PROMPT,
    ROLLOUT1_SYSTEM_PROMPT,
    resolve_system_prompt,
    system_prompt_sha256,
)
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.tasks import Task
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    ExperimentConfiguration,
    ExperimentConfigurationError,
    ExperimentRunner,
    load_experiment_configuration,
)
from graph_swarm.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
CANARY_CONFIG = ROOT / "configs" / "experiments" / "sprint3b_kilo_v4_canary.yaml"
V3_CONFIG = ROOT / "configs" / "experiments" / "sprint3b_kilo_v3.yaml"

# These tests intentionally call the runner's construction seam without
# executing a model/provider.
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false


def _settings() -> Settings:
    return Settings(
        neo4j_uri="neo4j+s://example.databases.neo4j.io",
        neo4j_username="example-user",
        neo4j_password="example-password",
        neo4j_database="example-db",
        agent_request_limit=3,
    )


def _base_config(**updates: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "experiment_id": "GS-E003",
        "rollout": "development",
        "model_config": "configs/models/kilo_coding.yaml",
        "conditions": ["B0", "T"],
        "limits": {"max_actions": 28, "max_requests": 24, "timeout_seconds": 300},
        "development": True,
        "pilot": True,
    }
    config.update(updates)
    return config


def test_supported_prompt_ids_resolve_to_exact_existing_text() -> None:
    assert resolve_system_prompt("ROLLOUT1_SYSTEM_PROMPT") == ROLLOUT1_SYSTEM_PROMPT
    assert resolve_system_prompt("R13B_SYSTEM_PROMPT") == R13B_SYSTEM_PROMPT


def test_unknown_prompt_id_fails_closed_with_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown system prompt identifier"):
        resolve_system_prompt("NOT_A_SUPPORTED_PROMPT")

    with pytest.raises(ValueError, match="unknown system prompt identifier"):
        ExperimentConfiguration.model_validate(
            _base_config(system_prompt="NOT_A_SUPPORTED_PROMPT")
        )

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(
        "\n".join(
            [
                "experiment_id: GS-E003",
                "rollout: development",
                "model_config: configs/models/kilo_coding.yaml",
                "conditions: [B0]",
                "limits: {max_actions: 28, max_requests: 24, timeout_seconds: 300}",
                "system_prompt: NOT_A_SUPPORTED_PROMPT",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigurationError, match="unknown system prompt identifier"):
        load_experiment_configuration(invalid_path, project_root=ROOT)


def test_configuration_carries_prompt_id_and_validates_optional_hash() -> None:
    configuration = ExperimentConfiguration.model_validate(
        _base_config(
            system_prompt="R13B_SYSTEM_PROMPT",
            system_prompt_sha256=system_prompt_sha256("R13B_SYSTEM_PROMPT"),
        )
    )
    assert configuration.system_prompt == "R13B_SYSTEM_PROMPT"
    assert configuration.system_prompt_sha256 == system_prompt_sha256("R13B_SYSTEM_PROMPT")


def test_canary_configuration_preserves_frozen_execution_limits() -> None:
    configuration = load_experiment_configuration(CANARY_CONFIG, project_root=ROOT)
    assert configuration.config.experiment_id == "GS-E003"
    assert configuration.config.system_prompt == "R13B_SYSTEM_PROMPT"
    assert configuration.config.limits.max_actions == 28
    assert configuration.config.limits.max_requests == 24
    assert configuration.resolved_system_prompt_sha256 == system_prompt_sha256(
        "R13B_SYSTEM_PROMPT"
    )


def test_v4_canary_has_exact_six_run_package_and_new_recurrence_metadata() -> None:
    protocol = yaml.safe_load(CANARY_CONFIG.read_text(encoding="utf-8"))
    assert protocol["task_ids"] == ["GS-T006", "GS-T007", "GS-T008"]
    assert protocol["task_order"] == ["GS-T006", "GS-T007", "GS-T008"]
    assert protocol["planned_primary_runs"] == 6
    assert protocol["execution_plan"] == [
        ["GS-T006", "B0"],
        ["GS-T006", "T"],
        ["GS-T007", "B0"],
        ["GS-T007", "T"],
        ["GS-T008", "B0"],
        ["GS-T008", "T"],
    ]
    assert protocol["recurrence_evaluator_version"] == (
        "frozen_recurrence_matcher_v2_structured_pytest"
    )
    assert "experiments/run_sprint3b_kilo_v3.py" in v4.V4_FREEZE_INPUT_PATHS
    assert "experiments/run_sprint3b_reduced_b0_t.py" in v4.V4_FREEZE_INPUT_PATHS
    assert "src/graph_swarm/memory/recovery_evidence.py" in v4.V4_FREEZE_INPUT_PATHS
    assert "src/graph_swarm/graph/neo4j_repository.py" in v4.V4_FREEZE_INPUT_PATHS
    assert v4.EXPECTED_PLAN == tuple(
        (task_id, condition)
        for task_id in ("GS-T006", "GS-T007", "GS-T008")
        for condition in ("B0", "T")
    )


def test_v3_metadata_remains_historical() -> None:
    v3_protocol = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8"))
    assert v3_protocol["system_prompt"] == "ROLLOUT1_SYSTEM_PROMPT"
    assert v3_protocol["recurrence_evaluator_version"] == "frozen_recurrence_matcher_v1"
    assert hashlib.sha256(V3_CONFIG.read_bytes()).hexdigest() == (
        "0c788ce16ac0c64e9da9296bee"
        "a10093e280893ab3941d2c76305cd0bbf32bd7"
    )


def test_v4_freeze_and_loader_honor_the_six_run_plan_offline(tmp_path: Path) -> None:
    freeze_path = tmp_path / "v4-freeze.json"
    freeze = v4.create_freeze_artifact(
        project_root=ROOT,
        config_path=CANARY_CONFIG,
        freeze_path=freeze_path,
    )
    context = v4.load_canary_context(
        project_root=ROOT,
        config_path=CANARY_CONFIG,
        freeze_path=freeze_path,
        allow_existing_artifacts=True,
    )
    assert freeze["planned_primary_runs"] == 6
    assert freeze["artifact_destination"]["run_artifact_root"].endswith(
        "sprint3b_kilo_v4_canary/runs"
    )
    assert len(context.plan) == 6
    assert context.config.artifact_root_path == ROOT / (
        "research/evidence/results/GS-E003/sprint3b_kilo_v4_canary/runs"
    )
    assert context.config.system_prompt_id == "R13B_SYSTEM_PROMPT"
    assert context.config.resolved_system_prompt_sha256 == system_prompt_sha256(
        "R13B_SYSTEM_PROMPT"
    )


def test_new_critical_file_hash_mismatch_rejects_existing_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_path = tmp_path / "v4-freeze.json"
    v4.create_freeze_artifact(
        project_root=ROOT,
        config_path=CANARY_CONFIG,
        freeze_path=freeze_path,
    )
    original_sha256 = v4._sha256

    def changed_recovery_evidence_hash(path: Path) -> str:
        if path.as_posix().endswith("src/graph_swarm/memory/recovery_evidence.py"):
            return "changed-recovery-evidence-hash"
        return original_sha256(path)

    monkeypatch.setattr(v4, "_sha256", changed_recovery_evidence_hash)
    with pytest.raises(v4.V4CanaryConfigurationError, match="freeze input hashes"):
        v4.load_canary_context(
            project_root=ROOT,
            config_path=CANARY_CONFIG,
            freeze_path=freeze_path,
        )


def test_live_artifact_root_is_fresh_or_rejected_without_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = {"artifact_root": "research/evidence/results/GS-E003/canary/runs"}
    fresh = v4._require_fresh_primary_artifact_root(protocol, tmp_path)
    assert fresh == tmp_path / "research/evidence/results/GS-E003/canary/runs"
    fresh.parent.mkdir(parents=True)
    fresh.mkdir()
    sentinel = fresh / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(v4.V4CanaryConfigurationError, match="refusing a second"):
        v4._require_fresh_primary_artifact_root(protocol, tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "preserve"

    received: dict[str, object] = {}

    def fail_on_existing_root(**kwargs: object) -> object:
        received.update(kwargs)
        raise v4.V4CanaryConfigurationError("refusing a second canary execution")

    monkeypatch.setattr(v4, "load_canary_context", fail_on_existing_root)
    with pytest.raises(v4.V4CanaryConfigurationError, match="refusing a second"):
        v4.execute_canary(project_root=tmp_path)
    assert received["allow_existing_artifacts"] is False


def test_preflight_covers_provider_treatment_and_selected_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        v4.v3,
        "validate_treatment_embedding_configuration",
        lambda *_args, **_kwargs: calls.append("embedding-config"),
    )
    monkeypatch.setattr(
        v4.v3,
        "validate_kilo_provider_preflight",
        lambda _context: calls.append("kilo-provider"),
    )
    monkeypatch.setattr(
        v4.v3,
        "validate_treatment_dependency_preflight",
        lambda *_args, **_kwargs: calls.append("treatment-dependencies"),
    )
    monkeypatch.setattr(
        v4,
        "validate_live_preflight",
        lambda _context: calls.append("environments"),
    )

    v4.validate_canary_preflight(
        cast(Any, SimpleNamespace(protocol={})), settings=cast(Any, object())
    )

    assert calls == [
        "embedding-config",
        "kilo-provider",
        "treatment-dependencies",
        "environments",
    ]


def test_b0_and_t_canary_runners_share_prompt_id_text_and_hash() -> None:
    configuration = load_experiment_configuration(CANARY_CONFIG, project_root=ROOT)

    class AdvisoryDouble:
        def evaluate_action(
            self,
            task: Task,
            planned_action: PlannedAction,
            environment: EnvironmentContext,
        ) -> AdviceResult:
            del task, planned_action, environment
            return cast(AdviceResult, None)

    def objective(_task: Task, _workspace: Path) -> bool:
        return True

    def recurrence(
        _case: BenchmarkTaskCase,
        _events: Sequence[AgentEvent],
        _result: AgentRunResult[str] | None,
        _workspace: Path,
    ) -> bool:
        return False

    b0 = ExperimentRunner(
        configuration,
        settings=_settings(),
        condition=ExperimentCondition.B0,
        objective_evaluator=objective,
        recurrence_evaluator=recurrence,
    )
    treatment = ExperimentRunner(
        configuration,
        settings=_settings(),
        condition=ExperimentCondition.T,
        advisory_service=AdvisoryDouble(),
        objective_evaluator=objective,
        recurrence_evaluator=recurrence,
    )

    assert b0.system_prompt_id == treatment.system_prompt_id == "R13B_SYSTEM_PROMPT"
    assert b0.system_prompt == treatment.system_prompt == R13B_SYSTEM_PROMPT
    assert b0.system_prompt_sha256 == treatment.system_prompt_sha256


def test_runner_passes_resolved_prompt_explicitly_to_agent_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(CANARY_CONFIG, project_root=ROOT)
    captured: dict[str, object] = {}

    def capture_create(*_args: object, **kwargs: object) -> Agent[AgentDependencies, str]:
        captured.update(kwargs)
        return cast(Agent[AgentDependencies, str], object())

    monkeypatch.setattr("graph_swarm.research.runner.create_coding_agent", capture_create)
    runner = ExperimentRunner(
        configuration,
        settings=_settings(),
        condition=ExperimentCondition.B0,
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
    )
    persistence = StepPersistence(
        store=SqliteStepStore(database=tmp_path / "steps.sqlite"),
        agent_name="prompt-wiring-test",
        run_id="prompt-wiring-test-run",
    )

    runner._build_agent(_settings(), persistence)

    assert captured["system_prompt"] == R13B_SYSTEM_PROMPT


def test_r13b_prompt_contains_only_generic_runtime_guidance() -> None:
    assert R13B_RUNTIME_GUIDANCE in R13B_SYSTEM_PROMPT
    for phrase in (
        "structured argv",
        "pipelines, redirection, &&, ||",
        "optional shell utilities",
        "Use read_file",
        "prefer\nedit_file",
        "broad,\nunrelated repository exploration",
    ):
        assert phrase in R13B_SYSTEM_PROMPT
    for forbidden in (
        "family_id",
        "FAIL_TO_PASS",
        "expected recovery",
        "gold patch",
        "future task",
    ):
        assert forbidden.lower() not in R13B_SYSTEM_PROMPT.lower()


def test_task_prompt_excludes_evaluator_only_metadata() -> None:
    task = Task(
        id="GS-T006",
        problem_statement="Fix the current defect.",
        family_id="GS-FAMILY-ONLY",
        repository="repo",
        chronological_index=6,
    )
    from graph_swarm.research.runner import build_task_prompt

    prompt = build_task_prompt(task)
    assert prompt == "Task problem statement:\n\nFix the current defect."
    assert "GS-FAMILY-ONLY" not in prompt
    assert "FAIL_TO_PASS" not in prompt
