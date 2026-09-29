from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import experiments.run_sprint3b_kilo_v2 as v2
from graph_swarm.agent.advisory import (
    TreatmentAdvisoryInfrastructureError,
    prepare_tool_action,
)
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import ExperimentRunArtifactStore
from graph_swarm.settings import Settings

ROOT = Path(__file__).resolve().parents[3]


def _settings(model: str = v2.FROZEN_EMBEDDING_MODEL) -> Settings:
    return Settings(
        neo4j_uri="neo4j://test",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        hf_token="offline-test-token",
        hf_embedding_model=model,
        kilo_api_key="offline-kilo-key",
        kilo_coding_model=v2.EXPECTED_CODING_MODEL,
    )


class _NoAdviceService:
    def evaluate_action(self, *_args: object) -> AdviceResult:
        return AdviceResult.no_advice("no applicable recovery")


class _FailingAdviceService:
    def evaluate_action(self, *_args: object) -> AdviceResult:
        raise RuntimeError("Neo4j query failed")


def _dependencies(service: object, *, fail_closed: bool) -> AgentDependencies:
    return AgentDependencies(
        ROOT,
        "run-1",
        "GS-T006",
        task=SimpleNamespace(id="GS-T006"),  # type: ignore[arg-type]
        environment=object(),  # type: ignore[arg-type]
        advisory_service=service,  # type: ignore[arg-type]
        fail_closed_advisory=fail_closed,
    )


def test_wrong_runtime_embedding_model_is_rejected_before_execution() -> None:
    with pytest.raises(v2.FreezeValidationError, match="HF_EMBEDDING_MODEL"):
        v2.validate_treatment_embedding_configuration(settings=_settings("wrong/model"))


def test_exact_minilm_embedding_model_is_accepted() -> None:
    v2.validate_treatment_embedding_configuration(settings=_settings())


def test_t_legitimate_no_advice_remains_valid_and_b0_semantics_are_unchanged() -> None:
    treatment_action = prepare_tool_action(
        _dependencies(_NoAdviceService(), fail_closed=True),
        "run_tests",
        "run_tests",
        {"argv": ["pytest"]},
    )
    b0_action = prepare_tool_action(
        _dependencies(_FailingAdviceService(), fail_closed=False),
        "run_tests",
        "run_tests",
        {"argv": ["pytest"]},
    )
    assert treatment_action.task_id == "GS-T006"
    assert b0_action.task_id == "GS-T006"


def test_t_advisory_infrastructure_failure_is_invalid_and_artifact_is_preserved(
    tmp_path: Path,
) -> None:
    with pytest.raises(TreatmentAdvisoryInfrastructureError, match="lookup failed"):
        prepare_tool_action(
            _dependencies(_FailingAdviceService(), fail_closed=True),
            "run_tests",
            "run_tests",
            {"argv": ["pytest"]},
        )

    context = v2.load_frozen_execution_context(project_root=ROOT)
    slot = v2.ExecutionSlot(1, "GS-T006", ExperimentCondition.T)
    context = dataclasses.replace(context, plan=(slot,), artifact_root=tmp_path / "runs")
    store = ExperimentRunArtifactStore(context.artifact_root)

    def fail_treatment(_slot: v2.ExecutionSlot, _case: Any) -> Any:
        raise RuntimeError("Hugging Face inference failed")

    report = v2.execute_primary_plan(context, fail_treatment, artifact_store=store)
    assert report.invalid_infrastructure_slots
    assert report.valid_completed_slots == 0
    observation = v2.inspect_primary_slots(context.artifact_root, context.plan)[0]
    assert observation.status is v2.SlotStatus.INVALID_INFRASTRUCTURE
    assert observation.artifact_path is not None and observation.artifact_path.is_file()


def test_treatment_dependency_preflight_mocks_hf_and_neo4j_without_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = v2.load_frozen_execution_context(project_root=ROOT)
    calls: list[str] = []

    class FakeEmbedder:
        def __init__(self, *, settings: Settings) -> None:
            assert settings.hf_embedding_model == v2.FROZEN_EMBEDDING_MODEL
            calls.append("hf-init")

        def embed_text(self, _text: str) -> list[float]:
            calls.append("hf-request")
            return [0.0] * 384

    class FakeRepository:
        def verify_connectivity(self) -> None:
            calls.append("neo4j-connectivity")

    class FakeTreatmentRepository:
        def get_recovery_pattern(self, pattern_id: str) -> object:
            calls.append(f"neo4j-read:{pattern_id}")
            return object()

    class FakeRuntime:
        repository = FakeRepository()
        treatment_repository = FakeTreatmentRepository()

        def close(self) -> None:
            calls.append("neo4j-close")

    monkeypatch.setattr(v2, "RecoveryPatternEmbedder", FakeEmbedder)

    def fake_runtime_factory(*_args: object, **_kwargs: object) -> FakeRuntime:
        return FakeRuntime()

    monkeypatch.setattr(
        v2, "create_neo4j_advisory_runtime", fake_runtime_factory
    )
    v2.validate_treatment_dependency_preflight(context, settings=_settings())

    assert calls[:3] == ["hf-init", "hf-request", "neo4j-connectivity"]
    assert calls[-1] == "neo4j-close"
    assert len([item for item in calls if item.startswith("neo4j-read:")]) == 5
    assert not any("write" in item for item in calls)


def test_treatment_dependency_preflight_precedes_kilo_provider_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def fake_settings() -> Settings:
        return _settings()

    def fake_treatment_preflight(*_args: object, **_kwargs: object) -> None:
        order.append("treatment")

    def fake_kilo_preflight(_context: v2.FrozenExecutionContext) -> str:
        order.append("kilo")
        return "READY"

    def fake_live_preflight(_context: v2.FrozenExecutionContext) -> None:
        return None

    monkeypatch.setattr(v2, "get_settings", fake_settings)
    monkeypatch.setattr(
        v2, "validate_treatment_dependency_preflight", fake_treatment_preflight
    )
    monkeypatch.setattr(v2, "validate_kilo_provider_preflight", fake_kilo_preflight)
    monkeypatch.setattr(v2, "validate_live_preflight", fake_live_preflight)
    monkeypatch.setattr("sys.argv", ["run_sprint3b_kilo_v2", "--preflight-only"])
    v2.main()
    assert order == ["treatment", "kilo"]


def test_v2_uses_fresh_artifact_root_and_v1_files_are_unchanged() -> None:
    v1_evidence_root = ROOT / "research/evidence/results/GS-E003/sprint3b_kilo"

    def snapshot(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    evidence_before = snapshot(v1_evidence_root)
    context = v2.load_frozen_execution_context(project_root=ROOT)
    assert context.artifact_root == ROOT / "research/evidence/results/GS-E003/sprint3b_kilo_v2/runs"
    assert context.artifact_root != ROOT / "research/evidence/results/GS-E003/sprint3b_kilo/runs"
    assert context.artifact_root.exists()
    with pytest.raises(v2.FreezeValidationError, match="already exists"):
        v2.load_frozen_execution_context(project_root=ROOT, allow_existing_artifacts=False)

    v1_paths = (
        ROOT / "configs/experiments/sprint3b_kilo.yaml",
        ROOT / "research/evidence/results/GS-E003/sprint3b_kilo/freeze.json",
    )
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in v1_paths}
    subprocess.run(["git", "diff", "--check", "--", *map(str, v1_paths)], cwd=ROOT, check=True)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in v1_paths}
    assert after == before
    assert snapshot(v1_evidence_root) == evidence_before
