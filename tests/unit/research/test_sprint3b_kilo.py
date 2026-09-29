from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.run_sprint3b_reduced_b0_t import (
    DEFAULT_CONFIG,
    DEFAULT_FREEZE,
    EXPECTED_CODING_MODEL,
    EXPECTED_PROVIDER,
    FreezeValidationError,
    load_frozen_execution_context,
)
from graph_swarm.agent import coding_agent
from graph_swarm.settings import Settings

ROOT = Path(__file__).resolve().parents[3]


def test_kilo_protocol_freeze_selects_exact_route_and_new_artifact_root() -> None:
    assert DEFAULT_CONFIG.as_posix() == "configs/experiments/sprint3b_kilo.yaml"
    assert (
        DEFAULT_FREEZE.as_posix()
        == "research/evidence/results/GS-E003/sprint3b_kilo/freeze.json"
    )
    text = (ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    assert "provider: kilo" in text
    assert f"model: {EXPECTED_CODING_MODEL}" in text
    assert "artifact_root: research/evidence/results/GS-E003/sprint3b_kilo/runs" in text


def test_kilo_context_validates_without_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    context = load_frozen_execution_context(project_root=ROOT)
    assert context.protocol["provider"] == EXPECTED_PROVIDER
    assert context.protocol["model"] == EXPECTED_CODING_MODEL
    assert context.artifact_root == ROOT / "research/evidence/results/GS-E003/sprint3b_kilo/runs"


def test_kilo_preflight_function_is_mockable_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[Settings] = []

    def fake_preflight(settings: Settings) -> str:
        called.append(settings)
        return "READY"

    monkeypatch.setattr(coding_agent, "preflight_kilo_provider", fake_preflight)
    assert coding_agent.preflight_kilo_provider(
        Settings(
            neo4j_uri="neo4j://test",
            neo4j_username="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            kilo_api_key="offline-key",
            kilo_coding_model=EXPECTED_CODING_MODEL,
        )
    ) == "READY"
    assert len(called) == 1


def test_kilo_model_mismatch_is_rejected_before_execution(tmp_path: Path) -> None:
    config_text = (ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8").replace(
        "model: nex-agi/nex-n2.5-pro\n",
        "model: wrong/model\n",
        1,
    )
    config_path = tmp_path / "sprint3b_kilo.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    freeze = json.loads(
        (ROOT / DEFAULT_FREEZE).read_text(encoding="utf-8")
    )
    freeze["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(FreezeValidationError, match="coding model"):
        load_frozen_execution_context(
            project_root=ROOT,
            config_path=config_path,
            freeze_path=path,
        )
