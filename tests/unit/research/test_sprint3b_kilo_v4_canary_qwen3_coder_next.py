from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from experiments import run_sprint3b_kilo_v4_qwen3_coder_next_canary as qwen
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import load_experiment_configuration
from graph_swarm.settings import Settings

# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/experiments/sprint3b_kilo_v4_canary_qwen3_coder_next.yaml"


def test_qwen3_coder_next_configuration_loads_with_the_frozen_model() -> None:
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)

    assert configuration.model.provider == "kilo"
    assert configuration.model.model == qwen.EXPECTED_CODING_MODEL
    assert configuration.config.limits.max_actions == 28
    assert configuration.config.limits.max_requests == 24


def test_qwen3_coder_next_protocol_rejects_a_non_qwen3_coder_next_model() -> None:
    protocol = qwen._protocol(CONFIG_PATH)
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)
    protocol["model"] = "nex-agi/nex-n2.5-pro"

    with pytest.raises(qwen.Qwen3CoderNextCanaryConfigurationError, match="model"):
        qwen._validate_protocol(protocol, configuration)


def test_qwen3_coder_next_plan_prompt_and_paths_match_the_canary_contract() -> None:
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)
    protocol = qwen._protocol(CONFIG_PATH)

    assert qwen.EXPECTED_PLAN == (
        ("GS-T006", "B0"),
        ("GS-T006", "T"),
        ("GS-T007", "B0"),
        ("GS-T007", "T"),
        ("GS-T008", "B0"),
        ("GS-T008", "T"),
    )
    assert protocol["planned_primary_runs"] == 6
    assert protocol["system_prompt"] == "R13B_SYSTEM_PROMPT"
    assert protocol["system_prompt_sha256"] == (
        "f3bf4187cc9bfabcf2551aa3988e592a2d0f2d0e61ad2984d7d90ddf13d54502"
    )
    assert configuration.config.artifact_root.endswith(
        "sprint3b_kilo_v4_qwen3_coder_next_canary/runs"
    )
    assert qwen.DEFAULT_FREEZE.as_posix().endswith(
        "sprint3b_kilo_v4_qwen3_coder_next_canary/freeze.json"
    )
    assert protocol["limits"]["max_actions"] == 28
    assert protocol["limits"]["max_requests"] == 24
    assert tuple(protocol["conditions"]) == tuple(
        condition.value for condition in qwen.EXPECTED_CONDITIONS
    )
    assert ExperimentCondition.B0 is qwen.EXPECTED_CONDITIONS[0]


def test_qwen3_coder_next_freeze_hashes_include_qwen3_coder_next_inputs(tmp_path: Path) -> None:
    freeze = qwen.create_freeze_artifact(
        project_root=ROOT,
        freeze_path=tmp_path / "freeze.json",
    )

    hashes = cast(dict[str, str], freeze["freeze_input_hashes"])
    assert isinstance(hashes, dict)
    assert set(
        (
            "configs/experiments/sprint3b_kilo_v4_canary_qwen3_coder_next.yaml",
            "experiments/run_sprint3b_kilo_v4_qwen3_coder_next_canary.py",
            "configs/models/kilo_coding_qwen3_coder_next.yaml",
        )
    ).issubset(hashes)
    assert freeze["artifact_destination"] == {
        "run_artifact_root": (
            "research/evidence/results/GS-E003/sprint3b_kilo_v4_qwen3_coder_next_canary/runs"
        ),
        "freeze_artifact": (
            "research/evidence/results/GS-E003/sprint3b_kilo_v4_qwen3_coder_next_canary/freeze.json"
        ),
    }


def test_qwen3_coder_next_provider_preflight_overrides_stale_model_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)
    base_settings = Settings(
        neo4j_uri="bolt://localhost",
        neo4j_username="user",
        neo4j_password="password",
        neo4j_database="neo4j",
        model_provider="kilo",
        kilo_coding_model="nex-agi/nex-n2.5-pro",
        kilo_api_key="test-key",
    )
    captured: dict[str, str | None] = {}

    def fake_preflight(settings: Settings) -> str:
        captured["model"] = settings.kilo_coding_model
        return "READY"

    monkeypatch.setattr(qwen, "get_settings", lambda: base_settings)
    monkeypatch.setattr(qwen.coding_agent, "preflight_kilo_provider", fake_preflight)

    context = SimpleNamespace(
        protocol={"provider": "kilo"},
        config=configuration,
    )
    assert qwen.validate_kilo_provider_preflight(
        cast(qwen.v1.FrozenExecutionContext, context)
    ) == "READY"
    assert captured["model"] == qwen.EXPECTED_CODING_MODEL
