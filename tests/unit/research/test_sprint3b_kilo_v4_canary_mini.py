from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from experiments import run_sprint3b_kilo_v4_mini_canary as mini
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import load_experiment_configuration
from graph_swarm.settings import Settings

# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/experiments/sprint3b_kilo_v4_canary_mini.yaml"


def test_mini_configuration_loads_with_the_frozen_model() -> None:
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)

    assert configuration.model.provider == "kilo"
    assert configuration.model.model == mini.EXPECTED_CODING_MODEL
    assert configuration.config.limits.max_actions == 28
    assert configuration.config.limits.max_requests == 24


def test_mini_protocol_rejects_a_non_mini_model() -> None:
    protocol = mini._protocol(CONFIG_PATH)
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)
    protocol["model"] = "nex-agi/nex-n2.5-pro"

    with pytest.raises(mini.MiniCanaryConfigurationError, match="model"):
        mini._validate_protocol(protocol, configuration)


def test_mini_plan_prompt_and_paths_match_the_canary_contract() -> None:
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)
    protocol = mini._protocol(CONFIG_PATH)

    assert mini.EXPECTED_PLAN == (
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
    assert configuration.config.artifact_root.endswith("sprint3b_kilo_v4_mini_canary/runs")
    assert mini.DEFAULT_FREEZE.as_posix().endswith(
        "sprint3b_kilo_v4_mini_canary/freeze.json"
    )
    assert protocol["limits"]["max_actions"] == 28
    assert protocol["limits"]["max_requests"] == 24
    assert tuple(protocol["conditions"]) == tuple(
        condition.value for condition in mini.EXPECTED_CONDITIONS
    )
    assert ExperimentCondition.B0 is mini.EXPECTED_CONDITIONS[0]


def test_mini_freeze_hashes_include_mini_inputs(tmp_path: Path) -> None:
    freeze = mini.create_freeze_artifact(
        project_root=ROOT,
        freeze_path=tmp_path / "freeze.json",
    )

    hashes = cast(dict[str, str], freeze["freeze_input_hashes"])
    assert isinstance(hashes, dict)
    assert set(
        (
            "configs/experiments/sprint3b_kilo_v4_canary_mini.yaml",
            "experiments/run_sprint3b_kilo_v4_mini_canary.py",
            "configs/models/kilo_coding_mini.yaml",
        )
    ).issubset(hashes)
    assert freeze["artifact_destination"] == {
        "run_artifact_root": "research/evidence/results/GS-E003/sprint3b_kilo_v4_mini_canary/runs",
        "freeze_artifact": (
            "research/evidence/results/GS-E003/sprint3b_kilo_v4_mini_canary/freeze.json"
        ),
    }


def test_mini_provider_preflight_overrides_stale_model_setting(
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

    monkeypatch.setattr(mini, "get_settings", lambda: base_settings)
    monkeypatch.setattr(mini.coding_agent, "preflight_kilo_provider", fake_preflight)

    context = SimpleNamespace(
        protocol={"provider": "kilo"},
        config=configuration,
    )
    assert mini.validate_kilo_provider_preflight(
        cast(mini.v1.FrozenExecutionContext, context)
    ) == "READY"
    assert captured["model"] == mini.EXPECTED_CODING_MODEL
