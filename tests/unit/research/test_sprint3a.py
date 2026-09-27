from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.sprint3a import (
    DEFAULT_CONFIG,
    EXPECTED_ABSTRACTION_MODEL,
    EXPECTED_CODING_MODEL,
    SPRINT3A_CONDITIONS,
    SPRINT3A_EXPERIMENT_ID,
    SPRINT3A_PATTERN_IDS,
    SPRINT3A_TASK_IDS,
    Sprint3AProtocolError,
    build_execution_plan,
    freeze_input_hashes,
    freeze_inputs_hash,
    freeze_protocol,
    load_protocol,
    preflight,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _temporary_protocol(tmp_path: Path) -> Path:
    source = PROJECT_ROOT / DEFAULT_CONFIG
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "research/evidence/results/GS-E003/sprint3a_reduced_b0_t/runs",
        str(tmp_path / "runs"),
    ).replace(
        "research/evidence/results/GS-E003/sprint3a_reduced_b0_t/freeze.json",
        str(tmp_path / "freeze.json"),
    )
    config = tmp_path / "protocol.yaml"
    config.write_text(text, encoding="utf-8")
    return config


def test_reduced_protocol_freezes_exact_task_major_b0_t_plan() -> None:
    protocol = load_protocol(PROJECT_ROOT / DEFAULT_CONFIG)

    assert protocol["experiment_id"] == SPRINT3A_EXPERIMENT_ID == "GS-E003"
    assert "GS-E004" not in str(protocol["artifact_root"])
    assert "GS-E004" not in str(protocol["freeze_artifact"])
    assert protocol["model"] == EXPECTED_CODING_MODEL
    assert protocol["model_source_evidence"]
    assert protocol["abstraction_model"]["name"] == EXPECTED_ABSTRACTION_MODEL
    assert protocol["abstraction_model"]["used_during_transfer_execution"] is False
    assert tuple(protocol["conditions"]) == SPRINT3A_CONDITIONS
    assert tuple(protocol["task_ids"]) == SPRINT3A_TASK_IDS
    assert tuple(protocol["task_order"]) == SPRINT3A_TASK_IDS
    assert build_execution_plan(protocol) == tuple(
        (task_id, condition) for task_id in SPRINT3A_TASK_IDS for condition in SPRINT3A_CONDITIONS
    )
    assert tuple(protocol["treatment_pattern_ids"]) == SPRINT3A_PATTERN_IDS
    assert protocol["task_workspace_network"] == "disabled"
    assert protocol["external_runtime_network"] == {
        "openrouter": "required",
        "huggingface_inference": "required",
    }


def test_freeze_input_hashes_are_complete_and_deterministic() -> None:
    hashes = freeze_input_hashes(PROJECT_ROOT)

    assert "benchmark/annotations/recurrence_validation.csv" in hashes
    assert len(hashes["benchmark/annotations/recurrence_validation.csv"]) == 64
    assert freeze_inputs_hash(hashes) == freeze_inputs_hash(dict(reversed(tuple(hashes.items()))))
    assert freeze_inputs_hash(hashes)


def test_preflight_is_ready_without_provider_or_neo4j(tmp_path: Path) -> None:
    report = preflight(_temporary_protocol(tmp_path), project_root=PROJECT_ROOT)

    assert report.status == "READY"
    assert report.checks["provider_calls"] == "NOT CALLED"
    assert report.checks["benchmark_execution"] == "NOT CALLED"
    assert report.checks["neo4j_read_only"] == "PASS"


def test_preflight_rejects_an_existing_run_root(tmp_path: Path) -> None:
    config = _temporary_protocol(tmp_path)
    (tmp_path / "runs").mkdir()

    with pytest.raises(Sprint3AProtocolError, match="artifact root already exists"):
        preflight(config, project_root=PROJECT_ROOT)


def test_model_mismatch_fails_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_CODING_MODEL", "different/model")

    with pytest.raises(Sprint3AProtocolError, match="OPENROUTER_CODING_MODEL"):
        preflight(PROJECT_ROOT / DEFAULT_CONFIG, project_root=PROJECT_ROOT)


def test_legacy_model_variable_cannot_satisfy_coding_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_CODING_MODEL", "")
    monkeypatch.setenv("OPENROUTER_MODEL", EXPECTED_CODING_MODEL)

    with pytest.raises(Sprint3AProtocolError, match="OPENROUTER_CODING_MODEL"):
        preflight(PROJECT_ROOT / DEFAULT_CONFIG, project_root=PROJECT_ROOT)


def test_freeze_artifact_contains_protocol_only_metadata(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    config = _temporary_protocol(tmp_path)

    freeze_protocol(
        config,
        project_root=PROJECT_ROOT,
        output_path=output,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["status"] == "READY"
    assert artifact["freeze_type"] == "protocol_only_no_experimental_results"
    assert artifact["conditions"] == ["B0", "T"]
    assert artifact["task_ids"] == list(SPRINT3A_TASK_IDS)
    assert artifact["experiment_id"] == "GS-E003"
    assert artifact["model"]["resolved_model"] == EXPECTED_CODING_MODEL
    assert artifact["model"]["resolution_variable"] == "OPENROUTER_CODING_MODEL"
    assert artifact["abstraction_model"]["name"] == EXPECTED_ABSTRACTION_MODEL
    assert artifact["abstraction_model"]["used_during_transfer_execution"] is False
    assert artifact["preflight_checks"]["abstraction_execution"] == "NOT CALLED"
    assert artifact["freeze_input_hashes"]["benchmark/annotations/recurrence_validation.csv"]
    assert artifact["freeze_inputs_sha256"]
    assert artifact["network_policy"] == {
        "task_workspace_network": "disabled",
        "external_runtime_network": {
            "openrouter": "required",
            "huggingface_inference": "required",
        },
    }
    assert "experimental_results" not in artifact
    assert artifact["preflight_checks"]["provider_calls"] == "NOT CALLED"
