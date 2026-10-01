from pathlib import Path

import yaml

from experiments import run_sprint3b_kilo_hardened_development_v5_mutation_equivalence as v5

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / v5.DEFAULT_CONFIG


def test_v5_is_a_single_t_only_mutation_equivalence_revision() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert protocol["config_version"] == v5.CONFIG_VERSION
    assert protocol["model"] == "qwen/qwen3-coder"
    assert protocol["system_prompt"] == "R13B_SYSTEM_PROMPT"
    assert protocol["limits"]["max_actions"] == 28
    assert protocol["limits"]["max_requests"] == 24
    assert protocol["limits"]["tool_retries"] == 3
    assert protocol["recurrence_evaluator_version"] == v5.EXPECTED_RECURRENCE_VERSION
    assert protocol["conditions"] == ["T"]
    assert protocol["execution_plan"] == [["GS-T018", "T"]]
    assert protocol["planned_primary_runs"] == 1
    assert protocol["memory_writes"] == "forbidden"
    assert protocol["neo4j_write_policy"] == "forbidden"
    assert protocol["pre_mutation_advisory"]["revision"] == v5.EXPECTED_ADVISORY_REVISION
    assert protocol["pre_mutation_advisory"]["real_planned_action_preserved"] is True
    assert protocol["pre_mutation_advisory"]["test_boundary_equivalence_disabled"] is True
    assert protocol["artifact_root"].endswith(
        "research/evidence/results/GS-E003/"
        "sprint3b_hardened_development_v5_mutation_equivalence/runs"
    )


def test_v5_freeze_creation_and_loader_are_provider_free(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    created = v5.create_freeze_artifact(ROOT, CONFIG, freeze)

    assert created["candidate"] == {
        "task_id": "GS-T018",
        "b0_t_runs_in_this_revision": 0,
        "model_provider_calls_in_this_revision": 0,
        "agent_runs_in_this_revision": 0,
        "neo4j_writes_in_this_revision": 0,
    }
    prepared = v5.prepare_runner(ROOT, CONFIG, freeze)
    assert [(slot.task_id, slot.condition.value) for slot in prepared.slots] == [
        ("GS-T018", "T")
    ]
