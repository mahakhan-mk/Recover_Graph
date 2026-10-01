from __future__ import annotations

from pathlib import Path

from experiments import run_gs_e003_hardened_development_v1 as hardened
from graph_swarm.research.benchmark_environments import load_benchmark_environment_policy
from graph_swarm.research.runner import load_experiment_configuration, load_task_cases

# pyright: reportPrivateUsage=false

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / hardened.DEFAULT_CONFIG


def test_hardened_plan_has_exactly_two_slots_for_gs_t017() -> None:
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)
    protocol = hardened._protocol(CONFIG_PATH)
    hardened._validate_protocol(protocol, configuration)

    assert hardened.EXPECTED_PLAN == (("GS-T017", "B0"), ("GS-T017", "T"))
    assert protocol["planned_primary_runs"] == 2
    assert hardened.build_execution_plan(protocol) == (
        hardened.v1.ExecutionSlot(1, "GS-T017", hardened.ExperimentCondition.B0),
        hardened.v1.ExecutionSlot(2, "GS-T017", hardened.ExperimentCondition.T),
    )


def test_hardened_protocol_preserves_frozen_execution_contract() -> None:
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)
    protocol = hardened._protocol(CONFIG_PATH)

    assert configuration.model.provider == "kilo"
    assert configuration.model.model == "qwen/qwen3-coder"
    assert configuration.config.system_prompt == "R13B_SYSTEM_PROMPT"
    assert configuration.config.limits.max_actions == 28
    assert configuration.config.limits.max_requests == 24
    assert protocol["limits"]["tool_retries"] == 3
    assert protocol["tools"] == ["read_file", "write_file", "edit_file", "run_tests", "run_command"]
    assert protocol["recurrence_evaluator_version"] == (
        "frozen_recurrence_matcher_v2_structured_pytest"
    )
    assert protocol["objective_evaluator_version"] == "frozen_swesmith_objective_v1"
    assert protocol["artifact_root"].endswith(
        "research/evidence/results/GS-E003/sprint3b_hardened_development_v1/runs"
    )
    assert "GS-T006" not in protocol["task_ids"]
    assert "GS-T007" not in protocol["task_ids"]
    assert "GS-T008" not in protocol["task_ids"]


def test_hardened_manifest_and_environment_are_single_task_and_read_only() -> None:
    configuration = load_experiment_configuration(CONFIG_PATH, project_root=ROOT)
    cases = load_task_cases(
        configuration.task_manifest_path,
        problem_statements_path=configuration.task_problems_path,
    )
    assert [(case.task.id, case.occurrence_index) for case in cases] == [("GS-T017", 4)]
    assert cases[0].task.chronological_index == 17

    environment_policy = load_benchmark_environment_policy(
        ROOT / hardened.ENVIRONMENT_POLICY,
        expected_task_order=("GS-T017",),
    )
    task_policy = environment_policy.task("GS-T017")
    assert task_policy.runtime == "manifest_container_required"
    assert task_policy.python_constraint_assertion == "==3.12.1"
    assert task_policy.network_during_execution is False

    protocol = hardened._protocol(CONFIG_PATH)
    assert protocol["treatment_memory_policy"] == (
        "fixed frozen acquisition corpus / read-only treatment memory"
    )
    assert protocol["treatment_repository"] == "R13bTreatmentRepository"
    assert protocol["memory_writes"] == "forbidden"
    assert protocol["online_learning"] is False
    assert protocol["neo4j_write_policy"] == "forbidden"
