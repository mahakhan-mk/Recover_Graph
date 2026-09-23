from pathlib import Path
from types import SimpleNamespace

from graph_swarm.agent.prompts import R13B_RUNTIME_GUIDANCE, R13B_SYSTEM_PROMPT
from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research.gate_a1_r8 import objective_final_check_policy
from graph_swarm.research.runner import load_experiment_configuration

ROOT = Path(__file__).resolve().parents[3]


def test_r13b_configuration_records_new_revision_and_frozen_runtime_limits() -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )

    assert configuration.config.run_revision == "R13b"
    assert configuration.config.rollout == "track_a_gate_a1_acquisition_r13b"
    assert configuration.config.config_version == "gate-a1-r13b-v2"
    assert configuration.config.revision_reason == acquisition.R13B_REVISION_REASON
    assert configuration.model.prompt_version == acquisition.R13B_PROMPT_VERSION
    assert configuration.config.limits.timeout_seconds == acquisition.R13B_TIMEOUT_SECONDS
    assert configuration.config.agent_timeout_seconds == acquisition.R13B_AGENT_TIMEOUT_SECONDS
    assert configuration.config.objective_timeout_seconds == 900
    assert configuration.config.recovery_event_semantics == (
        "objective_anchor_trusted_mutation_objective_success_v1"
    )
    assert acquisition.R13B_EXPECTED_CODING_MODEL == "nex-agi/nex-n2.5-pro:free"


def test_historical_r13b_v1_configuration_remains_reconstructable() -> None:
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/gate_a1_acquisition_r13b.yaml",
        project_root=ROOT,
    )

    assert configuration.config.config_version == "gate-a1-r13b-v1"
    assert configuration.config.agent_timeout_seconds == 600
    assert configuration.config.limits.timeout_seconds == 600


def test_r13b_uses_r13_objective_lifecycle_and_new_prompt_only() -> None:
    assert acquisition.R13B_EXPECTED_CODING_MODEL == acquisition.R13_EXPECTED_CODING_MODEL
    assert acquisition.R13B_EXPECTED_ABSTRACTION_MODEL == acquisition.R13_EXPECTED_ABSTRACTION_MODEL
    assert objective_final_check_policy(
        "R13b",
        None,
        has_repository_mutation=False,
    ) == "skipped_r13b_requires_mutation_bound_objective"
    assert R13B_RUNTIME_GUIDANCE in R13B_SYSTEM_PROMPT
    assert "GS-T001" not in R13B_SYSTEM_PROMPT


def test_r13b_configuration_validator_rejects_r13_revision() -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )
    invalid = SimpleNamespace(
        config=configuration.config.model_copy(update={"run_revision": "R13"})
    )

    try:
        acquisition._validate_r13b_harness_configuration(invalid)  # pyright: ignore[reportPrivateUsage]
    except acquisition.GateA1AcquisitionPreflightError:
        return
    raise AssertionError("R13b validator accepted an R13 configuration")


def test_r13b_configuration_validator_rejects_wrong_prompt_version() -> None:
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )
    invalid = SimpleNamespace(
        config=configuration.config,
        model=configuration.model.model_copy(update={"prompt_version": "v1"}),
    )

    try:
        acquisition._validate_r13b_harness_configuration(invalid)  # pyright: ignore[reportPrivateUsage]
    except acquisition.GateA1AcquisitionPreflightError:
        return
    raise AssertionError("R13b validator accepted the wrong prompt version")
