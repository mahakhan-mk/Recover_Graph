from pathlib import Path
from types import SimpleNamespace

from graph_swarm.research import gate_a1_acquisition as acquisition
from graph_swarm.research.gate_a1_r8 import objective_final_check_policy
from graph_swarm.research.gate_a1_r12 import R12ObjectiveController
from graph_swarm.research.gate_a1_r13 import R13ObjectiveController
from graph_swarm.research.runner import load_experiment_configuration

ROOT = Path(__file__).resolve().parents[3]


def test_r13_configuration_is_immutable_and_keeps_r12_objective_semantics() -> None:
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/gate_a1_acquisition_r13.yaml",
        project_root=ROOT,
    )

    assert configuration.config.run_revision == "R13"
    assert configuration.config.rollout == "track_a_gate_a1_acquisition_r13"
    assert configuration.config.config_version == "gate-a1-r13-v1"
    assert configuration.model.prompt_version == "v1"
    assert configuration.config.revision_reason == acquisition.R13_REVISION_REASON
    assert configuration.config.recovery_event_semantics == (
        "objective_anchor_trusted_mutation_objective_success_v1"
    )
    assert configuration.config.stopping_policy == (
        "objective_anchored_recovery_or_timeout_v1"
    )
    assert configuration.config.objective_mutation_check_policy == (
        "pre_agent_failure_then_post_mutation_success_v1"
    )
    assert configuration.config.limits.timeout_seconds == 600
    assert configuration.config.agent_timeout_seconds == 600
    assert configuration.config.objective_timeout_seconds == 900


def test_r13_controller_is_the_objective_anchored_r12_extension() -> None:
    assert issubclass(R13ObjectiveController, R12ObjectiveController)
    assert acquisition.R13_NAMESPACE == "GS-E003/Gate-A1/acquisition-r13"
    assert acquisition.R13_EXPECTED_CODING_MODEL == acquisition.R12_EXPECTED_CODING_MODEL
    assert objective_final_check_policy(
        "R13",
        None,
        has_repository_mutation=False,
    ) == "skipped_r13_requires_mutation_bound_objective"


def test_r13_configuration_validator_rejects_r12_revision() -> None:
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/gate_a1_acquisition_r13.yaml",
        project_root=ROOT,
    )
    invalid = SimpleNamespace(
        config=configuration.config.model_copy(update={"run_revision": "R12"})
    )

    try:
        acquisition._validate_r13_harness_configuration(invalid)  # pyright: ignore[reportPrivateUsage]
    except acquisition.GateA1AcquisitionPreflightError:
        return
    raise AssertionError("R13 validator accepted an R12 configuration")
