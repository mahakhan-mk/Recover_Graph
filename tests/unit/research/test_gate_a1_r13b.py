from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_r13_runtime_timeout_defaults_preserve_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )

    resolutions = acquisition.r13_runtime_timeout_resolutions(configuration)

    assert resolutions["agent"].configured_seconds == 900
    assert resolutions["agent"].effective_seconds == 900
    assert resolutions["agent"].override_applied is False
    assert resolutions["objective"].configured_seconds == 900
    assert resolutions["objective"].effective_seconds == 900
    assert resolutions["objective"].override_applied is False


@pytest.mark.parametrize(
    ("environment_variable", "value", "expected_agent", "expected_objective"),
    [
        (acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, "3600", 3600, 900),
        (acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, "1800", 900, 1800),
        ("both", "3600", 3600, 3600),
    ],
)
def test_r13_runtime_timeout_overrides_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    environment_variable: str,
    value: str,
    expected_agent: int,
    expected_objective: int,
) -> None:
    monkeypatch.delenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    if environment_variable == "both":
        monkeypatch.setenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, value)
        monkeypatch.setenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, value)
    else:
        monkeypatch.setenv(environment_variable, value)
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )

    resolutions = acquisition.r13_runtime_timeout_resolutions(configuration)

    assert resolutions["agent"].effective_seconds == expected_agent
    assert resolutions["objective"].effective_seconds == expected_objective
    assert resolutions["agent"].override_applied is (expected_agent != 900)
    assert resolutions["objective"].override_applied is (expected_objective != 900)


@pytest.mark.parametrize("value", ["0", "-1", "abc", "NaN", "Infinity", ""])
def test_invalid_r13_runtime_timeout_override_fails_before_agent_startup(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, value)
    agent_startup_attempted = False

    def fail_if_settings_are_resolved() -> object:
        nonlocal agent_startup_attempted
        agent_startup_attempted = True
        raise AssertionError("agent settings were resolved before invalid timeout failed")

    monkeypatch.setattr(
        "graph_swarm.research.gate_a1_r13._settings_for_agent",
        fail_if_settings_are_resolved,
    )

    with pytest.raises(ValueError, match=acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE):
        acquisition.run_gate_a1_acquisition_r13b(ROOT)

    assert agent_startup_attempted is False


def test_r13_runtime_timeout_provenance_records_configured_effective_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, "3600")
    monkeypatch.delenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    configuration = load_experiment_configuration(
        ROOT / acquisition.R13B_CONFIG,
        project_root=ROOT,
    )

    provenance = acquisition.runtime_timeout_provenance(
        acquisition.r13_runtime_timeout_resolutions(configuration)
    )

    assert provenance == {
        "agent": {
            "environment_variable": "GRAPH_SWARM_AGENT_TIMEOUT_SECONDS",
            "configured_seconds": 900,
            "effective_seconds": 3600,
            "override_applied": True,
        },
        "objective": {
            "environment_variable": "GRAPH_SWARM_OBJECTIVE_TIMEOUT_SECONDS",
            "configured_seconds": 900,
            "effective_seconds": 900,
            "override_applied": False,
        },
    }
