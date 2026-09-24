# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownLambdaType=false

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from graph_swarm.agent import coding_agent
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.stagnation import (
    PRE_MUTATION_STAGNATION_NUDGE,
    PreMutationStagnationConfig,
    PreMutationStagnationError,
    PreMutationStagnationGuard,
)
from graph_swarm.research import gate_a1_acquisition as acquisition


def _config(
    *,
    nudge: float = 1200,
    abort: float = 1800,
) -> PreMutationStagnationConfig:
    return PreMutationStagnationConfig(
        configured_nudge_seconds=1200,
        effective_nudge_seconds=nudge,
        nudge_env_var=acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE,
        nudge_override_applied=nudge != 1200,
        configured_abort_seconds=1800,
        effective_abort_seconds=abort,
        abort_env_var=acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE,
        abort_override_applied=abort != 1800,
    )


def test_r13b_pre_mutation_defaults_are_1200_and_1800(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, raising=False)

    resolved = acquisition.r13b_pre_mutation_stagnation_resolution()

    assert resolved.effective_nudge_seconds == 1200
    assert resolved.effective_abort_seconds == 1800
    assert resolved.nudge_override_applied is False
    assert resolved.abort_override_applied is False


def test_r13b_pre_mutation_environment_overrides_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, "30")
    monkeypatch.setenv(acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, "45")

    resolved = acquisition.r13b_pre_mutation_stagnation_resolution()

    assert resolved.effective_nudge_seconds == 30
    assert resolved.effective_abort_seconds == 45
    assert resolved.nudge_override_applied is True
    assert resolved.abort_override_applied is True


def test_existing_agent_timeout_override_accepts_future_7200_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from graph_swarm.research.runner import load_experiment_configuration

    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv(acquisition.AGENT_TIMEOUT_ENVIRONMENT_VARIABLE, "7200")
    monkeypatch.delenv(acquisition.OBJECTIVE_TIMEOUT_ENVIRONMENT_VARIABLE, raising=False)
    configuration = load_experiment_configuration(
        root / acquisition.R13B_CONFIG,
        project_root=root,
    )

    resolved = acquisition.r13_runtime_timeout_resolutions(configuration)

    assert resolved["agent"].effective_seconds == 7200
    assert resolved["objective"].effective_seconds == 900


@pytest.mark.parametrize(
    "variable,value",
    [
        (acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, ""),
        (acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, "abc"),
        (acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, "NaN"),
        (acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, "Infinity"),
        (acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, "0"),
        (acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, "-1"),
        (acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, ""),
        (acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, "abc"),
        (acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, "NaN"),
        (acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, "Infinity"),
        (acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, "0"),
        (acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, "-1"),
    ],
)
def test_invalid_pre_mutation_environment_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match=variable):
        acquisition.r13b_pre_mutation_stagnation_resolution()


def test_pre_mutation_nudge_must_precede_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, "1800")
    monkeypatch.setenv(acquisition.PRE_MUTATION_ABORT_ENVIRONMENT_VARIABLE, "1200")

    with pytest.raises(ValueError, match="less than abort"):
        acquisition.r13b_pre_mutation_stagnation_resolution()


def test_invalid_guard_configuration_fails_before_agent_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from graph_swarm.research import gate_a1_r13 as r13

    monkeypatch.setenv(acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE, "bad")
    startup_attempted = False

    def fail_if_agent_settings_are_resolved(**_kwargs: object) -> object:
        nonlocal startup_attempted
        startup_attempted = True
        raise AssertionError("agent settings were resolved before guard validation")

    monkeypatch.setattr(r13, "_settings_for_agent", fail_if_agent_settings_are_resolved)
    with pytest.raises(ValueError, match=acquisition.PRE_MUTATION_NUDGE_ENVIRONMENT_VARIABLE):
        acquisition.run_gate_a1_acquisition_r13b(Path.cwd())
    assert startup_attempted is False


def test_nudge_is_deterministic_one_shot_and_generic() -> None:
    now = [0.0]
    guard = PreMutationStagnationGuard(_config(), monotonic=lambda: now[0])
    guard.start()
    dependencies = SimpleNamespace(repository_mutation_evidence={})
    messages = [ModelRequest(parts=[UserPromptPart(content="continue")])]

    now[0] = 1199
    assert guard.before_model_request(messages, dependencies) is messages
    assert guard.nudge_sent is False

    now[0] = 1200
    nudged = guard.before_model_request(messages, dependencies)
    assert guard.nudge_sent is True
    assert isinstance(nudged[-1], ModelRequest)
    assert any(
        isinstance(part, UserPromptPart)
        and part.content == PRE_MUTATION_STAGNATION_NUDGE
        for part in nudged[-1].parts
    )
    assert "GS-T004" not in PRE_MUTATION_STAGNATION_NUDGE
    assert "MONAI" not in PRE_MUTATION_STAGNATION_NUDGE

    now[0] = 1500
    nudged_again = guard.before_model_request(nudged, dependencies)
    assert nudged_again == nudged
    assert sum(
        isinstance(part, UserPromptPart)
        and part.content == PRE_MUTATION_STAGNATION_NUDGE
        for message in nudged_again
        if isinstance(message, ModelRequest)
        for part in message.parts
    ) == 1


def test_trusted_mutation_before_nudge_disables_guard() -> None:
    now = [0.0]
    guard = PreMutationStagnationGuard(_config(), monotonic=lambda: now[0])
    guard.start()
    dependencies = SimpleNamespace(repository_mutation_evidence={})
    now[0] = 1000
    dependencies.repository_mutation_evidence["trusted-action"] = object()

    guard.observe_dependencies(dependencies)
    now[0] = 2000
    guard.before_model_request([], dependencies)

    assert guard.trusted_mutation_observed is True
    assert guard.first_trusted_mutation_action_id == "trusted-action"
    assert guard.nudge_sent is False
    assert guard.stagnation_abort_triggered is False


def test_trusted_mutation_after_nudge_disables_abort() -> None:
    now = [0.0]
    guard = PreMutationStagnationGuard(_config(), monotonic=lambda: now[0])
    guard.start()
    dependencies = SimpleNamespace(repository_mutation_evidence={})
    now[0] = 1200
    guard.before_model_request([], dependencies)
    dependencies.repository_mutation_evidence["trusted-action"] = object()
    now[0] = 1500
    guard.observe_dependencies(dependencies)
    now[0] = 1801

    guard.before_model_request([], dependencies)

    assert guard.nudge_sent is True
    assert guard.trusted_mutation_observed is True
    assert guard.stagnation_abort_triggered is False


def test_read_only_failed_and_noop_actions_do_not_count_without_trusted_evidence() -> None:
    now = [1800.0]
    guard = PreMutationStagnationGuard(_config(), monotonic=lambda: now[0])
    guard.start(0.0)
    dependencies = SimpleNamespace(repository_mutation_evidence={})

    guard.observe_dependencies(dependencies)
    with pytest.raises(PreMutationStagnationError):
        guard.before_model_request([], dependencies)

    assert guard.trusted_mutation_observed is False
    assert guard.stagnation_abort_triggered is True


def test_stagnation_provenance_distinguishes_nudge_and_abort() -> None:
    now = [0.0]
    guard = PreMutationStagnationGuard(_config(), monotonic=lambda: now[0])
    guard.start()
    dependencies = SimpleNamespace(repository_mutation_evidence={})
    now[0] = 1200
    guard.before_model_request([], dependencies)
    now[0] = 1800
    with pytest.raises(PreMutationStagnationError):
        guard.before_model_request([], dependencies)

    provenance = guard.provenance()
    assert provenance["pre_mutation_stagnation_policy"] == (
        "pre_mutation_stagnation_guard_v1"
    )
    assert provenance["nudge_sent"] is True
    assert provenance["trusted_mutation_observed"] is False
    assert provenance["stagnation_abort_triggered"] is True
    assert provenance["stagnation_abort_at_agent_elapsed_seconds"] == 1800


@pytest.mark.asyncio
async def test_stagnation_abort_cancels_agent_execution_without_orphan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard = PreMutationStagnationGuard(_config(), monotonic=lambda: 1800.0)
    guard.start(0.0)
    cancelled = False

    class FakeAgent:
        def run(self, *_args: object, **_kwargs: object):
            async def never_finishes() -> object:
                nonlocal cancelled
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled = True
                return object()

            return never_finishes()

    monkeypatch.setattr(
        coding_agent,
        "attach_provider_request_pacing",
        lambda *_args, **_kwargs: None,
    )
    dependencies = AgentDependencies(tmp_path, "run-1", "GS-T004")
    settings = SimpleNamespace(agent_request_limit=None)

    with pytest.raises(PreMutationStagnationError):
        await coding_agent.run_coding_agent_async(
            FakeAgent(),
            settings,
            dependencies,
            "run",
            pre_mutation_guard=guard,
        )

    assert cancelled is True
