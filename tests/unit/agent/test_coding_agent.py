from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import AgentRunResult, FunctionToolset, UsageLimits
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from graph_swarm.agent import coding_agent as coding_agent_module
from graph_swarm.agent.coding_agent import (
    MODEL_REQUEST_TIMEOUT_SECONDS,
    AgentConfigurationError,
    create_coding_agent,
    run_coding_agent,
    run_coding_agent_async,
)
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.model_output import (
    MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS,
    MODEL_OUTPUT_TRUNCATION_MARKER,
    model_visible_action_result,
)
from graph_swarm.agent.prompts import ROLLOUT1_SYSTEM_PROMPT
from graph_swarm.domain.action import ActionResult
from graph_swarm.settings import Settings


def make_settings(*, load_env_file: bool = True) -> Settings:
    settings_factory = cast(Callable[..., Settings], Settings)
    return settings_factory(
        _env_file=".env" if load_env_file else None,
        neo4j_uri="neo4j+s://example.databases.neo4j.io",
        neo4j_username="example-user",
        neo4j_password="example-password",
        neo4j_database="example-db",
        agent_request_limit=3,
        agent_command_timeout_seconds=5,
        agent_tests_timeout_seconds=5,
    )


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentDependencies(workspace, "run-001", "task-001")


def make_offline_agent() -> tuple[Settings, TestModel]:
    settings = make_settings()
    model = TestModel(call_tools=["read_file"], custom_output_text="offline complete")
    return settings, model


def test_run_coding_agent_passes_explicit_public_usage_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, model = make_offline_agent()
    dependencies = make_dependencies(tmp_path)
    agent = create_coding_agent(settings, model=model)
    captured: dict[str, object] = {}

    def fake_run_sync(*_args: object, **kwargs: object) -> AgentRunResult[str]:
        captured["usage_limits"] = kwargs["usage_limits"]
        return cast(AgentRunResult[str], object())

    monkeypatch.setattr(agent, "run_sync", fake_run_sync)
    run_coding_agent(
        agent,
        settings,
        dependencies,
        "Use the configured limits.",
        max_actions=20,
        max_requests=24,
    )

    usage_limits = cast(UsageLimits, captured["usage_limits"])
    assert usage_limits.request_limit == 24
    assert usage_limits.tool_calls_limit == 20


def test_run_coding_agent_forces_explicit_model_request_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, model = make_offline_agent()
    dependencies = make_dependencies(tmp_path)
    agent = create_coding_agent(settings, model=model)
    captured: dict[str, object] = {}

    def fake_run_sync(*_args: object, **kwargs: object) -> AgentRunResult[str]:
        captured["model_settings"] = kwargs["model_settings"]
        return cast(AgentRunResult[str], object())

    monkeypatch.setattr(agent, "run_sync", fake_run_sync)
    run_coding_agent(agent, settings, dependencies, "Use explicit timeout settings.")

    model_settings = cast(dict[str, object], captured["model_settings"])
    assert model_settings["timeout"] == MODEL_REQUEST_TIMEOUT_SECONDS


async def test_run_coding_agent_async_falls_back_to_settings_request_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, model = make_offline_agent()
    dependencies = make_dependencies(tmp_path)
    agent = create_coding_agent(settings, model=model)
    captured: dict[str, object] = {}

    async def fake_run(*_args: object, **kwargs: object) -> AgentRunResult[str]:
        captured["usage_limits"] = kwargs["usage_limits"]
        return cast(AgentRunResult[str], object())

    monkeypatch.setattr(agent, "run", fake_run)
    await run_coding_agent_async(
        agent,
        settings,
        dependencies,
        "Use the configured limits.",
        max_actions=7,
    )

    usage_limits = cast(UsageLimits, captured["usage_limits"])
    assert usage_limits.request_limit == settings.agent_request_limit
    assert usage_limits.tool_calls_limit == 7


def test_live_factory_requires_explicit_openrouter_configuration() -> None:
    with pytest.raises(AgentConfigurationError, match="OPENROUTER_CODING_MODEL"):
        create_coding_agent(make_settings(load_env_file=False))


def test_live_factory_uses_the_explicit_coding_model_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={
            "openrouter_coding_model": "qwen/qwen3-coder:free",
            "openrouter_api_key": "offline-key",
        }
    )
    selected: dict[str, str] = {}

    def fake_openrouter_model(model: str, *, provider: object) -> TestModel:
        selected["model"] = model
        return TestModel()

    monkeypatch.setattr(coding_agent_module, "OpenRouterModel", fake_openrouter_model)

    create_coding_agent(settings)

    assert selected["model"] == "qwen/qwen3-coder:free"


def test_live_factory_rejects_unsupported_provider() -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={"model_provider": "unsupported"}
    )
    with pytest.raises(AgentConfigurationError, match="supported live provider is openrouter"):
        create_coding_agent(settings)


def test_openrouter_factory_requires_key_only_when_selected() -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={"model_provider": "openrouter", "openrouter_coding_model": "qwen/test"}
    )
    with pytest.raises(AgentConfigurationError, match="OPENROUTER_API_KEY"):
        create_coding_agent(settings)


def test_openrouter_factory_requires_model_only_when_selected() -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={
            "model_provider": "openrouter",
            "openrouter_api_key": "offline-key",
        }
    )
    with pytest.raises(AgentConfigurationError, match="OPENROUTER_CODING_MODEL"):
        create_coding_agent(settings)


def test_openrouter_factory_uses_dedicated_model_and_preserves_configuration() -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={
            "model_provider": "openrouter",
            "openrouter_coding_model": "qwen/test",
            "openrouter_api_key": "offline-key",
        }
    )

    agent = create_coding_agent(settings)

    model = cast(OpenRouterModel, agent.model)
    assert type(model) is OpenRouterModel
    assert type(model) is not OpenAIChatModel
    assert model.model_name == "qwen/test"
    assert isinstance(model, OpenRouterModel)
    assert isinstance(model._provider, OpenRouterProvider)  # pyright: ignore[reportPrivateUsage]


def test_coding_agent_freezes_tool_retries_at_three() -> None:
    settings, model = make_offline_agent()

    agent = create_coding_agent(settings, model=model)

    assert agent._max_tool_retries == 3  # pyright: ignore[reportPrivateUsage]


def test_coding_agent_accepts_offline_model_injection() -> None:
    settings, model = make_offline_agent()

    agent = create_coding_agent(settings, model=model)

    assert agent.model is model


async def test_agent_has_rollout_prompt_and_exactly_four_controlled_tools(
    tmp_path: Path,
) -> None:
    settings, model = make_offline_agent()
    dependencies = make_dependencies(tmp_path)
    agent = create_coding_agent(settings, model=model)

    prompt_parts = await agent.system_prompt_parts(deps=dependencies)
    assert prompt_parts[0].content == ROLLOUT1_SYSTEM_PROMPT
    toolset = cast(FunctionToolset[AgentDependencies], agent.toolsets[0])
    assert set(toolset.tools) == {
        "read_file",
        "write_file",
        "run_tests",
        "run_command",
    }


def test_one_offline_agent_read_call_uses_dependencies_and_emits_one_event(
    tmp_path: Path,
) -> None:
    settings, model = make_offline_agent()
    dependencies = make_dependencies(tmp_path)
    (dependencies.workspace_root / "a").write_text("workspace content", encoding="utf-8")
    agent = create_coding_agent(settings, model=model)

    result = run_coding_agent(agent, settings, dependencies, "Read the workspace file.")

    assert result.output == "offline complete"
    assert len(dependencies.events) == 1
    event = dependencies.events[0]
    assert event.result.tool_name == "read_file"
    assert event.result.success is True
    tool_returns = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, ActionResult)
    assert tool_returns[0].content.output == "workspace content"


def test_offline_agent_preserves_controlled_read_failure(tmp_path: Path) -> None:
    settings, model = make_offline_agent()
    dependencies = make_dependencies(tmp_path)
    agent = create_coding_agent(settings, model=model)

    result = run_coding_agent(agent, settings, dependencies, "Read the missing workspace file.")

    assert result.output == "offline complete"
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result.success is False
    assert dependencies.events[0].result.error == "file does not exist"
    tool_returns = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, ActionResult)
    assert tool_returns[0].content.error == "file does not exist"


def test_model_visible_tool_output_is_bounded_without_mutating_durable_result(
    tmp_path: Path,
) -> None:
    started = datetime.now(UTC)
    raw = "beginning\n" + ("x" * (MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS + 100)) + "\nending"
    durable = ActionResult(
        action_id="action-long",
        tool_name="run_command",
        success=True,
        exit_code=0,
        output=raw,
        started_at=started,
        completed_at=started,
    )
    dependencies = make_dependencies(tmp_path)

    visible = model_visible_action_result(durable, dependencies.model_output_telemetry)

    assert durable.output == raw
    assert visible.output is not None
    assert len(visible.output) == MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS
    assert visible.output.startswith("beginning")
    assert visible.output.endswith("ending")
    assert MODEL_OUTPUT_TRUNCATION_MARKER in visible.output
    assert dependencies.model_output_telemetry == [
        {
            "tool_name": "run_command",
            "action_id": "action-long",
            "raw_output_chars": len(raw),
            "model_visible_output_chars": MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS,
            "output_truncated": True,
        }
    ]


def test_model_visible_output_and_error_share_one_total_cap(tmp_path: Path) -> None:
    started = datetime.now(UTC)
    raw_output = "output-begin\n" + ("o" * MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS) + "\noutput-end"
    raw_error = "error-begin\n" + ("e" * MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS) + "\nerror-end"
    durable = ActionResult(
        action_id="action-both-long",
        tool_name="run_command",
        success=False,
        exit_code=1,
        output=raw_output,
        error=raw_error,
        started_at=started,
        completed_at=started,
    )
    dependencies = make_dependencies(tmp_path)

    visible = model_visible_action_result(durable, dependencies.model_output_telemetry)

    assert durable.output == raw_output
    assert durable.error == raw_error
    assert visible.output is not None
    assert visible.error is not None
    assert len(visible.output) + len(visible.error) <= MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS
    assert visible.output.startswith("output-begin")
    assert visible.error.startswith("error-begin")
    assert visible.output.endswith("output-end")
    assert visible.error.endswith("error-end")
    assert MODEL_OUTPUT_TRUNCATION_MARKER in visible.output
    assert MODEL_OUTPUT_TRUNCATION_MARKER in visible.error
    assert dependencies.model_output_telemetry[-1] == {
        "tool_name": "run_command",
        "action_id": "action-both-long",
        "raw_output_chars": len(raw_output) + len(raw_error),
        "model_visible_output_chars": len(visible.output) + len(visible.error),
        "output_truncated": True,
    }
