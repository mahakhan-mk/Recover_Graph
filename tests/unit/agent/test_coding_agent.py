from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import AgentRunResult, FunctionToolset, UsageLimits
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from graph_swarm.agent.coding_agent import (
    AgentConfigurationError,
    create_coding_agent,
    run_coding_agent,
    run_coding_agent_async,
)
from graph_swarm.agent.dependencies import AgentDependencies
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


def test_live_factory_requires_explicit_groq_configuration() -> None:
    with pytest.raises(AgentConfigurationError, match="groq_model"):
        create_coding_agent(make_settings(load_env_file=False))


def test_groq_factory_preserves_groq_model_and_provider() -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={
            "model_provider": "groq",
            "groq_model": "llama-3.3-70b-versatile",
            "groq_api_key": "offline-key",
        }
    )

    agent = create_coding_agent(settings)

    model = cast(GroqModel, agent.model)
    assert type(model) is GroqModel
    assert isinstance(model, GroqModel)
    assert model.model_name == "llama-3.3-70b-versatile"
    assert model.system == "groq"
    assert isinstance(model._provider, GroqProvider)  # pyright: ignore[reportPrivateUsage]


def test_openrouter_factory_requires_key_only_when_selected() -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={"model_provider": "openrouter", "openrouter_model": "qwen/test"}
    )
    with pytest.raises(AgentConfigurationError, match="OPENROUTER_API_KEY"):
        create_coding_agent(settings)


def test_openrouter_factory_requires_model_only_when_selected() -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={"model_provider": "openrouter", "openrouter_api_key": "offline-key"}
    )
    with pytest.raises(AgentConfigurationError, match="OPENROUTER_MODEL"):
        create_coding_agent(settings)


def test_openrouter_factory_uses_dedicated_model_and_preserves_configuration() -> None:
    settings = make_settings(load_env_file=False).model_copy(
        update={
            "model_provider": "openrouter",
            "openrouter_model": "qwen/test",
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
