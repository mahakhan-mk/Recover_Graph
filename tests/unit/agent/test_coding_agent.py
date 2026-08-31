from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import FunctionToolset
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from graph_swarm.agent.coding_agent import (
    AgentConfigurationError,
    create_coding_agent,
    run_coding_agent,
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


def test_live_factory_requires_explicit_groq_configuration() -> None:
    with pytest.raises(AgentConfigurationError, match="groq_model"):
        create_coding_agent(make_settings(load_env_file=False))


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
