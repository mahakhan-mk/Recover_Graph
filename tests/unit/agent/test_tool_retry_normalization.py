from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from graph_swarm.agent.coding_agent import create_coding_agent, run_coding_agent
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.settings import Settings


def make_settings() -> Settings:
    return Settings(
        neo4j_uri="neo4j+s://example.databases.neo4j.io",
        neo4j_username="example-user",
        neo4j_password="example-password",
        neo4j_database="example-db",
        agent_request_limit=24,
        agent_command_timeout_seconds=5,
        agent_tests_timeout_seconds=5,
    )


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentDependencies(workspace, "run-001", "task-001")


def read_model(path: str) -> FunctionModel:
    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "read_file"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("read result received")])
        return ModelResponse(parts=[ToolCallPart("read_file", {"path": path})])

    return FunctionModel(respond)


def test_missing_read_file_failure_does_not_consume_model_retry_budget(
    tmp_path: Path,
) -> None:
    settings = make_settings()
    dependencies = make_dependencies(tmp_path)
    agent = create_coding_agent(settings, model=read_model("missing.txt"))

    result = run_coding_agent(
        agent,
        settings,
        dependencies,
        "Read the missing file and report the result.",
        max_actions=20,
        max_requests=24,
    )

    assert result.output is not None
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result.success is False
    assert dependencies.events[0].result.error == "file does not exist"
    assert not [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]


def test_directory_read_failure_does_not_consume_model_retry_budget(
    tmp_path: Path,
) -> None:
    settings = make_settings()
    dependencies = make_dependencies(tmp_path)
    (dependencies.workspace_root / "directory").mkdir()
    agent = create_coding_agent(settings, model=read_model("directory"))

    result = run_coding_agent(
        agent,
        settings,
        dependencies,
        "Read the directory and report the result.",
        max_actions=20,
        max_requests=24,
    )

    assert result.output is not None
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result.success is False
    assert dependencies.events[0].result.error == "path is not a regular file"
    assert not [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]


def test_failed_read_result_is_returned_to_model_and_recorded(tmp_path: Path) -> None:
    settings = make_settings()
    dependencies = make_dependencies(tmp_path)
    observed_returns: list[ActionResult] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        tool_returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "read_file"
        ]
        if tool_returns:
            assert isinstance(tool_returns[-1].content, ActionResult)
            observed_returns.append(tool_returns[-1].content)
            return ModelResponse(parts=[TextPart("failure was visible")])
        return ModelResponse(parts=[ToolCallPart("read_file", {"path": "missing.txt"})])

    agent = create_coding_agent(settings, model=FunctionModel(respond))
    result = run_coding_agent(
        agent,
        settings,
        dependencies,
        "Read the missing file and report the result.",
        max_actions=20,
        max_requests=24,
    )

    assert result.output == "failure was visible"
    assert len(observed_returns) == 1
    assert observed_returns[0] == dependencies.events[0].result
    assert observed_returns[0].error == "file does not exist"


def test_repeated_ordinary_filesystem_mistakes_do_not_exhaust_tool_retries(
    tmp_path: Path,
) -> None:
    settings = make_settings()
    dependencies = make_dependencies(tmp_path)

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        failures = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "read_file"
        ]
        if len(failures) >= 4:
            return ModelResponse(parts=[TextPart("four failures were recoverable")])
        return ModelResponse(parts=[ToolCallPart("read_file", {"path": "missing.txt"})])

    agent = create_coding_agent(settings, model=FunctionModel(respond))
    result = run_coding_agent(
        agent,
        settings,
        dependencies,
        "Keep exploring after recoverable filesystem failures.",
        max_actions=20,
        max_requests=24,
    )

    assert result.output == "four failures were recoverable"
    assert len(dependencies.events) == 4
    assert all(not event.result.success for event in dependencies.events)
    assert all(event.result.error == "file does not exist" for event in dependencies.events)


def test_argument_validation_failure_still_respects_three_tool_retries(
    tmp_path: Path,
) -> None:
    settings = make_settings()
    dependencies = make_dependencies(tmp_path)

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("read_file", {})])

    agent = create_coding_agent(settings, model=FunctionModel(respond))

    with pytest.raises(UnexpectedModelBehavior, match="read_file.*max retries count of 3"):
        run_coding_agent(
            agent,
            settings,
            dependencies,
            "Call read_file with invalid arguments.",
            max_actions=20,
            max_requests=24,
        )

    assert dependencies.events == []


def test_paginated_read_arguments_are_schema_valid_and_do_not_retry(
    tmp_path: Path,
) -> None:
    settings = make_settings()
    dependencies = make_dependencies(tmp_path)
    (dependencies.workspace_root / "source.py").write_text(
        "one\ntwo\nthree\n", encoding="utf-8"
    )

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "read_file"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("page received")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "read_file",
                    {"path": "source.py", "offset": 1, "length": 1},
                )
            ]
        )

    agent = create_coding_agent(settings, model=FunctionModel(respond))
    result = run_coding_agent(
        agent,
        settings,
        dependencies,
        "Read one page.",
        max_actions=20,
        max_requests=24,
    )

    assert result.output == "page received"
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result.success is True
    assert not [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]
