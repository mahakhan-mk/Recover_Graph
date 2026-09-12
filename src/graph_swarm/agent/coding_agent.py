"""Minimal PydanticAI coding-agent wiring for controlled Rollout 1 execution.

The existing controlled tools remain the only filesystem and process boundary.
This module constructs the provider lazily so importing it never requires Groq
credentials or makes a network request.
"""

import asyncio
from collections.abc import Sequence

from pydantic_ai import (
    Agent,
    AgentCapability,
    AgentRunResult,
    ModelSettings,
    RunContext,
    UsageLimits,
)
from pydantic_ai.models import Model
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from graph_swarm.agent.advisory import (
    finalize_pending_advice,
    prepare_tool_action,
    record_post_advice_action,
)
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.prompts import ROLLOUT1_SYSTEM_PROMPT
from graph_swarm.agent.tools.read_file import read_file as controlled_read_file
from graph_swarm.agent.tools.run_command import run_command as controlled_run_command
from graph_swarm.agent.tools.run_tests import run_tests as controlled_run_tests
from graph_swarm.agent.tools.write_file import write_file as controlled_write_file
from graph_swarm.domain.action import ActionResult
from graph_swarm.settings import Settings


class AgentConfigurationError(ValueError):
    """Raised when live coding-agent provider configuration is incomplete."""


def _build_groq_model(settings: Settings) -> GroqModel:
    if not settings.groq_model or not settings.groq_model.strip():
        raise AgentConfigurationError(
            "groq_model must be supplied through Settings or GROQ_MODEL"
        )
    if not settings.groq_api_key or not settings.groq_api_key.strip():
        raise AgentConfigurationError(
            "groq_api_key must be supplied through Settings or GROQ_API_KEY"
        )

    provider = GroqProvider(api_key=settings.groq_api_key)
    return GroqModel(settings.groq_model, provider=provider)


def create_coding_agent(
    settings: Settings,
    *,
    model: Model | None = None,
    capabilities: Sequence[AgentCapability[AgentDependencies]] | None = None,
) -> Agent[AgentDependencies, str]:
    """Build the controlled coding agent without performing a model request.

    ``model`` is an explicit injection point for PydanticAI's offline test
    models. Live construction uses only the Groq model configured in ``settings``.
    """
    selected_model = model if model is not None else _build_groq_model(settings)
    agent: Agent[AgentDependencies, str] = Agent(
        selected_model,
        deps_type=AgentDependencies,
        output_type=str,
        system_prompt=ROLLOUT1_SYSTEM_PROMPT,
        capabilities=capabilities,
    )

    def read_file(
        ctx: RunContext[AgentDependencies],
        path: str,
    ) -> ActionResult:
        """Read a UTF-8 text file within the workspace."""
        action = prepare_tool_action(ctx.deps, "read_file", "read_file", {"path": path})
        result = controlled_read_file(ctx.deps, path, action_id=action.id)
        record_post_advice_action(ctx.deps, action, ctx.deps.events[-1])
        return result
    agent.tool(read_file)

    def write_file(
        ctx: RunContext[AgentDependencies],
        path: str,
        content: str,
    ) -> ActionResult:
        """Write UTF-8 text to a file within the workspace."""
        action = prepare_tool_action(
            ctx.deps,
            "write_file",
            "write_file",
            {"path": path, "content": content},
        )
        result = controlled_write_file(ctx.deps, path, content, action_id=action.id)
        record_post_advice_action(ctx.deps, action, ctx.deps.events[-1])
        return result
    agent.tool(write_file)

    def run_tests(ctx: RunContext[AgentDependencies]) -> ActionResult:
        """Run the repository test suite within the configured timeout."""
        action = prepare_tool_action(ctx.deps, "run_tests", "run_tests", {})
        result = controlled_run_tests(
            ctx.deps,
            settings.agent_tests_timeout_seconds,
            action_id=action.id,
        )
        record_post_advice_action(ctx.deps, action, ctx.deps.events[-1])
        return result
    agent.tool(run_tests)

    def run_command(
        ctx: RunContext[AgentDependencies],
        command: list[str],
    ) -> ActionResult:
        """Run structured argv within the configured timeout."""
        action = prepare_tool_action(
            ctx.deps,
            "run_command",
            "run_command",
            {"command": command},
        )
        result = controlled_run_command(
            ctx.deps,
            command,
            settings.agent_command_timeout_seconds,
            action_id=action.id,
        )
        record_post_advice_action(ctx.deps, action, ctx.deps.events[-1])
        return result
    agent.tool(run_command)

    return agent


def run_coding_agent(
    agent: Agent[AgentDependencies, str],
    settings: Settings,
    dependencies: AgentDependencies,
    user_prompt: str,
    *,
    max_actions: int | None = None,
    timeout_seconds: float | None = None,
    model_settings: ModelSettings | None = None,
) -> AgentRunResult[str]:
    """Run a constructed agent with isolated identity and optional bounds."""
    if timeout_seconds is not None:
        return asyncio.run(
            run_coding_agent_async(
                agent,
                settings,
                dependencies,
                user_prompt,
                max_actions=max_actions,
                timeout_seconds=timeout_seconds,
                model_settings=model_settings,
            )
        )
    try:
        return agent.run_sync(
            user_prompt,
            deps=dependencies,
            message_history=None,
            conversation_id=dependencies.run_id,
            run_id=dependencies.run_id,
            model_settings=model_settings,
            usage_limits=_usage_limits(settings, max_actions),
        )
    finally:
        finalize_pending_advice(dependencies)


async def run_coding_agent_async(
    agent: Agent[AgentDependencies, str],
    settings: Settings,
    dependencies: AgentDependencies,
    user_prompt: str,
    *,
    max_actions: int | None = None,
    timeout_seconds: float | None = None,
    model_settings: ModelSettings | None = None,
) -> AgentRunResult[str]:
    """Async bounded variant used by the sequential Track B runner."""
    try:
        run = agent.run(
            user_prompt,
            deps=dependencies,
            message_history=None,
            conversation_id=dependencies.run_id,
            run_id=dependencies.run_id,
            model_settings=model_settings,
            usage_limits=_usage_limits(settings, max_actions),
        )
        if timeout_seconds is None:
            return await run
        async with asyncio.timeout(timeout_seconds):
            return await run
    finally:
        finalize_pending_advice(dependencies)


def _usage_limits(settings: Settings, max_actions: int | None) -> UsageLimits:
    return UsageLimits(
        request_limit=settings.agent_request_limit,
        tool_calls_limit=max_actions,
    )
