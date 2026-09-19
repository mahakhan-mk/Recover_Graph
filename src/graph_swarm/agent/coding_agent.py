"""Minimal PydanticAI coding-agent wiring for controlled Rollout 1 execution.

The existing controlled tools remain the only filesystem and process boundary.
This module constructs providers lazily so importing it never requires provider
credentials or makes a network request.
"""

import asyncio
from collections.abc import Sequence
from typing import cast

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
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from graph_swarm.agent.advisory import (
    finalize_pending_advice,
    prepare_tool_action,
    record_post_advice_action,
)
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.pacing import ProviderRequestPacing, attach_provider_request_pacing
from graph_swarm.agent.prompts import ROLLOUT1_SYSTEM_PROMPT
from graph_swarm.agent.tools.read_file import (
    DEFAULT_READ_FILE_LENGTH,
    DEFAULT_READ_FILE_OFFSET,
)
from graph_swarm.agent.tools.read_file import (
    read_file as controlled_read_file,
)
from graph_swarm.agent.tools.run_command import run_command as controlled_run_command
from graph_swarm.agent.tools.run_tests import run_tests as controlled_run_tests
from graph_swarm.agent.tools.write_file import write_file as controlled_write_file
from graph_swarm.domain.action import ActionResult
from graph_swarm.settings import Settings


class AgentConfigurationError(ValueError):
    """Raised when live coding-agent provider configuration is incomplete."""


CODING_AGENT_TOOL_RETRIES = 3
_CODING_AGENT_TOOL_RETRIES = CODING_AGENT_TOOL_RETRIES
MODEL_REQUEST_TIMEOUT_SECONDS = 300


class AgentWallClockTimeoutError(TimeoutError):
    """The configured overall agent wall-clock deadline expired."""

    timeout_layer = "agent_wall_clock"

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"agent wall-clock deadline expired after {timeout_seconds} seconds")


class ModelRequestTimeoutError(TimeoutError):
    """A model request raised a timeout before the agent deadline expired."""

    timeout_layer = "model_request"

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"model request timeout expired after {timeout_seconds} seconds")


def _build_groq_model(settings: Settings) -> GroqModel:
    if not settings.groq_model or not settings.groq_model.strip():
        raise AgentConfigurationError("groq_model must be supplied through Settings or GROQ_MODEL")
    if not settings.groq_api_key or not settings.groq_api_key.strip():
        raise AgentConfigurationError(
            "groq_api_key must be supplied through Settings or GROQ_API_KEY"
        )

    provider = GroqProvider(api_key=settings.groq_api_key)
    return GroqModel(settings.groq_model, provider=provider)


def _build_openrouter_model(settings: Settings) -> OpenRouterModel:
    if not settings.openrouter_model or not settings.openrouter_model.strip():
        raise AgentConfigurationError(
            "openrouter_model must be supplied through Settings or OPENROUTER_MODEL"
        )
    if not settings.openrouter_api_key or not settings.openrouter_api_key.strip():
        raise AgentConfigurationError(
            "openrouter_api_key must be supplied through Settings or OPENROUTER_API_KEY"
        )
    return OpenRouterModel(
        settings.openrouter_model,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key),
    )


def create_coding_agent(
    settings: Settings,
    *,
    model: Model | None = None,
    capabilities: Sequence[AgentCapability[AgentDependencies]] | None = None,
) -> Agent[AgentDependencies, str]:
    """Build the controlled coding agent without performing a model request.

    ``model`` is an explicit injection point for PydanticAI's offline test
    models. Live construction selects the provider configured in ``settings``.
    """
    if model is not None:
        selected_model = model
    elif settings.model_provider == "openrouter":
        selected_model = _build_openrouter_model(settings)
    else:
        selected_model = _build_groq_model(settings)
    agent: Agent[AgentDependencies, str] = Agent(
        selected_model,
        deps_type=AgentDependencies,
        output_type=str,
        system_prompt=ROLLOUT1_SYSTEM_PROMPT,
        retries={"tools": _CODING_AGENT_TOOL_RETRIES},
        capabilities=capabilities,
    )

    def read_file(
        ctx: RunContext[AgentDependencies],
        path: str,
        offset: int = DEFAULT_READ_FILE_OFFSET,
        length: int = DEFAULT_READ_FILE_LENGTH,
    ) -> ActionResult:
        """Read regular UTF-8 files only; use run_command/listing behavior for directories.

        Offset is a zero-based line offset. Length is a positive line count capped at
        400. Use offset and length to inspect large source files page by page.
        """
        action = prepare_tool_action(
            ctx.deps,
            "read_file",
            "read_file",
            {"path": path, "offset": offset, "length": length},
        )
        result = controlled_read_file(
            ctx.deps,
            path,
            offset=offset,
            length=length,
            action_id=action.id,
        )
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
    max_requests: int | None = None,
    timeout_seconds: float | None = None,
    model_settings: ModelSettings | None = None,
    request_pacing: ProviderRequestPacing | None = None,
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
                max_requests=max_requests,
                timeout_seconds=timeout_seconds,
                model_settings=model_settings,
                request_pacing=request_pacing,
            )
        )
    if request_pacing is not None:
        attach_provider_request_pacing(agent, request_pacing, dependencies.run_id)
    try:
        return agent.run_sync(
            user_prompt,
            deps=dependencies,
            message_history=None,
            conversation_id=dependencies.run_id,
            run_id=dependencies.run_id,
            model_settings=_effective_model_settings(model_settings),
            usage_limits=_usage_limits(settings, max_actions, max_requests),
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
    max_requests: int | None = None,
    timeout_seconds: float | None = None,
    model_settings: ModelSettings | None = None,
    request_pacing: ProviderRequestPacing | None = None,
) -> AgentRunResult[str]:
    """Async bounded variant used by the sequential Track B runner."""
    if request_pacing is not None:
        attach_provider_request_pacing(agent, request_pacing, dependencies.run_id)
    try:
        run = agent.run(
            user_prompt,
            deps=dependencies,
            message_history=None,
            conversation_id=dependencies.run_id,
            run_id=dependencies.run_id,
            model_settings=_effective_model_settings(model_settings),
            usage_limits=_usage_limits(settings, max_actions, max_requests),
        )
        if timeout_seconds is None:
            return await run
        deadline = asyncio.timeout(timeout_seconds)
        try:
            async with deadline:
                return await run
        except TimeoutError as error:
            if deadline.expired():
                raise AgentWallClockTimeoutError(timeout_seconds) from error
            raise ModelRequestTimeoutError(MODEL_REQUEST_TIMEOUT_SECONDS) from error
    finally:
        finalize_pending_advice(dependencies)


def _usage_limits(
    settings: Settings,
    max_actions: int | None,
    max_requests: int | None = None,
) -> UsageLimits:
    return UsageLimits(
        request_limit=(
            max_requests
            if max_requests is not None
            else settings.agent_request_limit
        ),
        tool_calls_limit=max_actions,
    )


def _effective_model_settings(model_settings: ModelSettings | None) -> ModelSettings:
    """Apply the explicit per-request timeout without changing research settings."""
    effective = dict(model_settings or {})
    effective["timeout"] = MODEL_REQUEST_TIMEOUT_SECONDS
    return cast(ModelSettings, effective)


def timeout_provenance(error: BaseException | None) -> dict[str, object] | None:
    """Return structured timeout provenance for persisted run evidence."""
    if error is None or not isinstance(error, TimeoutError):
        return None
    layer = getattr(error, "timeout_layer", "unknown")
    seconds = getattr(error, "timeout_seconds", None)
    return {
        "layer": str(layer),
        "timeout_seconds": seconds,
        "error_type": type(error).__name__,
        "message": str(error),
    }
