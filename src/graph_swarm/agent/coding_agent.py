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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from graph_swarm.agent.advisory import (
    finalize_pending_advice,
    prepare_tool_action,
    record_post_advice_action,
)
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.model_output import model_visible_action_result
from graph_swarm.agent.pacing import ProviderRequestPacing, attach_provider_request_pacing
from graph_swarm.agent.prompts import ROLLOUT1_SYSTEM_PROMPT
from graph_swarm.agent.stagnation import PreMutationStagnationGuard
from graph_swarm.agent.tools.edit_file import edit_file as controlled_edit_file
from graph_swarm.agent.tools.read_file import (
    DEFAULT_READ_FILE_LENGTH,
    DEFAULT_READ_FILE_OFFSET,
)
from graph_swarm.agent.tools.read_file import (
    read_file as controlled_read_file,
)
from graph_swarm.agent.tools.run_command import (
    CommandCanonicalizationError,
    canonicalize_command,
)
from graph_swarm.agent.tools.run_command import run_command as controlled_run_command
from graph_swarm.agent.tools.run_tests import run_tests as controlled_run_tests
from graph_swarm.agent.tools.write_file import write_file as controlled_write_file
from graph_swarm.domain.action import ActionResult
from graph_swarm.settings import Settings


class AgentConfigurationError(ValueError):
    """Raised when live coding-agent provider configuration is incomplete."""


class KiloPreflightError(RuntimeError):
    """Raised when the single Kilo provider preflight request fails."""


CODING_AGENT_TOOL_RETRIES = 3
_CODING_AGENT_TOOL_RETRIES = CODING_AGENT_TOOL_RETRIES
MODEL_REQUEST_TIMEOUT_SECONDS = 300
KILO_BASE_URL = "https://api.kilo.ai/api/gateway"


def _required_setting(value: str | None, setting_name: str, env_name: str) -> str:
    if not value or not value.strip():
        raise AgentConfigurationError(
            f"{setting_name} must be supplied through Settings or {env_name}"
        )
    return value


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


def _build_openrouter_model(settings: Settings) -> OpenRouterModel:
    model = _required_setting(
        settings.openrouter_coding_model,
        "openrouter_coding_model",
        "OPENROUTER_CODING_MODEL",
    )
    api_key = _required_setting(
        settings.openrouter_api_key,
        "openrouter_api_key",
        "OPENROUTER_API_KEY",
    )
    return OpenRouterModel(
        model,
        provider=OpenRouterProvider(api_key=api_key),
    )


def _build_kilo_model(settings: Settings) -> OpenAIChatModel:
    model = _required_setting(
        settings.kilo_coding_model,
        "kilo_coding_model",
        "KILO_CODING_MODEL",
    )
    api_key = _required_setting(settings.kilo_api_key, "kilo_api_key", "KILO_API_KEY")
    return OpenAIChatModel(
        model,
        provider=OpenAIProvider(base_url=KILO_BASE_URL, api_key=api_key),
    )


def _redact_kilo_error(error: BaseException, api_key: str) -> str:
    message = str(error).replace(api_key, "[redacted]")
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return f"HTTP {status_code}: {message}"
    return message or type(error).__name__


def preflight_kilo_provider(settings: Settings) -> str:
    """Make one tiny, tool-free Kilo request for provider readiness."""
    api_key = _required_setting(settings.kilo_api_key, "kilo_api_key", "KILO_API_KEY")
    model = _build_kilo_model(settings)
    try:
        probe = Agent(model, output_type=str, retries=0)
        result = probe.run_sync(
            "Reply only with READY",
            model_settings={"temperature": 0, "max_tokens": 8},
            usage_limits=UsageLimits(request_limit=1),
        )
        output = str(result.output).strip()
        if output != "READY":
            raise KiloPreflightError("Kilo provider preflight did not return READY")
    except Exception as error:
        if isinstance(error, KiloPreflightError):
            raise
        raise KiloPreflightError(
            f"Kilo provider preflight failed: {_redact_kilo_error(error, api_key)}"
        ) from error
    return "READY"


def create_coding_agent(
    settings: Settings,
    *,
    model: Model | None = None,
    capabilities: Sequence[AgentCapability[AgentDependencies]] | None = None,
    system_prompt: str | None = None,
) -> Agent[AgentDependencies, str]:
    """Build the controlled coding agent without performing a model request.

    ``model`` is an explicit injection point for PydanticAI's offline test
    models. Live construction selects the provider configured in ``settings``.
    """
    if model is not None:
        selected_model = model
    else:
        if settings.model_provider == "openrouter":
            selected_model = _build_openrouter_model(settings)
        elif settings.model_provider == "kilo":
            selected_model = _build_kilo_model(settings)
        else:
            raise AgentConfigurationError(
                "unsupported model_provider; the supported live provider is openrouter or kilo"
            )
    agent: Agent[AgentDependencies, str] = Agent(
        selected_model,
        deps_type=AgentDependencies,
        output_type=str,
        system_prompt=system_prompt or ROLLOUT1_SYSTEM_PROMPT,
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
        return model_visible_action_result(result, ctx.deps.model_output_telemetry)

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
        return model_visible_action_result(result, ctx.deps.model_output_telemetry)

    agent.tool(write_file)

    def edit_file(
        ctx: RunContext[AgentDependencies],
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> ActionResult:
        """Use for small exact edits to existing text files.

        ``old_text`` must match exactly. Prefer this over rewriting a large
        file with ``write_file`` when only a small section changes.
        """
        action = prepare_tool_action(
            ctx.deps,
            "edit_file",
            "edit_file",
            {
                "path": path,
                "old_text": old_text,
                "new_text": new_text,
                "expected_replacements": expected_replacements,
            },
        )
        result = controlled_edit_file(
            ctx.deps,
            path,
            old_text,
            new_text,
            expected_replacements=expected_replacements,
            action_id=action.id,
        )
        record_post_advice_action(ctx.deps, action, ctx.deps.events[-1])
        return model_visible_action_result(result, ctx.deps.model_output_telemetry)

    agent.tool(edit_file)

    def run_tests(ctx: RunContext[AgentDependencies]) -> ActionResult:
        """Run the repository test suite within the configured timeout."""
        action = prepare_tool_action(ctx.deps, "run_tests", "run_tests", {})
        result = controlled_run_tests(
            ctx.deps,
            settings.agent_tests_timeout_seconds,
            action_id=action.id,
        )
        record_post_advice_action(ctx.deps, action, ctx.deps.events[-1])
        return model_visible_action_result(result, ctx.deps.model_output_telemetry)

    agent.tool(run_tests)

    def run_command(
        ctx: RunContext[AgentDependencies],
        command: list[str],
    ) -> ActionResult:
        """Run canonical structured argv with shell=False.

        Correct examples: ["git", "status", "--short"] and
        ["python", "-m", "pytest", "tests/test_locales.py::Test...", "-q"].
        Incorrect: ["git status --short"].  Shell pipelines and redirection
        are unsupported.  ``rg`` is not guaranteed in benchmark containers;
        prefer portable tools or Python when necessary.
        """
        try:
            canonical_command = canonicalize_command(command)
        except CommandCanonicalizationError:
            # Keep the rejected action inside the normal tool-result boundary;
            # controlled_run_command emits the validation failure event.
            canonical_command = list(command)
        action = prepare_tool_action(
            ctx.deps,
            "run_command",
            "run_command",
            {"command": canonical_command},
        )
        result = controlled_run_command(
            ctx.deps,
            canonical_command,
            settings.agent_command_timeout_seconds,
            action_id=action.id,
        )
        record_post_advice_action(ctx.deps, action, ctx.deps.events[-1])
        return model_visible_action_result(result, ctx.deps.model_output_telemetry)

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
    disable_request_limit: bool = False,
    pre_mutation_guard: PreMutationStagnationGuard | None = None,
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
                disable_request_limit=disable_request_limit,
            )
        )
    if request_pacing is not None:
        attach_provider_request_pacing(
            agent,
            request_pacing,
            dependencies.run_id,
            pre_mutation_guard=pre_mutation_guard,
            dependencies=dependencies,
        )
    elif pre_mutation_guard is not None:
        attach_provider_request_pacing(
            agent,
            ProviderRequestPacing(),
            dependencies.run_id,
            pre_mutation_guard=pre_mutation_guard,
            dependencies=dependencies,
        )
    try:
        return agent.run_sync(
            user_prompt,
            deps=dependencies,
            message_history=None,
            conversation_id=dependencies.run_id,
            run_id=dependencies.run_id,
            model_settings=_effective_model_settings(model_settings),
            usage_limits=_usage_limits(
                settings,
                max_actions,
                max_requests,
                disable_request_limit=disable_request_limit,
            ),
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
    disable_request_limit: bool = False,
    pre_mutation_guard: PreMutationStagnationGuard | None = None,
) -> AgentRunResult[str]:
    """Async bounded variant used by the sequential Track B runner."""
    if request_pacing is not None:
        attach_provider_request_pacing(
            agent,
            request_pacing,
            dependencies.run_id,
            pre_mutation_guard=pre_mutation_guard,
            dependencies=dependencies,
        )
    elif pre_mutation_guard is not None:
        attach_provider_request_pacing(
            agent,
            ProviderRequestPacing(),
            dependencies.run_id,
            pre_mutation_guard=pre_mutation_guard,
            dependencies=dependencies,
        )
    try:
        run = agent.run(
            user_prompt,
            deps=dependencies,
            message_history=None,
            conversation_id=dependencies.run_id,
            run_id=dependencies.run_id,
            model_settings=_effective_model_settings(model_settings),
            usage_limits=_usage_limits(
                settings,
                max_actions,
                max_requests,
                disable_request_limit=disable_request_limit,
            ),
        )
        if pre_mutation_guard is None:
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

        run_task = asyncio.create_task(run)
        guard_task = asyncio.create_task(pre_mutation_guard.wait_for_abort())
        deadline = asyncio.timeout(timeout_seconds) if timeout_seconds is not None else None
        try:
            if timeout_seconds is None:
                done, _ = await asyncio.wait(
                    {run_task, guard_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            else:
                assert deadline is not None
                async with deadline:
                    done, _ = await asyncio.wait(
                        {run_task, guard_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
            if guard_task in done:
                guard_error = guard_task.exception()
                if guard_error is not None:
                    run_task.cancel()
                    await asyncio.gather(run_task, return_exceptions=True)
                    raise guard_error
            return await run_task
        except TimeoutError as error:
            if deadline is not None and deadline.expired():
                assert timeout_seconds is not None
                raise AgentWallClockTimeoutError(timeout_seconds) from error
            raise ModelRequestTimeoutError(MODEL_REQUEST_TIMEOUT_SECONDS) from error
        finally:
            if not guard_task.done():
                guard_task.cancel()
                await asyncio.gather(guard_task, return_exceptions=True)
            if not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
    finally:
        finalize_pending_advice(dependencies)


def _usage_limits(
    settings: Settings,
    max_actions: int | None,
    max_requests: int | None = None,
    *,
    disable_request_limit: bool = False,
) -> UsageLimits:
    return UsageLimits(
        request_limit=(None if disable_request_limit else (
            max_requests if max_requests is not None else settings.agent_request_limit
        )),
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
