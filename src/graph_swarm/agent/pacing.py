"""Deterministic pacing for provider-backed model request starts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

MIN_PROVIDER_REQUEST_INTERVAL_SECONDS = 5.0
MAX_NOMINAL_PROVIDER_REQUESTS_PER_MINUTE = 12.0

MonotonicClock = Callable[[], float]
AsyncSleeper = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class ProviderRequestPacingConfig:
    """Frozen provider request pacing contract."""

    min_interval_seconds: float = MIN_PROVIDER_REQUEST_INTERVAL_SECONDS
    max_nominal_requests_per_minute: float = MAX_NOMINAL_PROVIDER_REQUESTS_PER_MINUTE


class ProviderRequestPacing:
    """Share request-start pacing state across sequential agent runs."""

    def __init__(
        self,
        config: ProviderRequestPacingConfig | None = None,
        *,
        monotonic: MonotonicClock = time.monotonic,
        sleeper: AsyncSleeper = asyncio.sleep,
    ) -> None:
        self.config = config or ProviderRequestPacingConfig()
        if self.config.min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds must be greater than zero")
        if self.config.max_nominal_requests_per_minute <= 0:
            raise ValueError("max_nominal_requests_per_minute must be greater than zero")
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._last_request_start: float | None = None
        self._total_wait_seconds = 0.0
        self._wait_seconds_by_run: dict[str, float] = {}

    async def wait_for_request(self, run_id: str) -> float:
        """Wait until this request may start and return deliberate wait time."""
        now = self._monotonic()
        wait_seconds = 0.0
        if self._last_request_start is not None:
            wait_seconds = max(
                0.0,
                self.config.min_interval_seconds - (now - self._last_request_start),
            )
        if wait_seconds > 0:
            await self._sleeper(wait_seconds)
            self._total_wait_seconds += wait_seconds
            self._wait_seconds_by_run[run_id] = (
                self._wait_seconds_by_run.get(run_id, 0.0) + wait_seconds
            )
        self._last_request_start = self._monotonic()
        return wait_seconds

    @property
    def total_wait_seconds(self) -> float:
        """Return deliberate pacing sleep across the shared lifecycle."""
        return self._total_wait_seconds

    def wait_seconds_for_run(self, run_id: str) -> float:
        """Return deliberate pacing sleep attributed to one run."""
        return self._wait_seconds_by_run.get(run_id, 0.0)


class ProviderPacingModel(WrapperModel):
    """Apply pacing immediately before every non-streaming or streaming request."""

    def __init__(self, wrapped: Model, pacing: ProviderRequestPacing, run_id: str) -> None:
        super().__init__(wrapped)
        self.pacing = pacing
        self.run_id = run_id

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        await self.pacing.wait_for_request(self.run_id)
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse, None]:
        await self.pacing.wait_for_request(self.run_id)
        async with self.wrapped.request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as response_stream:
            yield response_stream


def attach_provider_request_pacing(
    agent: Any,
    pacing: ProviderRequestPacing,
    run_id: str,
) -> None:
    """Wrap an agent's selected model at the PydanticAI request boundary."""
    model = agent.model
    if not isinstance(model, Model):
        raise TypeError("provider request pacing requires an explicitly selected model")
    if isinstance(model, ProviderPacingModel):
        if model.pacing is pacing:
            model.run_id = run_id
            return
        model = model.wrapped
    agent.model = ProviderPacingModel(cast(Model[Any], model), pacing, run_id)


__all__ = [
    "MAX_NOMINAL_PROVIDER_REQUESTS_PER_MINUTE",
    "MIN_PROVIDER_REQUEST_INTERVAL_SECONDS",
    "ProviderPacingModel",
    "ProviderRequestPacing",
    "ProviderRequestPacingConfig",
    "attach_provider_request_pacing",
]
