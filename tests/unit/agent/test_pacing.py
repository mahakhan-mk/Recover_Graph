from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from graph_swarm.agent.pacing import (
    ProviderPacingModel,
    ProviderRequestPacing,
    ProviderRequestPacingConfig,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.starts: list[float] = []
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _pacing(clock: FakeClock) -> ProviderRequestPacing:
    return ProviderRequestPacing(
        ProviderRequestPacingConfig(),
        monotonic=clock,
        sleeper=clock.sleep,
    )


def test_first_model_request_is_not_delayed() -> None:
    async def exercise() -> None:
        clock = FakeClock()
        pacing = _pacing(clock)

        assert await pacing.wait_for_request("B0") == 0
        assert clock.sleeps == []
        assert pacing.wait_seconds_for_run("B0") == 0

    asyncio.run(exercise())


def test_request_starts_are_separated_and_only_remaining_interval_is_slept() -> None:
    async def exercise() -> None:
        clock = FakeClock()
        pacing = _pacing(clock)

        await pacing.wait_for_request("B0")
        clock.now += 1.3
        await pacing.wait_for_request("B0")
        clock.now += 2.0
        await pacing.wait_for_request("B0")

        assert clock.sleeps == pytest.approx([3.7, 3.0])
        assert pacing.total_wait_seconds == pytest.approx(6.7)

    asyncio.run(exercise())


def test_shared_state_covers_sequential_b0_to_o1_runs() -> None:
    async def exercise() -> None:
        clock = FakeClock()
        pacing = _pacing(clock)

        await pacing.wait_for_request("B0-run")
        clock.now += 1.0
        await pacing.wait_for_request("O1-run")

        assert clock.sleeps == pytest.approx([4.0])
        assert pacing.wait_seconds_for_run("B0-run") == 0
        assert pacing.wait_seconds_for_run("O1-run") == pytest.approx(4.0)

    asyncio.run(exercise())


def test_follow_up_model_request_is_paced_without_delaying_tool_work() -> None:
    async def exercise() -> None:
        clock = FakeClock()
        pacing = _pacing(clock)
        starts: list[float] = []

        def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            starts.append(clock.now)
            return ModelResponse(parts=[TextPart("done")])

        model = ProviderPacingModel(
            FunctionModel(respond),
            pacing,
            "B0-run",
        )
        parameters = ModelRequestParameters()
        await model.request([], None, parameters)
        clock.now += 1.0
        await model.request([], None, parameters)

        assert starts == pytest.approx([100.0, 105.0])
        assert clock.sleeps == pytest.approx([4.0])

    asyncio.run(exercise())


def test_provider_429_is_propagated_without_automatic_retry() -> None:
    async def exercise() -> None:
        calls = 0

        def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            raise ModelHTTPError(429, "test-model", {"error": "rate limited"})

        model = ProviderPacingModel(
            FunctionModel(respond),
            ProviderRequestPacing(),
            "B0-run",
        )
        with pytest.raises(ModelHTTPError) as error:
            await model.request([], None, ModelRequestParameters())

        assert error.value.status_code == 429
        assert calls == 1

    asyncio.run(exercise())
