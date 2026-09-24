"""Pre-mutation runtime discipline for bounded coding-agent attempts."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

PRE_MUTATION_STAGNATION_POLICY = "pre_mutation_stagnation_guard_v1"
PRE_MUTATION_STAGNATION_NUDGE = (
    "You have not made a repository mutation yet.\n\n"
    "Stop broad investigation. Based on the evidence already gathered, identify "
    "the smallest plausible fix and use edit_file now. Then run the frozen objective. "
    "Do not continue broad repository exploration unless the edit cannot be made safely "
    "from the evidence already available."
)

MonotonicClock = Callable[[], float]


@dataclass(frozen=True)
class PreMutationStagnationConfig:
    """Resolved runtime policy for one R13b agent attempt."""

    configured_nudge_seconds: float
    effective_nudge_seconds: float
    nudge_env_var: str
    nudge_override_applied: bool
    configured_abort_seconds: float
    effective_abort_seconds: float
    abort_env_var: str
    abort_override_applied: bool
    policy: str = PRE_MUTATION_STAGNATION_POLICY

    def __post_init__(self) -> None:
        if not math.isfinite(self.effective_nudge_seconds) or self.effective_nudge_seconds <= 0:
            raise ValueError("pre-mutation nudge threshold must be positive")
        if not math.isfinite(self.effective_abort_seconds) or self.effective_abort_seconds <= 0:
            raise ValueError("pre-mutation abort threshold must be positive")
        if self.effective_nudge_seconds >= self.effective_abort_seconds:
            raise ValueError("pre-mutation nudge threshold must be less than abort threshold")


class PreMutationStagnationError(RuntimeError):
    """The agent reached the pre-mutation stagnation deadline."""

    termination_reason = "pre_mutation_stagnation"

    def __init__(self, elapsed_seconds: float, abort_seconds: float) -> None:
        self.elapsed_seconds = elapsed_seconds
        self.abort_seconds = abort_seconds
        super().__init__(
            "pre-mutation stagnation deadline expired after "
            f"{elapsed_seconds:.3f} seconds (abort threshold {abort_seconds:g})"
        )


class PreMutationStagnationGuard:
    """Track one attempt until trusted repository mutation or clean abort."""

    def __init__(
        self,
        config: PreMutationStagnationConfig,
        *,
        monotonic: MonotonicClock = time.monotonic,
    ) -> None:
        self.config = config
        self._monotonic = monotonic
        self._started_at: float | None = None
        self._mutation_event = asyncio.Event()
        self.nudge_sent = False
        self.nudge_sent_at_agent_elapsed_seconds: float | None = None
        self.trusted_mutation_observed = False
        self.first_trusted_mutation_action_id: str | None = None
        self.first_trusted_mutation_at_agent_elapsed_seconds: float | None = None
        self.stagnation_abort_triggered = False
        self.stagnation_abort_at_agent_elapsed_seconds: float | None = None

    def start(self, started_at: float | None = None) -> None:
        """Start the attempt clock immediately before agent execution."""
        self._started_at = self._monotonic() if started_at is None else started_at

    @property
    def active(self) -> bool:
        return self._started_at is not None and not self.trusted_mutation_observed

    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, self._monotonic() - self._started_at)

    def observe_dependencies(self, dependencies: Any) -> None:
        """Observe the existing trusted mutation evidence, without redefining it."""
        if self.trusted_mutation_observed:
            return
        evidence = getattr(dependencies, "repository_mutation_evidence", None)
        if not evidence:
            return
        self.trusted_mutation_observed = True
        self.first_trusted_mutation_action_id = next(iter(evidence))
        self.first_trusted_mutation_at_agent_elapsed_seconds = self.elapsed_seconds()
        self._mutation_event.set()

    def before_model_request(
        self,
        messages: list[ModelMessage],
        dependencies: Any,
    ) -> list[ModelMessage]:
        """Apply the one nudge or raise the clean stagnation termination."""
        self.observe_dependencies(dependencies)
        if not self.active:
            return messages
        elapsed = self.elapsed_seconds()
        if elapsed >= self.config.effective_abort_seconds:
            raise self._abort()
        if elapsed < self.config.effective_nudge_seconds or self.nudge_sent:
            return messages
        self.nudge_sent = True
        self.nudge_sent_at_agent_elapsed_seconds = elapsed
        return _append_runtime_instruction(messages)

    def after_tool_event(self, dependencies: Any) -> None:
        """Check the abort boundary after a completed tool action."""
        self.observe_dependencies(dependencies)
        if self.active and self.elapsed_seconds() >= self.config.effective_abort_seconds:
            raise self._abort()

    async def wait_for_abort(self) -> None:
        """Watch the independent abort deadline until mutation or termination."""
        if self._started_at is None:
            return
        while self.active:
            remaining = self.config.effective_abort_seconds - self.elapsed_seconds()
            if remaining <= 0:
                raise self._abort()
            try:
                await asyncio.wait_for(self._mutation_event.wait(), timeout=remaining)
            except TimeoutError:
                if self.active:
                    raise self._abort() from None

    def provenance(self) -> dict[str, object]:
        return {
            "pre_mutation_stagnation_policy": self.config.policy,
            "configured_nudge_seconds": self.config.configured_nudge_seconds,
            "effective_nudge_seconds": self.config.effective_nudge_seconds,
            "nudge_env_var": self.config.nudge_env_var,
            "nudge_override_applied": self.config.nudge_override_applied,
            "configured_abort_seconds": self.config.configured_abort_seconds,
            "effective_abort_seconds": self.config.effective_abort_seconds,
            "abort_env_var": self.config.abort_env_var,
            "abort_override_applied": self.config.abort_override_applied,
            "nudge_sent": self.nudge_sent,
            "nudge_sent_at_agent_elapsed_seconds": (
                self.nudge_sent_at_agent_elapsed_seconds
            ),
            "trusted_mutation_observed": self.trusted_mutation_observed,
            "first_trusted_mutation_action_id": self.first_trusted_mutation_action_id,
            "first_trusted_mutation_at_agent_elapsed_seconds": (
                self.first_trusted_mutation_at_agent_elapsed_seconds
            ),
            "stagnation_abort_triggered": self.stagnation_abort_triggered,
            "stagnation_abort_at_agent_elapsed_seconds": (
                self.stagnation_abort_at_agent_elapsed_seconds
            ),
        }

    def _abort(self) -> PreMutationStagnationError:
        elapsed = self.elapsed_seconds()
        self.stagnation_abort_triggered = True
        self.stagnation_abort_at_agent_elapsed_seconds = elapsed
        return PreMutationStagnationError(elapsed, self.config.effective_abort_seconds)


def _append_runtime_instruction(messages: list[ModelMessage]) -> list[ModelMessage]:
    instruction = UserPromptPart(content=PRE_MUTATION_STAGNATION_NUDGE)
    updated = list(messages)
    if updated and isinstance(updated[-1], ModelRequest):
        last_request = cast(Any, updated[-1])
        updated[-1] = dataclasses.replace(
            last_request,
            parts=(*last_request.parts, instruction),
        )
    else:
        updated.append(ModelRequest(parts=[instruction]))
    return updated


__all__ = [
    "PRE_MUTATION_STAGNATION_NUDGE",
    "PRE_MUTATION_STAGNATION_POLICY",
    "PreMutationStagnationConfig",
    "PreMutationStagnationError",
    "PreMutationStagnationGuard",
]
