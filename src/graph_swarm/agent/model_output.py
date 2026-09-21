"""Bounded representations returned from controlled tools to the model."""

from __future__ import annotations

from collections.abc import MutableSequence

from graph_swarm.domain.action import ActionResult

MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS = 16_000
MODEL_OUTPUT_TRUNCATION_MARKER = "\n...[model-visible output truncated]...\n"


class ModelOutputTelemetry(dict[str, object]):
    """Durable, prompt-independent accounting for one model-visible result."""


def bound_model_visible_text(
    value: str | None,
    *,
    cap: int = MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS,
) -> str | None:
    """Keep the beginning and end of tool text within the model-facing cap."""
    if value is None or len(value) <= cap:
        return value
    if cap <= len(MODEL_OUTPUT_TRUNCATION_MARKER):
        return value[:cap]
    content_budget = cap - len(MODEL_OUTPUT_TRUNCATION_MARKER)
    prefix_chars = content_budget // 2
    suffix_chars = content_budget - prefix_chars
    return (
        value[:prefix_chars]
        + MODEL_OUTPUT_TRUNCATION_MARKER
        + value[-suffix_chars:]
    )


def model_visible_action_result(
    result: ActionResult,
    telemetry: MutableSequence[ModelOutputTelemetry],
    *,
    cap: int = MAX_MODEL_VISIBLE_TOOL_OUTPUT_CHARS,
) -> ActionResult:
    """Return a bounded copy without mutating the durable action result."""
    visible_output, visible_error = _bound_combined_text_fields(
        result.output,
        result.error,
        cap=cap,
    )
    raw_chars = len(result.output or "") + len(result.error or "")
    visible_chars = len(visible_output or "") + len(visible_error or "")
    telemetry.append(
        ModelOutputTelemetry(
            tool_name=result.tool_name,
            action_id=result.action_id,
            raw_output_chars=raw_chars,
            model_visible_output_chars=visible_chars,
            output_truncated=(raw_chars != visible_chars),
        )
    )
    return result.model_copy(update={"output": visible_output, "error": visible_error})


def _bound_combined_text_fields(
    output: str | None,
    error: str | None,
    *,
    cap: int,
) -> tuple[str | None, str | None]:
    """Bound both textual result fields against one shared character budget."""
    raw_lengths = (len(output or ""), len(error or ""))
    if sum(raw_lengths) <= cap:
        return output, error

    active = [index for index, length in enumerate(raw_lengths) if length]
    if len(active) == 1:
        capacities = [cap if index == active[0] else 0 for index in range(2)]
    else:
        marker_reserve = min(
            len(MODEL_OUTPUT_TRUNCATION_MARKER) + 2,
            cap // len(active),
        )
        remaining = cap - marker_reserve * len(active)
        weighted_lengths = [max(raw_lengths[index] - marker_reserve, 0) for index in active]
        weighted_total = sum(weighted_lengths)
        capacities = [0, 0]
        assigned = 0
        for position, index in enumerate(active):
            if position == len(active) - 1:
                share = remaining - assigned
            elif weighted_total:
                share = remaining * weighted_lengths[position] // weighted_total
            else:
                share = remaining // len(active)
            capacities[index] = marker_reserve + share
            assigned += share

    visible_values = (
        bound_model_visible_text(output, cap=capacities[0]),
        bound_model_visible_text(error, cap=capacities[1]),
    )
    return visible_values
