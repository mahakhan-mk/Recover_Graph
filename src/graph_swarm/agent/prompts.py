"""System prompts for controlled coding-agent experiments."""

from __future__ import annotations

import hashlib
from typing import Literal, cast

SystemPromptId = Literal["ROLLOUT1_SYSTEM_PROMPT", "R13B_SYSTEM_PROMPT"]

ROLLOUT1_SYSTEM_PROMPT = """\
Work only inside the provided workspace. Inspect relevant files before editing them.
Use the registered tools for all filesystem and process operations; never claim a
command or test ran unless you actually used a tool. Make the smallest reasonable
change, then run tests to verify it. Continue only within the configured resource
limits and stop when tests pass or the allowed run budget is exhausted. Do not
access hidden benchmark or research metadata.
"""

R13B_RUNTIME_GUIDANCE = """\
Runtime guidance:
Use run_command only for a single executable with structured argv. Shell
pipelines, redirection, &&, ||, and shell composition are unsupported.
Do not assume optional shell utilities such as rg or file are installed.
Prefer the registered repository tools and portable commands already known to
be available.
Use read_file to inspect relevant source regions.
For a small exact source-code change after locating the relevant text, prefer
edit_file instead of reconstructing an entire file or trying to perform
shell-based text editing.
Once the relevant defect and expected behavior are understood, make the
smallest reasonable edit and validate it instead of continuing broad,
unrelated repository exploration.
"""

R13B_SYSTEM_PROMPT = ROLLOUT1_SYSTEM_PROMPT + "\n" + R13B_RUNTIME_GUIDANCE

_SYSTEM_PROMPTS: dict[SystemPromptId, str] = {
    "ROLLOUT1_SYSTEM_PROMPT": ROLLOUT1_SYSTEM_PROMPT,
    "R13B_SYSTEM_PROMPT": R13B_SYSTEM_PROMPT,
}


def resolve_system_prompt(prompt_id: str) -> str:
    """Resolve one supported experiment prompt identifier to exact text."""
    try:
        return _SYSTEM_PROMPTS[cast(SystemPromptId, prompt_id)]
    except KeyError as error:
        supported = ", ".join(sorted(_SYSTEM_PROMPTS))
        raise ValueError(
            f"unknown system prompt identifier {prompt_id!r}; supported identifiers: {supported}"
        ) from error


def system_prompt_sha256(prompt_id: str) -> str:
    """Return the SHA256 of the exact UTF-8 text selected by ``prompt_id``."""
    return hashlib.sha256(resolve_system_prompt(prompt_id).encode("utf-8")).hexdigest()


__all__ = [
    "R13B_RUNTIME_GUIDANCE",
    "R13B_SYSTEM_PROMPT",
    "ROLLOUT1_SYSTEM_PROMPT",
    "SystemPromptId",
    "resolve_system_prompt",
    "system_prompt_sha256",
]
