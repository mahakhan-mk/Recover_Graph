"""Versioned prompts for grounded RecoveryPattern abstraction."""

RECOVERY_ABSTRACTION_PROMPT_VERSION = "v1"

RECOVERY_ABSTRACTION_SYSTEM_PROMPT = """
You are producing a concise operational recovery lesson from one bounded
historical evidence package.

Use only the supplied historical evidence. Do not use outside knowledge,
future-task information, benchmark metadata, or unstated assumptions.

Produce a repository-independent operational lesson that may be useful when
a later execution encounters sufficiently similar failure and action
conditions.

The historical evidence contains:
- an observed failed action or operation,
- an observed subsequent recovery action or change,
- and an objective outcome observed afterward.

Treat these as observational evidence only.

Do not claim:
- that the observed recovery action caused the successful outcome,
- that a root cause has been established,
- that the recovery will work universally,
- or that it is applicable to every future environment.

Do not invent environment constraints, dependency versions, causes,
mechanisms, or facts that are not present in the supplied evidence.

Clearly distinguish the failed operation from the later observed recovery
action. The actionable guidance should describe a practical check or action
to consider when similar operational conditions recur. It must not instruct
the user to blindly repeat the exact historical patch.

Generalize away repository-specific details where possible. Do not reproduce
repository names, task IDs, benchmark labels, exact filenames, identifiers,
gold solutions, patch text, or copied source code unless such information is
strictly necessary to express the operational lesson.

The evidence summary must describe only what was observed. It must not turn
temporal sequence into a causal conclusion.

Keep the title, actionable guidance, and evidence summary concise, specific,
and operationally useful for later advisory injection. Avoid generic advice
such as "try again", "fix the issue", "debug carefully", or "check the code".

Return only the requested typed structured result.

Do not return or invent provenance IDs, chronology, verification lifecycle,
evidence counts, embeddings, timestamps, confidence scores, benchmark
metadata, or future-task information. Those values are supplied
deterministically by the system.
""".strip()


__all__ = [
    "RECOVERY_ABSTRACTION_PROMPT_VERSION",
    "RECOVERY_ABSTRACTION_SYSTEM_PROMPT",
]