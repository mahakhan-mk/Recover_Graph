"""Deterministic, concise rendering of historical recovery advice."""

from graph_swarm.domain.advice import AdviceResult, HistoricalRecoveryAdvice


def format_advice(result: AdviceResult) -> str:
    """Render only the structured facts needed for model reconsideration."""
    if not isinstance(result.advice, HistoricalRecoveryAdvice):
        return ""

    advice = result.advice
    summary = " ".join(advice.recovery_summary.split())
    matched_fields = ", ".join(advice.applicability.matched_fields)
    evidence = advice.recovery_evidence
    return (
        "A comparable historical action previously failed: "
        f"{advice.failed_tool}/{advice.failed_operation}. "
        f"An observed successful recovery was: {summary}. "
        f"Evidence status: {advice.resolution_status.value}; "
        f"successful observations: {evidence.successful_observations}; "
        f"applicable by: {matched_fields}. "
        "Consider this before executing the planned action; the recovery is not guaranteed. "
        f"Provenance: failure_episode_id={advice.provenance.failure_episode_id}; "
        f"resolution_id={advice.provenance.resolution_id}."
    )
