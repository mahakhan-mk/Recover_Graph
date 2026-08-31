"""Deterministic ranking of structurally applicable recovery candidates."""

from collections.abc import Sequence

from graph_swarm.retrieval.candidates import HistoricalRecoveryCandidate


def candidate_sort_key(
    candidate: HistoricalRecoveryCandidate,
) -> tuple[int, int, float, str, str]:
    """Prefer stronger observed evidence and newer incidents; IDs break ties."""
    return (
        -candidate.resolution.successful_observations,
        -candidate.successful_outcome_count,
        -candidate.failure.observed_at.timestamp(),
        candidate.resolution.id,
        candidate.failure.id,
    )


def select_best_candidate(
    candidates: Sequence[HistoricalRecoveryCandidate],
) -> HistoricalRecoveryCandidate:
    """Select one candidate using the documented stable ranking rule."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    return min(candidates, key=candidate_sort_key)
