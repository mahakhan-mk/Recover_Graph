"""Construction boundary for the real Track B treatment advisory runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

from graph_swarm.advisory.service import AdvisoryService
from graph_swarm.graph.neo4j_repository import Neo4jRepository
from graph_swarm.graph.read_models import RecoveryPatternLineage, RecoveryPatternVectorCandidate
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.memory.recovery_embeddings import RECOVERY_PATTERN_EMBEDDING_DIMENSION
from graph_swarm.settings import Settings, get_settings

R13B_TREATMENT_PATTERN_IDS = frozenset(
    {
        "recovery-pattern-ad07a6a45718b848a30ad377",
        "recovery-pattern-162e3999c4a2c66a1ff647ed",
        "recovery-pattern-f490f62ab931191c6eac6db1",
        "recovery-pattern-99a54266f940e1d4648f4698",
        "recovery-pattern-fd7b65022b22dc5f2a42816f",
    }
)


class NonCanonicalTreatmentPatternError(ValueError):
    """Raised when treatment asks for a RecoveryPattern outside the R13b set."""


class TreatmentVectorRepository(Protocol):
    """Read-only source surface required by the treatment allowlist."""

    def count_recovery_pattern_vectors(self) -> int:
        ...

    def query_recovery_pattern_vectors(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> tuple[RecoveryPatternVectorCandidate, ...]:
        ...

    def get_recovery_pattern(self, pattern_id: str) -> RecoveryPatternLineage:
        ...


class R13bTreatmentRepository:
    """Read-only allowlist view over the complete Neo4j vector repository.

    The underlying repository remains untouched.  Every vector query asks the
    native index for the complete embedded pool before applying the treatment
    allowlist, so historical development nodes cannot occupy a top-k slot and
    hide a canonical R13b pattern.
    """

    def __init__(self, repository: TreatmentVectorRepository) -> None:
        self._repository = repository

    def count_recovery_pattern_vectors(self) -> int:
        """Count only canonical R13b patterns with persisted embeddings."""
        raw_count = self._repository.count_recovery_pattern_vectors()
        if raw_count == 0:
            return 0
        probe = [1.0] + [0.0] * (RECOVERY_PATTERN_EMBEDDING_DIMENSION - 1)
        return len(
            {
                candidate.pattern.id
                for candidate in self._repository.query_recovery_pattern_vectors(
                    probe,
                    limit=raw_count,
                )
                if candidate.pattern.id in R13B_TREATMENT_PATTERN_IDS
            }
        )

    def query_recovery_pattern_vectors(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> tuple[RecoveryPatternVectorCandidate, ...]:
        """Return only canonical candidates, preserving native scores."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        raw_count = self._repository.count_recovery_pattern_vectors()
        if raw_count == 0:
            return ()
        candidates = self._repository.query_recovery_pattern_vectors(
            query_embedding,
            limit=raw_count,
        )
        return tuple(
            candidate
            for candidate in candidates
            if candidate.pattern.id in R13B_TREATMENT_PATTERN_IDS
        )[:limit]

    def get_recovery_pattern(self, pattern_id: str) -> RecoveryPatternLineage:
        """Read canonical provenance only; reject all historical extras."""
        if pattern_id not in R13B_TREATMENT_PATTERN_IDS:
            raise NonCanonicalTreatmentPatternError(
                f"pattern {pattern_id!r} is outside the R13b treatment corpus"
            )
        return self._repository.get_recovery_pattern(pattern_id)


@dataclass
class Neo4jAdvisoryRuntime:
    """Owned resources for one real, read-only advisory-service runtime."""

    repository: Neo4jRepository
    treatment_repository: R13bTreatmentRepository
    advisory_service: AdvisoryService

    def close(self) -> None:
        """Close the Neo4j driver owned by this runtime."""
        self.repository.close()

    def __enter__(self) -> Neo4jAdvisoryRuntime:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def create_neo4j_advisory_runtime(
    settings: Settings | None = None,
    *,
    fail_closed_advisory: bool = False,
) -> Neo4jAdvisoryRuntime:
    """Build the production Neo4j -> V2 retrieval -> advisory stack.

    The caller owns the returned runtime and must close it.  This function does
    not verify connectivity, create indexes, or otherwise mutate the graph.
    ``ExperimentRunner`` remains unaware of Neo4j and receives only the
    returned ``advisory_service`` through its existing dependency-injection
    boundary.
    """
    selected_settings = settings or get_settings()
    _inject_tls_truststore(selected_settings)
    repository = Neo4jRepository(
        uri=selected_settings.neo4j_uri,
        username=selected_settings.neo4j_username,
        password=selected_settings.neo4j_password,
        database=selected_settings.neo4j_database,
    )
    treatment_repository = R13bTreatmentRepository(repository)
    return Neo4jAdvisoryRuntime(
        repository=repository,
        treatment_repository=treatment_repository,
        advisory_service=AdvisoryService(
            cast(OperationalMemoryRepository, treatment_repository),
            fail_closed=fail_closed_advisory,
        ),
    )


def _inject_tls_truststore(settings: Settings) -> None:
    """Apply the existing Neo4j TLS setup used by live integration tests."""
    if not settings.neo4j_uri.startswith("neo4j+s://"):
        return
    truststore = import_module("truststore")
    inject = cast(Callable[[], None], truststore.__dict__["inject_into_ssl"])
    inject()


__all__ = [
    "NonCanonicalTreatmentPatternError",
    "Neo4jAdvisoryRuntime",
    "R13B_TREATMENT_PATTERN_IDS",
    "R13bTreatmentRepository",
    "TreatmentVectorRepository",
    "create_neo4j_advisory_runtime",
]
