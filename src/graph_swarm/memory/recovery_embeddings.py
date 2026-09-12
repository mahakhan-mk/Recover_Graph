"""Local, deterministic RecoveryPattern embedding infrastructure."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, cast

from graph_swarm.domain.recovery_patterns import (
    RecoveryPattern,
    RecoveryPatternStatus,
)

RECOVERY_PATTERN_EMBEDDING_TEXT_VERSION = "v1"
RECOVERY_PATTERN_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RECOVERY_PATTERN_EMBEDDING_DIMENSION = 384
RECOVERY_PATTERN_EMBEDDING_NORMALIZED = True


class RecoveryEmbeddingError(RuntimeError):
    """Base error for local RecoveryPattern embedding infrastructure."""


class EmbeddingModelUnavailableError(RecoveryEmbeddingError):
    """Raised when the frozen local SentenceTransformer cannot be loaded."""


class EmbeddingValidationError(RecoveryEmbeddingError, ValueError):
    """Raised when an encoder returns an invalid vector."""


class EmbeddingEncoder(Protocol):
    """Narrow injectable boundary shared by production and deterministic tests."""

    def encode(self, text: str, *, normalize_embeddings: bool) -> object:
        ...


class _SentenceTransformerModel(Protocol):
    def encode(
        self,
        text: str,
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> object:
        ...


class RecoveryPatternEmbeddingRepository(Protocol):
    """Minimal persistence boundary for the embedding-only update."""

    def update_recovery_pattern_embedding(
        self,
        pattern_id: str,
        embedding: list[float],
    ) -> None:
        ...


def recovery_pattern_embedding_text(pattern: RecoveryPattern) -> str:
    """Return the frozen semantic-only text representation for v1 embeddings."""
    title = " ".join(pattern.title.split())
    guidance = " ".join(pattern.guidance.split())
    return f"{title}\n{guidance}"


def _load_sentence_transformer() -> EmbeddingEncoder:
    try:
        from sentence_transformers import SentenceTransformer

        model = cast(
            _SentenceTransformerModel,
            SentenceTransformer(RECOVERY_PATTERN_EMBEDDING_MODEL),
        )
    except Exception as error:
        raise EmbeddingModelUnavailableError(
            "Unable to load local SentenceTransformer model "
            f"{RECOVERY_PATTERN_EMBEDDING_MODEL!r}: {type(error).__name__}"
        ) from error

    class _SentenceTransformerEncoder:
        def encode(self, text: str, *, normalize_embeddings: bool) -> object:
            return model.encode(
                text,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )

    return _SentenceTransformerEncoder()


def validate_embedding_vector(vector: object) -> list[float]:
    """Validate and normalize an encoder result into a native float list."""
    tolist = getattr(vector, "tolist", None)
    if callable(tolist):
        vector = tolist()
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise EmbeddingValidationError("embedding must be a numeric sequence")
    values = list(cast(Sequence[object], vector))
    if len(values) != RECOVERY_PATTERN_EMBEDDING_DIMENSION:
        raise EmbeddingValidationError(
            "embedding dimension must be "
            f"{RECOVERY_PATTERN_EMBEDDING_DIMENSION}, got {len(values)}"
        )
    floats: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingValidationError("embedding values must be numeric and not boolean")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise EmbeddingValidationError("embedding values must be finite")
        floats.append(numeric_value)
    return floats


class RecoveryPatternEmbedder:
    """Generate frozen local embeddings without any fallback model."""

    model_name = RECOVERY_PATTERN_EMBEDDING_MODEL
    dimension = RECOVERY_PATTERN_EMBEDDING_DIMENSION
    normalize_embeddings = RECOVERY_PATTERN_EMBEDDING_NORMALIZED

    def __init__(self, encoder: EmbeddingEncoder | None = None) -> None:
        self._encoder = encoder if encoder is not None else _load_sentence_transformer()

    def embed_text(self, text: str) -> list[float]:
        """Encode text with the frozen model and validate its native vector."""
        return validate_embedding_vector(
            self._encoder.encode(
                text,
                normalize_embeddings=self.normalize_embeddings,
            )
        )

    def embed_pattern(self, pattern: RecoveryPattern) -> RecoveryPattern:
        """Return a copy with only the embedding field populated."""
        if pattern.verification_status is RecoveryPatternStatus.INVALIDATED:
            raise RecoveryEmbeddingError(
                "invalidated RecoveryPatterns must not receive embeddings"
            )
        vector = self.embed_text(recovery_pattern_embedding_text(pattern))
        return pattern.model_copy(update={"embedding": vector})


def embed_and_persist_recovery_pattern(
    pattern: RecoveryPattern,
    repository: RecoveryPatternEmbeddingRepository,
    embedder: RecoveryPatternEmbedder,
) -> RecoveryPattern:
    """Embed an existing pattern and update only its native vector property."""
    embedded_pattern = embedder.embed_pattern(pattern)
    if embedded_pattern.embedding is None:
        raise EmbeddingValidationError("embedded RecoveryPattern has no vector")
    repository.update_recovery_pattern_embedding(
        embedded_pattern.id,
        embedded_pattern.embedding,
    )
    return embedded_pattern


__all__ = [
    "EmbeddingEncoder",
    "EmbeddingModelUnavailableError",
    "EmbeddingValidationError",
    "RecoveryPatternEmbeddingRepository",
    "RECOVERY_PATTERN_EMBEDDING_DIMENSION",
    "RECOVERY_PATTERN_EMBEDDING_MODEL",
    "RECOVERY_PATTERN_EMBEDDING_NORMALIZED",
    "RECOVERY_PATTERN_EMBEDDING_TEXT_VERSION",
    "RecoveryEmbeddingError",
    "RecoveryPatternEmbedder",
    "embed_and_persist_recovery_pattern",
    "recovery_pattern_embedding_text",
    "validate_embedding_vector",
]
