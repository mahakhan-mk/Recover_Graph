"""Hugging Face RecoveryPattern embedding infrastructure."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, cast

from huggingface_hub import InferenceClient

from graph_swarm.domain.recovery_patterns import (
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.settings import Settings, get_settings

RECOVERY_PATTERN_EMBEDDING_TEXT_VERSION = "v1"
RECOVERY_PATTERN_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RECOVERY_PATTERN_EMBEDDING_DIMENSION = 384
RECOVERY_PATTERN_EMBEDDING_NORMALIZED = True


class RecoveryEmbeddingError(RuntimeError):
    """Base error for frozen Hugging Face RecoveryPattern embedding infrastructure."""


class EmbeddingModelUnavailableError(RecoveryEmbeddingError):
    """Raised when the frozen Hugging Face embedding model is unavailable."""


class EmbeddingValidationError(RecoveryEmbeddingError, ValueError):
    """Raised when an encoder returns an invalid vector."""


class EmbeddingEncoder(Protocol):
    """Narrow injectable boundary shared by production and deterministic tests."""

    def encode(self, text: str, *, normalize_embeddings: bool) -> object:
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


def _load_huggingface_encoder(
    *,
    token: str,
    model_name: str,
) -> EmbeddingEncoder:
    try:
        client = InferenceClient(
            provider="hf-inference",
            api_key=token,
        )
    except Exception as error:
        raise EmbeddingModelUnavailableError(
            "Unable to initialize Hugging Face embedding client for "
            f"{model_name!r}: {type(error).__name__}"
        ) from error

    class _HuggingFaceEncoder:
        def encode(self, text: str, *, normalize_embeddings: bool) -> object:
            try:
                return client.feature_extraction(
                    text,
                    model=model_name,
                    normalize=normalize_embeddings,
                )
            except Exception as error:
                raise EmbeddingModelUnavailableError(
                    "Unable to obtain Hugging Face embedding for "
                    f"{model_name!r}: {type(error).__name__}"
                ) from error

    return _HuggingFaceEncoder()


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
    """Generate frozen embeddings without any fallback model."""

    model_name = RECOVERY_PATTERN_EMBEDDING_MODEL
    dimension = RECOVERY_PATTERN_EMBEDDING_DIMENSION
    normalize_embeddings = RECOVERY_PATTERN_EMBEDDING_NORMALIZED

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        settings: Settings | None = None,
    ) -> None:
        if encoder is not None:
            self._encoder = encoder
            return

        selected_settings = settings or get_settings()
        token = selected_settings.hf_token
        if token is None:
            raise EmbeddingModelUnavailableError(
                "HF_TOKEN is required for RecoveryPattern embeddings"
            )
        self.model_name = selected_settings.hf_embedding_model
        self._encoder = _load_huggingface_encoder(
            token=token,
            model_name=self.model_name,
        )

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
