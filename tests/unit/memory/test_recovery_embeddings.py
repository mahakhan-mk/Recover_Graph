from datetime import UTC, datetime
from math import isclose
from unittest.mock import Mock

import pytest

import graph_swarm.memory.recovery_embeddings as recovery_embeddings
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.memory.recovery_embeddings import (
    RECOVERY_PATTERN_EMBEDDING_DIMENSION,
    RECOVERY_PATTERN_EMBEDDING_MODEL,
    RECOVERY_PATTERN_EMBEDDING_NORMALIZED,
    EmbeddingModelUnavailableError,
    EmbeddingValidationError,
    RecoveryEmbeddingError,
    RecoveryPatternEmbedder,
    embed_and_persist_recovery_pattern,
    recovery_pattern_embedding_text,
    validate_embedding_vector,
)


def make_pattern() -> RecoveryPattern:
    return RecoveryPattern(
        id="pattern-001",
        title="  Restore   branch condition ",
        guidance=" Inspect polarity and restore the intended condition. ",
        source_failure_id="failure-001",
        source_resolution_id="resolution-001",
        source_outcome_id="outcome-001",
        source_task_id="task-001",
        source_chronological_index=7,
        source_tool="run_tests",
        source_operation="pytest",
        source_failure_type="test_failure",
        environment_constraints=EnvironmentConstraints(
            runtime="python-3.13",
            versions={"pytest": "8.0"},
            dependencies={"package": "1.2"},
            markers={"platform": "linux"},
        ),
        verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        evidence_count=1,
        evidence_summary="Failure evidence and recovery outcome.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class FakeEncoder:
    def __init__(self, value: float = 0.25) -> None:
        self.value = value
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, normalize_embeddings: bool) -> object:
        self.calls.append((text, normalize_embeddings))
        return [self.value] * RECOVERY_PATTERN_EMBEDDING_DIMENSION


def test_embedding_text_v1_contains_only_normalized_semantic_fields() -> None:
    pattern = make_pattern()

    assert recovery_pattern_embedding_text(pattern) == (
        "Restore branch condition\nInspect polarity and restore the intended condition."
    )
    text = recovery_pattern_embedding_text(pattern)
    for excluded in (
        pattern.id,
        pattern.source_failure_id,
        pattern.source_task_id,
        "failure evidence",
        "example/repository",
        "python-3.13",
        "package",
    ):
        assert excluded not in text


def test_injected_encoder_is_normalized_and_returns_native_384_float_list() -> None:
    encoder = FakeEncoder()
    embedder = RecoveryPatternEmbedder(encoder=encoder)

    vector = embedder.embed_text("restore the condition")

    assert embedder.model_name == RECOVERY_PATTERN_EMBEDDING_MODEL
    assert embedder.dimension == RECOVERY_PATTERN_EMBEDDING_DIMENSION
    assert embedder.normalize_embeddings is RECOVERY_PATTERN_EMBEDDING_NORMALIZED
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)
    assert encoder.calls == [("restore the condition", True)]


def test_injected_encoder_does_not_require_huggingface_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_settings() -> object:
        raise AssertionError("injected encoders must not load settings")

    monkeypatch.setattr(recovery_embeddings, "get_settings", fail_get_settings)

    embedder = RecoveryPatternEmbedder(encoder=FakeEncoder())

    assert len(embedder.embed_text("restore the condition")) == 384


def test_missing_huggingface_token_raises_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery_embeddings,
        "get_settings",
        lambda: Mock(hf_token=None, hf_embedding_model=RECOVERY_PATTERN_EMBEDDING_MODEL),
    )

    with pytest.raises(
        EmbeddingModelUnavailableError,
        match="HF_TOKEN is required for RecoveryPattern embeddings",
    ):
        RecoveryPatternEmbedder()


def test_huggingface_loader_uses_configured_model_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = [0.25] * RECOVERY_PATTERN_EMBEDDING_DIMENSION
    client = Mock()
    client.feature_extraction.return_value = vector
    inference_client = Mock(return_value=client)
    monkeypatch.setattr(recovery_embeddings, "InferenceClient", inference_client)
    monkeypatch.setattr(
        recovery_embeddings,
        "get_settings",
        lambda: Mock(hf_token="test-token", hf_embedding_model=RECOVERY_PATTERN_EMBEDDING_MODEL),
    )

    embedder = RecoveryPatternEmbedder()
    result = embedder.embed_text("restore the condition")

    inference_client.assert_called_once_with(provider="hf-inference", api_key="test-token")
    client.feature_extraction.assert_called_once_with(
        "restore the condition",
        model=RECOVERY_PATTERN_EMBEDDING_MODEL,
        normalize=True,
    )
    assert result == vector
    assert len(result) == RECOVERY_PATTERN_EMBEDDING_DIMENSION


@pytest.mark.parametrize(
    "invalid_vector",
    (
        [0.1] * 383,
        [0.1] * 385,
        [True] * 384,
        [float("nan")] * 384,
        [float("inf")] * 384,
        ["0.1"] * 384,
    ),
)
def test_embedding_validation_rejects_non_384_finite_numeric_vectors(
    invalid_vector: list[object],
) -> None:
    with pytest.raises(EmbeddingValidationError):
        validate_embedding_vector(invalid_vector)


def test_embed_pattern_changes_only_embedding_and_rejects_invalidated_patterns() -> None:
    pattern = make_pattern()
    embedded = RecoveryPatternEmbedder(encoder=FakeEncoder()).embed_pattern(pattern)

    assert embedded.model_dump(exclude={"embedding"}) == pattern.model_dump(
        exclude={"embedding"}
    )
    assert embedded.embedding is not None

    invalidated = pattern.model_copy(
        update={"verification_status": RecoveryPatternStatus.INVALIDATED}
    )
    with pytest.raises(RecoveryEmbeddingError, match="invalidated"):
        RecoveryPatternEmbedder(encoder=FakeEncoder()).embed_pattern(invalidated)


def test_embedding_persistence_updates_only_existing_pattern_vector() -> None:
    repository = Mock(spec=OperationalMemoryRepository)
    pattern = make_pattern()
    embedder = RecoveryPatternEmbedder(encoder=FakeEncoder(0.125))

    embedded = embed_and_persist_recovery_pattern(pattern, repository, embedder)

    repository.update_recovery_pattern_embedding.assert_called_once_with(
        pattern.id,
        [0.125] * 384,
    )
    repository.save_recovery_pattern.assert_not_called()
    assert embedded.embedding == [0.125] * 384


def test_same_text_is_deterministic_and_fake_distinct_texts_differ() -> None:
    class TextEncoder(FakeEncoder):
        def encode(self, text: str, *, normalize_embeddings: bool) -> object:
            self.calls.append((text, normalize_embeddings))
            value = 0.1 if text == "first" else 0.2
            return [value] * RECOVERY_PATTERN_EMBEDDING_DIMENSION

    embedder = RecoveryPatternEmbedder(encoder=TextEncoder())
    first = embedder.embed_text("first")
    repeated = embedder.embed_text("first")
    different = embedder.embed_text("second")

    assert first == repeated
    assert different != first
    assert isclose(first[0], 0.1)
