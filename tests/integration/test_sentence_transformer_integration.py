"""Opt-in validation of the frozen local SentenceTransformer boundary."""

import os
from math import sqrt

import pytest

from graph_swarm.memory.recovery_embeddings import (
    RECOVERY_PATTERN_EMBEDDING_DIMENSION,
    RECOVERY_PATTERN_EMBEDDING_MODEL,
    RecoveryPatternEmbedder,
)

RUN_EMBEDDING_INTEGRATION = os.getenv("GRAPH_SWARM_RUN_EMBEDDING_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_EMBEDDING_INTEGRATION,
    reason="Set GRAPH_SWARM_RUN_EMBEDDING_INTEGRATION=1 to load the frozen local model",
)
def test_frozen_sentence_transformer_is_local_deterministic_and_384_dimensional() -> None:
    embedder = RecoveryPatternEmbedder()
    first = embedder.embed_text("Restore the branch condition.")
    repeated = embedder.embed_text("Restore the branch condition.")
    different = embedder.embed_text("Update the dependency lockfile.")

    assert embedder.model_name == RECOVERY_PATTERN_EMBEDDING_MODEL
    assert len(first) == RECOVERY_PATTERN_EMBEDDING_DIMENSION
    assert all(isinstance(value, float) for value in first)
    assert sqrt(sum(value * value for value in first)) == pytest.approx(1.0, rel=1e-4)
    assert first == repeated
    assert different != first
