CREATE VECTOR INDEX recovery_pattern_embedding_idx IF NOT EXISTS
FOR (pattern:RecoveryPattern) ON (pattern.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }
};
