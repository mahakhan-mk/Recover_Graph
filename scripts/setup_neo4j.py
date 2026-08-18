"""Initialize the Neo4j schema and verify AuraDB connectivity."""

from collections.abc import Iterator
from pathlib import Path

from graph_swarm.graph.neo4j_repository import Neo4jRepository
from graph_swarm.settings import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations" / "neo4j"


def iter_migration_statements(migrations_dir: Path = MIGRATIONS_DIR) -> Iterator[str]:
    """Yield non-empty Cypher statements from migrations in lexical order."""
    for migration_path in sorted(migrations_dir.glob("*.cypher")):
        for statement in migration_path.read_text(encoding="utf-8").split(";"):
            statement = statement.strip()
            if statement:
                yield statement


def apply_schema(repository: Neo4jRepository, migrations_dir: Path = MIGRATIONS_DIR) -> None:
    """Apply all Neo4j migrations through the configured repository."""
    for statement in iter_migration_statements(migrations_dir):
        repository.execute_query(statement)


def main() -> None:
    # AuraDB connections in this environment require the platform trust store.
    import truststore

    truststore.inject_into_ssl()
    settings = get_settings()

    with Neo4jRepository(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    ) as repository:
        repository.verify_connectivity()
        print("Neo4j connection successful.")
        apply_schema(repository)
        print("Neo4j schema initialization successful.")


if __name__ == "__main__":
    main()
