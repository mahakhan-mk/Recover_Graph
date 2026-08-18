"""Smoke test Neo4j AuraDB connectivity."""

import truststore

truststore.inject_into_ssl()

from graph_swarm.graph.neo4j_repository import Neo4jRepository  # noqa: E402
from graph_swarm.settings import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()

    with Neo4jRepository(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    ) as repository:
        repository.verify_connectivity()
        print("Neo4j connection successful.")

        result = repository.execute_query("RETURN 1 AS connection_test")
        connection_test = result.records[0]["connection_test"]

        if connection_test != 1:
            msg = f"Unexpected connection_test value: {connection_test!r}"
            raise RuntimeError(msg)

        print(f"Database query successful: connection_test={connection_test}")


if __name__ == "__main__":
    main()
