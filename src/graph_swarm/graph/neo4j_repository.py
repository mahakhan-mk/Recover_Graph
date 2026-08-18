"""Neo4j connectivity wrapper for the operational memory repository."""

from types import TracebackType

from neo4j import Driver, EagerResult, GraphDatabase


class Neo4jRepository:
    """Small synchronous wrapper around the official Neo4j driver."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str,
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )
        self._database = database

    def verify_connectivity(self) -> None:
        """Verify that the driver can connect to Neo4j."""
        self._driver.verify_connectivity()

    def execute_query(
        self,
        query: str,
        **parameters: object,
    ) -> EagerResult:
        """Execute a Cypher query against the configured database."""
        return self._driver.execute_query(
            query,
            parameters_=parameters,
            database_=self._database,
        )

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    def __enter__(self) -> "Neo4jRepository":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
