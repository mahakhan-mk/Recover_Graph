from unittest.mock import Mock, patch

from graph_swarm.graph.neo4j_repository import Neo4jRepository


def test_repository_creates_driver_with_uri_and_authentication() -> None:
    with patch("graph_swarm.graph.neo4j_repository.GraphDatabase.driver") as driver_factory:
        Neo4jRepository(
            uri="neo4j+s://example.databases.neo4j.io",
            username="example-user",
            password="example-password",
            database="example-db",
        )

    driver_factory.assert_called_once_with(
        "neo4j+s://example.databases.neo4j.io",
        auth=("example-user", "example-password"),
    )


def test_verify_connectivity_delegates_to_driver() -> None:
    driver = Mock()
    with patch(
        "graph_swarm.graph.neo4j_repository.GraphDatabase.driver",
        return_value=driver,
    ):
        repository = Neo4jRepository("uri", "username", "password", "database")

    repository.verify_connectivity()

    driver.verify_connectivity.assert_called_once_with()


def test_execute_query_passes_database_and_parameters_separately() -> None:
    driver = Mock()
    with patch(
        "graph_swarm.graph.neo4j_repository.GraphDatabase.driver",
        return_value=driver,
    ):
        repository = Neo4jRepository("uri", "username", "password", "graph")

    repository.execute_query("MATCH (n {name: $name}) RETURN n", name="Ada")

    driver.execute_query.assert_called_once_with(
        "MATCH (n {name: $name}) RETURN n",
        parameters_={"name": "Ada"},
        database_="graph",
    )


def test_close_closes_driver() -> None:
    driver = Mock()
    with patch(
        "graph_swarm.graph.neo4j_repository.GraphDatabase.driver",
        return_value=driver,
    ):
        repository = Neo4jRepository("uri", "username", "password", "database")

    repository.close()

    driver.close.assert_called_once_with()


def test_context_manager_exit_closes_driver() -> None:
    driver = Mock()
    with patch(
        "graph_swarm.graph.neo4j_repository.GraphDatabase.driver",
        return_value=driver,
    ):
        with Neo4jRepository("uri", "username", "password", "database"):
            pass

    driver.close.assert_called_once_with()
