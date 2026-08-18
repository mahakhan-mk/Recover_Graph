import importlib
import inspect
import sys
from unittest.mock import Mock, patch

from graph_swarm.graph.repository import OperationalMemoryRepository


def _repository_class():
    from graph_swarm.graph.neo4j_repository import Neo4jRepository

    return Neo4jRepository


def test_import_does_not_inject_ssl() -> None:
    sys.modules.pop("graph_swarm.graph.neo4j_repository", None)

    with patch("truststore.inject_into_ssl") as inject_into_ssl:
        importlib.import_module("graph_swarm.graph.neo4j_repository")

    inject_into_ssl.assert_not_called()


def test_repository_methods_are_synchronous() -> None:
    repository_class = _repository_class()

    assert not inspect.iscoroutinefunction(repository_class.verify_connectivity)
    assert not inspect.iscoroutinefunction(repository_class.execute_query)
    assert not inspect.iscoroutinefunction(repository_class.close)


def test_operational_memory_repository_methods_are_synchronous() -> None:
    for method_name in ("save_failure", "save_resolution", "save_outcome"):
        assert not inspect.iscoroutinefunction(
            getattr(OperationalMemoryRepository, method_name)
        )


def test_repository_creates_driver_with_uri_and_authentication() -> None:
    repository_class = _repository_class()

    with patch("graph_swarm.graph.neo4j_repository.GraphDatabase.driver") as driver_factory:
        repository_class(
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
    repository_class = _repository_class()
    driver = Mock()
    with patch(
        "graph_swarm.graph.neo4j_repository.GraphDatabase.driver",
        return_value=driver,
    ):
        repository = repository_class("uri", "username", "password", "database")

    repository.verify_connectivity()

    driver.verify_connectivity.assert_called_once_with()


def test_execute_query_passes_database_and_parameters_separately() -> None:
    repository_class = _repository_class()
    driver = Mock()
    with patch(
        "graph_swarm.graph.neo4j_repository.GraphDatabase.driver",
        return_value=driver,
    ):
        repository = repository_class("uri", "username", "password", "graph")

    repository.execute_query("MATCH (n {name: $name}) RETURN n", name="Ada")

    driver.execute_query.assert_called_once_with(
        "MATCH (n {name: $name}) RETURN n",
        parameters_={"name": "Ada"},
        database_="graph",
    )


def test_close_closes_driver() -> None:
    repository_class = _repository_class()
    driver = Mock()
    with patch(
        "graph_swarm.graph.neo4j_repository.GraphDatabase.driver",
        return_value=driver,
    ):
        repository = repository_class("uri", "username", "password", "database")

    repository.close()

    driver.close.assert_called_once_with()


def test_context_manager_exit_closes_driver() -> None:
    repository_class = _repository_class()
    driver = Mock()
    with patch(
        "graph_swarm.graph.neo4j_repository.GraphDatabase.driver",
        return_value=driver,
    ):
        with repository_class("uri", "username", "password", "database"):
            pass

    driver.close.assert_called_once_with()
