from collections.abc import Callable
from typing import cast

import pytest

from graph_swarm.settings import Settings


def test_settings_loads_neo4j_values_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://example.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_USERNAME", "example-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "example-password")
    monkeypatch.setenv("NEO4J_DATABASE", "example-db")

    settings_factory = cast(Callable[[], Settings], Settings)
    settings = settings_factory()

    assert settings.neo4j_uri == "neo4j+s://example.databases.neo4j.io"
    assert settings.neo4j_username == "example-user"
    assert settings.neo4j_password == "example-password"
    assert settings.neo4j_database == "example-db"
