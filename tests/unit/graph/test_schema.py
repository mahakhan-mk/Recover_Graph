from pathlib import Path
from unittest.mock import Mock

from scripts.setup_neo4j import apply_schema, iter_migration_statements


def test_rollout_1_constraints_cover_current_graph_identities() -> None:
    migration = Path("migrations/neo4j/001_constraints.cypher").read_text(
        encoding="utf-8"
    )

    expected_constraints = {
        "run_id_unique": "FOR (n:Run) REQUIRE n.id IS UNIQUE",
        "task_id_unique": "FOR (n:Task) REQUIRE n.id IS UNIQUE",
        "action_id_unique": "FOR (n:Action) REQUIRE n.id IS UNIQUE",
        "tool_name_unique": "FOR (n:Tool) REQUIRE n.name IS UNIQUE",
        "environment_id_unique": "FOR (n:Environment) REQUIRE n.id IS UNIQUE",
        "failure_id_unique": "FOR (n:FailureEpisode) REQUIRE n.id IS UNIQUE",
        "resolution_id_unique": "FOR (n:Resolution) REQUIRE n.id IS UNIQUE",
        "outcome_id_unique": "FOR (n:Outcome) REQUIRE n.id IS UNIQUE",
    }

    for constraint_name, definition in expected_constraints.items():
        assert f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS" in migration
        assert definition in migration

    assert "(n.repository, n.runtime) IS NODE KEY" not in migration
    assert "REQUIRE n.action_id IS UNIQUE" not in migration


def test_apply_schema_executes_migrations_in_order(tmp_path: Path) -> None:
    (tmp_path / "002_second.cypher").write_text("RETURN 2;", encoding="utf-8")
    (tmp_path / "001_first.cypher").write_text("RETURN 1;", encoding="utf-8")
    repository = Mock()

    apply_schema(repository, tmp_path)

    assert [call.args[0] for call in repository.execute_query.call_args_list] == [
        "RETURN 1",
        "RETURN 2",
    ]


def test_iter_migration_statements_ignores_empty_statements(tmp_path: Path) -> None:
    (tmp_path / "001_schema.cypher").write_text(
        "CREATE CONSTRAINT example IF NOT EXISTS FOR (n:Example) REQUIRE n.id IS UNIQUE;\n;",
        encoding="utf-8",
    )

    assert list(iter_migration_statements(tmp_path)) == [
        "CREATE CONSTRAINT example IF NOT EXISTS FOR (n:Example) REQUIRE n.id IS UNIQUE"
    ]
