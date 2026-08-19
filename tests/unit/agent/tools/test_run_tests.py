from pathlib import Path

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.domain.action import ActionResult


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentDependencies(workspace, "run-001", "task-001")


def assert_timestamps_are_valid(result: ActionResult) -> None:
    assert result.started_at.tzinfo is not None
    assert result.started_at.utcoffset() is not None
    assert result.completed_at.tzinfo is not None
    assert result.completed_at.utcoffset() is not None
    assert result.completed_at >= result.started_at


def test_passing_fixture_returns_successful_run_tests_result(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    (dependencies.workspace_root / "test_example.py").write_text(
        "def test_ok() -> None:\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    result = run_tests(dependencies, timeout_seconds=30)

    assert type(result) is ActionResult
    assert result.tool_name == "run_tests"
    assert result.success is True
    assert result.exit_code == 0
    assert result.output is not None and "1 passed" in result.output
    assert result.action_id
    assert_timestamps_are_valid(result)
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result == result
    assert dependencies.events[0].action_id == result.action_id


def test_failing_fixture_returns_failed_run_tests_result(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    (dependencies.workspace_root / "test_example.py").write_text(
        "def test_failure() -> None:\n    assert 2 + 2 == 5\n",
        encoding="utf-8",
    )

    result = run_tests(dependencies, timeout_seconds=30)

    assert result.tool_name == "run_tests"
    assert result.success is False
    assert result.exit_code != 0
    assert result.output is not None and "failed" in result.output
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result == result
    assert dependencies.events[0].action_id == result.action_id


def test_run_tests_timeout_returns_failed_result(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    (dependencies.workspace_root / "test_example.py").write_text(
        "import time\n\ndef test_slow() -> None:\n    time.sleep(1)\n",
        encoding="utf-8",
    )

    result = run_tests(dependencies, timeout_seconds=0.05)

    assert result.tool_name == "run_tests"
    assert result.success is False
    assert result.exit_code is None
    assert result.error is not None and "timed out" in result.error
