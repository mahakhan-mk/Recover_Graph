import shutil
from pathlib import Path

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.read_file import read_file
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.events import AgentEvent

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "fixtures"
    / "repositories"
    / "rollout1_agent_smoke"
)


def test_controlled_tools_repair_local_fixture_and_record_event_stream(
    tmp_path: Path,
) -> None:
    copied_fixture = tmp_path / "rollout1_agent_smoke"
    shutil.copytree(FIXTURE_PATH, copied_fixture)
    dependencies = AgentDependencies(
        workspace_root=copied_fixture,
        run_id="LOCAL-RUN-ROLLOUT1-SMOKE",
        task_id="LOCAL-ROLLOUT1-SMOKE",
    )

    first_test_result = run_tests(dependencies, timeout_seconds=30)
    assert first_test_result.tool_name == "run_tests"
    assert first_test_result.success is False
    assert first_test_result.exit_code != 0
    assert first_test_result.output is not None
    assert "test_add" in first_test_result.output

    read_result = read_file(dependencies, "calculator.py")
    assert read_result.tool_name == "read_file"
    assert read_result.success is True
    assert read_result.output is not None
    assert "return a - b" in read_result.output

    repaired_source = read_result.output.replace("return a - b", "return a + b")
    write_result = write_file(dependencies, "calculator.py", repaired_source)
    assert write_result.tool_name == "write_file"
    assert write_result.success is True

    final_test_result = run_tests(dependencies, timeout_seconds=30)
    assert final_test_result.tool_name == "run_tests"
    assert final_test_result.success is True
    assert final_test_result.exit_code == 0

    assert [event.result.tool_name for event in dependencies.events] == [
        "run_tests",
        "read_file",
        "write_file",
        "run_tests",
    ]
    assert len(dependencies.events) == 4
    assert dependencies.events[0].result.success is False
    assert dependencies.events[0].result.exit_code != 0
    assert dependencies.events[3].result.success is True
    assert dependencies.events[3].result.exit_code == 0

    event_ids = {event.event_id for event in dependencies.events}
    action_ids = {event.action_id for event in dependencies.events}
    assert len(event_ids) == 4
    assert len(action_ids) == 4
    for event in dependencies.events:
        assert event.run_id == dependencies.run_id
        assert event.task_id == "LOCAL-ROLLOUT1-SMOKE"
        assert event.action_id == event.result.action_id
        assert AgentEvent.model_validate_json(event.model_dump_json()) == event

    assert "return a - b" in (FIXTURE_PATH / "calculator.py").read_text(encoding="utf-8")
