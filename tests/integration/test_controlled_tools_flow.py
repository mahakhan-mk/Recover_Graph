import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

from pydantic_ai import AgentRunResult
from pydantic_ai.models.test import TestModel

from graph_swarm.agent.coding_agent import create_coding_agent, run_coding_agent
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.read_file import read_file
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.events import AgentEvent
from graph_swarm.settings import Settings

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "fixtures"
    / "repositories"
    / "rollout1_agent_smoke"
)


class WriteFileBoundaryModel(TestModel):
    """Deterministic model that supplies concrete arguments to the real wrapper."""

    def gen_tool_args(self, tool_def: object) -> object:
        if getattr(tool_def, "name", None) == "write_file":
            return {"path": "recovered.py", "content": "value = 1\n"}
        return super().gen_tool_args(tool_def)  # type: ignore[arg-type]


def make_offline_settings() -> Settings:
    settings_factory = cast(Callable[..., Settings], Settings)
    return settings_factory(
        _env_file=None,
        neo4j_uri="neo4j+s://example.databases.neo4j.io",
        neo4j_username="example-user",
        neo4j_password="example-password",
        neo4j_database="example-db",
        agent_request_limit=3,
        agent_command_timeout_seconds=5,
        agent_tests_timeout_seconds=5,
    )


def test_actual_coding_agent_wrapper_hands_off_exact_planned_action(
    tmp_path: Path,
) -> None:
    dependencies = AgentDependencies(tmp_path, "run-wrapper", "task-wrapper")
    agent = create_coding_agent(
        make_offline_settings(),
        model=WriteFileBoundaryModel(
            call_tools=["write_file"],
            custom_output_text="complete",
        ),
    )

    result: AgentRunResult[str] = run_coding_agent(
        agent,
        make_offline_settings(),
        dependencies,
        "Write the recovery file.",
    )

    assert result.output == "complete"
    assert len(dependencies.events) == 1
    event = dependencies.events[0]
    action = dependencies.planned_action_for(event.action_id)
    assert action is not None
    assert action.id == event.action_id == event.result.action_id
    assert action.operation == "write_file"
    assert action.arguments == {"path": "recovered.py", "content": "value = 1\n"}
    assert action.run_id == event.run_id == dependencies.run_id
    assert action.task_id == event.task_id == dependencies.task_id
    assert action.planned_at <= event.occurred_at
    assert (tmp_path / "recovered.py").read_text(encoding="utf-8") == "value = 1\n"


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
