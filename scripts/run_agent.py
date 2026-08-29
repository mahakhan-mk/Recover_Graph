"""Run one bounded live Groq smoke test against an isolated fixture copy."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from pydantic import ValidationError
from pydantic_ai.exceptions import UsageLimitExceeded

from graph_swarm.agent.coding_agent import (
    AgentConfigurationError,
    create_coding_agent,
    run_coding_agent,
)
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.events import AgentEvent
from graph_swarm.settings import Settings, get_settings

TASK_ID = "LOCAL-ROLLOUT1-SMOKE"
AGENT_OBJECTIVE = (
    "Fix the repository so all tests pass. Inspect the relevant files, make the "
    "smallest necessary change, and verify the result by running tests."
)


def _fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "fixtures"
        / "repositories"
        / "rollout1_agent_smoke"
    )


def _validate_event_stream(
    events: list[AgentEvent],
    run_id: str,
) -> tuple[int | None, int | None, int | None, bool]:
    if not events:
        return None, None, None, False

    event_ids = {event.event_id for event in events}
    action_ids = {event.action_id for event in events}
    if len(event_ids) != len(events) or len(action_ids) != len(events):
        return None, None, None, False

    if any(event.run_id != run_id or event.task_id != TASK_ID for event in events):
        return None, None, None, False
    if any(event.action_id != event.result.action_id for event in events):
        return None, None, None, False
    if any(
        previous.occurred_at > current.occurred_at
        for previous, current in zip(events, events[1:], strict=False)
    ):
        return None, None, None, False

    try:
        for event in events:
            AgentEvent.model_validate_json(event.model_dump_json())
    except ValidationError:
        return None, None, None, False

    failed_index = next(
        (
            index
            for index, event in enumerate(events)
            if event.result.tool_name == "run_tests"
            and not event.result.success
            and event.result.exit_code is not None
            and event.result.exit_code != 0
        ),
        None,
    )
    inspection_index = next(
        (
            index
            for index, event in enumerate(events)
            if event.result.tool_name == "read_file" and event.result.success
        ),
        None,
    )
    write_index = next(
        (
            index
            for index, event in enumerate(events)
            if event.result.tool_name == "write_file" and event.result.success
        ),
        None,
    )
    passing_index = next(
        (
            index
            for index, event in enumerate(events)
            if event.result.tool_name == "run_tests"
            and event.result.success
            and event.result.exit_code == 0
            and failed_index is not None
            and write_index is not None
            and index > failed_index
            and index > write_index
        ),
        None,
    )
    ordered = (
        failed_index is not None
        and write_index is not None
        and passing_index is not None
        and failed_index < write_index < passing_index
    )
    return failed_index, inspection_index, passing_index, ordered


def _run_independent_tests(workspace: Path, timeout_seconds: float) -> int:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest"],
            cwd=workspace,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return -1
    return result.returncode


def _run_smoke(settings: Settings) -> int:
    if not settings.groq_model or not settings.groq_model.strip():
        raise AgentConfigurationError(
            "GROQ_MODEL is required for the live smoke run"
        )

    fixture = _fixture_path()
    run_id = f"LOCAL-ROLLOUT1-SMOKE-{uuid4().hex}"
    resource_limit_reached = False
    agent_error: str | None = None

    with TemporaryDirectory(prefix="graph_swarm_rollout1_") as temporary_root:
        workspace = Path(temporary_root) / fixture.name
        shutil.copytree(fixture, workspace)
        dependencies = AgentDependencies(workspace, run_id, TASK_ID)

        try:
            agent = create_coding_agent(settings)
            run_coding_agent(agent, settings, dependencies, AGENT_OBJECTIVE)
        except UsageLimitExceeded:
            resource_limit_reached = True
            agent_error = "UsageLimitExceeded"
        except Exception as error:
            agent_error = type(error).__name__

        final_pytest_exit_code = _run_independent_tests(
            workspace,
            settings.agent_tests_timeout_seconds,
        )
        failed_index, inspection_index, passing_index, ordered = _validate_event_stream(
            dependencies.events,
            run_id,
        )
        write_index = next(
            (
                index
                for index, event in enumerate(dependencies.events)
                if event.result.tool_name == "write_file" and event.result.success
            ),
            None,
        )

        canonical_source = (fixture / "calculator.py").read_text(encoding="utf-8")
        canonical_fixture_broken = "return a - b" in canonical_source

        print(f"model_id: {settings.groq_model}")
        print(f"task_id: {TASK_ID}")
        print(f"run_id: {run_id}")
        print(f"tool_actions: {len(dependencies.events)}")
        print(f"agent_events: {len(dependencies.events)}")
        print(f"initial_failed_run_tests: {failed_index is not None}")
        print(f"read_file_observed: {inspection_index is not None}")
        print(f"write_file_observed: {write_index is not None}")
        print(f"final_passing_run_tests: {passing_index is not None}")
        print(f"event_order_valid: {ordered}")
        print(f"independent_pytest_exit_code: {final_pytest_exit_code}")
        print(f"resource_limit_reached: {resource_limit_reached}")
        print(f"canonical_fixture_broken: {canonical_fixture_broken}")
        if agent_error is not None:
            print(f"agent_error: {agent_error}")

        succeeded = (
            agent_error is None
            and not resource_limit_reached
            and failed_index is not None
            and inspection_index is not None
            and write_index is not None
            and passing_index is not None
            and ordered
            and final_pytest_exit_code == 0
            and canonical_fixture_broken
        )
        return 0 if succeeded else 1


def main() -> int:
    """Load configuration and run exactly one live smoke attempt."""
    try:
        settings = get_settings()
        return _run_smoke(settings)
    except AgentConfigurationError as error:
        print(f"configuration_error: {error}")
        return 2
    except ValidationError:
        print("configuration_error: required application settings are missing")
        return 2
    except OSError:
        print("runner_error: unable to prepare the isolated smoke workspace")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
