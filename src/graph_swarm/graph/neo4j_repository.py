"""Synchronous Neo4j implementation of the operational memory repository."""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import TracebackType
from typing import LiteralString, Protocol, cast

from neo4j import Driver, EagerResult, GraphDatabase, Query, Record

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph import queries
from graph_swarm.graph._validation import (
    validate_action_persistence,
    validate_relationship_ids,
)
from graph_swarm.graph.read_models import ActionLineageRecord, IncidentLineage
from graph_swarm.retrieval.candidates import (
    HistoricalActionContext,
    HistoricalRecoveryCandidate,
)


class EntityNotFoundError(LookupError):
    """Raised when a required Neo4j entity or lineage endpoint is absent."""


class _GraphDatabaseApi(Protocol):
    """Typed view of the synchronous public driver factory boundary."""

    @staticmethod
    def driver(uri: str, *, auth: tuple[str, str]) -> Driver:
        ...


class _DriverApi(Protocol):
    """Typed view of the synchronous public driver operations used here."""

    def verify_connectivity(self) -> None:
        ...

    def execute_query(
        self,
        query_: LiteralString | Query,
        *,
        parameters_: dict[str, object],
        database_: str,
    ) -> EagerResult:
        ...


def _isoformat(value: datetime) -> str:
    return value.isoformat()


def _json_object(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _properties(record: Record, key: str) -> Mapping[str, object]:
    value = record[key]
    if value is None:
        raise ValueError(f"Neo4j record field {key!r} is null")
    return cast(Mapping[str, object], value)


def _required_text(properties: Mapping[str, object], key: str) -> str:
    return cast(str, properties[key])


def _required_int(properties: Mapping[str, object], key: str) -> int:
    return cast(int, properties[key])


def _datetime(properties: Mapping[str, object], key: str) -> datetime:
    return datetime.fromisoformat(_required_text(properties, key))


def _json_dict(properties: Mapping[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(_required_text(properties, key)))


def _json_string_dict(properties: Mapping[str, object], key: str) -> dict[str, str]:
    value = _json_dict(properties, key)
    if not all(isinstance(item, str) for item in value.values()):
        raise ValueError(f"Neo4j property {key!r} must contain only string values")
    return cast(dict[str, str], value)


def _optional_value(properties: Mapping[str, object], key: str) -> object | None:
    return properties.get(key)


def _optional_int(properties: Mapping[str, object], key: str) -> int | None:
    return cast(int | None, _optional_value(properties, key))


def _optional_datetime(properties: Mapping[str, object], key: str) -> datetime | None:
    value = _optional_value(properties, key)
    if value is None:
        return None
    return datetime.fromisoformat(cast(str, value))


def _read_run(properties: Mapping[str, object]) -> Run:
    return Run(
        id=_required_text(properties, "id"),
        task_id=_required_text(properties, "task_id"),
        started_at=_datetime(properties, "started_at"),
    )


def _read_task(properties: Mapping[str, object]) -> Task:
    return Task(
        id=_required_text(properties, "id"),
        family_id=_required_text(properties, "family_id"),
        repository=_required_text(properties, "repository"),
        chronological_index=_required_int(properties, "chronological_index"),
    )


def _read_tool(properties: Mapping[str, object]) -> Tool:
    return Tool(name=_required_text(properties, "name"))


def _read_environment(properties: Mapping[str, object]) -> EnvironmentContext:
    return EnvironmentContext(
        id=_required_text(properties, "id"),
        repository=_required_text(properties, "repository"),
        runtime=_required_text(properties, "runtime"),
        versions=_json_string_dict(properties, "versions_json"),
        markers=_json_string_dict(properties, "markers_json"),
    )


def _read_failure(properties: Mapping[str, object]) -> FailureEpisode:
    return FailureEpisode(
        id=_required_text(properties, "id"),
        action_id=_required_text(properties, "action_id"),
        failure_type=FailureType(_required_text(properties, "failure_type")),
        signature=_required_text(properties, "signature"),
        symptom=_required_text(properties, "symptom"),
        observed_at=_datetime(properties, "observed_at"),
    )


def _read_resolution(properties: Mapping[str, object]) -> Resolution:
    return Resolution(
        id=_required_text(properties, "id"),
        failure_id=_required_text(properties, "failure_id"),
        description=_required_text(properties, "description"),
        status=ResolutionStatus(_required_text(properties, "status")),
        successful_observations=_required_int(properties, "successful_observations"),
        failed_observations=_required_int(properties, "failed_observations"),
        observed_at=_optional_datetime(properties, "observed_at"),
    )


def _read_outcome(properties: Mapping[str, object]) -> Outcome:
    return Outcome(
        id=_required_text(properties, "id"),
        action_id=_required_text(properties, "action_id"),
        success=cast(bool, properties["success"]),
        tests_passed=_optional_int(properties, "tests_passed"),
        tests_failed=_optional_int(properties, "tests_failed"),
        exit_code=_optional_int(properties, "exit_code"),
        observed_at=_datetime(properties, "observed_at"),
    )


def _read_action(properties: Mapping[str, object]) -> ActionLineageRecord:
    planned_action = PlannedAction(
        id=_required_text(properties, "id"),
        run_id=_required_text(properties, "run_id"),
        task_id=_required_text(properties, "task_id"),
        tool=_required_text(properties, "tool"),
        operation=_required_text(properties, "operation"),
        arguments=_json_dict(properties, "arguments_json"),
        planned_at=_datetime(properties, "planned_at"),
    )
    result = ActionResult(
        action_id=planned_action.id,
        tool_name=_required_text(properties, "result_tool_name"),
        success=cast(bool, properties["success"]),
        exit_code=_optional_int(properties, "exit_code"),
        output=cast(str | None, _optional_value(properties, "output")),
        error=cast(str | None, _optional_value(properties, "error")),
        started_at=_datetime(properties, "started_at"),
        completed_at=_datetime(properties, "completed_at"),
    )
    return ActionLineageRecord(planned_action=planned_action, result=result)


def _read_historical_candidate(record: Record) -> HistoricalRecoveryCandidate:
    """Map one candidate query row without leaking Neo4j values upward."""
    failure = FailureEpisode(
        id=cast(str, record["failure_id"]),
        action_id=cast(str, record["failure_action_id"]),
        failure_type=FailureType(cast(str, record["failure_type"])),
        signature=cast(str, record["failure_signature"]),
        symptom=cast(str, record["symptom"]),
        observed_at=datetime.fromisoformat(cast(str, record["failure_observed_at"])),
    )
    failed_action = HistoricalActionContext(
        id=cast(str, record["failed_action_id"]),
        source_run_id=cast(str, record["source_run_id"]),
        tool=cast(str, record["tool"]),
        operation=cast(str, record["operation"]),
        planned_at=datetime.fromisoformat(cast(str, record["planned_at"])),
    )
    environment = EnvironmentContext(
        id=cast(str, record["environment_id"]),
        repository=cast(str, record["repository"]),
        runtime=cast(str, record["runtime"]),
        versions=_json_string_dict(
            {"versions_json": cast(str, record["versions_json"])},
            "versions_json",
        ),
        markers=_json_string_dict(
            {"markers_json": cast(str, record["markers_json"])},
            "markers_json",
        ),
    )
    resolution = Resolution(
        id=cast(str, record["resolution_id"]),
        failure_id=cast(str, record["resolution_failure_id"]),
        description=cast(str, record["resolution_description"]),
        status=ResolutionStatus(cast(str, record["resolution_status"])),
        successful_observations=cast(int, record["successful_observations"]),
        failed_observations=cast(int, record["failed_observations"]),
        observed_at=_optional_datetime(
            {"resolution_observed_at": record["resolution_observed_at"]},
            "resolution_observed_at",
        ),
    )

    outcomes: list[Outcome] = []
    outcome_values = cast(Sequence[object], record["outcomes"])
    for outcome_value in outcome_values:
        if outcome_value is None:
            continue
        outcome_properties = cast(Mapping[str, object], outcome_value)
        if outcome_properties.get("id") is None:
            continue
        outcomes.append(_read_outcome(outcome_properties))

    return HistoricalRecoveryCandidate(
        failure=failure,
        failed_action=failed_action,
        environment=environment,
        resolution=resolution,
        outcomes=tuple(outcomes),
    )


class Neo4jRepository:
    """Synchronous Rollout 1 repository backed by the official Neo4j driver."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str,
    ) -> None:
        driver_factory = cast(_GraphDatabaseApi, GraphDatabase).driver
        self._driver: Driver = driver_factory(
            uri,
            auth=(username, password),
        )
        self._database = database

    def verify_connectivity(self) -> None:
        """Verify that the driver can connect to Neo4j."""
        cast(_DriverApi, self._driver).verify_connectivity()

    def save_run(self, run: Run) -> None:
        self.execute_query(
            queries.SAVE_RUN,
            id=run.id,
            task_id=run.task_id,
            started_at=_isoformat(run.started_at),
        )

    def save_task(self, task: Task) -> None:
        self.execute_query(
            queries.SAVE_TASK,
            id=task.id,
            family_id=task.family_id,
            repository=task.repository,
            chronological_index=task.chronological_index,
        )

    def save_action(self, action: PlannedAction, result: ActionResult) -> None:
        validate_action_persistence(action, result)
        self.execute_query(
            queries.SAVE_ACTION,
            id=action.id,
            run_id=action.run_id,
            task_id=action.task_id,
            tool=action.tool,
            operation=action.operation,
            arguments_json=_json_object(action.arguments),
            planned_at=_isoformat(action.planned_at),
            result_tool_name=result.tool_name,
            success=result.success,
            exit_code=result.exit_code,
            output=result.output,
            error=result.error,
            started_at=_isoformat(result.started_at),
            completed_at=_isoformat(result.completed_at),
        )

    def save_tool(self, tool: Tool) -> None:
        self.execute_query(queries.SAVE_TOOL, name=tool.name)

    def save_environment(self, environment: EnvironmentContext) -> None:
        self.execute_query(
            queries.SAVE_ENVIRONMENT,
            id=environment.id,
            repository=environment.repository,
            runtime=environment.runtime,
            versions_json=_json_object(environment.versions),
            markers_json=_json_object(environment.markers),
        )

    def save_failure(self, failure: FailureEpisode) -> None:
        self.execute_query(
            queries.SAVE_FAILURE,
            id=failure.id,
            action_id=failure.action_id,
            failure_type=failure.failure_type.value,
            signature=failure.signature,
            symptom=failure.symptom,
            observed_at=_isoformat(failure.observed_at),
        )

    def save_resolution(self, resolution: Resolution) -> None:
        self.execute_query(
            queries.SAVE_RESOLUTION,
            id=resolution.id,
            failure_id=resolution.failure_id,
            description=resolution.description,
            status=resolution.status.value,
            successful_observations=resolution.successful_observations,
            failed_observations=resolution.failed_observations,
            observed_at=_isoformat(resolution.observed_at)
            if resolution.observed_at is not None
            else None,
        )

    def save_outcome(self, outcome: Outcome) -> None:
        self.execute_query(
            queries.SAVE_OUTCOME,
            id=outcome.id,
            action_id=outcome.action_id,
            success=outcome.success,
            tests_passed=outcome.tests_passed,
            tests_failed=outcome.tests_failed,
            exit_code=outcome.exit_code,
            observed_at=_isoformat(outcome.observed_at),
        )

    def _link(self, query: str, relationship: str, **identifiers: str) -> None:
        validate_relationship_ids(**identifiers)
        result = self.execute_query(query, **identifiers)
        if not result.records:
            raise EntityNotFoundError(
                f"Cannot create {relationship}: one or more endpoint entities are missing"
            )

    def link_task_action(self, task_id: str, action_id: str) -> None:
        self._link(
            queries.LINK_TASK_ACTION,
            "HAS_ACTION",
            task_id=task_id,
            action_id=action_id,
        )

    def link_action_tool(self, action_id: str, tool_name: str) -> None:
        self._link(
            queries.LINK_ACTION_TOOL,
            "USED",
            action_id=action_id,
            tool_name=tool_name,
        )

    def link_action_run(self, action_id: str, run_id: str) -> None:
        self._link(
            queries.LINK_ACTION_RUN,
            "PART_OF",
            action_id=action_id,
            run_id=run_id,
        )

    def link_action_failure(self, action_id: str, failure_id: str) -> None:
        self._link(
            queries.LINK_ACTION_FAILURE,
            "PART_OF_FAILURE",
            action_id=action_id,
            failure_id=failure_id,
        )

    def link_failure_environment(self, failure_id: str, environment_id: str) -> None:
        self._link(
            queries.LINK_FAILURE_ENVIRONMENT,
            "OCCURRED_IN",
            failure_id=failure_id,
            environment_id=environment_id,
        )

    def link_failure_resolution(self, failure_id: str, resolution_id: str) -> None:
        self._link(
            queries.LINK_FAILURE_RESOLUTION,
            "RESOLVED_BY",
            failure_id=failure_id,
            resolution_id=resolution_id,
        )

    def link_resolution_outcome(self, resolution_id: str, outcome_id: str) -> None:
        self._link(
            queries.LINK_RESOLUTION_OUTCOME,
            "VERIFIED_BY",
            resolution_id=resolution_id,
            outcome_id=outcome_id,
        )

    def get_incident_lineage(self, failure_id: str) -> IncidentLineage:
        validate_relationship_ids(failure_id=failure_id)
        failure_result = self.execute_query(queries.GET_FAILURE, failure_id=failure_id)
        if not failure_result.records:
            raise EntityNotFoundError(f"FailureEpisode {failure_id!r} was not found")
        failure = _read_failure(_properties(failure_result.records[0], "failure"))

        environment_result = self.execute_query(
            queries.GET_ENVIRONMENT,
            failure_id=failure_id,
        )
        if not environment_result.records:
            raise EntityNotFoundError(
                f"FailureEpisode {failure_id!r} has no Environment lineage"
            )
        environment = _read_environment(
            _properties(environment_result.records[0], "environment")
        )

        action_result = self.execute_query(
            queries.GET_ACTION_CONTEXT,
            failure_id=failure_id,
        )
        if not action_result.records:
            raise EntityNotFoundError(
                f"FailureEpisode {failure_id!r} has no complete Action lineage"
            )

        actions_by_id: dict[str, ActionLineageRecord] = {}
        task_by_id: dict[str, Task] = {}
        run_by_id: dict[str, Run] = {}
        for record in action_result.records:
            action = _read_action(_properties(record, "action"))
            task = _read_task(_properties(record, "task"))
            run = _read_run(_properties(record, "run"))
            actions_by_id[action.planned_action.id] = action
            task_by_id[task.id] = task
            run_by_id[run.id] = run

        if len(task_by_id) != 1 or len(run_by_id) != 1:
            raise ValueError(
                f"FailureEpisode {failure_id!r} has ambiguous Task or Run lineage"
            )

        tool_result = self.execute_query(queries.GET_TOOLS, failure_id=failure_id)
        tools_by_name = {
            _required_text(_properties(record, "tool"), "name"):
            _read_tool(_properties(record, "tool"))
            for record in tool_result.records
        }

        resolution_result = self.execute_query(
            queries.GET_RESOLUTIONS,
            failure_id=failure_id,
        )
        resolutions_by_id: dict[str, Resolution] = {}
        outcomes_by_id: dict[str, Outcome] = {}
        for record in resolution_result.records:
            resolution = _read_resolution(_properties(record, "resolution"))
            resolutions_by_id[resolution.id] = resolution
            outcome_values = cast(Sequence[object], record["outcomes"])
            for outcome_value in outcome_values:
                if outcome_value is None:
                    continue
                outcome = _read_outcome(cast(Mapping[str, object], outcome_value))
                outcomes_by_id[outcome.id] = outcome

        return IncidentLineage(
            run=next(iter(run_by_id.values())),
            task=next(iter(task_by_id.values())),
            actions=tuple(actions_by_id.values()),
            tools=tuple(tools_by_name.values()),
            failure=failure,
            environment=environment,
            resolutions=tuple(resolutions_by_id.values()),
            outcomes=tuple(outcomes_by_id.values()),
        )

    def find_historical_recovery_candidates(
        self,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
    ) -> tuple[HistoricalRecoveryCandidate, ...]:
        """Retrieve eligible candidates using only persisted structural fields."""
        result = self.execute_query(
            queries.FIND_HISTORICAL_RECOVERY_CANDIDATES,
            tool=planned_action.tool,
            operation=planned_action.operation,
            repository=environment.repository,
            current_run_id=planned_action.run_id,
            planned_at=_isoformat(planned_action.planned_at),
        )
        return tuple(_read_historical_candidate(record) for record in result.records)

    def execute_query(
        self,
        query: str,
        **parameters: object,
    ) -> EagerResult:
        """Execute a Cypher query against the configured database."""
        return cast(_DriverApi, self._driver).execute_query(
            cast(LiteralString, query),
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
