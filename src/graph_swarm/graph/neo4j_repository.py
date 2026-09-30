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
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
    RecoveryTrigger,
)
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph import queries
from graph_swarm.graph._validation import (
    validate_action_persistence,
    validate_relationship_ids,
)
from graph_swarm.graph.read_models import (
    ActionLineageRecord,
    IncidentLineage,
    RecoveryEvidenceLineage,
    RecoveryEvidenceTask,
    RecoveryPatternLineage,
    RecoveryPatternTask,
    RecoveryPatternVectorCandidate,
)
from graph_swarm.memory.recovery_embeddings import (
    validate_embedding_vector,
)
from graph_swarm.retrieval.candidates import (
    HistoricalActionContext,
    HistoricalRecoveryCandidate,
)


class EntityNotFoundError(LookupError):
    """Raised when a required Neo4j entity or lineage endpoint is absent."""


class _GraphDatabaseApi(Protocol):
    """Typed view of the synchronous public driver factory boundary."""

    @staticmethod
    def driver(uri: str, *, auth: tuple[str, str]) -> Driver: ...


class _DriverApi(Protocol):
    """Typed view of the synchronous public driver operations used here."""

    def verify_connectivity(self) -> None: ...

    def execute_query(
        self,
        query_: LiteralString | Query,
        *,
        parameters_: dict[str, object],
        database_: str,
    ) -> EagerResult: ...


def _isoformat(value: datetime) -> str:
    return value.isoformat()


def _json_object(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _properties(record: Record, key: str) -> Mapping[str, object]:
    value = record[key]
    if value is None:
        raise ValueError(f"Neo4j record field {key!r} is null")
    return cast(Mapping[str, object], value)


def _optional_properties(record: Record, key: str) -> Mapping[str, object] | None:
    """Read an optional related node without weakening required-node checks."""
    try:
        value = record[key]
    except (KeyError, IndexError):
        return None
    if value is None:
        return None
    return cast(Mapping[str, object], value)


def _required_text(properties: Mapping[str, object], key: str) -> str:
    return cast(str, properties[key])


def _required_int(properties: Mapping[str, object], key: str) -> int:
    return cast(int, properties[key])


def _datetime(properties: Mapping[str, object], key: str) -> datetime:
    return datetime.fromisoformat(_required_text(properties, key))


def _json_dict(properties: Mapping[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(_required_text(properties, key)))


def _optional_json_dict(
    properties: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    value = _optional_value(properties, key)
    if value is None:
        return {}
    return cast(dict[str, object], json.loads(cast(str, value)))


def _optional_numeric_list(
    properties: Mapping[str, object],
    key: str,
) -> list[float] | None:
    value = _optional_value(properties, key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Neo4j property {key!r} must contain a numeric list")
    values = cast(list[object], value)
    numeric_values: list[int | float] = []
    for item in values:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"Neo4j property {key!r} must contain a numeric list")
        numeric_values.append(item)
    return [float(item) for item in numeric_values]


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
        problem_statement=_required_text(properties, "problem_statement"),
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


def _read_recovery_pattern(properties: Mapping[str, object]) -> RecoveryPattern:
    return RecoveryPattern(
        id=_required_text(properties, "id"),
        title=_required_text(properties, "title"),
        guidance=_required_text(properties, "guidance"),
        source_failure_id=_required_text(properties, "source_failure_id"),
        source_resolution_id=_required_text(properties, "source_resolution_id"),
        source_outcome_id=_required_text(properties, "source_outcome_id"),
        source_task_id=_required_text(properties, "source_task_id"),
        source_chronological_index=_required_int(
            properties,
            "source_chronological_index",
        ),
        source_tool=_required_text(properties, "source_tool"),
        source_operation=_required_text(properties, "source_operation"),
        applicability_tool=cast(
            str | None,
            _optional_value(properties, "applicability_tool"),
        ),
        applicability_operation=cast(
            str | None,
            _optional_value(properties, "applicability_operation"),
        ),
        source_failure_type=_required_text(properties, "source_failure_type"),
        environment_constraints=EnvironmentConstraints(
            runtime=cast(str | None, _optional_value(properties, "environment_runtime")),
            versions=cast(
                dict[str, str],
                _optional_json_dict(properties, "environment_versions_json"),
            ),
            dependencies=cast(
                dict[str, str],
                _optional_json_dict(properties, "environment_dependencies_json"),
            ),
            markers=cast(
                dict[str, str],
                _optional_json_dict(properties, "environment_markers_json"),
            ),
        ),
        verification_status=RecoveryPatternStatus(
            _required_text(properties, "verification_status")
        ),
        evidence_count=_required_int(properties, "evidence_count"),
        evidence_summary=_required_text(properties, "evidence_summary"),
        embedding=_optional_numeric_list(properties, "embedding"),
        created_at=_datetime(properties, "created_at"),
        invalidated_at=_optional_datetime(properties, "invalidated_at"),
    )


def _attach_historical_trigger(
    pattern: RecoveryPattern,
    failure: Mapping[str, object] | None,
    source_task: Mapping[str, object] | None,
    failed_action: Mapping[str, object] | None,
) -> RecoveryPattern:
    """Enrich old immutable nodes from their existing provenance edges."""
    if pattern.trigger is not None or failure is None or source_task is None:
        return pattern
    if failed_action is None:
        # The trigger action is required for the narrow intent gate.  A legacy
        # vector row without its provenance remains safely ineligible on the
        # corrected path rather than being treated as a generic command.
        return pattern
    return pattern.model_copy(
        update={
            "trigger": RecoveryTrigger(
                failure_type=_required_text(failure, "failure_type"),
                failure_signature=_required_text(failure, "signature"),
                failure_context=_required_text(failure, "symptom"),
                source_task_problem_statement=_required_text(source_task, "problem_statement"),
                source_tool=_required_text(failed_action, "tool"),
                source_operation=_required_text(failed_action, "operation"),
                source_action_arguments=(
                    _json_dict(failed_action, "arguments_json")
                    if "arguments_json" in failed_action
                    else cast(dict[str, object], failed_action.get("arguments", {}))
                ),
                version_sensitive=True,
            )
        }
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
            problem_statement=task.problem_statement,
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

    def save_recovery_pattern(self, pattern: RecoveryPattern) -> None:
        result = self.execute_query(
            queries.SAVE_RECOVERY_PATTERN,
            id=pattern.id,
            title=pattern.title,
            guidance=pattern.guidance,
            source_failure_id=pattern.source_failure_id,
            source_resolution_id=pattern.source_resolution_id,
            source_outcome_id=pattern.source_outcome_id,
            source_task_id=pattern.source_task_id,
            source_chronological_index=pattern.source_chronological_index,
            source_tool=pattern.source_tool,
            source_operation=pattern.source_operation,
            applicability_tool=pattern.applicability_tool,
            applicability_operation=pattern.applicability_operation,
            source_failure_type=pattern.source_failure_type,
            environment_runtime=pattern.environment_constraints.runtime,
            environment_versions_json=_json_object(pattern.environment_constraints.versions),
            environment_dependencies_json=_json_object(
                pattern.environment_constraints.dependencies
            ),
            environment_markers_json=_json_object(pattern.environment_constraints.markers),
            verification_status=pattern.verification_status.value,
            evidence_count=pattern.evidence_count,
            evidence_summary=pattern.evidence_summary,
            embedding=pattern.embedding,
            created_at=_isoformat(pattern.created_at),
            invalidated_at=(
                _isoformat(pattern.invalidated_at) if pattern.invalidated_at is not None else None
            ),
        )
        if not result.records:
            raise EntityNotFoundError(
                "Cannot save RecoveryPattern: required historical provenance was not found"
            )
        if result.records[0].get("provenance_matches") is False:
            raise ValueError(f"RecoveryPattern {pattern.id!r} has conflicting immutable provenance")

    def update_recovery_pattern_embedding(
        self,
        pattern_id: str,
        embedding: list[float],
    ) -> None:
        """Update only the native embedding of an existing RecoveryPattern."""
        validate_relationship_ids(pattern_id=pattern_id)
        validated_embedding = validate_embedding_vector(embedding)
        result = self.execute_query(
            queries.UPDATE_RECOVERY_PATTERN_EMBEDDING,
            pattern_id=pattern_id,
            embedding=validated_embedding,
        )
        if not result.records:
            raise EntityNotFoundError(
                f"RecoveryPattern {pattern_id!r} was not found for embedding update"
            )

    def ensure_recovery_pattern_vector_index(self) -> None:
        """Create the native RecoveryPattern vector index idempotently."""
        self.execute_query(queries.CREATE_RECOVERY_PATTERN_VECTOR_INDEX)

    def query_recovery_pattern_vectors(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> tuple[RecoveryPatternVectorCandidate, ...]:
        """Return native vector-index results without retrieval-policy decisions."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        validated_embedding = validate_embedding_vector(query_embedding)
        result = self.execute_query(
            queries.QUERY_RECOVERY_PATTERN_VECTORS,
            query_embedding=validated_embedding,
            limit=limit,
        )
        return tuple(
            RecoveryPatternVectorCandidate(
                pattern=_attach_historical_trigger(
                    _read_recovery_pattern(_properties(record, "pattern")),
                    _optional_properties(record, "failure"),
                    _optional_properties(record, "source_task"),
                    _optional_properties(record, "failed_action"),
                ),
                vector_score=record["vector_score"],
            )
            for record in result.records
        )

    def count_recovery_pattern_vectors(self) -> int:
        """Count embedded patterns before policy-free vector search."""
        result = self.execute_query(queries.COUNT_RECOVERY_PATTERN_VECTORS)
        if not result.records:
            raise ValueError("Neo4j did not return an embedded RecoveryPattern count")
        count = result.records[0]["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Neo4j returned an invalid embedded RecoveryPattern count")
        return count

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

    def link_resolution_observed_change(
        self,
        resolution_id: str,
        action_id: str,
    ) -> None:
        self._link(
            queries.LINK_RESOLUTION_OBSERVED_CHANGE,
            "OBSERVED_CHANGE",
            resolution_id=resolution_id,
            action_id=action_id,
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
            raise EntityNotFoundError(f"FailureEpisode {failure_id!r} has no Environment lineage")
        environment = _read_environment(_properties(environment_result.records[0], "environment"))

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
            raise ValueError(f"FailureEpisode {failure_id!r} has ambiguous Task or Run lineage")

        tool_result = self.execute_query(queries.GET_TOOLS, failure_id=failure_id)
        tools_by_name = {
            _required_text(_properties(record, "tool"), "name"): _read_tool(
                _properties(record, "tool")
            )
            for record in tool_result.records
        }

        resolution_result = self.execute_query(
            queries.GET_RESOLUTIONS,
            failure_id=failure_id,
        )
        resolutions_by_id: dict[str, Resolution] = {}
        outcomes_by_id: dict[str, Outcome] = {}
        recovery_actions_by_id: dict[str, ActionLineageRecord] = {}
        for record in resolution_result.records:
            resolution = _read_resolution(_properties(record, "resolution"))
            resolutions_by_id[resolution.id] = resolution
            outcome_values = cast(Sequence[object], record["outcomes"])
            for outcome_value in outcome_values:
                if outcome_value is None:
                    continue
                outcome = _read_outcome(cast(Mapping[str, object], outcome_value))
                outcomes_by_id[outcome.id] = outcome
            observed_change_values = cast(
                Sequence[object],
                record.get("observed_changes", []),
            )
            for observed_change_value in observed_change_values:
                if observed_change_value is None:
                    continue
                observed_change = _read_action(cast(Mapping[str, object], observed_change_value))
                recovery_actions_by_id[observed_change.planned_action.id] = observed_change

        return IncidentLineage(
            run=next(iter(run_by_id.values())),
            task=next(iter(task_by_id.values())),
            actions=tuple(actions_by_id.values()),
            tools=tuple(tools_by_name.values()),
            failure=failure,
            environment=environment,
            resolutions=tuple(resolutions_by_id.values()),
            outcomes=tuple(outcomes_by_id.values()),
            recovery_actions=tuple(recovery_actions_by_id.values()),
        )

    def get_recovery_evidence(self, failure_id: str) -> RecoveryEvidenceLineage:
        validate_relationship_ids(failure_id=failure_id)
        result = self.execute_query(
            queries.GET_RECOVERY_EVIDENCE,
            failure_id=failure_id,
        )
        if not result.records:
            raise EntityNotFoundError(
                f"Recovery evidence for FailureEpisode {failure_id!r} was not found"
            )
        if len(result.records) != 1:
            raise ValueError(f"Recovery evidence for FailureEpisode {failure_id!r} is ambiguous")
        record = result.records[0]
        failure = _read_failure(_properties(record, "failure"))
        resolution = _read_resolution(_properties(record, "resolution"))
        outcome = _read_outcome(_properties(record, "outcome"))
        task_properties = _properties(record, "task")
        task = RecoveryEvidenceTask(
            id=_required_text(task_properties, "id"),
            problem_statement=_required_text(task_properties, "problem_statement"),
            repository=_required_text(task_properties, "repository"),
            chronological_index=_required_int(task_properties, "chronological_index"),
        )
        environment = _read_environment(_properties(record, "environment"))
        failed_action = _read_action(_properties(record, "failed_action"))
        recovery_action = _read_action(_properties(record, "recovery_action"))
        if (
            failure.id != failure_id
            or failure.action_id != failed_action.planned_action.id
            or resolution.failure_id != failure.id
            or failed_action.planned_action.task_id != task.id
            or recovery_action.planned_action.task_id != task.id
            or failed_action.planned_action.run_id != recovery_action.planned_action.run_id
        ):
            raise ValueError(f"Recovery evidence for FailureEpisode {failure_id!r} is inconsistent")
        return RecoveryEvidenceLineage(
            failure=failure,
            resolution=resolution,
            outcome=outcome,
            task=task,
            environment=environment,
            failed_action=failed_action,
            recovery_action=recovery_action,
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

    def get_recovery_pattern(self, pattern_id: str) -> RecoveryPatternLineage:
        validate_relationship_ids(pattern_id=pattern_id)
        result = self.execute_query(
            queries.GET_RECOVERY_PATTERN,
            pattern_id=pattern_id,
        )
        if not result.records:
            raise EntityNotFoundError(
                f"RecoveryPattern {pattern_id!r} or required provenance was not found"
            )
        record = result.records[0]
        pattern = _read_recovery_pattern(_properties(record, "pattern"))
        failure = _read_failure(_properties(record, "failure"))
        resolution = _read_resolution(_properties(record, "resolution"))
        outcome = _read_outcome(_properties(record, "outcome"))
        task_properties = _properties(record, "task")
        task = RecoveryPatternTask(
            id=_required_text(task_properties, "id"),
            problem_statement=_required_text(task_properties, "problem_statement"),
            repository=_required_text(task_properties, "repository"),
            chronological_index=_required_int(task_properties, "chronological_index"),
        )
        environment = _read_environment(_properties(record, "environment"))
        failed_action = _read_action(_properties(record, "failed_action"))
        recovery_action = _read_action(_properties(record, "recovery_action"))
        pattern = _attach_historical_trigger(
            pattern,
            failure.model_dump(mode="json"),
            _properties(record, "task"),
            failed_action.planned_action.model_dump(mode="json"),
        )
        if (
            pattern.source_failure_id != failure.id
            or pattern.source_resolution_id != resolution.id
            or pattern.source_outcome_id != outcome.id
            or pattern.source_task_id != task.id
            or pattern.source_chronological_index != task.chronological_index
            or pattern.source_failure_type != failure.failure_type.value
            or pattern.source_tool != failed_action.planned_action.tool
            or pattern.source_operation != failed_action.planned_action.operation
            or resolution.failure_id != failure.id
            or recovery_action.planned_action.task_id != task.id
        ):
            raise ValueError(
                f"RecoveryPattern {pattern_id!r} has inconsistent historical provenance"
            )
        return RecoveryPatternLineage(
            pattern=pattern,
            failure=failure,
            resolution=resolution,
            outcome=outcome,
            task=task,
            environment=environment,
            failed_action=failed_action,
            recovery_action=recovery_action,
        )

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
