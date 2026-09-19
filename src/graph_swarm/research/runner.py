"""Track B experiment configuration, task loading, and B0 execution.

This module is deliberately limited to the no-persistent-memory control.  It
does not construct an advisory service or read the operational memory graph.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_ai import Agent, AgentCapability, AgentRunResult, ModelSettings
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.models import Model
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from graph_swarm.agent.advisory import prepare_task_start_guidance
from graph_swarm.agent.coding_agent import (
    MODEL_REQUEST_TIMEOUT_SECONDS,
    create_coding_agent,
    run_coding_agent,
    timeout_provenance,
)
from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.agent.pacing import ProviderRequestPacing
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.failures import FailureType
from graph_swarm.domain.tasks import Task
from graph_swarm.research.artifacts import JsonlResearchArtifactWriter
from graph_swarm.research.contracts import (
    BoundedTermination,
    ExperimentCondition,
    ExperimentRunArtifact,
    ProviderRequestPacingContract,
    TimeoutContract,
)
from graph_swarm.settings import Settings, get_settings


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


class RecurrenceEvaluationRequired(ExperimentConfigurationError):
    """Raised when the frozen artifact cannot represent an unevaluated recurrence."""


class ObjectiveEvaluationRequired(ExperimentConfigurationError):
    """Raised when no objective benchmark/SWE-smith evaluator was supplied."""


class WorkspaceIsolationError(ExperimentConfigurationError):
    """Raised when a clean benchmark baseline cannot be materialized."""


class OracleEvidenceRequired(ExperimentConfigurationError):
    """Raised when O1 lacks validated frozen transfer evidence."""


class ExperimentLimits(BaseModel):
    """Resource limits shared by comparable Track B conditions."""

    max_actions: int = Field(gt=0)
    max_requests: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(gt=0)


class ModelConfiguration(BaseModel):
    """The frozen provider/model settings recorded with every run."""

    provider: str
    model: str
    settings: dict[str, object] = Field(default_factory=dict)
    prompt_version: str

    @field_validator("provider", "model", "prompt_version")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model configuration text must be non-empty")
        return value


class ExperimentConfiguration(BaseModel):
    """Validated YAML configuration for one experiment rollout."""

    model_config = ConfigDict(populate_by_name=True)

    experiment_id: str
    rollout: str
    model_config_path: str = Field(alias="model_config")
    conditions: tuple[ExperimentCondition, ...]
    limits: ExperimentLimits
    task_manifest: str = "benchmark/manifests/pilot.jsonl"
    task_problems: str = "benchmark/annotations/recurrence_validation.csv"
    artifact_root: str = "research/evidence/results"
    workspace_baseline_root: str = "benchmark/workspaces"
    workspace_execution_root: str = "research/evidence/workspaces"
    config_version: str = "v1"
    development: bool = False
    pilot: bool = False

    @field_validator("experiment_id", "rollout", "model_config_path", "config_version")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("experiment configuration text must be non-empty")
        return value

    @model_validator(mode="after")
    def require_conditions(self) -> ExperimentConfiguration:
        if not self.conditions:
            raise ValueError("at least one experiment condition is required")
        return self


@dataclass(frozen=True)
class LoadedExperimentConfiguration:
    """Configuration plus resolved paths and the referenced model settings."""

    config: ExperimentConfiguration
    model: ModelConfiguration
    config_path: Path
    project_root: Path

    @property
    def task_manifest_path(self) -> Path:
        return resolve_config_path(self.config.task_manifest, self.project_root)

    @property
    def task_problems_path(self) -> Path:
        return resolve_config_path(self.config.task_problems, self.project_root)

    @property
    def artifact_root_path(self) -> Path:
        return resolve_config_path(self.config.artifact_root, self.project_root)

    @property
    def workspace_baseline_root_path(self) -> Path:
        return resolve_config_path(self.config.workspace_baseline_root, self.project_root)

    @property
    def workspace_execution_root_path(self) -> Path:
        return resolve_config_path(self.config.workspace_execution_root, self.project_root)


def resolve_config_path(value: str | Path, project_root: Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_experiment_configuration(
    config_path: Path,
    *,
    project_root: Path | None = None,
) -> LoadedExperimentConfiguration:
    """Load an experiment YAML file and its referenced model YAML file."""
    resolved_config_path = config_path.expanduser().resolve()
    root = (project_root or Path.cwd()).expanduser().resolve()
    try:
        with resolved_config_path.open(encoding="utf-8") as handle:
            config = ExperimentConfiguration.model_validate(yaml.safe_load(handle))
    except (OSError, TypeError, ValueError) as error:
        raise ExperimentConfigurationError(
            f"could not load experiment configuration {resolved_config_path}: {error}"
        ) from error

    model_path = resolve_config_path(config.model_config_path, root)
    try:
        with model_path.open(encoding="utf-8") as handle:
            model = ModelConfiguration.model_validate(yaml.safe_load(handle))
    except (OSError, TypeError, ValueError) as error:
        raise ExperimentConfigurationError(
            f"could not load model configuration {model_path}: {error}"
        ) from error

    return LoadedExperimentConfiguration(config, model, resolved_config_path, root)


@dataclass(frozen=True)
class BenchmarkTaskCase:
    """Task plus benchmark-only recurrence evidence from the manifest."""

    task: Task
    occurrence_index: int


def load_task_cases(
    manifest_path: Path,
    *,
    problem_statements_path: Path | None = None,
) -> list[BenchmarkTaskCase]:
    """Load manifest tasks in chronological order without exposing annotations.

    The manifest is intentionally metadata-only.  The annotation source is
    used solely as a keyed source of problem statements; recovery labels,
    patches, and all other columns are discarded before a :class:`Task` is
    constructed.
    """
    manifest_records = list(_read_jsonl(manifest_path))
    statements = (
        _read_problem_statements(problem_statements_path) if problem_statements_path else {}
    )
    task_cases: list[BenchmarkTaskCase] = []
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()

    for record in manifest_records:
        task_id = _required_record_text(record, "task_id")
        review_id = _required_record_text(record, "review_id")
        problem_statement = record.get("problem_statement")
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            problem_statement = statements.get(review_id)
        if problem_statement is None:
            raise ExperimentConfigurationError(
                f"no problem_statement found for manifest review_id {review_id}"
            )
        chronological_index = _required_record_int(record, "chronological_index")
        occurrence_index = _required_record_int(record, "occurrence_index")
        if task_id in seen_ids or chronological_index in seen_indexes:
            raise ExperimentConfigurationError(
                "manifest task IDs and chronological indexes must be unique"
            )
        seen_ids.add(task_id)
        seen_indexes.add(chronological_index)
        task_cases.append(
            BenchmarkTaskCase(
                task=Task(
                    id=task_id,
                    problem_statement=problem_statement,
                    family_id=_required_record_text(record, "family_id"),
                    repository=_required_record_text(record, "repository"),
                    chronological_index=chronological_index,
                ),
                occurrence_index=occurrence_index,
            )
        )

    return sorted(task_cases, key=lambda case: case.task.chronological_index)


def load_tasks(
    manifest_path: Path,
    *,
    problem_statements_path: Path | None = None,
) -> list[Task]:
    """Load only the Task domain objects, retaining no benchmark annotations."""
    return [
        case.task
        for case in load_task_cases(
            manifest_path,
            problem_statements_path=problem_statements_path,
        )
    ]


def build_task_prompt(task: Task, *, task_start_guidance: str | None = None) -> str:
    """Build the task input, keeping optional guidance outside the problem statement."""
    prompt = f"Task problem statement:\n\n{task.problem_statement}"
    if task_start_guidance:
        prompt += (
            "\n\n--- Recovery guidance intervention (task_start) ---\n\n"
            f"{task_start_guidance}\n"
            "--- End recovery guidance intervention ---"
        )
    return prompt


class BaselineWorkspaceManager:
    """Materialize a fresh task workspace from an immutable benchmark baseline."""

    def __init__(
        self,
        baseline_root: Path,
        execution_root: Path,
        *,
        experiment_id: str = "GS-E003",
        condition: ExperimentCondition = ExperimentCondition.B0,
    ) -> None:
        self.baseline_root = baseline_root.expanduser().resolve()
        self.execution_root = execution_root.expanduser().resolve()
        self.experiment_id = experiment_id
        self.condition = condition

    def materialize(self, task: Task, run_id: str) -> Path:
        baseline = (self.baseline_root / task.repository).resolve()
        if not baseline.is_dir():
            raise WorkspaceIsolationError(
                f"benchmark baseline workspace does not exist: {baseline}"
            )
        destination = (
            self.execution_root
            / self.experiment_id
            / self.condition.value
            / task.id
            / run_id
            / "workspace"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(baseline, destination, dirs_exist_ok=False)
        except FileExistsError as error:
            raise WorkspaceIsolationError(
                f"refusing to reuse an existing task workspace: {destination}"
            ) from error
        except OSError as error:
            raise WorkspaceIsolationError(
                f"could not materialize benchmark baseline {baseline}: {error}"
            ) from error
        return destination


@dataclass(frozen=True)
class BenchmarkEvaluationInput:
    """Pydantic Evals case input; only the prompt is model-visible."""

    task: Task
    prompt: str


@dataclass(frozen=True)
class BenchmarkEvaluationOutput:
    """Output passed to the objective benchmark evaluator after agent execution."""

    task: Task
    workspace: Path
    events: tuple[AgentEvent, ...]
    agent_result: AgentRunResult[str] | None
    error: Exception | None


ObjectiveTaskEvaluator = Callable[[Task, Path], bool]
RecurrenceEvaluator = Callable[
    [
        BenchmarkTaskCase,
        Sequence[AgentEvent],
        AgentRunResult[str] | None,
        Path,
    ],
    bool | None,
]
LegacyTestEvaluator = Callable[[Task, Sequence[AgentEvent], AgentRunResult[str] | None], bool]


class OracleAdviceResolver(Protocol):
    """O1-only adapter for validated, pre-action recovery guidance."""

    def validate_case(self, case: BenchmarkTaskCase) -> None:
        """Validate frozen Oracle evidence before model execution."""
        ...

    def evaluate_action(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
    ) -> AdviceResult:
        """Return Oracle advice for a planned action, without changing it."""
        ...

    def evaluate_task_start(
        self,
        task: Task,
        environment: EnvironmentContext,
        planned_action: PlannedAction,
    ) -> AdviceResult:
        """Return one guidance item for the pre-first-request boundary."""
        ...

    def render_advice(self, advice: AdviceResult) -> str:
        """Render only recovery guidance for the agent-visible retry."""
        ...


@dataclass
class ObjectiveBenchmarkEvaluator(
    Evaluator[BenchmarkEvaluationInput, BenchmarkEvaluationOutput, None]
):
    """Pydantic Evals adapter for the objective SWE-smith benchmark boundary."""

    objective: ObjectiveTaskEvaluator

    def evaluate(
        self,
        ctx: EvaluatorContext[
            BenchmarkEvaluationInput,
            BenchmarkEvaluationOutput,
            None,
        ],
    ) -> bool:
        return self.objective(ctx.inputs.task, ctx.output.workspace)


class BenchmarkEvaluation:
    """Dataset/Case/Evaluator wrapper used by the sequential runner."""

    def __init__(self, name: str, objective: ObjectiveTaskEvaluator) -> None:
        self.evaluator = ObjectiveBenchmarkEvaluator(objective)
        self.dataset = Dataset[BenchmarkEvaluationInput, BenchmarkEvaluationOutput, None](
            name=name,
            cases=[],
            evaluators=[self.evaluator],
        )

    def evaluate(
        self,
        task: Task,
        workspace: Path,
        events: Sequence[AgentEvent],
        agent_result: AgentRunResult[str] | None,
        error: Exception | None,
        duration_seconds: float,
    ) -> bool:
        case = Case[BenchmarkEvaluationInput, BenchmarkEvaluationOutput, None](
            name=task.id,
            inputs=BenchmarkEvaluationInput(task=task, prompt=build_task_prompt(task)),
        )
        output = BenchmarkEvaluationOutput(
            task=task,
            workspace=workspace,
            events=tuple(events),
            agent_result=agent_result,
            error=error,
        )
        self.dataset = Dataset[BenchmarkEvaluationInput, BenchmarkEvaluationOutput, None](
            name=self.dataset.name,
            cases=[case],
            evaluators=[self.evaluator],
        )
        report = self.dataset.evaluate_sync(
            lambda _inputs: output,
            max_concurrency=1,
            progress=False,
        )
        case_report = report.cases[0]
        evaluation = next(iter(case_report.assertions.values()))
        return bool(evaluation.value)


class ExperimentRunArtifactStore:
    """Append-only store for canonical per-run artifacts and raw evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def run_directory(
        self,
        experiment_id: str,
        condition: ExperimentCondition,
        task_id: str,
        run_id: str,
    ) -> Path:
        return self.root / experiment_id / condition.value / task_id / run_id

    def prepare_run_directory(
        self,
        experiment_id: str,
        condition: ExperimentCondition,
        task_id: str,
        run_id: str,
    ) -> Path:
        """Create and return the unique directory shared by run evidence."""
        run_dir = self.run_directory(experiment_id, condition, task_id, run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def write_artifact(self, artifact: ExperimentRunArtifact) -> Path:
        run_dir = self.run_directory(
            artifact.experiment_id,
            artifact.condition,
            artifact.task_id,
            artifact.run_id,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "artifact.json"
        serialized = json.dumps(
            artifact.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized + "\n")
        return path

    def write_raw_evidence(
        self,
        *,
        artifact: ExperimentRunArtifact,
        dependencies: AgentDependencies,
        result: AgentRunResult[str] | None,
        error: Exception | None,
        step_database_path: Path,
    ) -> Path:
        run_dir = self.run_directory(
            artifact.experiment_id,
            artifact.condition,
            artifact.task_id,
            artifact.run_id,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        evidence = {
            "experiment_id": artifact.experiment_id,
            "gate_id": artifact.gate_id,
            "execution_manifest_id": artifact.execution_manifest_id,
            "run_id": artifact.run_id,
            "task_id": artifact.task_id,
            "error": None if error is None else type(error).__name__,
            "error_message": None if error is None else str(error),
            "step_database": str(step_database_path),
            "events": [event.model_dump(mode="json") for event in dependencies.events],
            "messages": [] if result is None else json.loads(result.all_messages_json()),
            "output": None if result is None else result.output,
            "conversation_id": None if result is None else result.conversation_id,
            "agent_run_id": None if result is None else result.run_id,
            "usage": None if result is None else dataclasses.asdict(result.usage),
            "timeout_contract": artifact.timeout_contract.model_dump(mode="json"),
            "timeout_provenance": timeout_provenance(error),
            "provider_request_pacing": artifact.provider_request_pacing.model_dump(mode="json"),
            "provider_pacing_wait_seconds": artifact.provider_pacing_wait_seconds,
            "advice_count": artifact.advice_count,
            "advice_review_id": artifact.advice_review_id,
            "advice_intervention_boundary": artifact.advice_intervention_boundary,
            "advice_delivery_timing": artifact.advice_delivery_timing,
            "termination": (
                None
                if artifact.termination is None
                else artifact.termination.model_dump(mode="json")
            ),
            "budget_exhausted": artifact.termination is not None,
            "objective_result": artifact.task_success,
        }
        if artifact.condition is ExperimentCondition.O1:
            evidence.update(
                {
                    "condition": artifact.condition.value,
                    "advice_source": "O1",
                    "advice_events": [
                        event.model_dump(mode="json") for event in dependencies.advice_events
                    ],
                    "behavior_evidence": [
                        item.model_dump(mode="json") for item in dependencies.behavior_evidence
                    ],
                }
            )
        path = run_dir / "raw_evidence.json"
        serialized = json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=str)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized + "\n")
        return path


AgentFactory = Callable[
    [Settings, Sequence[AgentCapability[AgentDependencies]]],
    Agent[AgentDependencies, str],
]
WorkspaceResolver = Callable[[Task], Path]
PythonExecutableResolver = Callable[[Task, Path], Path]
ExecutionRuntimeResolver = Callable[[Task, Path], ExecutionRuntime]


@dataclass(frozen=True)
class ExperimentExecution:
    """Result of one task/run, including its canonical artifact location."""

    artifact: ExperimentRunArtifact
    artifact_path: Path
    raw_evidence_path: Path
    dependencies: AgentDependencies
    agent_result: AgentRunResult[str] | None
    error: Exception | None
    step_database_path: Path
    actual_environment_fingerprint: str | None = None


class ExperimentRunner:
    """Sequential B0/O1 runner with isolated PydanticAI executions."""

    def __init__(
        self,
        configuration: LoadedExperimentConfiguration,
        *,
        settings: Settings | None = None,
        agent_factory: AgentFactory | None = None,
        model: Model | None = None,
        workspace_resolver: WorkspaceResolver | None = None,
        python_executable_resolver: PythonExecutableResolver | None = None,
        execution_runtime_resolver: ExecutionRuntimeResolver | None = None,
        objective_evaluator: ObjectiveTaskEvaluator | None = None,
        evaluator: LegacyTestEvaluator | None = None,
        recurrence_evaluator: RecurrenceEvaluator | None = None,
        oracle_resolver: OracleAdviceResolver | None = None,
        condition: ExperimentCondition | None = None,
        artifact_store: ExperimentRunArtifactStore | None = None,
        request_pacing: ProviderRequestPacing | None = None,
        gate_id: str | None = None,
        execution_manifest_id: str | None = None,
    ) -> None:
        if not configuration.config.development or not configuration.config.pilot:
            raise ExperimentConfigurationError(
                "GS-E003 must be explicitly marked development/pilot"
            )
        if configuration.model.provider != "openrouter":
            raise ExperimentConfigurationError(
                "unsupported Track B model provider: "
                f"{configuration.model.provider}; the supported provider is openrouter"
            )
        selected_condition = condition or _configured_condition(configuration)
        if selected_condition not in configuration.config.conditions:
            raise ExperimentConfigurationError(
                "condition "
                f"{selected_condition.value} is not enabled in the experiment configuration"
            )
        if selected_condition is ExperimentCondition.T:
            raise ExperimentConfigurationError("Track B Sprint 2A does not implement T")
        if selected_condition not in (ExperimentCondition.B0, ExperimentCondition.O1):
            raise ExperimentConfigurationError(
                f"unsupported Track B condition: {selected_condition.value}"
            )
        if selected_condition is ExperimentCondition.O1 and oracle_resolver is None:
            raise OracleEvidenceRequired("O1 requires a validated Oracle resolver")
        if selected_condition is ExperimentCondition.O1 and not callable(
            getattr(oracle_resolver, "render_advice", None)
        ):
            raise OracleEvidenceRequired(
                "O1 requires an Oracle resolver with a guidance-only renderer"
            )
        self.configuration = configuration
        self.condition = selected_condition
        self.settings = settings
        self.agent_factory = agent_factory
        self.model = model
        self.workspace_resolver = workspace_resolver
        self.python_executable_resolver = python_executable_resolver
        self.execution_runtime_resolver = execution_runtime_resolver
        if objective_evaluator is None and evaluator is not None:
            legacy_evaluator = evaluator

            def legacy_objective(task: Task, _workspace: Path) -> bool:
                return legacy_evaluator(task, (), None)

            objective_evaluator = legacy_objective
        self.objective_evaluator = objective_evaluator
        self.recurrence_evaluator = recurrence_evaluator or benchmark_recurrence_determination
        self.oracle_resolver = oracle_resolver
        self.artifact_store = artifact_store or ExperimentRunArtifactStore(
            configuration.artifact_root_path
        )
        self.request_pacing = request_pacing or ProviderRequestPacing()
        self.gate_id = gate_id
        self.execution_manifest_id = execution_manifest_id

    def run_all(self, tasks: Sequence[Task] | None = None) -> list[ExperimentExecution]:
        """Run each selected task once, strictly in chronological order."""
        selected_cases = (
            [BenchmarkTaskCase(task=task, occurrence_index=1) for task in tasks]
            if tasks is not None
            else load_task_cases(
                self.configuration.task_manifest_path,
                problem_statements_path=self.configuration.task_problems_path,
            )
        )
        ordered_cases = sorted(
            selected_cases,
            key=lambda case: case.task.chronological_index,
        )
        return [self.run_case(case) for case in ordered_cases]

    def run_task(self, task: Task) -> ExperimentExecution:
        """Run one configured Track B task using explicit test determinations."""
        return self.run_case(BenchmarkTaskCase(task=task, occurrence_index=1))

    def run_case(self, case: BenchmarkTaskCase) -> ExperimentExecution:
        """Run one configured B0/O1 case with benchmark evaluation."""
        task = case.task
        condition = self.condition
        run_id = self._new_run_id(condition, task)
        if case.occurrence_index < 1:
            raise RecurrenceEvaluationRequired(
                f"no benchmark recurrence determination for task {task.id}"
            )
        if self.objective_evaluator is None:
            raise ObjectiveEvaluationRequired(
                "a benchmark/SWE-smith objective evaluator is required"
            )
        if condition is ExperimentCondition.O1 and self.oracle_resolver is not None:
            self.oracle_resolver.validate_case(case)
        run_dir = self.artifact_store.prepare_run_directory(
            self.configuration.config.experiment_id,
            condition,
            task.id,
            run_id,
        )
        step_database_path = run_dir / "steps.sqlite"
        workspace = self._workspace_for(task, run_id, condition).expanduser().resolve()
        execution_runtime = (
            self.execution_runtime_resolver(task, workspace)
            if self.execution_runtime_resolver is not None
            else None
        )
        python_executable = None
        if execution_runtime is None and self.python_executable_resolver is not None:
            python_executable = (
                self.python_executable_resolver(task, workspace).expanduser().resolve()
            )
        local_executable = (
            execution_runtime.python_executable
            if execution_runtime is not None and execution_runtime.runtime_type == "local"
            else python_executable
        )
        if local_executable is not None and not local_executable.is_file():
            raise WorkspaceIsolationError(
                f"benchmark task {task.id} has no executable isolated environment: "
                f"{local_executable}"
            )
        environment = (
            _environment_for(task, run_id) if condition is ExperimentCondition.O1 else None
        )
        dependencies = AgentDependencies(
            workspace,
            run_id,
            task.id,
            task=task,
            advisory_service=(
                cast(Any, self.oracle_resolver) if condition is ExperimentCondition.O1 else None
            ),
            environment=environment,
            python_executable=python_executable,
            execution_runtime=execution_runtime,
            artifact_writer=(
                JsonlResearchArtifactWriter(run_dir / "advisory_evidence.jsonl")
                if condition is ExperimentCondition.O1
                else None
            ),
        )
        started = datetime.now(UTC)
        started_counter = time.perf_counter()
        result: AgentRunResult[str] | None = None
        error: Exception | None = None
        task_start_guidance: str | None = None
        run_settings: Settings | None = None
        try:
            settings = self._settings_for_model()
            run_settings = settings
            step_persistence = StepPersistence(
                store=SqliteStepStore(database=step_database_path),
                agent_name="graph_swarm_coding_agent",
                run_id=run_id,
                metadata={
                    "experiment_id": self.configuration.config.experiment_id,
                    "condition": condition.value,
                    "task_id": task.id,
                },
            )
            agent = self._build_agent(settings, step_persistence)
            task_start_guidance = prepare_task_start_guidance(dependencies)
            result = run_coding_agent(
                agent,
                settings,
                dependencies,
                build_task_prompt(task, task_start_guidance=task_start_guidance),
                max_actions=self.configuration.config.limits.max_actions,
                max_requests=self.configuration.config.limits.max_requests,
                timeout_seconds=self.configuration.config.limits.timeout_seconds,
                model_settings=cast(ModelSettings, self.configuration.model.settings),
                request_pacing=self.request_pacing,
            )
        except Exception as caught:  # preserve bounded-run failures in evidence
            error = caught

        task_success = BenchmarkEvaluation(
            self.configuration.config.experiment_id,
            self.objective_evaluator,
        ).evaluate(
            task,
            workspace,
            dependencies.events,
            result,
            error if isinstance(error, Exception) else None,
            (time.perf_counter() - started_counter),
        )
        known_failure_repeated = self.recurrence_evaluator(
            case,
            tuple(dependencies.events),
            result,
            workspace,
        )
        if known_failure_repeated is None:
            raise RecurrenceEvaluationRequired(
                f"no post-execution recurrence determination for task {task.id}"
            )
        artifact = self._make_artifact(
            task=task,
            condition=condition,
            run_id=run_id,
            started=started,
            elapsed_ms=(time.perf_counter() - started_counter) * 1000,
            dependencies=dependencies,
            settings=run_settings,
            result=result,
            error=error,
            task_success=task_success,
            known_failure_repeated=known_failure_repeated,
            advice_received=_advice_received(dependencies),
            task_start_guidance=task_start_guidance,
            termination=_budget_termination(error, self.configuration.config.limits),
        )
        raw_path = self.artifact_store.write_raw_evidence(
            artifact=artifact,
            dependencies=dependencies,
            result=result,
            error=error,
            step_database_path=step_database_path,
        )
        artifact_path = self.artifact_store.write_artifact(artifact)
        return ExperimentExecution(
            artifact,
            artifact_path,
            raw_path,
            dependencies,
            result,
            error,
            step_database_path,
        )

    def _settings_for_model(self) -> Settings:
        base = self.settings or get_settings()
        if self.configuration.model.provider != "openrouter":
            raise ExperimentConfigurationError(
                "unsupported Track B model provider: "
                f"{self.configuration.model.provider}; the supported provider is openrouter"
            )
        import os

        return base.model_copy(
            update={
                "model_provider": "openrouter",
                "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY")
                or base.openrouter_api_key,
                "openrouter_model": os.environ.get("OPENROUTER_MODEL")
                or base.openrouter_model,
            }
        )

    def _workspace_for(
        self,
        task: Task,
        run_id: str,
        condition: ExperimentCondition,
    ) -> Path:
        if self.workspace_resolver is not None:
            return self.workspace_resolver(task)
        manager = BaselineWorkspaceManager(
            self.configuration.workspace_baseline_root_path,
            self.configuration.workspace_execution_root_path,
            experiment_id=self.configuration.config.experiment_id,
            condition=condition,
        )
        return manager.materialize(task, run_id)

    def _build_agent(
        self,
        settings: Settings,
        step_persistence: StepPersistence,
    ) -> Agent[AgentDependencies, str]:
        if self.agent_factory is not None:
            agent = self.agent_factory(settings, [step_persistence])
            if not any(
                capability is step_persistence for capability in agent.root_capability.capabilities
            ):
                raise ExperimentConfigurationError(
                    "custom agent factory did not attach the mandatory StepPersistence capability"
                )
            return agent
        return create_coding_agent(
            settings,
            model=self.model,
            capabilities=[step_persistence],
        )

    def _new_run_id(self, condition: ExperimentCondition, task: Task) -> str:
        return (
            f"{self.configuration.config.experiment_id}-{condition.value}-{task.id}-{uuid4().hex}"
        )

    def _make_artifact(
        self,
        *,
        task: Task,
        condition: ExperimentCondition,
        run_id: str,
        started: datetime,
        elapsed_ms: float,
        dependencies: AgentDependencies,
        settings: Settings | None,
        result: AgentRunResult[str] | None,
        error: Exception | None,
        task_success: bool,
        known_failure_repeated: bool,
        advice_received: AdviceResult | None,
        task_start_guidance: str | None,
        termination: BoundedTermination | None,
    ) -> ExperimentRunArtifact:
        if dependencies.events:
            event = dependencies.events[-1]
            executed_action = event.result
            planned_action = _planned_action(event)
            latency_ms = max(0.0, (executed_action.completed_at - started).total_seconds() * 1000)
        else:
            now = datetime.now(UTC)
            action_id = f"{run_id}-agent-run"
            executed_action = ActionResult(
                action_id=action_id,
                tool_name="agent",
                success=task_success,
                output=None if result is None else result.output,
                error=None if error is None else type(error).__name__,
                started_at=started,
                completed_at=now,
            )
            planned_action = PlannedAction(
                id=action_id,
                run_id=run_id,
                task_id=task.id,
                tool="agent",
                operation="task_completion",
                arguments={},
                planned_at=started,
            )
            latency_ms = max(0.0, elapsed_ms)

        usage = None if result is None else result.usage
        retries = (
            0
            if result is None
            else sum(
                isinstance(part, RetryPromptPart)
                for message in result.all_messages()
                for part in message.parts
            )
        )
        failure_type = _failure_type(executed_action)
        return ExperimentRunArtifact(
            experiment_id=self.configuration.config.experiment_id,
            gate_id=self.gate_id,
            execution_manifest_id=self.execution_manifest_id,
            condition=condition,
            run_id=run_id,
            task_id=task.id,
            family_id=task.family_id,
            chronological_index=task.chronological_index,
            model=self._artifact_model_name(),
            model_settings=self.configuration.model.settings,
            prompt_version=self.configuration.model.prompt_version,
            planned_action=planned_action,
            executed_action=executed_action,
            advice_received=advice_received,
            advice_count=len(dependencies.advice_events),
            advice_intervention_boundary=(
                "task_start" if task_start_guidance is not None else None
            ),
            advice_delivery_timing=(
                "pre_first_model_request" if task_start_guidance is not None else None
            ),
            advice_review_id=self._advice_review_id(condition, task),
            advice_accepted=False,
            failure_type=failure_type,
            task_success=task_success,
            known_failure_repeated=known_failure_repeated,
            tool_calls=len(dependencies.events),
            retries=retries,
            input_tokens=None if usage is None else usage.input_tokens,
            output_tokens=None if usage is None else usage.output_tokens,
            latency_ms=latency_ms,
            retrieved_incident_id=None,
            retrieval_score=None,
            timeout_contract=_timeout_contract(
                self.configuration.config.limits.timeout_seconds,
                settings,
            ),
            timeout_provenance=timeout_provenance(error),
            provider_request_pacing=ProviderRequestPacingContract(
                min_interval_seconds=self.request_pacing.config.min_interval_seconds,
                max_nominal_requests_per_minute=(
                    self.request_pacing.config.max_nominal_requests_per_minute
                ),
            ),
            provider_pacing_wait_seconds=self.request_pacing.wait_seconds_for_run(run_id),
            termination=termination,
        )

    def _artifact_model_name(self) -> str:
        if self.configuration.model.provider != "openrouter":
            raise ExperimentConfigurationError(
                "unsupported Track B model provider: "
                f"{self.configuration.model.provider}; the supported provider is openrouter"
            )
        base = self.settings or get_settings()
        return (
            os.environ.get("OPENROUTER_MODEL")
            or base.openrouter_model
            or self.configuration.model.model
        )

    def _advice_review_id(
        self,
        condition: ExperimentCondition,
        task: Task,
    ) -> str | None:
        if condition is not ExperimentCondition.O1 or self.oracle_resolver is None:
            return None
        reviewer = cast(
            Callable[[str], str | None] | None,
            getattr(self.oracle_resolver, "review_id_for", None),
        )
        return None if reviewer is None else reviewer(task.id)


def _planned_action(event: AgentEvent) -> PlannedAction:
    return PlannedAction(
        id=event.action_id,
        run_id=event.run_id,
        task_id=event.task_id,
        tool=event.result.tool_name,
        operation=event.result.tool_name,
        arguments={},
        planned_at=event.result.started_at,
    )


def _failure_type(result: ActionResult) -> FailureType | None:
    if result.success:
        return None
    return {
        "run_tests": FailureType.TEST_FAILURE,
        "run_command": FailureType.COMMAND_FAILURE,
    }.get(result.tool_name, FailureType.TOOL_PARAMETER_ERROR)


def _budget_termination(
    error: Exception | None,
    limits: ExperimentLimits,
) -> BoundedTermination | None:
    """Recognize only the configured action/request budget exceptions."""
    if not isinstance(error, UsageLimitExceeded):
        return None
    message = str(error)
    if f"tool_calls_limit of {limits.max_actions}" in message:
        return BoundedTermination(budget="tool_calls", configured_limit=limits.max_actions)
    if limits.max_requests is not None and f"request_limit of {limits.max_requests}" in message:
        return BoundedTermination(budget="requests", configured_limit=limits.max_requests)
    return None


def _configured_condition(
    configuration: LoadedExperimentConfiguration,
) -> ExperimentCondition:
    conditions = configuration.config.conditions
    if len(conditions) != 1:
        raise ExperimentConfigurationError(
            "select exactly one Track B condition when constructing ExperimentRunner"
        )
    return conditions[0]


def _environment_for(task: Task, run_id: str) -> EnvironmentContext:
    return EnvironmentContext(
        id=f"{run_id}-environment",
        repository=task.repository,
        runtime="python",
    )


def _advice_received(dependencies: AgentDependencies) -> AdviceResult | None:
    if not dependencies.advice_events:
        return None
    return AdviceResult(advice=dependencies.advice_events[0].advice)


def _timeout_contract(
    experiment_timeout_seconds: float,
    settings: Settings | None,
) -> TimeoutContract:
    return TimeoutContract(
        agent_wall_clock_seconds=experiment_timeout_seconds,
        model_request_timeout_seconds=MODEL_REQUEST_TIMEOUT_SECONDS,
        tool_timeout_seconds={
            "run_command": 30 if settings is None else settings.agent_command_timeout_seconds,
            "run_tests": 120 if settings is None else settings.agent_tests_timeout_seconds,
        },
        experiment_timeout_seconds=experiment_timeout_seconds,
    )


def benchmark_recurrence_determination(
    case: BenchmarkTaskCase,
    _events: Sequence[AgentEvent],
    _agent_result: AgentRunResult[str] | None,
    _workspace: Path,
) -> bool | None:
    """Return only determinations defensible from the frozen benchmark data.

    The pilot data identifies recurrence opportunities but does not contain a
    task-specific normalized known-failure signature.  Transfer cases must
    therefore supply an explicit benchmark matcher to compare
    :func:`detect_failure` output with that frozen evidence.
    """
    if case.occurrence_index == 1:
        return False
    if case.occurrence_index < 1:
        return None
    return None


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ExperimentConfigurationError(
                        f"invalid JSON in {path} at line {line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise ExperimentConfigurationError(
                        f"manifest record {line_number} is not an object"
                    )
                yield record
    except OSError as error:
        raise ExperimentConfigurationError(
            f"could not read task manifest {path}: {error}"
        ) from error


def _read_problem_statements(path: Path) -> dict[str, str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "review_id" not in (reader.fieldnames or ()) or "problem_statement" not in (
                reader.fieldnames or ()
            ):
                raise ExperimentConfigurationError(
                    f"problem statement source {path} must contain review_id and problem_statement"
                )
            statements: dict[str, str] = {}
            for row in reader:
                review_id = row.get("review_id")
                problem_statement = row.get("problem_statement")
                if review_id and problem_statement and problem_statement.strip():
                    statements[review_id] = problem_statement
            return statements
    except OSError as error:
        raise ExperimentConfigurationError(
            f"could not read problem statement source {path}: {error}"
        ) from error


def _required_record_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigurationError(f"manifest field {field!r} must be non-empty text")
    return value


def _required_record_int(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentConfigurationError(f"manifest field {field!r} must be an integer")
    return value


__all__ = [
    "AgentFactory",
    "BaselineWorkspaceManager",
    "BenchmarkEvaluation",
    "BenchmarkEvaluationInput",
    "BenchmarkEvaluationOutput",
    "BenchmarkTaskCase",
    "ExperimentConfiguration",
    "ExperimentConfigurationError",
    "ExperimentExecution",
    "ExperimentLimits",
    "ExperimentRunArtifactStore",
    "ExperimentRunner",
    "LoadedExperimentConfiguration",
    "ModelConfiguration",
    "ObjectiveBenchmarkEvaluator",
    "ObjectiveEvaluationRequired",
    "ObjectiveTaskEvaluator",
    "OracleAdviceResolver",
    "OracleEvidenceRequired",
    "RecurrenceEvaluationRequired",
    "RecurrenceEvaluator",
    "WorkspaceResolver",
    "PythonExecutableResolver",
    "WorkspaceIsolationError",
    "build_task_prompt",
    "benchmark_recurrence_determination",
    "load_experiment_configuration",
    "load_task_cases",
    "load_tasks",
]
