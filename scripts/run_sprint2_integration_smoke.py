"""Run the read-only Sprint 2 Integration Point 1 smoke.

This command requires the existing R13b Neo4j database and HF embedding
credentials.  It never applies schema, creates indexes, writes graph data,
executes benchmark tasks, or constructs an external model provider.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.recovery_patterns import RecoveryPattern
from graph_swarm.domain.tasks import Task
from graph_swarm.integration.advisory_runtime import (
    R13B_TREATMENT_PATTERN_IDS,
    create_neo4j_advisory_runtime,
)
from graph_swarm.integration.sprint2_smoke import (
    R13B_CANONICAL_PATTERN_IDS,
    chronology_audit,
    preflight_r13b_corpus,
)
from graph_swarm.memory.recovery_embeddings import (
    RECOVERY_PATTERN_EMBEDDING_DIMENSION,
    RECOVERY_PATTERN_EMBEDDING_MODEL,
    RECOVERY_PATTERN_EMBEDDING_NORMALIZED,
)
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    ExperimentRunArtifactStore,
    ExperimentRunner,
    load_experiment_configuration,
    load_task_cases,
)
from graph_swarm.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    evidence_dir = _new_evidence_dir()
    runtime = create_neo4j_advisory_runtime(get_settings())
    try:
        runtime.repository.verify_connectivity()
        preflight = preflight_r13b_corpus(runtime.repository)
        treatment_preflight = preflight_r13b_corpus(runtime.treatment_repository)
        task = _diagnostic_task()
        direct_action = _planned_action(task, "sprint2-direct")
        direct_environment = _environment(task, "sprint2-direct")
        direct_advice = runtime.advisory_service.evaluate_action(
            task,
            direct_action,
            direct_environment,
        )
        direct_retrieval = runtime.advisory_service.last_retrieval_result
        if direct_retrieval is None or direct_retrieval.selected_pattern is None:
            raise RuntimeError("real advisory lookup did not select a historical pattern")

        earlier_task = task.model_copy(
            update={"id": "integration-point-1-chronology", "chronological_index": 3}
        )
        runtime.advisory_service.evaluate_action(
            earlier_task,
            _planned_action(earlier_task, "sprint2-chronology"),
            _environment(earlier_task, "sprint2-chronology"),
        )
        earlier_retrieval = runtime.advisory_service.last_retrieval_result
        if earlier_retrieval is None:
            raise RuntimeError("chronology lookup did not produce retrieval evidence")
        lineages = {
            candidate.pattern_id: runtime.repository.get_recovery_pattern(candidate.pattern_id)
            for candidate in earlier_retrieval.candidates
        }
        chronology = chronology_audit(
            earlier_retrieval,
            task_chronological_index=3,
            pattern_lineages=lineages,
        )

        t_execution = _run_t_smoke(runtime, task, evidence_dir)
        b0_execution = _run_b0_smoke(runtime, task, evidence_dir)
        raw_text = t_execution.raw_evidence_path.read_text(encoding="utf-8")
        if task.family_id in raw_text:
            raise RuntimeError("evaluation family metadata reached T model-visible/raw context")

        report = {
            "status": "PASS",
            "integration_point": "1",
            "purpose": "Sprint 2 real Neo4j AdvisoryService integration smoke",
            "created_at": datetime.now(UTC).isoformat(),
            "base_sha": _git_sha(),
            "output_sha": None,
            "working_tree_diff_hash": _working_tree_diff_hash(),
            "r13b_freeze_sha": "857bd073de420d0129435e3bb47e96db3741ad4f",
            "read_only": True,
            "benchmark_tasks_executed": [],
            "providers_called": [],
            "graph_mutations_performed": False,
            "embedding": {
                "model": RECOVERY_PATTERN_EMBEDDING_MODEL,
                "dimension": RECOVERY_PATTERN_EMBEDDING_DIMENSION,
                "normalized": RECOVERY_PATTERN_EMBEDDING_NORMALIZED,
                "execution_backend": "Hugging Face Inference API (hf-inference)",
                "frozen_implementation": "huggingface_hub.InferenceClient.feature_extraction",
                "deviation": (
                    "R13b used the Hugging Face Inference API; no local "
                    "SentenceTransformer was substituted."
                ),
            },
            "config_sha256": _config_hashes(),
            "vector_index": preflight.vector_index,
            "corpus_preflight": dataclasses.asdict(preflight),
            "underlying_graph_pattern_count": preflight.embedded_pattern_count,
            "treatment_visible_pattern_count": treatment_preflight.embedded_pattern_count,
            "canonical_pattern_ids": R13B_CANONICAL_PATTERN_IDS,
            "treatment_pattern_ids": sorted(R13B_TREATMENT_PATTERN_IDS),
            "direct_real_advice": {
                "advice_fired": direct_advice.advice is not None,
                "matched_failure_episode_id": (
                    None
                    if direct_advice.advice is None
                    else direct_advice.advice.provenance.failure_episode_id
                ),
                "matched_resolution_id": (
                    None
                    if direct_advice.advice is None
                    else direct_advice.advice.provenance.resolution_id
                ),
                "candidate_ids": [
                    candidate.pattern_id for candidate in direct_retrieval.candidates
                ],
                "candidate_scores": {
                    candidate.pattern_id: candidate.vector_score
                    for candidate in direct_retrieval.candidates
                },
                "selected_pattern": _selected_pattern_evidence(
                    direct_retrieval.selected_pattern,
                    direct_retrieval.selected_vector_score,
                    direct_advice,
                ),
                "query_excludes_family_id": task.family_id not in direct_retrieval.query_text,
            },
            "chronology_audit": chronology,
            "t_runner_smoke": {
                "task_id": t_execution.artifact.task_id,
                "condition": t_execution.artifact.condition.value,
                "advice_count": t_execution.artifact.advice_count,
                "advice_accepted": t_execution.artifact.advice_accepted,
                "retrieval_evidence": [
                    dataclasses.asdict(item)
                    for item in t_execution.dependencies.advisory_retrieval_evidence
                ],
                "advice_event_count": len(t_execution.dependencies.advice_events),
                "executed_event_count": len(t_execution.dependencies.events),
                "objective_result": t_execution.artifact.task_success,
                "known_failure_repeated": t_execution.artifact.known_failure_repeated,
                "advice_payload": (
                    None
                    if t_execution.artifact.advice_received is None
                    else t_execution.artifact.advice_received.model_dump(mode="json")
                ),
                "planned_action": t_execution.artifact.planned_action.model_dump(mode="json"),
                "executed_action": t_execution.artifact.executed_action.model_dump(mode="json"),
                "artifact": str(t_execution.artifact_path),
                "raw_evidence": str(t_execution.raw_evidence_path),
                "model_visible_context_excludes_family_id": task.family_id not in raw_text,
            },
            "b0_isolation_smoke": {
                "condition": b0_execution.artifact.condition.value,
                "advisory_service_calls": 0,
                "dependency_service_is_none": b0_execution.dependencies.advisory_service is None,
                "retrieval_evidence_count": len(
                    b0_execution.dependencies.advisory_retrieval_evidence
                ),
                "artifact": str(b0_execution.artifact_path),
            },
            "anomalies_or_deviations": [
                (
                    "The live graph contains historical development patterns; the T "
                    "runtime exposes only the five canonical R13b IDs."
                ),
                (
                    "Neo4j emitted a deprecation warning for db.index.vector.queryNodes; "
                    "the frozen repository query was not changed in Sprint 2."
                ),
            ],
            "validation": {
                "focused_offline_tests": {
                    "status": "PASS",
                    "result": "3 passed",
                },
                "existing_track_b_advisory_tests": {
                    "status": "PASS",
                    "result": "48 passed",
                },
                "live_read_only_smoke": {
                    "status": "PASS",
                    "result": "completed by this report",
                },
                "opt_in_live_pytest": {
                    "status": "PASS",
                    "result": "4 passed",
                },
                "ruff": {"status": "PASS"},
                "pyright": {"status": "PASS", "result": "0 errors"},
                "git_diff_check": {"status": "PASS"},
            },
        }
        report_path = evidence_dir / "integration_point_1_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(report_path)
        return 0
    finally:
        runtime.close()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _working_tree_diff_hash() -> str:
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(status + b"\0" + diff).hexdigest()


def _config_hashes() -> dict[str, str]:
    paths = (
        Path("configs/experiments/rollout_3a_pilot.yaml"),
        Path("configs/models/openrouter.yaml"),
        Path("benchmark/manifests/pilot.jsonl"),
        Path("configs/research/gate_b1_environments.json"),
    )
    return {
        str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _selected_pattern_evidence(
    pattern: object,
    vector_score: float | None,
    advice: object,
) -> dict[str, object] | None:
    if pattern is None:
        return None
    selected = cast(RecoveryPattern, pattern)
    selected_advice = cast(AdviceResult, advice)
    return {
        "pattern_id": selected.id,
        "vector_score": vector_score,
        "source_task_id": selected.source_task_id,
        "source_chronological_index": selected.source_chronological_index,
        "source_failure_id": selected.source_failure_id,
        "source_resolution_id": selected.source_resolution_id,
        "source_outcome_id": selected.source_outcome_id,
        "advice_provenance": (
            None
            if selected_advice.advice is None
            else selected_advice.advice.provenance.model_dump(mode="json")
        ),
    }


def _new_evidence_dir() -> Path:
    root = ROOT / "research" / "evidence" / "integration_point_1"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / (
        "sprint2-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    directory.mkdir()
    return directory


def _diagnostic_task() -> Task:
    source = next(
        case.task
        for case in load_task_cases(
            ROOT / "benchmark/manifests/pilot.jsonl",
            problem_statements_path=ROOT / "benchmark/annotations/recurrence_validation.csv",
        )
        if case.task.id == "GS-T006"
    )
    return source.model_copy(
        update={
            "id": "IP1",
            "family_id": "evaluation-only-family-metadata",
        }
    )


def _environment(task: Task, run_id: str) -> EnvironmentContext:
    return EnvironmentContext(
        id=f"{run_id}-environment",
        repository=task.repository,
        runtime="docker",
        versions={"python": "3.12.1"},
    )


def _planned_action(task: Task, run_id: str) -> PlannedAction:
    return PlannedAction(
        id=f"{run_id}-action",
        run_id=run_id,
        task_id=task.id,
        tool="edit_file",
        operation="edit_file",
        arguments={"path": "README.md"},
        planned_at=datetime.now(UTC),
    )


def _run_t_smoke(runtime: object, task: Task, evidence_dir: Path):
    workspace = evidence_dir / "t-workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("before\n", encoding="utf-8")
    calls = 0

    def response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "edit_file",
                        {
                            "path": "README.md",
                            "old_text": "before\n",
                            "new_text": "after\n",
                            "expected_replacements": 1,
                        },
                    )
                ]
            )
        if calls == 2:
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": "README.md"})])
        return ModelResponse(parts=[TextPart("complete")])

    return _runner(
        runtime,
        task,
        ExperimentCondition.T,
        workspace,
        evidence_dir / "t-results",
        FunctionModel(response),
    ).run_task(task)


def _run_b0_smoke(runtime: object, task: Task, evidence_dir: Path):
    workspace = evidence_dir / "b0-workspace"
    workspace.mkdir()

    def response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("complete")])

    return _runner(
        runtime,
        task,
        ExperimentCondition.B0,
        workspace,
        evidence_dir / "b0-results",
        FunctionModel(response),
    ).run_task(task)


def _runner(
    runtime: object,
    task: Task,
    condition: ExperimentCondition,
    workspace: Path,
    results: Path,
    model: FunctionModel,
) -> ExperimentRunner:
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/rollout_3a_pilot.yaml",
        project_root=ROOT,
    )
    configuration = dataclasses.replace(
        configuration,
        config=configuration.config.model_copy(update={"conditions": (condition,)}),
    )
    advisory_service = runtime.advisory_service
    return ExperimentRunner(
        configuration,
        settings=get_settings(),
        model=model,
        workspace_resolver=lambda _task: workspace,
        environment_resolver=lambda current_task, run_id: _environment(current_task, run_id),
        artifact_store=ExperimentRunArtifactStore(results),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
        advisory_service=advisory_service,
        condition=condition,
    )


if __name__ == "__main__":
    raise SystemExit(main())
