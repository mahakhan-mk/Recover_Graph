"""Deterministic JSONL recording for advisory experiment evidence."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from graph_swarm.domain.behavior import BehaviorChangeEvidence
from graph_swarm.domain.events import AdviceEvent, AgentEvent


class AdviceArtifact(BaseModel):
    record_type: Literal["advice_event"] = "advice_event"
    run_id: str
    task_id: str
    action_id: str
    advice_event_id: str
    matched_failure_episode_id: str
    matched_resolution_id: str
    timestamp: datetime
    event: AdviceEvent


class RetrievalArtifact(BaseModel):
    record_type: Literal["advisory_retrieval"] = "advisory_retrieval"
    run_id: str
    task_id: str
    action_id: str
    advice_event_id: str
    pattern_id: str | None
    vector_score: float | None
    matched_failure_episode_id: str
    matched_resolution_id: str
    retrieval: dict[str, object] | None = None


class BehaviorArtifact(BaseModel):
    record_type: Literal["behavior_change_evidence"] = "behavior_change_evidence"
    run_id: str
    task_id: str
    action_id: str
    advice_event_id: str
    timestamp: datetime
    evidence: BehaviorChangeEvidence


class OutcomeArtifact(BaseModel):
    record_type: Literal["subsequent_outcome"] = "subsequent_outcome"
    run_id: str
    task_id: str
    action_id: str
    advice_event_id: str
    action_event_id: str
    timestamp: datetime
    success: bool
    exit_code: int | None


class JsonlResearchArtifactWriter:
    """Append typed advisory evidence as one deterministic JSON object per line."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write_advice_event(self, event: AdviceEvent) -> None:
        advice = event.advice
        self._append(
            AdviceArtifact(
                run_id=event.run_id,
                task_id=event.task_id,
                action_id=event.planned_action.id,
                advice_event_id=event.event_id,
                matched_failure_episode_id=advice.provenance.failure_episode_id,
                matched_resolution_id=advice.provenance.resolution_id,
                timestamp=event.issued_at,
                event=event,
            )
        )

    def write_behavior_evidence(self, evidence: BehaviorChangeEvidence) -> None:
        action = evidence.actual_action_after_advice or evidence.planned_action_before_advice
        self._append(
            BehaviorArtifact(
                run_id=action.run_id,
                task_id=action.task_id,
                action_id=action.id,
                advice_event_id=evidence.advice_event_id,
                timestamp=evidence.observed_at,
                evidence=evidence,
            )
        )

    def write_retrieval_evidence(
        self,
        *,
        run_id: str,
        task_id: str,
        action_id: str,
        advice_event_id: str,
        pattern_id: str | None,
        vector_score: float | None,
        matched_failure_episode_id: str,
        matched_resolution_id: str,
        retrieval: dict[str, object] | None,
    ) -> None:
        self._append(
            RetrievalArtifact(
                run_id=run_id,
                task_id=task_id,
                action_id=action_id,
                advice_event_id=advice_event_id,
                pattern_id=pattern_id,
                vector_score=vector_score,
                matched_failure_episode_id=matched_failure_episode_id,
                matched_resolution_id=matched_resolution_id,
                retrieval=retrieval,
            )
        )

    def write_subsequent_outcome(
        self,
        advice_event_id: str,
        event: AgentEvent,
    ) -> None:
        self._append(
            OutcomeArtifact(
                run_id=event.run_id,
                task_id=event.task_id,
                action_id=event.action_id,
                advice_event_id=advice_event_id,
                action_event_id=event.event_id,
                timestamp=event.occurred_at,
                success=event.result.success,
                exit_code=event.result.exit_code,
            )
        )

    def _append(self, record: BaseModel) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._path.open("a", encoding="utf-8", newline="\n") as artifact_file:
            artifact_file.write(serialized + "\n")
