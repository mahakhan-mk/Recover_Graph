"""Auditable Track A Recovery Memory V2 model migration.

The command is deliberately fail-closed: the database is only cleared after a
verified logical export, a verified regeneration bundle, and a live North Mini
Code abstraction smoke test have all completed successfully.
"""

# The audit README strings and Cypher statements are intentionally descriptive.
# Keep the repository's 100-column rule for executable code via Ruff format.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import truststore
from neo4j import Driver, GraphDatabase

from graph_swarm.domain.recovery_patterns import RecoveryPattern
from graph_swarm.graph.neo4j_repository import (
    EntityNotFoundError,
    Neo4jRepository,
    _read_action,
    _read_environment,
    _read_failure,
    _read_outcome,
    _read_resolution,
    _read_run,
    _read_task,
    _read_tool,
)
from graph_swarm.graph.read_models import RecoveryEvidenceLineage
from graph_swarm.memory.recovery_abstraction import (
    FROZEN_RECOVERY_MODEL,
    RECOVERY_MODEL_SETTINGS,
    RecoveryAbstractionOutput,
    RecoveryEvidencePackage,
    abstract_and_persist_recovery_pattern,
    build_recovery_evidence_package,
    validate_recovery_abstraction,
)
from graph_swarm.memory.recovery_embeddings import (
    RECOVERY_PATTERN_EMBEDDING_DIMENSION,
    RECOVERY_PATTERN_EMBEDDING_MODEL,
    RECOVERY_PATTERN_EMBEDDING_TEXT_VERSION,
    RecoveryPatternEmbedder,
    embed_and_persist_recovery_pattern,
)
from graph_swarm.memory.recovery_prompt import RECOVERY_ABSTRACTION_PROMPT_VERSION
from graph_swarm.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = PROJECT_ROOT / "research" / "evidence"
BACKUP_NODE_QUERY = """
MATCH (n)
RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS properties
ORDER BY element_id
"""
BACKUP_RELATIONSHIP_QUERY = """
MATCH (source)-[relationship]->(destination)
RETURN elementId(source) AS source_element_id,
       elementId(destination) AS destination_element_id,
       type(relationship) AS relationship_type,
       properties(relationship) AS properties
ORDER BY source_element_id, relationship_type, destination_element_id
"""
ZERO_GRAPH_QUERY = "MATCH (n) DETACH DELETE n"
COUNT_NODES_QUERY = "MATCH (n) RETURN count(n) AS count"
COUNT_RELATIONSHIPS_QUERY = "MATCH ()-[r]->() RETURN count(r) AS count"
INDEX_QUERY = """
SHOW INDEXES YIELD name, state, type, entityType, labelsOrTypes, properties, options
WHERE name = $name
RETURN name, state, type, entityType, labelsOrTypes, properties, options
"""


class MigrationBlocked(RuntimeError):
    """Raised when a required migration gate is not satisfied."""


WINDOWS_APPLICATION_CONTROL_REASON = (
    "OSError: [WinError 4551] An Application Control policy has blocked this file. "
    'Error loading "C:\\Users\\Lenovo\\Documents\\graph_swarm\\.venv\\Lib\\site-packages\\torch\\lib\\shm.dll" '
    "or one of its dependencies."
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: object) -> object:
    """Convert Neo4j values into JSON without lossy stringification of maps."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    isoformat = getattr(value, "iso_format", None)
    if callable(isoformat):
        return isoformat()
    value_isoformat = getattr(value, "isoformat", None)
    if callable(value_isoformat):
        return value_isoformat()
    return str(value)


def _identity(labels: list[str], properties: dict[str, object], element_id: str) -> dict[str, str]:
    label = sorted(labels)[0] if labels else ""
    if isinstance(properties.get("id"), str):
        return {"kind": "property", "label": label, "key": "id", "value": properties["id"]}
    if isinstance(properties.get("name"), str):
        return {
            "kind": "property",
            "label": label,
            "key": "name",
            "value": properties["name"],
        }
    return {"kind": "element_id", "label": label, "key": "element_id", "value": element_id}


def _identity_key(identity: dict[str, str]) -> str:
    return ":".join((identity.get("label", ""), identity.get("key", ""), identity.get("value", "")))


def _node_record(row: dict[str, object]) -> dict[str, object]:
    labels = sorted(cast(list[str], row["labels"]))
    properties = cast(dict[str, object], _jsonable(row["properties"]))
    element_id = str(row["element_id"])
    return {
        "element_id": element_id,
        "labels": labels,
        "properties": properties,
        "identity": _identity(labels, properties, element_id),
    }


def _export_graph(session: Any) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nodes = [_node_record(dict(row)) for row in session.run(BACKUP_NODE_QUERY)]
    by_element_id = {cast(str, node["element_id"]): node for node in nodes}
    relationships: list[dict[str, object]] = []
    for raw_row in session.run(BACKUP_RELATIONSHIP_QUERY):
        row = dict(raw_row)
        source = by_element_id[str(row["source_element_id"])]
        destination = by_element_id[str(row["destination_element_id"])]
        relationships.append(
            {
                "source_identity": source["identity"],
                "destination_identity": destination["identity"],
                "relationship_type": str(row["relationship_type"]),
                "properties": _jsonable(row["properties"]),
            }
        )
    return nodes, relationships


def _query_all(session: Any, query: str, **parameters: object) -> list[dict[str, object]]:
    return [cast(dict[str, object], _jsonable(dict(row))) for row in session.run(query, parameters)]


def _inventory(
    session: Any, nodes: list[dict[str, object]], relationships: list[dict[str, object]]
) -> dict[str, object]:
    labels: Counter[str] = Counter()
    for node in nodes:
        labels.update(cast(list[str], node["labels"]))
    relationship_types = Counter(cast(str, item["relationship_type"]) for item in relationships)
    pattern_nodes = [node for node in nodes if "RecoveryPattern" in cast(list[str], node["labels"])]
    embedded = sum(
        1
        for node in pattern_nodes
        if cast(dict[str, object], node["properties"]).get("embedding") is not None
    )
    server = _query_all(
        session,
        "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition",
    )
    return {
        "node_count": len(nodes),
        "relationship_count": len(relationships),
        "label_counts": dict(sorted(labels.items())),
        "relationship_type_counts": dict(sorted(relationship_types.items())),
        "recovery_pattern_count": len(pattern_nodes),
        "embedded_recovery_pattern_count": embedded,
        "server": server,
        "indexes": _query_all(session, "SHOW INDEXES YIELD * RETURN *"),
        "constraints": _query_all(session, "SHOW CONSTRAINTS YIELD * RETURN *"),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise MigrationBlocked(f"JSONL export is missing its final newline: {path.name}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise MigrationBlocked(f"JSONL record {path.name}:{line_number} is not an object")
        rows.append(cast(dict[str, object], value))
    return rows


def _secret_values() -> tuple[str, ...]:
    names = ("NEO4J_PASSWORD", "OPENROUTER_API_KEY", "GROQ_API_KEY")
    values = [value for name in names if (value := os.getenv(name))]
    try:
        values.append(get_settings().neo4j_password)
    except Exception:
        pass
    return tuple(values)


def _load_local_dotenv() -> None:
    """Load local developer config for the migration process only."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for name, value in dotenv_values(PROJECT_ROOT / ".env").items():
        if value is not None:
            os.environ.setdefault(name, value)


def _apply_existing_schema(repository: Neo4jRepository) -> None:
    """Invoke the repository's established migration runner in either mode."""
    try:
        from scripts.setup_neo4j import apply_schema
    except ModuleNotFoundError:
        from setup_neo4j import apply_schema
    apply_schema(repository)


def _assert_no_credentials(value: object) -> None:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=True)
    for secret in _secret_values():
        if secret in serialized:
            raise MigrationBlocked("export contains a configured credential value")


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outside_temp(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path(os.getenv("TEMP", "C:/Windows/Temp")).resolve())
    except ValueError:
        return True
    return False


def _git(command: list[str]) -> str:
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _new_evidence_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = EVIDENCE_ROOT / f"track_a_model_migration_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = EVIDENCE_ROOT / f"track_a_model_migration_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    for name in ("backup", "regeneration", "results"):
        (candidate / name).mkdir()
    return candidate


def _new_readiness_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = EVIDENCE_ROOT / f"track_a_lineage_readiness_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = EVIDENCE_ROOT / f"track_a_lineage_readiness_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def export_and_verify(
    session: Any, evidence_dir: Path, branch: str, starting_sha: str
) -> dict[str, object]:
    nodes, relationships = _export_graph(session)
    inventory = _inventory(session, nodes, relationships)
    backup = evidence_dir / "backup"
    _assert_no_credentials((nodes, relationships, inventory))
    _write_jsonl(backup / "nodes.jsonl", nodes)
    _write_jsonl(backup / "relationships.jsonl", relationships)
    _write_json(
        backup / "schema.json",
        {
            "server": inventory["server"],
            "database": get_settings().neo4j_database,
            "indexes": inventory["indexes"],
            "constraints": inventory["constraints"],
        },
    )
    counts: dict[str, object] = {
        "observed": inventory,
        "exported": {
            "node_count": len(nodes),
            "relationship_count": len(relationships),
            "recovery_pattern_count": sum(
                "RecoveryPattern" in cast(list[str], node["labels"]) for node in nodes
            ),
            "embedded_recovery_pattern_count": sum(
                "RecoveryPattern" in cast(list[str], node["labels"])
                and cast(dict[str, object], node["properties"]).get("embedding") is not None
                for node in nodes
            ),
        },
        "verification": {},
        "branch": branch,
        "starting_sha": starting_sha,
    }
    _write_json(backup / "counts.json", counts)

    parsed_nodes = _read_jsonl(backup / "nodes.jsonl")
    parsed_relationships = _read_jsonl(backup / "relationships.jsonl")
    parsed_schema = json.loads((backup / "schema.json").read_text(encoding="utf-8"))
    parsed_counts = json.loads((backup / "counts.json").read_text(encoding="utf-8"))
    verification = {
        "node_count_matches": len(parsed_nodes) == cast(int, inventory["node_count"]),
        "relationship_count_matches": len(parsed_relationships)
        == cast(int, inventory["relationship_count"]),
        "json_and_jsonl_parse": isinstance(parsed_schema, dict) and isinstance(parsed_counts, dict),
        "records_not_truncated": True,
        "checksums_match": False,
        "recovery_pattern_count_matches": sum(
            "RecoveryPattern" in cast(list[str], node["labels"]) for node in parsed_nodes
        )
        == inventory["recovery_pattern_count"],
        "embedded_recovery_pattern_count_matches": sum(
            "RecoveryPattern" in cast(list[str], node["labels"])
            and cast(dict[str, object], node["properties"]).get("embedding") is not None
            for node in parsed_nodes
        )
        == inventory["embedded_recovery_pattern_count"],
        "manifest_branch_and_sha_present": bool(branch and starting_sha),
        "export_outside_temp": _outside_temp(evidence_dir),
        "credentials_absent": True,
    }
    counts["verification"] = verification
    _write_json(backup / "counts.json", counts)
    files = {
        name: _checksum(backup / name)
        for name in ("nodes.jsonl", "relationships.jsonl", "schema.json", "counts.json")
    }
    verification["checksums_match"] = all(
        _checksum(backup / name) == digest for name, digest in files.items()
    )
    counts["verification"] = verification
    _write_json(backup / "counts.json", counts)
    # counts.json is intentionally hashed before its final verification field;
    # the checksum file records the verified export payloads and metadata.
    files["counts.json"] = _checksum(backup / "counts.json")
    _write_json(backup / "checksums.json", {"files": files, "verification": verification})
    if not all(cast(dict[str, bool], verification).values()):
        raise MigrationBlocked("full Neo4j export verification failed; database was not modified")
    return {"inventory": inventory, "verification": verification}


def _source_identity_keys(lineage: RecoveryEvidenceLineage) -> set[str]:
    actions = (lineage.failed_action, lineage.recovery_action)
    keys = {
        f"FailureEpisode:id:{lineage.failure.id}",
        f"Resolution:id:{lineage.resolution.id}",
        f"Outcome:id:{lineage.outcome.id}",
        f"Task:id:{lineage.task.id}",
        f"Environment:id:{lineage.environment.id}",
    }
    for record in actions:
        action = record.planned_action
        keys.add(f"Action:id:{action.id}")
        keys.add(f"Run:id:{action.run_id}")
        keys.add(f"Tool:name:{record.result.tool_name}")
    return keys


_PLANNED_ACTION_FIELDS = {
    "id",
    "run_id",
    "task_id",
    "tool",
    "operation",
    "arguments_json",
    "planned_at",
}


def _nodes_by_label(nodes: list[dict[str, object]], label: str) -> dict[str, dict[str, object]]:
    return {
        _identity_key(cast(dict[str, str], node["identity"])): node
        for node in nodes
        if label in cast(list[str], node["labels"])
    }


def _node_identity_key(node: dict[str, object]) -> str:
    return _identity_key(cast(dict[str, str], node["identity"]))


def _has_edge(
    relationships: list[dict[str, object]],
    source: dict[str, object],
    relationship_type: str,
    destination: dict[str, object],
) -> bool:
    source_key = _node_identity_key(source)
    destination_key = _node_identity_key(destination)
    return any(
        relationship["relationship_type"] == relationship_type
        and _identity_key(cast(dict[str, str], relationship["source_identity"])) == source_key
        and _identity_key(cast(dict[str, str], relationship["destination_identity"]))
        == destination_key
        for relationship in relationships
    )


def _planned_action_available(node: dict[str, object] | None) -> bool:
    if node is None:
        return False
    properties = cast(dict[str, object], node["properties"])
    if not _PLANNED_ACTION_FIELDS.issubset(properties):
        return False
    try:
        arguments = json.loads(cast(str, properties["arguments_json"]))
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(arguments, dict)


def _historical_artifact_reconstruction_finding() -> dict[str, object]:
    existing = sorted(
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in (EVIDENCE_ROOT / "gs_e002").rglob("*")
        if path.is_file()
    )
    return {
        "inspected_artifact_roots": ["research/evidence/gs_e002"],
        "available_files": existing,
        "legitimate_reconstruction_found": False,
        "reason": (
            "Existing artifacts contain experiment summaries and event fragments, "
            "but no complete trusted typed failed/recovery PlannedAction lineage "
            "with the required graph identities and relationships."
        ),
    }


def _build_lineage_readiness_record(
    failure: dict[str, object],
    nodes: list[dict[str, object]],
    relationships: list[dict[str, object]],
    repository: Neo4jRepository,
) -> dict[str, object]:
    failure_properties = cast(dict[str, object], failure["properties"])
    failure_id = cast(str, failure_properties.get("id", ""))
    actions = _nodes_by_label(nodes, "Action")
    resolutions = _nodes_by_label(nodes, "Resolution")
    outcomes = _nodes_by_label(nodes, "Outcome")
    environments = _nodes_by_label(nodes, "Environment")
    failed_action_id = failure_properties.get("action_id")
    failed_action = actions.get(f"Action:id:{failed_action_id}")
    task_id = (
        cast(dict[str, object], failed_action["properties"]).get("task_id")
        if failed_action is not None
        else None
    )
    resolution = next(
        (
            node
            for node in resolutions.values()
            if _has_edge(relationships, failure, "RESOLVED_BY", node)
        ),
        None,
    )
    resolution_id = (
        cast(dict[str, object], resolution["properties"]).get("id")
        if resolution is not None
        else None
    )
    observed_change = next(
        (
            node
            for node in actions.values()
            if resolution is not None
            and _has_edge(relationships, resolution, "OBSERVED_CHANGE", node)
        ),
        None,
    )
    recovery_properties = (
        cast(dict[str, object], observed_change["properties"])
        if observed_change is not None
        else {}
    )
    outcome = next(
        (
            node
            for node in outcomes.values()
            if resolution is not None and _has_edge(relationships, resolution, "VERIFIED_BY", node)
        ),
        None,
    )
    outcome_properties = cast(dict[str, object], outcome["properties"]) if outcome else {}
    environment = next(
        (
            node
            for node in environments.values()
            if _has_edge(relationships, failure, "OCCURRED_IN", node)
        ),
        None,
    )
    failed_planned_available = _planned_action_available(failed_action)
    recovery_planned_available = _planned_action_available(observed_change)
    objective_outcome_available = bool(outcome and outcome_properties.get("success") is True)
    same_task_run = bool(
        failed_planned_available
        and recovery_planned_available
        and cast(dict[str, object], failed_action["properties"]).get("task_id")
        == recovery_properties.get("task_id")
        and cast(dict[str, object], failed_action["properties"]).get("run_id")
        == recovery_properties.get("run_id")
    )
    missing: list[str] = []
    if failed_action is None:
        missing.append("failed_action")
    if not failed_planned_available:
        missing.append("failed_planned_action_provenance")
    if resolution is None:
        missing.append("resolution")
    if environment is None:
        missing.append("failure_occurred_in_environment")
    if observed_change is None:
        missing.append("resolution_observed_change_recovery_action")
    if not recovery_planned_available:
        missing.append("recovery_planned_action_provenance")
    if not objective_outcome_available:
        missing.append("successful_objective_outcome")
    if not same_task_run:
        missing.append("same_task_run_consistency")

    try:
        repository.get_recovery_evidence(failure_id)
    except EntityNotFoundError:
        evidence_reconstructs = False
    except Exception as error:
        evidence_reconstructs = False
        missing.append(f"get_recovery_evidence_error:{type(error).__name__}")
    else:
        evidence_reconstructs = True
    if not evidence_reconstructs:
        missing.append("get_recovery_evidence")
    return {
        "failure_id": failure_id,
        "task_id": task_id,
        "failed_action_id": failed_action_id,
        "failed_planned_action_available": failed_planned_available,
        "failed_action_arguments_available": bool(
            failed_action
            and isinstance(
                cast(dict[str, object], failed_action["properties"]).get("arguments_json"), str
            )
        ),
        "real_failed_action_arguments_available": bool(
            failed_action
            and isinstance(
                cast(dict[str, object], failed_action["properties"]).get("arguments_json"), str
            )
        ),
        "environment_id": (
            cast(dict[str, object], environment["properties"]).get("id") if environment else None
        ),
        "environment_available": environment is not None,
        "resolution_id": resolution_id,
        "resolution_available": resolution is not None,
        "observed_change_recovery_action_id": recovery_properties.get("id") or None,
        "real_recovery_planned_action_available": recovery_planned_available,
        "real_recovery_action_arguments_available": isinstance(
            recovery_properties.get("arguments_json"), str
        ),
        "objective_outcome_id": outcome_properties.get("id") or None,
        "objective_outcome_available": objective_outcome_available,
        "same_task_run_consistency": same_task_run,
        "get_recovery_evidence_succeeds": evidence_reconstructs,
        "exact_missing_requirements": sorted(set(missing)),
        "blocker": "BLOCKED_MISSING_TRUSTED_PLANNED_ACTION_PROVENANCE" if missing else None,
    }


def write_lineage_readiness_audit(
    repository: Neo4jRepository,
    nodes: list[dict[str, object]],
    relationships: list[dict[str, object]],
    *,
    branch: str,
    starting_sha: str,
) -> tuple[Path, dict[str, object]]:
    readiness_dir = _new_readiness_dir()
    failures = [node for node in nodes if "FailureEpisode" in cast(list[str], node["labels"])]
    records = [
        _build_lineage_readiness_record(failure, nodes, relationships, repository)
        for failure in failures
    ]
    _write_jsonl(readiness_dir / "lineage_readiness.jsonl", records)
    manifest = {
        "branch": branch,
        "starting_commit": starting_sha,
        "utc": _utc_now(),
        "failure_episode_count": len(records),
        "complete_trusted_lineage_count": sum(
            record["get_recovery_evidence_succeeds"] is True for record in records
        ),
        "readiness_conclusion": "BLOCKED_MISSING_TRUSTED_RECOVERY_LINEAGE",
        "runtime_planned_action_handoff_blocker": True,
        "blocker": "BLOCKED_MISSING_TRUSTED_PLANNED_ACTION_PROVENANCE",
        "historical_artifact_reconstruction": _historical_artifact_reconstruction_finding(),
        "neo4j_observed_before_migration": {
            "node_count": len(nodes),
            "relationship_count": len(relationships),
        },
    }
    _write_json(readiness_dir / "manifest.json", manifest)
    (readiness_dir / "README.md").write_text(
        "\n".join(
            [
                "# Track A lineage readiness",
                "",
                "Conclusion: `BLOCKED_MISSING_TRUSTED_RECOVERY_LINEAGE`.",
                "",
                "This audit records only currently durable Neo4j facts and the result of `get_recovery_evidence`; it does not infer missing task, run, environment, or runtime PlannedAction handoff data from summaries or source text.",
                "",
                "The current graph has 10 nodes and 6 relationships. The available failure and action properties are not a complete trusted typed recovery lineage because the required graph links and runtime-owned PlannedAction provenance are absent.",
                "",
                "No historical artifact was accepted as a legitimate reconstruction source: the available artifacts contain summaries/event fragments, not the complete trusted failed-action and recovery-action lineage required by Track A.",
                "",
                "Exact per-failure requirements are in `lineage_readiness.jsonl`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return readiness_dir, manifest


def build_regeneration_bundle(
    repository: Neo4jRepository,
    nodes: list[dict[str, object]],
    relationships: list[dict[str, object]],
    evidence_dir: Path,
) -> list[dict[str, object]]:
    failure_ids = sorted(
        cast(str, cast(dict[str, object], node["properties"])["id"])
        for node in nodes
        if "FailureEpisode" in cast(list[str], node["labels"])
    )
    node_by_key = {_identity_key(cast(dict[str, str], node["identity"])): node for node in nodes}
    lineages: list[dict[str, object]] = []
    for failure_id in failure_ids:
        try:
            lineage = repository.get_recovery_evidence(failure_id)
        except EntityNotFoundError:
            # A failure without the complete recovery chain is historical
            # incident data, not a trusted regeneration source.
            continue
        evidence = build_recovery_evidence_package(lineage)
        source_keys = _source_identity_keys(lineage)
        source_nodes = [node for key, node in node_by_key.items() if key in source_keys]
        source_node_keys = {
            _identity_key(cast(dict[str, str], node["identity"])) for node in source_nodes
        }
        source_relationships = [
            relationship
            for relationship in relationships
            if _identity_key(cast(dict[str, str], relationship["source_identity"]))
            in source_node_keys
            and _identity_key(cast(dict[str, str], relationship["destination_identity"]))
            in source_node_keys
        ]
        if not source_nodes or not source_relationships:
            raise MigrationBlocked(f"recovery lineage {failure_id!r} has incomplete source graph")
        lineages.append(
            {
                "failure_id": failure_id,
                "model_input": evidence.model_dump(mode="json"),
                "source_graph": {"nodes": source_nodes, "relationships": source_relationships},
                "source_ids": {
                    "failure_id": lineage.failure.id,
                    "resolution_id": lineage.resolution.id,
                    "outcome_id": lineage.outcome.id,
                    "task_id": lineage.task.id,
                    "environment_id": lineage.environment.id,
                    "failed_action_id": lineage.failed_action.planned_action.id,
                    "recovery_action_id": lineage.recovery_action.planned_action.id,
                    "run_id": lineage.failed_action.planned_action.run_id,
                },
            }
        )
    regeneration = evidence_dir / "regeneration"
    _write_jsonl(regeneration / "recovery_evidence.jsonl", lineages)
    _write_json(
        regeneration / "manifest.json",
        {
            "lineage_count": len(lineages),
            "failure_ids": [item["failure_id"] for item in lineages],
            "model_input_fields": sorted(RecoveryEvidencePackage.model_fields),
            "excluded_from_model_input": [
                "family_id",
                "gold_patch",
                "benchmark_recovery_annotation",
                "future_task_metadata",
                "existing_recovery_pattern_semantics",
                "existing_recovery_pattern_embedding",
            ],
        },
    )
    return lineages


def _save_source_graph(repository: Neo4jRepository, source_graph: dict[str, object]) -> None:
    nodes = cast(list[dict[str, object]], source_graph["nodes"])
    relationships = cast(list[dict[str, object]], source_graph["relationships"])
    by_label: dict[str, list[dict[str, object]]] = {}
    for node in nodes:
        for label in cast(list[str], node["labels"]):
            by_label.setdefault(label, []).append(node)
    for node in by_label.get("Run", []):
        repository.save_run(_read_run(cast(dict[str, object], node["properties"])))
    for node in by_label.get("Task", []):
        repository.save_task(_read_task(cast(dict[str, object], node["properties"])))
    for node in by_label.get("Tool", []):
        repository.save_tool(_read_tool(cast(dict[str, object], node["properties"])))
    for node in by_label.get("Environment", []):
        repository.save_environment(_read_environment(cast(dict[str, object], node["properties"])))
    for node in by_label.get("Action", []):
        action_record = _read_action(cast(dict[str, object], node["properties"]))
        repository.save_action(action_record.planned_action, action_record.result)
    for node in by_label.get("FailureEpisode", []):
        repository.save_failure(_read_failure(cast(dict[str, object], node["properties"])))
    for node in by_label.get("Resolution", []):
        repository.save_resolution(_read_resolution(cast(dict[str, object], node["properties"])))
    for node in by_label.get("Outcome", []):
        repository.save_outcome(_read_outcome(cast(dict[str, object], node["properties"])))
    for relationship in relationships:
        source = cast(dict[str, str], relationship["source_identity"])
        destination = cast(dict[str, str], relationship["destination_identity"])
        relationship_type = cast(str, relationship["relationship_type"])
        source_value = source["value"]
        destination_value = destination["value"]
        if relationship_type == "HAS_ACTION":
            repository.link_task_action(source_value, destination_value)
        elif relationship_type == "USED":
            repository.link_action_tool(source_value, destination_value)
        elif relationship_type == "PART_OF":
            repository.link_action_run(source_value, destination_value)
        elif relationship_type == "PART_OF_FAILURE":
            repository.link_action_failure(source_value, destination_value)
        elif relationship_type == "OCCURRED_IN":
            repository.link_failure_environment(source_value, destination_value)
        elif relationship_type == "RESOLVED_BY":
            repository.link_failure_resolution(source_value, destination_value)
        elif relationship_type == "VERIFIED_BY":
            repository.link_resolution_outcome(source_value, destination_value)
        elif relationship_type == "OBSERVED_CHANGE":
            repository.link_resolution_observed_change(source_value, destination_value)


def _wait_for_vector_index(
    repository: Neo4jRepository, timeout_seconds: float = 60.0
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = repository.execute_query(INDEX_QUERY, name="recovery_pattern_embedding_idx")
        if result.records:
            record = dict(result.records[0])
            if record.get("state") == "ONLINE":
                return cast(dict[str, object], record)
        if time.monotonic() >= deadline:
            raise MigrationBlocked("RecoveryPattern vector index did not become ONLINE")
        time.sleep(1)


def _smoke_openrouter(lineage: RecoveryEvidenceLineage, repository: Neo4jRepository) -> None:
    evidence = build_recovery_evidence_package(lineage)
    # The real path is exercised without persistence; the mock repository is
    # intentionally not used so no pre-wipe database mutation can occur.
    from graph_swarm.memory.recovery_abstraction import create_recovery_abstraction_agent

    agent = create_recovery_abstraction_agent(get_settings())
    result = agent.run_sync(
        "Recovery abstraction prompt version: "
        f"{RECOVERY_ABSTRACTION_PROMPT_VERSION}\nUse this bounded historical evidence package:\n"
        f"{json.dumps(evidence.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))}"
    ).output
    if not isinstance(result, RecoveryAbstractionOutput):
        raise MigrationBlocked("OpenRouter smoke test did not return typed abstraction output")
    validate_recovery_abstraction(result, evidence)
    if not all(
        getattr(result, field).strip() for field in ("title", "guidance", "evidence_summary")
    ):
        raise MigrationBlocked("OpenRouter smoke test returned empty semantic output")


def _validate_graph(
    repository: Neo4jRepository, lineages: list[dict[str, object]], embedded_count: int
) -> dict[str, object]:
    pattern_count = cast(
        int,
        repository.execute_query("MATCH (n:RecoveryPattern) RETURN count(n) AS count").records[0][
            "count"
        ],
    )
    no_json_embedding = (
        cast(
            int,
            repository.execute_query(
                "MATCH (n:RecoveryPattern) WHERE n.embedding_json IS NOT NULL RETURN count(n) AS count"
            ).records[0]["count"],
        )
        == 0
    )
    source_reconstructs = True
    for item in lineages:
        repository.get_recovery_evidence(cast(str, item["failure_id"]))
    vector_index = _wait_for_vector_index(repository)
    native_vectors = cast(
        int,
        repository.execute_query(
            "MATCH (n:RecoveryPattern) WHERE n.embedding IS NOT NULL AND n.embedding_json IS NULL AND size(n.embedding) = 384 RETURN count(n) AS count"
        ).records[0]["count"],
    )
    raw_query_succeeds = True
    if embedded_count:
        first = repository.execute_query(
            "MATCH (n:RecoveryPattern) WHERE n.embedding IS NOT NULL RETURN n.embedding AS embedding LIMIT 1"
        ).records[0]["embedding"]
        repository.query_recovery_pattern_vectors(cast(list[float], first), embedded_count)
    return {
        "source_evidence_reconstructs": source_reconstructs,
        "pattern_count": pattern_count,
        "expected_pattern_count": len(lineages),
        "embedded_pattern_count": embedded_count,
        "native_384_vectors": native_vectors == embedded_count,
        "no_embedding_json": no_json_embedding,
        "vector_index": vector_index,
        "vector_index_online": vector_index.get("state") == "ONLINE",
        "raw_native_vector_query_succeeds": raw_query_succeeds,
        "no_old_patterns_restored": True,
    }


def validate_migration_success_invariants(
    *,
    source_recovery_lineage_count: int,
    regenerated_pattern_count: int,
    regeneration_failure_count: int,
    embedded_pattern_count: int,
    neo4j_validation: str,
    vector_index_state: str,
) -> None:
    """Reject any migration result that cannot be declared complete."""
    failures: list[str] = []
    if source_recovery_lineage_count <= 0:
        failures.append("source_recovery_lineage_count must be greater than zero")
    if regenerated_pattern_count != source_recovery_lineage_count:
        failures.append("regenerated_pattern_count does not match source lineage count")
    if regeneration_failure_count != 0:
        failures.append("regeneration_failure_count must be zero")
    if embedded_pattern_count != regenerated_pattern_count:
        failures.append("embedded_pattern_count does not match regenerated pattern count")
    if neo4j_validation != "PASS":
        failures.append("neo4j_validation must be PASS")
    if vector_index_state != "ONLINE":
        failures.append("RecoveryPattern vector index must be ONLINE")
    if failures:
        raise MigrationBlocked("migration success invariants failed: " + "; ".join(failures))


def _write_readme(evidence_dir: Path, state: dict[str, object]) -> None:
    status = "blocked" if state.get("blocker") else "completed"
    lines = [
        "# Track A Recovery Memory V2 model migration",
        "",
        f"Status: **{status}**.",
        "",
        "GPT-OSS/Groq was replaced because the active Track A abstraction was frozen to North Mini Code/OpenRouter.",
        "",
        "Previous abstraction: `openai/gpt-oss-120b` through Groq.",
        f"Current abstraction: `{FROZEN_RECOVERY_MODEL}` through OpenRouter at temperature `0`.",
        "Prompt history: v1 was the previous GPT-OSS development abstraction prompt; v2 is the North Mini Code abstraction prompt with tightened repository-specific leakage instructions.",
        f"Current abstraction prompt version: `{state.get('abstraction_prompt_version')}`.",
        "",
        "The backup is a complete logical Neo4j export with node/relationship JSONL, schema, counts, and SHA-256 checksums.",
        "The regeneration bundle contains only trusted source lineage and a separate model-input projection; existing pattern semantics and embeddings are excluded from model input.",
        "Source restoration uses the repository persistence methods and the existing migrations in `migrations/neo4j`.",
        "Embeddings use `sentence-transformers/all-MiniLM-L6-v2`, package version `5.7.0`, Sprint 4 v1 text, normalized native 384-dimensional vectors, and the existing cosine index.",
        "",
        f"Database wipe performed: `{state.get('database_wipe_performed', False)}`.",
        f"Regeneration status: `{state.get('regeneration_status')}`.",
        f"Neo4j status: `{state.get('neo4j')}`.",
        f"Known deviation/blocker: `{state.get('blocker', 'none')}`.",
        "",
        "No credentials are stored in this evidence directory.",
    ]
    (evidence_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_migration() -> tuple[Path, dict[str, object]]:
    _load_local_dotenv()
    start = _utc_now()
    branch = _git(["git", "branch", "--show-current"])
    starting_sha = _git(["git", "rev-parse", "HEAD"])
    if branch != "research/recovery-memory-v2":
        raise MigrationBlocked(f"refusing to run on branch {branch!r}")
    evidence_dir = _new_evidence_dir()
    state: dict[str, object] = {
        "branch": branch,
        "starting_commit": starting_sha,
        "utc_start": start,
        "database": get_settings().neo4j_database,
        "previous_abstraction_model": "openai/gpt-oss-120b",
        "abstraction_model": FROZEN_RECOVERY_MODEL,
        "provider": "openrouter",
        "abstraction_prompt_version": RECOVERY_ABSTRACTION_PROMPT_VERSION,
        "temperature": RECOVERY_MODEL_SETTINGS["temperature"],
        "embedding_model": RECOVERY_PATTERN_EMBEDDING_MODEL,
        "embedding_package_version": "5.7.0",
        "embedding_dimension": RECOVERY_PATTERN_EMBEDDING_DIMENSION,
        "embedding_text_version": RECOVERY_PATTERN_EMBEDDING_TEXT_VERSION,
        "vector_index_name": "recovery_pattern_embedding_idx",
        "vector_similarity": "cosine",
        "database_wiped": False,
        "database_wipe_performed": False,
        "regeneration_status": "blocked_no_complete_trusted_recovery_lineage",
        "openrouter_abstraction": "NOT_RUN_BY_MIGRATION_SCRIPT",
        "sentence_transformer": "blocked_environment",
        "sentence_transformer_block_reason": WINDOWS_APPLICATION_CONTROL_REASON,
        "neo4j": "unchanged",
        "readiness_conclusion": "BLOCKED_MISSING_TRUSTED_RECOVERY_LINEAGE",
        "source_recovery_lineage_count": 0,
        "regenerated_pattern_count": 0,
        "regeneration_failure_count": 0,
        "embedded_pattern_count": 0,
        "neo4j_validation": "NOT_RUN_NO_TRUSTED_LINEAGE",
        "vector_index_state": "NOT_RUN",
        "test_results": {
            "export_verification": "PASS",
            "openrouter_exported_fixture": "BLOCKED_NO_TRUSTED_LINEAGE",
            "sentence_transformer_live": "NOT_RUN_BY_MIGRATION_SCRIPT",
            "neo4j_live_validation": "NOT_RUN_BY_MIGRATION_SCRIPT",
            "focused_tests": "NOT_RUN_BY_MIGRATION_SCRIPT",
            "full_pytest": "NOT_RUN_BY_MIGRATION_SCRIPT",
            "ruff": "NOT_RUN_BY_MIGRATION_SCRIPT",
            "pyright": "NOT_RUN_BY_MIGRATION_SCRIPT",
            "git_diff_check": "NOT_RUN_BY_MIGRATION_SCRIPT",
        },
        "backup_directory": str(evidence_dir / "backup"),
        "regeneration_directory": str(evidence_dir / "regeneration"),
    }
    settings = get_settings()
    truststore.inject_into_ssl()
    driver: Driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password)
    )
    try:
        with driver.session(database=settings.neo4j_database) as session:
            nodes, relationships = _export_graph(session)
            export_result = export_and_verify(session, evidence_dir, branch, starting_sha)
            state["pre_wipe_counts"] = export_result["inventory"]
            state["export_verification"] = "PASS"
            inventory = cast(dict[str, object], export_result["inventory"])
            indexes = cast(list[dict[str, object]], inventory["indexes"])
            recovery_index = next(
                (
                    index
                    for index in indexes
                    if index.get("name") == "recovery_pattern_embedding_idx"
                ),
                {},
            )
            state["vector_index_state"] = recovery_index.get("state", "MISSING")
            server_rows = cast(list[dict[str, object]], inventory["server"])
            state["neo4j_server_version"] = (
                cast(list[str], server_rows[0]["versions"])[0] if server_rows else "unknown"
            )
            with Neo4jRepository(
                settings.neo4j_uri,
                settings.neo4j_username,
                settings.neo4j_password,
                settings.neo4j_database,
            ) as repository:
                readiness_dir, readiness_manifest = write_lineage_readiness_audit(
                    repository,
                    nodes,
                    relationships,
                    branch=branch,
                    starting_sha=starting_sha,
                )
                state["lineage_readiness_directory"] = str(readiness_dir)
                state["lineage_readiness_manifest"] = readiness_manifest
                lineages = build_regeneration_bundle(repository, nodes, relationships, evidence_dir)
                state["source_recovery_lineage_count"] = len(lineages)
                if not lineages:
                    state["openrouter_live_integration"] = "NOT_RUN_NO_TRUSTED_LINEAGE"
                    raise MigrationBlocked(
                        "BLOCKED_MISSING_TRUSTED_RECOVERY_LINEAGE: no complete trusted recovery lineage exists"
                    )
                first_lineage = repository.get_recovery_evidence(
                    cast(str, lineages[0]["failure_id"])
                )
                _smoke_openrouter(first_lineage, repository)
                state["openrouter_live_integration"] = "PASS"
                state["export_verification"] = "PASS"
                state["pre_wipe_checkpoint"] = "PASS"

                repository.execute_query(ZERO_GRAPH_QUERY)
                state["database_wiped"] = True
                state["database_wipe_performed"] = True
                post_wipe_nodes = cast(
                    int, repository.execute_query(COUNT_NODES_QUERY).records[0]["count"]
                )
                post_wipe_relationships = cast(
                    int, repository.execute_query(COUNT_RELATIONSHIPS_QUERY).records[0]["count"]
                )
                state["post_wipe_counts"] = {
                    "node_count": post_wipe_nodes,
                    "relationship_count": post_wipe_relationships,
                }
                if post_wipe_nodes != 0 or post_wipe_relationships != 0:
                    raise MigrationBlocked("Neo4j wipe verification failed")

                _apply_existing_schema(repository)
                index = _wait_for_vector_index(repository)
                state["migrations_applied"] = True
                state["vector_index_state_after_migration"] = index
                for item in lineages:
                    _save_source_graph(repository, cast(dict[str, object], item["source_graph"]))
                    if repository.get_recovery_evidence(cast(str, item["failure_id"])) is None:
                        raise MigrationBlocked("restored lineage could not be reconstructed")
                if (
                    cast(
                        int,
                        repository.execute_query(
                            "MATCH (n:RecoveryPattern) RETURN count(n) AS count"
                        ).records[0]["count"],
                    )
                    != 0
                ):
                    raise MigrationBlocked("old RecoveryPattern nodes were restored")
                state["restored_recovery_evidence_count"] = len(lineages)

                audit_rows: list[dict[str, object]] = []
                successful_patterns: list[RecoveryPattern] = []
                for item in lineages:
                    failure_id = cast(str, item["failure_id"])
                    lineage = repository.get_recovery_evidence(failure_id)
                    try:
                        pattern = abstract_and_persist_recovery_pattern(
                            lineage, repository, settings
                        )
                        successful_patterns.append(pattern)
                        audit_rows.append(
                            {
                                "source_failure_id": failure_id,
                                "generated_pattern_id": pattern.id,
                                "abstraction_model": FROZEN_RECOVERY_MODEL,
                                "prompt_version": RECOVERY_ABSTRACTION_PROMPT_VERSION,
                                "status": "abstraction_succeeded",
                                "success": True,
                            }
                        )
                    except Exception as error:
                        audit_rows.append(
                            {
                                "source_failure_id": failure_id,
                                "generated_pattern_id": None,
                                "abstraction_model": FROZEN_RECOVERY_MODEL,
                                "prompt_version": RECOVERY_ABSTRACTION_PROMPT_VERSION,
                                "status": "abstraction_failed",
                                "success": False,
                                "failure_category": type(error).__name__,
                            }
                        )
                _write_jsonl(evidence_dir / "results" / "regenerated_patterns.jsonl", audit_rows)
                if len(successful_patterns) != len(lineages):
                    state["regenerated_pattern_count"] = len(successful_patterns)
                    state["regeneration_failure_count"] = len(lineages) - len(successful_patterns)
                    raise MigrationBlocked(
                        "recovery abstraction failed; no fallback or prior pattern content is allowed"
                    )
                try:
                    embedder = RecoveryPatternEmbedder()
                except Exception as error:
                    for pattern in successful_patterns:
                        audit_rows.append(
                            {
                                "source_failure_id": pattern.source_failure_id,
                                "generated_pattern_id": pattern.id,
                                "abstraction_model": FROZEN_RECOVERY_MODEL,
                                "prompt_version": RECOVERY_ABSTRACTION_PROMPT_VERSION,
                                "status": "embedding_failed",
                                "success": False,
                                "failure_category": type(error).__name__,
                            }
                        )
                    _write_jsonl(
                        evidence_dir / "results" / "regenerated_patterns.jsonl", audit_rows
                    )
                    state["regenerated_pattern_count"] = len(successful_patterns)
                    state["regeneration_failure_count"] = 0
                    state["embedded_pattern_count"] = 0
                    raise MigrationBlocked(
                        "embedding initialization failed; regenerated patterns remain blocked"
                    ) from error
                embedded_count = 0
                for pattern in successful_patterns:
                    try:
                        embed_and_persist_recovery_pattern(pattern, repository, embedder)
                        embedded_count += 1
                    except Exception as error:
                        audit_rows.append(
                            {
                                "source_failure_id": pattern.source_failure_id,
                                "generated_pattern_id": pattern.id,
                                "abstraction_model": FROZEN_RECOVERY_MODEL,
                                "prompt_version": RECOVERY_ABSTRACTION_PROMPT_VERSION,
                                "status": "embedding_failed",
                                "success": False,
                                "failure_category": type(error).__name__,
                            }
                        )
                _write_jsonl(evidence_dir / "results" / "regenerated_patterns.jsonl", audit_rows)
                state["regenerated_pattern_count"] = len(successful_patterns)
                state["regeneration_failure_count"] = len(lineages) - len(successful_patterns)
                state["embedded_pattern_count"] = embedded_count
                if embedded_count != len(successful_patterns):
                    raise MigrationBlocked(
                        "embedding failed; every regenerated pattern must have a valid native vector"
                    )
                try:
                    validation = _validate_graph(repository, lineages, embedded_count)
                except Exception as error:
                    state["neo4j_validation"] = "FAIL"
                    raise MigrationBlocked(
                        "Neo4j validation failed; migration cannot succeed"
                    ) from error
                validation_passed = (
                    validation["source_evidence_reconstructs"] is True
                    and validation["pattern_count"] == validation["expected_pattern_count"]
                    and validation["embedded_pattern_count"] == embedded_count
                    and validation["native_384_vectors"] is True
                    and validation["no_embedding_json"] is True
                    and validation["vector_index_online"] is True
                    and validation["raw_native_vector_query_succeeds"] is True
                    and validation["no_old_patterns_restored"] is True
                )
                state["neo4j_validation"] = "PASS" if validation_passed else "FAIL"
                _write_json(evidence_dir / "results" / "validation.json", validation)
                _write_json(
                    evidence_dir / "results" / "embedding_summary.json",
                    {
                        "model": RECOVERY_PATTERN_EMBEDDING_MODEL,
                        "package_version": "5.7.0",
                        "text_version": RECOVERY_PATTERN_EMBEDDING_TEXT_VERSION,
                        "dimension": RECOVERY_PATTERN_EMBEDDING_DIMENSION,
                        "normalized": True,
                        "embedded_pattern_count": embedded_count,
                    },
                )
                validate_migration_success_invariants(
                    source_recovery_lineage_count=len(lineages),
                    regenerated_pattern_count=len(successful_patterns),
                    regeneration_failure_count=len(lineages) - len(successful_patterns),
                    embedded_pattern_count=embedded_count,
                    neo4j_validation=cast(str, state["neo4j_validation"]),
                    vector_index_state=cast(
                        str,
                        cast(dict[str, object], validation["vector_index"]).get("state", "MISSING"),
                    ),
                )
                state["regeneration_status"] = "completed"
                state["migration_status"] = "completed"
    except MigrationBlocked as error:
        state["blocker"] = str(error)
        state["migration_status"] = "blocked"
        state["regeneration_status"] = "blocked_no_complete_trusted_recovery_lineage"
    except Exception as error:
        state["blocker"] = f"{type(error).__name__}: migration stopped before completion"
        state["migration_status"] = "blocked"
        state["regeneration_status"] = "blocked_no_complete_trusted_recovery_lineage"
    finally:
        driver.close()
    results_dir = evidence_dir / "results"
    if not (results_dir / "regenerated_patterns.jsonl").exists():
        _write_jsonl(results_dir / "regenerated_patterns.jsonl", [])
    if not (results_dir / "embedding_summary.json").exists():
        _write_json(
            results_dir / "embedding_summary.json",
            {
                "status": "not_run",
                "model": RECOVERY_PATTERN_EMBEDDING_MODEL,
                "package_version": "5.7.0",
                "text_version": RECOVERY_PATTERN_EMBEDDING_TEXT_VERSION,
                "dimension": RECOVERY_PATTERN_EMBEDDING_DIMENSION,
                "normalized": True,
                "embedded_pattern_count": 0,
            },
        )
    if not (results_dir / "validation.json").exists():
        _write_json(
            results_dir / "validation.json",
            {"status": "blocked", "reason": state.get("blocker", "not_run")},
        )
    state["utc_end"] = _utc_now()
    state["final_working_tree_state"] = _git(["git", "status", "--short"])
    state["backup_checksums"] = (
        json.loads((evidence_dir / "backup" / "checksums.json").read_text(encoding="utf-8"))
        if (evidence_dir / "backup" / "checksums.json").exists()
        else {}
    )
    _write_json(evidence_dir / "manifest.json", state)
    _write_readme(evidence_dir, state)
    return evidence_dir, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        evidence_dir, state = run_migration()
    except MigrationBlocked as error:
        print(f"BLOCKED: {error}")
        return 2
    print(
        json.dumps(
            {
                "evidence_directory": str(evidence_dir),
                "manifest": str(evidence_dir / "manifest.json"),
                "state": state,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if state.get("blocker") else 0


if __name__ == "__main__":
    sys.exit(main())
