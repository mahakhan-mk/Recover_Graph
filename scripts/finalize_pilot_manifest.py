"""Freeze approved manual annotations and create the leakage-safe pilot manifest."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ANNOTATION_PATH = Path("benchmark/annotations/recurrence_validation.csv")
MANIFEST_PATH = Path("benchmark/manifests/pilot.jsonl")
SOURCE = "SWE-bench/SWE-smith-py"

ANNOTATION_FIELDS: tuple[str, ...] = (
    "review_id",
    "instance_id",
    "mutation_family",
    "repository",
    "image_name",
    "fail_to_pass_count",
    "pass_to_pass_count",
    "problem_statement",
    "keep",
    "transferable",
    "recovery_pattern",
    "transfer_rationale",
    "chronological_index",
    "notes",
)

MANIFEST_FIELDS: tuple[str, ...] = (
    "task_id",
    "review_id",
    "source",
    "instance_id",
    "repository",
    "image_name",
    "family_id",
    "family_label",
    "occurrence_index",
    "chronological_index",
    "split",
    "validated",
)


@dataclass(frozen=True)
class FamilySpec:
    """Frozen recurrence-family metadata approved for the pilot."""

    family_id: str
    family_label: str
    source_mutation: str
    recovery_pattern: str
    transfer_rationale: str


@dataclass(frozen=True)
class ApprovedTask:
    """One approved review row and its frozen chronological placement."""

    review_id: str
    family_id: str
    occurrence_index: int
    chronological_index: int


FAMILY_SPECS: dict[str, FamilySpec] = {
    "GS-F001": FamilySpec(
        family_id="GS-F001",
        family_label="conditional_polarity_error",
        source_mutation="func_pm_ctrl_invert_if",
        recovery_pattern=(
            "When behavior is reversed around a boolean branch or guard, inspect condition "
            "polarity and restore the intended true/false branch mapping before changing "
            "downstream logic."
        ),
        transfer_rationale=(
            "Across repositories the failure is produced by an inverted decision boundary. "
            "A previously verified recovery that checks branch polarity is plausibly reusable "
            "without depending on repository-specific names or implementation details."
        ),
    ),
    "GS-F002": FamilySpec(
        family_id="GS-F002",
        family_label="missing_initialization_or_assignment",
        source_mutation="func_pm_remove_assign",
        recovery_pattern=(
            "When a required value or state is undefined or missing, inspect its dataflow "
            "for a removed initialization or assignment and restore the required assignment "
            "before first use."
        ),
        transfer_rationale=(
            "Across repositories the failure occurs because required state is no longer "
            "assigned before later use. The reusable recovery is to trace initialization and "
            "dataflow and restore the missing assignment."
        ),
    ),
    "GS-F003": FamilySpec(
        family_id="GS-F003",
        family_label="prerequisite_ordering_error",
        source_mutation="func_pm_ctrl_shuffle",
        recovery_pattern=(
            "When execution references values, state, or checks before they are established, "
            "restore dependency order so prerequisite computation or validation occurs before "
            "use or return."
        ),
        transfer_rationale=(
            "Across repositories statements are reordered so a prerequisite occurs after the "
            "operation that depends on it. A recovery based on prerequisite-before-use ordering "
            "can plausibly transfer across implementations."
        ),
    ),
    "GS-F004": FamilySpec(
        family_id="GS-F004",
        family_label="operator_semantics_error",
        source_mutation="func_pm_op_change",
        recovery_pattern=(
            "When arithmetic, comparison, string, or other operator semantics become incorrect "
            "after a code change, verify the operator against the intended behavior and restore "
            "the semantically correct operator."
        ),
        transfer_rationale=(
            "Across repositories the mutation replaces an operator with one that changes "
            "program semantics. A previously verified recovery that inspects operator intent "
            "and failing behavior is plausibly reusable."
        ),
    ),
    "GS-F005": FamilySpec(
        family_id="GS-F005",
        family_label="missing_iteration",
        source_mutation="func_pm_remove_loop",
        recovery_pattern=(
            "When a collection or candidate sequence is only partially processed or produces "
            "empty/incomplete output, inspect for removed iteration and restore traversal over "
            "all required elements."
        ),
        transfer_rationale=(
            "Across repositories required iteration has been removed, producing incomplete "
            "processing. A recovery that checks collection traversal and restores iteration is "
            "plausibly reusable across implementations."
        ),
    ),
}

APPROVED_TASKS: tuple[ApprovedTask, ...] = (
    ApprovedTask("GS-R007", "GS-F001", 1, 1),
    ApprovedTask("GS-R012", "GS-F002", 1, 2),
    ApprovedTask("GS-R021", "GS-F003", 1, 3),
    ApprovedTask("GS-R026", "GS-F004", 1, 4),
    ApprovedTask("GS-R032", "GS-F005", 1, 5),
    ApprovedTask("GS-R009", "GS-F001", 2, 6),
    ApprovedTask("GS-R014", "GS-F002", 2, 7),
    ApprovedTask("GS-R022", "GS-F003", 2, 8),
    ApprovedTask("GS-R027", "GS-F004", 2, 9),
    ApprovedTask("GS-R033", "GS-F005", 2, 10),
    ApprovedTask("GS-R010", "GS-F001", 3, 11),
    ApprovedTask("GS-R015", "GS-F002", 3, 12),
    ApprovedTask("GS-R023", "GS-F003", 3, 13),
    ApprovedTask("GS-R030", "GS-F004", 3, 14),
    ApprovedTask("GS-R034", "GS-F005", 3, 15),
)


def read_annotations(path: Path) -> list[dict[str, str]]:
    """Read and validate the complete manual annotation table."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != ANNOTATION_FIELDS:
            raise ValueError(
                f"Unexpected annotation CSV columns: {fieldnames}; "
                f"expected: {ANNOTATION_FIELDS}"
            )
        rows = list(reader)

    if len(rows) != 35:
        raise RuntimeError(f"Expected exactly 35 annotation rows, found {len(rows)}")
    if any(any(value is None for value in row.values()) for row in rows):
        raise ValueError("Annotation CSV contains a malformed row")

    review_ids = [row["review_id"] for row in rows]
    duplicate_review_ids = [
        review_id
        for review_id, count in Counter(review_ids).items()
        if count > 1
    ]
    if duplicate_review_ids:
        raise RuntimeError(f"Duplicate review_id values found: {duplicate_review_ids}")

    return rows


def validate_approved_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Validate the frozen review IDs, source mutations, and instance uniqueness."""
    rows_by_review_id = {row["review_id"]: row for row in rows}
    approved_rows: dict[str, dict[str, str]] = {}

    for approved_task in APPROVED_TASKS:
        row = rows_by_review_id.get(approved_task.review_id)
        if row is None:
            raise RuntimeError(f"Approved review row is missing: {approved_task.review_id}")
        spec = FAMILY_SPECS[approved_task.family_id]
        if row["mutation_family"] != spec.source_mutation:
            raise RuntimeError(
                f"Review {approved_task.review_id} has mutation family "
                f"{row['mutation_family']!r}; expected {spec.source_mutation!r}"
            )
        approved_rows[approved_task.review_id] = row

    approved_instance_ids = [row["instance_id"] for row in approved_rows.values()]
    if len(set(approved_instance_ids)) != len(approved_instance_ids):
        raise RuntimeError("Approved rows contain duplicate instance_id values")

    return approved_rows


def update_annotations(
    rows: list[dict[str, str]],
    approved_rows: dict[str, dict[str, str]],
) -> None:
    """Apply only the frozen manual-review decisions to the annotation rows."""
    approved_by_review_id = {task.review_id: task for task in APPROVED_TASKS}
    for row in rows:
        approved_task = approved_by_review_id.get(row["review_id"])
        if approved_task is None:
            row["keep"] = "no"
            row["transferable"] = ""
            row["recovery_pattern"] = ""
            row["transfer_rationale"] = ""
            row["chronological_index"] = ""
            row["notes"] = "Not selected for the frozen 15-task pilot."
            continue

        if row is not approved_rows[approved_task.review_id]:
            raise RuntimeError(f"Approved row identity mismatch: {approved_task.review_id}")
        spec = FAMILY_SPECS[approved_task.family_id]
        row["keep"] = "yes"
        row["transferable"] = "yes"
        row["recovery_pattern"] = spec.recovery_pattern
        row["transfer_rationale"] = spec.transfer_rationale
        row["chronological_index"] = str(approved_task.chronological_index)
        row["notes"] = (
            f"Selected for pilot as {spec.family_id} ({spec.family_label}), "
            f"occurrence {approved_task.occurrence_index} of 3."
        )


def build_manifest_records(
    approved_rows: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    """Build the exact leakage-safe manifest in frozen chronological order."""
    records: list[dict[str, object]] = []
    for task_number, approved_task in enumerate(APPROVED_TASKS, start=1):
        row = approved_rows[approved_task.review_id]
        spec = FAMILY_SPECS[approved_task.family_id]
        records.append(
            {
                "task_id": f"GS-T{task_number:03d}",
                "review_id": approved_task.review_id,
                "source": SOURCE,
                "instance_id": row["instance_id"],
                "repository": row["repository"],
                "image_name": row["image_name"],
                "family_id": spec.family_id,
                "family_label": spec.family_label,
                "occurrence_index": approved_task.occurrence_index,
                "chronological_index": approved_task.chronological_index,
                "split": "pilot",
                "validated": True,
            }
        )
    return records


def write_annotations(path: Path, rows: list[dict[str, str]]) -> None:
    """Overwrite the approved annotation table with frozen manual decisions."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    """Overwrite the pilot manifest as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Validate frozen approvals, update annotations, and write the pilot manifest."""
    rows = read_annotations(ANNOTATION_PATH)
    approved_rows = validate_approved_rows(rows)
    update_annotations(rows, approved_rows)
    manifest_records = build_manifest_records(approved_rows)
    if len(manifest_records) != 15:
        raise RuntimeError(f"Expected exactly 15 manifest records, found {len(manifest_records)}")

    write_annotations(ANNOTATION_PATH, rows)
    write_manifest(MANIFEST_PATH, manifest_records)

    family_counts = Counter(str(record["family_id"]) for record in manifest_records)
    occurrence_counts = Counter(int(record["occurrence_index"]) for record in manifest_records)
    unique_repositories = len({str(record["repository"]) for record in manifest_records})

    print(f"Validation rows read: {len(rows)}")
    print(f"Approved rows found: {len(approved_rows)}")
    print(f"Rows marked keep=yes: {sum(row['keep'] == 'yes' for row in rows)}")
    print(f"Rows marked keep=no: {sum(row['keep'] == 'no' for row in rows)}")
    print(f"Pilot manifest records written: {len(manifest_records)}")
    print(f"Count per final family: {dict(family_counts)}")
    print(f"Occurrence counts: {dict(occurrence_counts)}")
    print(f"Unique repositories: {unique_repositories}")
    print("Chronological index range: 1-15")
    print(f"Annotation CSV path: {ANNOTATION_PATH}")
    print(f"Pilot manifest path: {MANIFEST_PATH}")
    print("Ruff result: run separately")


if __name__ == "__main__":
    main()
