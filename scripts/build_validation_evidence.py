"""Extract researcher-only SWE-smith mutation evidence for manual Graph Swarm
recurrence validation. The output must never be exposed to the experimental agent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_NAME = "SWE-bench/SWE-smith-py"
DATASET_SPLIT = "train"
SHORTLIST_PATH = Path("benchmark/annotations/recurrence_validation.csv")
OUTPUT_PATH = Path("benchmark/annotations/recurrence_evidence.jsonl")

SHORTLIST_FIELDS = {
    "review_id",
    "instance_id",
    "mutation_family",
    "repository",
    "image_name",
}
REQUIRED_DATASET_FIELDS = {
    "instance_id",
    "repo",
    "problem_statement",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}
OPTIONAL_DATASET_FIELDS = ("test_patch", "base_commit", "version")


def read_shortlist(path: Path) -> list[dict[str, str]]:
    """Read the shortlist metadata and reject duplicate instance IDs."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_fields = SHORTLIST_FIELDS - fieldnames
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Shortlist is missing required fields: {missing}")

        rows = list(reader)

    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 shortlist rows, found {len(rows)}")

    instance_ids = [row["instance_id"] for row in rows]
    duplicate_ids = sorted(
        instance_id
        for instance_id in set(instance_ids)
        if instance_ids.count(instance_id) > 1
    )
    if duplicate_ids:
        raise RuntimeError(f"Duplicate shortlist instance IDs found: {duplicate_ids}")

    return rows


def require_dataset_columns(columns: list[str]) -> None:
    """Validate the dataset schema, including the unverified mutation patch."""
    missing_fields = REQUIRED_DATASET_FIELDS - set(columns)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise RuntimeError(
            f"Dataset is missing required columns: {missing}. "
            f"Available dataset columns: {columns}"
        )
    if "patch" not in columns:
        raise RuntimeError(
            "Dataset does not contain the required 'patch' column for mutation evidence. "
            f"Available dataset columns: {columns}"
        )


def build_evidence_record(
    shortlist_row: dict[str, str],
    dataset_row: dict[str, Any],
) -> dict[str, Any]:
    """Combine shortlist metadata with the matching dataset evidence."""
    return {
        "review_id": shortlist_row["review_id"],
        "instance_id": shortlist_row["instance_id"],
        "mutation_family": shortlist_row["mutation_family"],
        "repository": shortlist_row["repository"],
        "image_name": shortlist_row["image_name"],
        "problem_statement": dataset_row["problem_statement"],
        "dataset_patch": dataset_row["patch"],
        "test_patch": dataset_row.get("test_patch"),
        "base_commit": dataset_row.get("base_commit"),
        "version": dataset_row.get("version"),
        "fail_to_pass": dataset_row["FAIL_TO_PASS"],
        "pass_to_pass": dataset_row["PASS_TO_PASS"],
        "patch_role": "unverified_swesmith_mutation_patch",
    }


def write_evidence(path: Path, records: list[dict[str, Any]]) -> None:
    """Overwrite the researcher-only evidence JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Extract evidence for exactly the shortlisted instance IDs."""
    shortlist_rows = read_shortlist(SHORTLIST_PATH)
    shortlist_by_id = {row["instance_id"]: row for row in shortlist_rows}

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    dataset_columns = list(dataset.column_names)
    print(f"Dataset columns: {dataset_columns}")
    require_dataset_columns(dataset_columns)

    matched_rows: dict[str, dict[str, Any]] = {}
    for dataset_row in dataset:
        instance_id = str(dataset_row["instance_id"])
        if instance_id not in shortlist_by_id:
            continue
        if instance_id in matched_rows:
            raise RuntimeError(f"Duplicate matching dataset instance ID found: {instance_id}")
        matched_rows[instance_id] = dataset_row

    missing_ids = [
        instance_id
        for instance_id in shortlist_by_id
        if instance_id not in matched_rows
    ]
    if missing_ids:
        raise RuntimeError(f"Shortlist instance IDs missing from dataset: {missing_ids}")
    if len(matched_rows) != 35:
        raise RuntimeError(f"Expected 35 extracted records, found {len(matched_rows)}")

    records = [
        build_evidence_record(shortlist_row, matched_rows[shortlist_row["instance_id"]])
        for shortlist_row in shortlist_rows
    ]
    if len(records) != 35:
        raise RuntimeError(f"Expected 35 evidence records, found {len(records)}")
    write_evidence(OUTPUT_PATH, records)

    non_empty_patches = sum(bool(record["dataset_patch"]) for record in records)
    print(f"Shortlist rows read: {len(shortlist_rows)}")
    print(f"Dataset matches found: {len(matched_rows)}")
    print(f"Non-empty patches found: {non_empty_patches}")
    for field in OPTIONAL_DATASET_FIELDS:
        print(f"{field} exists in dataset: {field in dataset_columns}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
