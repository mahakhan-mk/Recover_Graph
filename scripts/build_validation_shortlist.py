"""Prepare SWE-smith candidates for manual recurrence validation; not agent-visible input."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INPUT_PATH = Path("artifacts/dataset_cache/swesmith_candidates.jsonl")
OUTPUT_PATH = Path("benchmark/annotations/recurrence_validation.csv")

MUTATION_FAMILY_ORDER: tuple[str, ...] = (
    "func_pm_op_swap",
    "func_pm_ctrl_invert_if",
    "func_pm_remove_assign",
    "func_pm_remove_cond",
    "func_pm_ctrl_shuffle",
    "func_pm_op_change",
    "func_pm_remove_loop",
)

REQUIRED_INPUT_FIELDS: frozenset[str] = frozenset(
    {
        "source",
        "instance_id",
        "repo",
        "image_name",
        "mutation_family",
        "problem_statement",
        "fail_to_pass_count",
        "pass_to_pass_count",
    }
)

CSV_FIELDS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class Candidate:
    """Candidate fields used by the deterministic shortlist procedure."""

    instance_id: str
    mutation_family: str
    repository: str
    image_name: str
    fail_to_pass_count: int
    pass_to_pass_count: int
    problem_statement: str


def text_value(value: Any) -> str:
    """Return a dataset field as text for sorting and CSV output."""
    if value is None:
        return ""
    return str(value)


def count_value(value: Any) -> int | None:
    """Return an integer test count, or None when the input type is invalid."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def candidate_from_record(record: dict[str, Any]) -> Candidate | None:
    """Convert a JSONL row into a candidate when it passes all shortlist filters."""
    mutation_family = text_value(record["mutation_family"])
    problem_statement = text_value(record["problem_statement"]).strip()
    fail_to_pass_count = count_value(record["fail_to_pass_count"])
    pass_to_pass_count = count_value(record["pass_to_pass_count"])

    if mutation_family not in MUTATION_FAMILY_ORDER:
        return None
    if not problem_statement:
        return None
    if fail_to_pass_count is None or not 1 <= fail_to_pass_count <= 30:
        return None
    if pass_to_pass_count is None or pass_to_pass_count < 1:
        return None

    return Candidate(
        instance_id=text_value(record["instance_id"]),
        mutation_family=mutation_family,
        repository=text_value(record["repo"]),
        image_name=text_value(record["image_name"]),
        fail_to_pass_count=fail_to_pass_count,
        pass_to_pass_count=pass_to_pass_count,
        problem_statement=problem_statement,
    )


def read_candidates(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read JSONL rows and validate the required input schema."""
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Input row {line_number} is not a JSON object")

            missing_fields = REQUIRED_INPUT_FIELDS - record.keys()
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"Input row {line_number} is missing fields: {missing}")
            records.append(record)

    return records, len(records)


def sort_key(candidate: Candidate) -> tuple[int, int, str, str]:
    """Sort by execution cost, then repository and instance ID."""
    return (
        candidate.fail_to_pass_count,
        candidate.pass_to_pass_count,
        candidate.repository,
        candidate.instance_id,
    )


def select_family_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Select five sorted candidates while preferring unused repositories."""
    sorted_candidates = sorted(candidates, key=sort_key)
    selected_indexes: set[int] = set()
    used_repositories: set[str] = set()

    for index, candidate in enumerate(sorted_candidates):
        if candidate.repository in used_repositories:
            continue
        selected_indexes.add(index)
        used_repositories.add(candidate.repository)
        if len(selected_indexes) == 5:
            break

    if len(selected_indexes) < 5:
        for index in range(len(sorted_candidates)):
            if index in selected_indexes:
                continue
            selected_indexes.add(index)
            if len(selected_indexes) == 5:
                break

    if len(selected_indexes) != 5:
        raise RuntimeError(
            f"Unable to select exactly 5 candidates; only {len(selected_indexes)} available"
        )

    return [
        sorted_candidates[index]
        for index in range(len(sorted_candidates))
        if index in selected_indexes
    ]


def build_shortlist(records: list[dict[str, Any]]) -> tuple[list[Candidate], int]:
    """Filter all records and select five candidates for every mutation family."""
    valid_candidates: list[Candidate] = []
    candidates_by_family: dict[str, list[Candidate]] = {
        family: [] for family in MUTATION_FAMILY_ORDER
    }

    for record in records:
        candidate = candidate_from_record(record)
        if candidate is None:
            continue
        valid_candidates.append(candidate)
        candidates_by_family[candidate.mutation_family].append(candidate)

    selected: list[Candidate] = []
    for family in MUTATION_FAMILY_ORDER:
        family_candidates = candidates_by_family[family]
        if len(family_candidates) < 5:
            raise RuntimeError(
                f"Unable to select exactly 5 valid candidates for {family}; "
                f"only {len(family_candidates)} available"
            )
        selected.extend(select_family_candidates(family_candidates))

    return selected, len(valid_candidates)


def csv_row(review_number: int, candidate: Candidate) -> dict[str, object]:
    """Create one CSV row, leaving all manual-review fields blank."""
    return {
        "review_id": f"GS-R{review_number:03d}",
        "instance_id": candidate.instance_id,
        "mutation_family": candidate.mutation_family,
        "repository": candidate.repository,
        "image_name": candidate.image_name,
        "fail_to_pass_count": candidate.fail_to_pass_count,
        "pass_to_pass_count": candidate.pass_to_pass_count,
        "problem_statement": candidate.problem_statement,
        "keep": "",
        "transferable": "",
        "recovery_pattern": "",
        "transfer_rationale": "",
        "chronological_index": "",
        "notes": "",
    }


def write_shortlist(path: Path, candidates: list[Candidate]) -> None:
    """Overwrite the manual-review CSV with the selected candidates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            csv_row(review_number, candidate)
            for review_number, candidate in enumerate(candidates, start=1)
        )


def main() -> None:
    """Build and write the deterministic manual-validation shortlist."""
    records, input_count = read_candidates(INPUT_PATH)
    selected, valid_count = build_shortlist(records)
    if len(selected) != 35:
        raise RuntimeError(f"Expected 35 selected candidates, got {len(selected)}")
    write_shortlist(OUTPUT_PATH, selected)

    family_counts = {family: 0 for family in MUTATION_FAMILY_ORDER}
    for candidate in selected:
        family_counts[candidate.mutation_family] += 1
    unique_repositories = len({candidate.repository for candidate in selected})

    print(f"Input candidates read: {input_count}")
    print(f"Valid candidates considered: {valid_count}")
    print(f"Candidates selected: {len(selected)}")
    print(f"Count per mutation family: {family_counts}")
    print(f"Unique repositories: {unique_repositories}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
