"""Build a deterministic, mutation-aware SWE-smith candidate pool."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_NAME = "SWE-bench/SWE-smith-py"
DATASET_SPLIT = "train"
SEED = 42
MAX_PER_MUTATION_FAMILY = 15
MAX_TOTAL_CANDIDATES = 105

OUTPUT_PATH = Path("benchmark/derived/swesmith_candidates.jsonl")
SUMMARY_PATH = Path("benchmark/derived/swesmith_candidate_summary.txt")

ALLOWED_MUTATION_FAMILIES: tuple[str, ...] = (
    "func_pm_op_swap",
    "func_pm_ctrl_invert_if",
    "func_pm_remove_assign",
    "func_pm_remove_cond",
    "func_pm_ctrl_shuffle",
    "func_pm_op_change",
    "func_pm_remove_loop",
)

# Mutation names are separated from the rest of an instance ID by punctuation
# or by the double-underscore convention used by SWE-smith IDs.
MUTATION_FAMILY_PATTERN = re.compile(
    r"(?:(?<![A-Za-z0-9_])|(?<=__))"
    rf"({'|'.join(re.escape(family) for family in ALLOWED_MUTATION_FAMILIES)})"
    r"(?:(?=$)|(?=[^A-Za-z0-9_])|(?=__))",
    re.IGNORECASE,
)
EXCLUDED_MUTATION_PATTERN = re.compile(
    r"(?:(?<![A-Za-z0-9_])|(?<=__))"
    r"(?:lm_rewrite|combine_file|combine_module|func_basic|pr_[A-Za-z0-9_]+)"
    r"(?:(?=$)|(?=[^A-Za-z0-9_])|(?=__))",
    re.IGNORECASE,
)

OPERATION_HINT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "parameter_or_argument",
        re.compile(
            r"\b(?:parameter|parameters|argument|arguments|keyword argument|keyword arguments)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "configuration",
        re.compile(
            r"\b(?:configuration|config|environment variable|environment variables|"
            r"setting|settings)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dependency_or_import",
        re.compile(
            r"\b(?:dependency|dependencies|import|imports|importerror|modulenotfounderror|package|packages)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "command_or_cli",
        re.compile(
            r"\b(?:command|commands|cli|command line|subprocess|shell|invocation|exit code)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "test_or_assertion",
        re.compile(
            r"\b(?:test|tests|testing|assert|assertion|assertionerror)\b",
            re.IGNORECASE,
        ),
    ),
)


def normalize_text(value: Any) -> str:
    """Convert a dataset value to trimmed text without leaking nulls."""
    if value is None:
        return ""
    return str(value).strip()


def count_tests(value: Any) -> int:
    """Count a SWE-smith test-list field conservatively."""
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def mutation_family_from_instance_id(instance_id: str) -> str | None:
    """Return an allowed mutation family encoded in an instance ID."""
    if EXCLUDED_MUTATION_PATTERN.search(instance_id):
        return None

    match = MUTATION_FAMILY_PATTERN.search(instance_id)
    if match is None:
        return None
    return match.group(1).casefold()


def infer_operation_hint(problem_statement: str) -> str | None:
    """Infer one conservative researcher-only hint using bounded regexes."""
    for hint, pattern in OPERATION_HINT_PATTERNS:
        if pattern.search(problem_statement):
            return hint
    return None


def compact_record(item: Mapping[str, Any], mutation_family: str) -> dict[str, object]:
    """Build the intentionally small candidate record."""
    problem_statement = normalize_text(item.get("problem_statement"))
    return {
        "source": DATASET_NAME,
        "instance_id": normalize_text(item.get("instance_id")),
        "repo": normalize_text(item.get("repo")),
        "image_name": normalize_text(item.get("image_name")),
        "mutation_family": mutation_family,
        "operation_hint": infer_operation_hint(problem_statement),
        "problem_statement": problem_statement,
        "fail_to_pass_count": count_tests(item.get("FAIL_TO_PASS")),
        "pass_to_pass_count": count_tests(item.get("PASS_TO_PASS")),
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Overwrite the JSONL output with one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    records: list[dict[str, object]],
    dataset_rows_scanned: int,
) -> None:
    """Write candidate counts and methodological safeguards for researchers."""
    path.parent.mkdir(parents=True, exist_ok=True)

    family_counts: Counter[str] = Counter(
        str(record["mutation_family"]) for record in records
    )
    hint_counts: Counter[str] = Counter(
        record["operation_hint"] if isinstance(record["operation_hint"], str) else "null"
        for record in records
    )
    repository_counts: Counter[str] = Counter(str(record["repo"]) for record in records)

    lines = [
        "Graph Swarm SWE-smith Mutation-Aware Candidate Pool",
        "=" * 52,
        f"Dataset name: {DATASET_NAME}",
        f"Dataset split: {DATASET_SPLIT}",
        f"Total dataset rows scanned: {dataset_rows_scanned}",
        f"Candidate rows written: {len(records)}",
        "FAIL_TO_PASS filter: count >= 1 and count <= 30",
        "PASS_TO_PASS minimum: count >= 1",
        "",
        "Counts per mutation family",
        "----------------------------",
    ]

    for family in ALLOWED_MUTATION_FAMILIES:
        lines.append(f"{family}: {family_counts[family]}")

    lines.extend(
        [
            "",
            "Counts of operation hints",
            "-------------------------",
        ]
    )
    for hint, _ in OPERATION_HINT_PATTERNS:
        lines.append(f"{hint}: {hint_counts[hint]}")
    lines.append(f"null: {hint_counts['null']}")

    lines.extend(
        [
            "",
            "Top repositories",
            "----------------",
        ]
    )
    if repository_counts:
        for repo, count in repository_counts.most_common(20):
            lines.append(f"{repo}: {count}")
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "Methodological notes",
            "--------------------",
            (
                "mutation_family and operation_hint are researcher-only metadata and must not "
                "be shown to the agent."
            ),
            "Final recurrence families require manual validation.",
            (
                "Candidates are deterministically sampled from the shuffled train split with "
                "seed 42, capped at 15 per allowed mutation family and 105 total."
            ),
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Extract and write the mutation-aware candidate pool."""
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT).shuffle(seed=SEED)
    dataset_rows_scanned = len(dataset)
    family_records: dict[str, list[dict[str, object]]] = {
        family: [] for family in ALLOWED_MUTATION_FAMILIES
    }

    for item in dataset:
        instance_id = normalize_text(item.get("instance_id"))
        mutation_family = mutation_family_from_instance_id(instance_id)
        if mutation_family is None:
            continue
        if len(family_records[mutation_family]) >= MAX_PER_MUTATION_FAMILY:
            continue

        fail_to_pass_count = count_tests(item.get("FAIL_TO_PASS"))
        pass_to_pass_count = count_tests(item.get("PASS_TO_PASS"))
        if not 1 <= fail_to_pass_count <= 30:
            continue
        if pass_to_pass_count < 1:
            continue

        family_records[mutation_family].append(compact_record(item, mutation_family))

    records = [
        record
        for family in ALLOWED_MUTATION_FAMILIES
        for record in family_records[family]
    ][:MAX_TOTAL_CANDIDATES]

    write_jsonl(OUTPUT_PATH, records)
    write_summary(SUMMARY_PATH, records, dataset_rows_scanned)

    print(f"Wrote {len(records)} candidate tasks to: {OUTPUT_PATH}")
    print(f"Wrote summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
