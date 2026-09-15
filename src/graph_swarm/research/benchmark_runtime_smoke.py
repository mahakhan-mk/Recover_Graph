"""Validation helpers and CLI for the zero-provider benchmark runtime smoke."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from graph_swarm.domain.action import ActionResult


class RuntimeSmokeValidationError(RuntimeError):
    """Raised when an agent-tool runtime smoke result is not trustworthy."""


def validate_run_command_evidence(
    result: ActionResult,
    *,
    sentinel: str,
    source_prefix: str = "/workspace/src/",
) -> dict[str, Any]:
    """Require a successful mounted-workspace and source-precedence check."""
    output = result.output or ""
    if not result.success:
        raise RuntimeSmokeValidationError(
            f"run_command failed: exit_code={result.exit_code}, error={result.error!r}"
        )
    if sentinel not in output:
        raise RuntimeSmokeValidationError("run_command could not read the host sentinel")

    source_marker = "JINJA2_SOURCE="
    source_location = next(
        (
            line[len(source_marker) :].strip()
            for line in output.splitlines()
            if line.startswith(source_marker)
        ),
        "",
    )
    if not source_location:
        raise RuntimeSmokeValidationError(
            "run_command did not report the Jinja import location"
        )
    if not source_location.startswith(source_prefix):
        raise RuntimeSmokeValidationError(
            f"Jinja was not imported from the workspace source: {source_location}"
        )
    return {
        "sentinel": sentinel,
        "sentinel_visible": True,
        "import": "jinja2",
        "import_location": source_location,
        "source_first": True,
        "output": output,
    }


def validate_mutated_test_failure(
    result: ActionResult,
    expected_selectors: Sequence[str],
) -> dict[str, Any]:
    """Accept the frozen mutation failure while rejecting infrastructure errors."""
    output = "\n".join(value for value in (result.output, result.error) if value)
    if result.success:
        raise RuntimeSmokeValidationError("mutated run_tests unexpectedly passed")
    if result.exit_code is None:
        raise RuntimeSmokeValidationError("run_tests did not produce a test exit code")
    if result.exit_code in (2, 3, 4, 5):
        raise RuntimeSmokeValidationError(
            f"run_tests reported an infrastructure/collection exit code: {result.exit_code}"
        )
    collection_markers = (
        "ERROR collecting",
        "ImportError while loading conftest",
        "ModuleNotFoundError:",
        "No module named",
    )
    if any(marker in output for marker in collection_markers):
        raise RuntimeSmokeValidationError(
            "run_tests reported an infrastructure/collection failure"
        )
    matched_selectors = tuple(selector for selector in expected_selectors if selector in output)
    if not matched_selectors:
        raise RuntimeSmokeValidationError(
            "run_tests failed without identifying a frozen FAIL_TO_PASS selector"
        )
    return {
        "expected_failure": True,
        "matched_selectors": matched_selectors,
        "output": result.output or "",
        "error": result.error,
    }


def write_runtime_smoke_evidence(
    artifact_root: Path,
    payload: dict[str, Any],
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Write one exclusive, timestamped runtime-smoke evidence document."""
    result_root = artifact_root / "GS-E003" / "sprint3b"
    result_root.mkdir(parents=True, exist_ok=True)
    moment = timestamp or datetime.now(UTC)
    result_path = result_root / (
        "runtime-smoke-"
        + moment.strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid4().hex
        + ".json"
    )
    with result_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the zero-provider Track B prepared-container agent runtime smoke "
            "(frozen to GS-T007)"
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-root", type=Path, default=Path("benchmark/workspaces"))
    parser.add_argument(
        "--execution-root",
        type=Path,
        default=Path("research/evidence/workspaces"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("research/evidence/results"),
    )
    parser.add_argument(
        "--task-id",
        default="GS-T007",
        help="Smoke task; Sprint 3C-B accepts only GS-T007 (the default).",
    )
    args = parser.parse_args()

    from experiments.sprint3 import BenchmarkPreflightError, run_runtime_smoke

    project_root = args.project_root.resolve()
    try:
        result_path = run_runtime_smoke(
            project_root=project_root,
            baseline_root=args.baseline_root.resolve(),
            execution_root=args.execution_root.resolve(),
            artifact_root=(
                args.artifact_root.resolve()
                if args.artifact_root.is_absolute()
                else (project_root / args.artifact_root).resolve()
            ),
            task_id=args.task_id,
        )
    except BenchmarkPreflightError as error:
        parser.error(str(error))
    print(f"AGENT_RUNTIME_SMOKE_READY {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
