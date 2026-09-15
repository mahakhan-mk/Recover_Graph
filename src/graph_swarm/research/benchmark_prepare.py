"""Command-line entry point for Sprint 3B benchmark environment preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.sprint3 import BenchmarkPreflightError, run_prepare


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or reuse prepared Sprint 3B benchmark environments"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-root", type=Path, default=Path("benchmark/workspaces"))
    parser.add_argument(
        "--execution-root", type=Path, default=Path("research/evidence/workspaces")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("research/evidence/results"))
    args = parser.parse_args()
    try:
        result_path = run_prepare(
            project_root=args.project_root.resolve(),
            baseline_root=args.baseline_root.resolve(),
            execution_root=args.execution_root.resolve(),
            artifact_root=args.artifact_root.resolve(),
        )
    except BenchmarkPreflightError as error:
        parser.error(str(error))
    print(f"BENCHMARK_ENVIRONMENTS_PREPARED {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
