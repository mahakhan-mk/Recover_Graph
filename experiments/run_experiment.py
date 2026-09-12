"""Entrypoint for reproducible Graph Swarm experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_swarm.research.runner import (
    ExperimentRunner,
    load_experiment_configuration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    configuration = load_experiment_configuration(args.config)
    runner = ExperimentRunner(configuration)
    executions = runner.run_all()
    print(f"Loaded experiment: {configuration.config.experiment_id}")
    print(f"Condition: {configuration.config.conditions[0].value}")
    print(f"Runs completed: {len(executions)}")
    for execution in executions:
        print(f"{execution.artifact.run_id}: {execution.artifact_path}")


if __name__ == "__main__":
    main()
