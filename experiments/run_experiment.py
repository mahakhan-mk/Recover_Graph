"""Entrypoint for reproducible Graph Swarm experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    print(f"Loaded experiment: {config['experiment_id']}")
    print("Implementation intentionally starts with the GS-E001 vertical slice.")


if __name__ == "__main__":
    main()
