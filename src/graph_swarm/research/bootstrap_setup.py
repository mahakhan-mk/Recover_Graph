"""Explicitly clone and pin missing benchmark repositories.

This command is intentionally separate from :mod:`verify_setup`. It never
overwrites an existing directory, dirty repository, or mismatched checkout.
It performs no Docker, Neo4j, provider, or experiment work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, cast


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _manifest(root: Path) -> list[dict[str, Any]]:
    raw = cast(
        dict[str, Any],
        json.loads(
            (root / "benchmark/manifests/repositories.json").read_text(encoding="utf-8")
        ),
    )
    repositories = raw.get("repositories")
    if not isinstance(repositories, list):
        raise ValueError("repository manifest must contain a repositories list")
    repositories = cast(list[Any], repositories)
    return [
        cast(dict[str, Any], record)
        for record in repositories
        if isinstance(record, dict)
    ]


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _git(path: Path, *arguments: str) -> tuple[bool, str]:
    completed = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0, (completed.stdout or completed.stderr).strip()


def _verify_existing(path: Path, expected: str, logical_name: str) -> bool:
    if not path.is_dir():
        return False
    ok, observed = _git(path, "rev-parse", "HEAD")
    if not ok:
        print(f"REFUSED {logical_name}: existing path is not a Git repository")
        return True
    dirty_ok, dirty = _git(path, "status", "--porcelain")
    if observed.lower() != expected.lower():
        print(f"REFUSED {logical_name}: existing checkout is {observed}, expected {expected}")
    elif not dirty_ok or dirty:
        print(f"REFUSED {logical_name}: existing repository is dirty")
    else:
        print(f"OK {logical_name}: already pinned at {expected}")
    return True


def bootstrap(root: Path, *, clone_missing: bool) -> int:
    root = root.expanduser().resolve()
    failures = 0
    for record in _manifest(root):
        logical_name = str(record["logical_name"])
        path = root / str(record["path"])
        expected = str(record["commit"])
        if not _inside(root, path):
            print(f"REFUSED {logical_name}: manifest path escapes project root")
            failures += 1
            continue
        if _verify_existing(path, expected, logical_name):
            ok, observed = _git(path, "rev-parse", "HEAD")
            dirty_ok, dirty = _git(path, "status", "--porcelain")
            if not ok or observed.lower() != expected.lower() or not dirty_ok or dirty:
                failures += 1
            continue
        if not clone_missing:
            print(f"MISSING {logical_name}: {path} (rerun with --clone to acquire it)")
            failures += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"CLONE {logical_name}: {record['clone_url']}")
        cloned = subprocess.run(
            ["git", "clone", "--no-checkout", str(record["clone_url"]), str(path)],
            check=False,
        )
        if cloned.returncode != 0:
            failures += 1
            continue
        checked_out = subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", expected],
            check=False,
        )
        if checked_out.returncode != 0:
            failures += 1
            continue
        print(f"OK {logical_name}: cloned and pinned at {expected}")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_project_root())
    parser.add_argument(
        "--clone",
        action="store_true",
        help="clone only missing repositories; never overwrite existing paths",
    )
    args = parser.parse_args()
    return bootstrap(args.root, clone_missing=args.clone)


if __name__ == "__main__":
    raise SystemExit(main())
