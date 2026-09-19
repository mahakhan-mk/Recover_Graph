"""Run safe, read-only Graph Swarm teammate setup checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

REQUIRED_ENVIRONMENT = (
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
)
REQUIRED_MODULES = {
    "packaging": "packaging",
    "datasets": "datasets",
    "neo4j": "neo4j",
    "pydantic": "pydantic",
    "pydantic_ai": "pydantic_ai",
    "pydantic_ai_harness": "pydantic_ai_harness",
    "pydantic_evals": "pydantic_evals",
    "pydantic_settings": "pydantic_settings",
    "truststore": "truststore",
    "yaml": "yaml",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected an object in {path}")
    return cast(dict[str, Any], raw)


def _dotenv_keys(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _available_environment_keys(root: Path) -> set[str]:
    dotenv = _dotenv_keys(root / ".env")
    return {
        key
        for key in REQUIRED_ENVIRONMENT + ("MODEL_PROVIDER",)
        if os.environ.get(key, "").strip() or dotenv.get(key, "").strip()
    }


def _run_git(path: Path, *arguments: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    output = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, output


def _check_repositories(root: Path) -> list[tuple[bool, str]]:
    manifest = _load_json(root / "benchmark/manifests/repositories.json")
    results: list[tuple[bool, str]] = []
    repositories = manifest.get("repositories", [])
    if not isinstance(repositories, list):
        return [(False, "repository manifest does not contain a list")]
    repositories = cast(list[Any], repositories)
    for raw_repository in repositories:
        if not isinstance(raw_repository, dict):
            results.append((False, "repository manifest contains a non-object entry"))
            continue
        repository = cast(dict[str, Any], raw_repository)
        path = root / str(repository["path"])
        expected = str(repository["commit"])
        if not path.is_dir():
            results.append((False, f"{repository['logical_name']}: missing ({path})"))
            continue
        ok, observed = _run_git(path, "rev-parse", "HEAD")
        if not ok:
            results.append((False, f"{repository['logical_name']}: not a readable Git repository"))
            continue
        dirty_ok, dirty = _run_git(path, "status", "--porcelain")
        dirty_state = "clean" if dirty_ok and not dirty else "dirty"
        matches = observed.lower() == expected.lower()
        results.append(
            (
                matches and dirty_state == "clean",
                f"{repository['logical_name']}: expected {expected[:12]}, "
                f"observed {observed[:12]} ({dirty_state})",
            )
        )
    return results


def _check_docker_metadata(root: Path) -> list[tuple[bool, str]]:
    manifest = _load_json(root / "configs/research/docker_environments.json")
    results: list[tuple[bool, str]] = []
    environments = manifest.get("environments", [])
    if not isinstance(environments, list):
        return [(False, "Docker manifest does not contain a list")]
    environments = cast(list[Any], environments)
    for raw_environment in environments:
        if not isinstance(raw_environment, dict):
            results.append((False, "Docker manifest contains a non-object entry"))
            continue
        environment = cast(dict[str, Any], raw_environment)
        marker = root / str(environment["marker"])
        if not marker.is_file():
            results.append((False, f"{environment['task_id']}: marker missing ({marker})"))
            continue
        try:
            metadata = _load_json(marker)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            results.append((False, f"{environment['task_id']}: marker is unreadable"))
            continue
        expected = {
            "task_id": environment["task_id"],
            "environment_fingerprint": environment["environment_fingerprint"],
            "container_image": environment["prepared_image"],
            "base_container_image": environment["upstream_image"],
            "base_image_digest": (
                f"{environment['upstream_image']}@{environment['upstream_digest']}"
            ),
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        results.append(
            (
                not mismatches and metadata.get("validated") is True,
                f"{environment['task_id']}: {environment['prepared_image']}"
                + (
                    " (validated marker)"
                    if not mismatches
                    else f" (metadata mismatch: {', '.join(mismatches)})"
                ),
            )
        )
    return results


def _check_research_settings(root: Path) -> tuple[bool, str]:
    try:
        import yaml
    except ImportError as error:
        return False, f"Gate B1 configuration unreadable: {error}"
    try:
        configuration = yaml.safe_load(
            (root / "configs/experiments/gate_b1.yaml").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        return False, f"Gate B1 configuration unreadable: {error}"
    if not isinstance(configuration, dict):
        return False, "Gate B1 configuration is not a mapping"
    configuration = cast(dict[str, Any], configuration)
    limits = configuration.get("limits", {})
    pacing = configuration.get("pacing", {})
    if not isinstance(limits, dict):
        limits = {}
    if not isinstance(pacing, dict):
        pacing = {}
    limits = cast(dict[str, Any], limits)
    pacing = cast(dict[str, Any], pacing)
    expected = {
        "provider": "openrouter",
        "temperature": 0,
        "max_actions": 20,
        "max_requests": 24,
        "tool_retries": 3,
        "agent_timeout_seconds": 300,
        "model_request_timeout_seconds": 300,
        "min_request_start_interval_seconds": 5.0,
        "nominal_requests_per_minute": 12,
        "t_enabled": False,
    }
    observed: dict[str, Any] = {
        "provider": configuration.get("provider"),
        "temperature": configuration.get("temperature"),
        "max_actions": limits.get("max_actions"),
        "max_requests": limits.get("max_requests"),
        "tool_retries": limits.get("tool_retries"),
        "agent_timeout_seconds": limits.get("agent_timeout_seconds"),
        "model_request_timeout_seconds": limits.get("model_request_timeout_seconds"),
        "min_request_start_interval_seconds": pacing.get("min_request_start_interval_seconds"),
        "nominal_requests_per_minute": pacing.get("nominal_requests_per_minute"),
        "t_enabled": configuration.get("t_enabled"),
    }
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    task_ids = configuration.get("task_ids", [])
    task_count = len(cast(list[Any], task_ids)) if isinstance(task_ids, list) else 0
    return (
        not mismatches and task_count == 10,
        "provider=openrouter, model=OPENROUTER_MODEL, limits=20/24/3/300/300, "
        f"pacing=5s/{expected['nominal_requests_per_minute']}RPM, tasks={task_count}"
        + ("" if not mismatches else f"; mismatches={', '.join(mismatches)}"),
    )


def check_setup(root: Path) -> tuple[bool, list[str]]:
    """Return readiness and safe diagnostic lines without changing local state."""
    lines: list[str] = []
    ready = True

    supported = (3, 12) <= sys.version_info[:2] < (3, 14)
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    lines.append(
        f"Python: {'OK' if supported else 'FAIL'} {version} (requires 3.12-3.13)"
    )
    ready &= supported

    missing_modules = [
        label
        for label, module in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    lines.append(
        "Project imports: "
        + ("OK" if not missing_modules else f"FAIL missing {', '.join(missing_modules)}")
    )
    ready &= not missing_modules

    available_keys = _available_environment_keys(root)
    missing_environment = [key for key in REQUIRED_ENVIRONMENT if key not in available_keys]
    provider = os.environ.get("MODEL_PROVIDER", "").strip().lower() or _dotenv_keys(
        root / ".env"
    ).get("MODEL_PROVIDER", "openrouter").strip().lower()
    lines.append(
        "Environment: "
        + (
            "OK required keys present"
            if not missing_environment
            else f"FAIL missing {', '.join(missing_environment)}"
        )
        + "; secret values hidden"
    )
    ready &= not missing_environment
    provider_ok = provider == "openrouter"
    lines.append(f"Model provider: {'OK' if provider_ok else 'FAIL'} {provider or '<unset>'}")
    ready &= provider_ok

    try:
        repository_results = _check_repositories(root)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        repository_results = [(False, f"manifest unreadable: {error}")]
    lines.append("Benchmark repositories:")
    for result, detail in repository_results:
        lines.append(f"  {'OK' if result else 'FAIL'} {detail}")
        ready &= result

    try:
        docker_results = _check_docker_metadata(root)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        docker_results = [(False, f"manifest unreadable: {error}")]
    lines.append("Docker metadata (read-only marker check; Docker not invoked):")
    for result, detail in docker_results:
        lines.append(f"  {'OK' if result else 'FAIL'} {detail}")
        ready &= result

    research_ok, research_detail = _check_research_settings(root)
    lines.append(f"Research controls: {'OK' if research_ok else 'FAIL'} {research_detail}")
    ready &= research_ok
    neo4j_keys = all(key in available_keys for key in REQUIRED_ENVIRONMENT[:4])
    lines.append(
        f"Neo4j configuration: {'OK' if neo4j_keys else 'FAIL'} "
        "required keys present; values hidden"
    )
    ready &= neo4j_keys
    lines.append(f"Readiness summary: {'READY' if ready else 'NOT READY'}")
    return ready, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_project_root())
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    ready, lines = check_setup(root)
    print("\n".join(lines))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
