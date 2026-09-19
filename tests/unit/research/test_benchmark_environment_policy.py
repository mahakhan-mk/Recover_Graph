from __future__ import annotations

from pathlib import Path

import pytest

from graph_swarm.research.benchmark_environments import (
    BenchmarkEnvironmentConfigurationError,
    load_benchmark_environment_policy,
)

POLICY_PATH = Path("configs/research/benchmark_environments.toml")


def _policy_with_timeout(tmp_path: Path, value: str) -> Path:
    path = tmp_path / "benchmark_environments.toml"
    path.write_text(
        POLICY_PATH.read_text(encoding="utf-8").replace(
            "docker_build_timeout_seconds = 7200",
            f"docker_build_timeout_seconds = {value}",
        ),
        encoding="utf-8",
    )
    return path


def test_configured_docker_build_timeout_is_loaded() -> None:
    policy = load_benchmark_environment_policy(POLICY_PATH)

    assert policy.docker_build_timeout_seconds == 7200


def test_gs_t014_has_frozen_python_compatibility_assertion() -> None:
    policy = load_benchmark_environment_policy(POLICY_PATH)

    assert policy.task("GS-T014").python_constraint_assertion == ">=3.7,<3.12"


def test_zero_docker_build_timeout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        load_benchmark_environment_policy(_policy_with_timeout(tmp_path, "0"))


def test_negative_docker_build_timeout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        load_benchmark_environment_policy(_policy_with_timeout(tmp_path, "-1"))


def test_non_integer_docker_build_timeout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        load_benchmark_environment_policy(_policy_with_timeout(tmp_path, '"7200"'))
TASK_HEADER = '[tasks."GS-T006"]\n'


def _load_with_t006_extras(tmp_path: Path, declaration: str):
    text = POLICY_PATH.read_text(encoding="utf-8")
    policy_path = tmp_path / "benchmark_environments.toml"
    policy_path.write_text(
        text.replace(TASK_HEADER, f"{TASK_HEADER}required_extras = {declaration}\n", 1),
        encoding="utf-8",
    )
    return load_benchmark_environment_policy(policy_path)


def test_required_extras_absent_defaults_to_empty_tuple():
    policy = load_benchmark_environment_policy(POLICY_PATH)

    assert policy.tasks["GS-T006"].required_extras == ()


def test_required_extras_explicit_empty_list_is_empty_tuple(tmp_path: Path):
    policy = _load_with_t006_extras(tmp_path, "[]")

    assert policy.tasks["GS-T006"].required_extras == ()


def test_required_extras_preserves_order_and_deduplicates(tmp_path: Path):
    policy = _load_with_t006_extras(tmp_path, '["dev", "dev"]')

    assert policy.tasks["GS-T006"].required_extras == ("dev",)


def test_required_extras_scalar_string_is_rejected(tmp_path: Path):
    with pytest.raises(BenchmarkEnvironmentConfigurationError):
        _load_with_t006_extras(tmp_path, '"test"')


def test_required_extras_empty_string_member_is_rejected(tmp_path: Path):
    with pytest.raises(BenchmarkEnvironmentConfigurationError):
        _load_with_t006_extras(tmp_path, '["", "test"]')
