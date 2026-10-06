"""Safe public interface for the frozen RecoverGraph reported cases.

This entry point validates and prepares pinned source trees. Experiment execution
is intentionally not exposed until a single-case runtime adapter is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark/manifests/reported_evaluation.jsonl"
PROMPT = ROOT / "configs/experiments/reported/system_prompt_r13b.txt"
PROMPT_SHA256 = "f3bf4187cc9bfabcf2551aa3988e592a2d0f2d0e61ad2984d7d90ddf13d54502"
MEMORY = ROOT / "research/evidence/reported/frozen_memory/recovery_patterns.jsonl"
CASES = {"EXP1": "GS-T006", "EXP2": "GS-T007", "EXP3": "GS-T008", "EXP4": "GS-T017"}


def load_cases() -> dict[str, dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {item["paper_label"]: item for item in cases}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(case_label: str, condition: str) -> list[str]:
    """Validate immutable package inputs without initializing a provider/runtime."""
    case = load_cases()[case_label]
    problems: list[str] = []
    prompt_hash = hashlib.sha256(PROMPT.read_bytes()).hexdigest()
    if prompt_hash != PROMPT_SHA256:
        problems.append(f"prompt snapshot hash mismatch: {prompt_hash}")
    config_path = ROOT / case["config"]
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        problems.append(f"cannot read reported config: {exc}")
        config = {}
    if config.get("provider") != "kilo" or config.get("model") != "qwen/qwen3-coder":
        problems.append("reported config must preserve Kilo with qwen/qwen3-coder")
    model_config = config.get("model_config")
    if not isinstance(model_config, str) or not (ROOT / model_config).is_file():
        problems.append("reported model configuration is missing")
    if config.get("tools") != ["read_file", "write_file", "edit_file", "run_tests", "run_command"]:
        problems.append("reported controlled tool set has changed")
    limits = config.get("limits", {})
    for limit_name in ("max_actions", "max_requests", "timeout_seconds", "tool_retries"):
        if not isinstance(limits.get(limit_name), int):
            problems.append(f"reported request/action limit is missing: {limit_name}")
    if config.get("objective_evaluator") != "frozen_swesmith_fail_to_pass":
        problems.append("reported objective evaluator has changed")
    if config.get("task_workspace_network") != "disabled":
        problems.append("reported task workspace network must remain disabled")
    if case.get("environment", {}).get("network_during_execution") is not False:
        problems.append("canonical case metadata does not disable task network")
    if config.get("system_prompt_sha256") != prompt_hash:
        problems.append("reported config prompt hash does not match frozen snapshot")
    source_config = ROOT / case["config_source"] if case.get("config_source") else None
    evidence_case = case["task_id"].removeprefix("GS-")
    case_manifest_path = ROOT / "research/evidence/reported" / evidence_case / "case_manifest.json"
    try:
        case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        case_manifest = {}
        problems.append(
            f"canonical case provenance manifest is missing or invalid: {case_manifest_path}"
        )
    if not isinstance(case_manifest, dict):
        case_manifest = {}
        problems.append("canonical case provenance manifest must be a JSON object")
    if case_manifest:
        if case_manifest.get("canonical_run_ids") != case["runs"]:
            problems.append("canonical case provenance run IDs do not match the reported manifest")
        if case_manifest.get("source_experiment_id") != "GS-E003":
            problems.append("canonical case provenance source experiment is missing or incorrect")
        try:
            if case_manifest.get("copied_config_sha256") != _sha256(config_path):
                problems.append("canonical reported config hash does not match case provenance")
            if source_config is None or case_manifest.get("config_source_sha256") != _sha256(
                source_config
            ):
                problems.append("source config hash does not match canonical case provenance")
        except OSError as exc:
            problems.append(f"cannot verify canonical config provenance: {exc}")
        for relative_path, expected_hash in case_manifest.get(
            "copied_artifact_sha256", {}
        ).items():
            copied_path = ROOT / "research/evidence/reported" / evidence_case / relative_path
            try:
                if _sha256(copied_path) != expected_hash:
                    problems.append(f"canonical artifact hash mismatch: {relative_path}")
            except OSError:
                problems.append(f"canonical artifact is missing: {relative_path}")
    memory_manifest_path = MEMORY.parent / "memory_manifest.json"
    try:
        memory_manifest = json.loads(memory_manifest_path.read_text(encoding="utf-8"))
        memory_bytes = MEMORY.read_bytes()
        patterns = [
            json.loads(line)
            for line in memory_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError):
        memory_manifest, patterns = {}, []
        problems.append("frozen memory or its manifest is missing or invalid")
    if not isinstance(memory_manifest, dict):
        memory_manifest, patterns = {}, []
        problems.append("frozen memory manifest must be a JSON object")
    if memory_manifest:
        checksums = memory_manifest.get("checksums", {})
        provenance = memory_manifest.get("provenance", {})
        if not isinstance(checksums, dict) or not isinstance(provenance, dict):
            checksums, provenance = {}, {}
            problems.append("frozen memory checksums and provenance must be JSON objects")
        if checksums.get("recovery_patterns_jsonl_sha256") != (
            hashlib.sha256(memory_bytes).hexdigest()
        ):
            problems.append("frozen memory corpus hash does not match its manifest")
        if [item.get("id") for item in patterns] != memory_manifest.get("pattern_ids"):
            problems.append("frozen memory pattern IDs do not match its manifest")
        provenance_root = ROOT / provenance.get("root", "")
        for relative_path, expected_hash in provenance.get("files_sha256", {}).items():
            try:
                if _sha256(provenance_root / relative_path) != expected_hash:
                    problems.append(f"frozen-memory provenance hash mismatch: {relative_path}")
            except OSError:
                problems.append(f"frozen-memory provenance file is missing: {relative_path}")
        freeze_metadata_path = ROOT / provenance.get("freeze_metadata", "")
        try:
            if _sha256(freeze_metadata_path) != checksums.get("freeze_metadata_sha256"):
                problems.append("frozen-memory freeze metadata hash does not match its manifest")
        except OSError:
            problems.append("frozen-memory freeze metadata is missing")
    runtime = config.get("condition_runtime", {}).get(condition, {})
    if condition == "B0" and runtime.get("advisory_service") != "none":
        problems.append("B0 must disable advisory service")
    if condition == "T":
        patterns = [item["id"] for item in patterns]
        if runtime.get("advisory_service") != "AdvisoryService":
            problems.append("T must enable AdvisoryService")
        if patterns != config.get("treatment_pattern_ids"):
            problems.append("T config does not reference exact frozen five-pattern corpus")
    for path_key in ("case_definition", "config"):
        if not (ROOT / case[path_key]).is_file():
            problems.append(f"missing {path_key}: {case[path_key]}")
    task_key = case["task_id"].removeprefix("GS-")
    for run_condition, run_id in case["runs"].items():
        run_dir = ROOT / "research/evidence/reported" / task_key / run_condition
        run_path = run_dir / "run_metadata.json"
        try:
            metadata = json.loads(run_path.read_text(encoding="utf-8"))
            artifact = json.loads((run_dir / "artifact.json").read_text(encoding="utf-8"))
            raw = json.loads((run_dir / "raw_evidence.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            metadata, artifact, raw = {}, {}, {}
        if metadata.get("run_id") != run_id:
            problems.append(
                f"canonical {run_condition} metadata missing or run id mismatch: {run_id}"
            )
        if metadata.get("experiment_id") != case_manifest.get("source_experiment_id"):
            problems.append(
                f"canonical {run_condition} source experiment does not match provenance"
            )
        if metadata.get("execution_commit_sha") != case_manifest.get(
            "source_commit_sha_by_condition", {}
        ).get(run_condition):
            problems.append(f"canonical {run_condition} commit does not match provenance")
        recorded_prompt_hashes = {
            metadata.get("system_prompt_sha256"),
            artifact.get("system_prompt_sha256"),
            raw.get("system_prompt_sha256"),
        }
        recorded_prompt_hashes.discard(None)
        if recorded_prompt_hashes != {prompt_hash}:
            problems.append(f"canonical {run_condition} prompt hash does not match frozen snapshot")
        if artifact.get("system_prompt_id") != config.get("system_prompt") or raw.get(
            "system_prompt_id"
        ) != config.get("system_prompt"):
            problems.append(f"canonical {run_condition} prompt id does not match reported config")
    return problems


def prepare(case_label: str, destination: Path | None = None) -> Path:
    """Clone and pin one source repository in a runtime-only directory."""
    case = load_cases()[case_label]
    base = destination or Path(tempfile.gettempdir()) / "recovergraph-repositories"
    base.mkdir(parents=True, exist_ok=True)
    target = base / case["repository_name"]
    if target.exists():
        raise FileExistsError(f"refusing to replace existing preparation directory: {target}")
    subprocess.run(["git", "clone", "--no-checkout", case["clone_url"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", case["revision"]], check=True)
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != case["revision"]:
        raise RuntimeError(f"revision verification failed: expected {case['revision']}, got {head}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list canonical cases and revisions")
    parser.add_argument("--case", choices=(*CASES, *CASES.values()))
    parser.add_argument("--condition", choices=("B0", "T"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    cases = load_cases()
    if args.list:
        for label, case in cases.items():
            print(f"{label}\t{case['task_id']}\t{case['clone_url']}\t{case['revision']}")
        return 0
    if args.case is None:
        parser.error("select --list or provide --case")
    label = next((key for key, task in CASES.items() if args.case in (key, task)), None)
    assert label is not None
    if args.prepare_only:
        path = prepare(label)
        print(f"prepared {cases[label]['revision']} at {path}")
        return 0
    if args.condition is None:
        parser.error("--condition is required for validation or dry-run")
    problems = validate(label, args.condition)
    if problems:
        print("VALIDATION FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    case = cases[label]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "case": label,
                    "task_id": case["task_id"],
                    "condition": args.condition,
                    "revision": case["revision"],
                    "provider_called": False,
                    "execution": "not started; explicit runtime adapter required",
                },
                sort_keys=True,
            )
        )
        return 0
    print(f"VALID: {label} {case['task_id']} {args.condition}; no provider initialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
