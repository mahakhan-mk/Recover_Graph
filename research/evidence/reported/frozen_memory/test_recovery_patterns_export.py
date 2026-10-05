"""Offline integrity checks for the frozen memory export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPECTED_IDS = {
    "recovery-pattern-ad07a6a45718b848a30ad377",
    "recovery-pattern-f490f62ab931191c6eac6db1",
    "recovery-pattern-162e3999c4a2c66a1ff647ed",
    "recovery-pattern-fd7b65022b22dc5f2a42816f",
    "recovery-pattern-99a54266f940e1d4648f4698",
}


def _load() -> tuple[list[dict[str, object]], dict[str, object]]:
    records = [
        json.loads(line)
        for line in (HERE / "recovery_patterns.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((HERE / "memory_manifest.json").read_text(encoding="utf-8"))
    return records, manifest


def test_frozen_corpus_records_and_manifest_are_consistent() -> None:
    records, manifest = _load()
    ids = [record["id"] for record in records]

    assert len(records) == 5
    assert len(ids) == len(set(ids))
    assert set(ids) == EXPECTED_IDS
    assert manifest["pattern_count"] == 5
    assert set(manifest["pattern_ids"]) == EXPECTED_IDS
    assert manifest["embedding_dimension"] == 384

    for record in records:
        assert record["source_task_id"] in {"GS-T001", "GS-T002", "GS-T003", "GS-T004", "GS-T005"}
        assert record["source_failure_id"]
        assert record["source_resolution_id"]
        assert record["source_outcome_id"]
        assert record["verification_status"] == "observed_successful"
        assert record["invalidated_at"] is None
        embedding = record["embedding"]
        if embedding is not None:
            assert len(embedding) == 384
            assert record["embedding_dimension"] == 384
        else:
            assert record["embedding_dimension"] is None

    for config_path in manifest["treatment_configs"]:
        config = (ROOT / config_path).read_text(encoding="utf-8")
        assert all(pattern_id in config for pattern_id in EXPECTED_IDS)
    config_hashes = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in manifest["treatment_configs"]
    }
    for freeze_path in manifest["treatment_freezes"]:
        freeze = json.loads((ROOT / freeze_path).read_text(encoding="utf-8"))
        config_path = freeze["config_path"]
        assert config_path in config_hashes
        assert freeze["config_sha256"] == config_hashes[config_path]

    digest = hashlib.sha256((HERE / "recovery_patterns.jsonl").read_bytes()).hexdigest()
    assert manifest["checksums"]["recovery_patterns_jsonl_sha256"] == digest
    assert all((ROOT / path).is_file() for path in manifest["source_acquisition_paths"])

    forbidden_keys = {"api_key", "token", "gold_patch", "expected_solution", "future_task_metadata"}

    def check_keys(value: object) -> None:
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(key.lower() for key in value)
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    check_keys(records)
