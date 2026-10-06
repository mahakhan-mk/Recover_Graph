"""Offline integrity checks for the canonical RecoverGraph case package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmark/manifests/reported_evaluation.jsonl"
EXPECTED = {
    "GS-T006": {
        "paper_label": "EXP1",
        "B0": "GS-E003-B0-GS-T006-58d1773e4e1840128fa4a7fff92f866b",
        "T": "GS-E003-T-GS-T006-f72adb31053f46bf8cb7930aba136e17",
    },
    "GS-T007": {
        "paper_label": "EXP2",
        "B0": "GS-E003-B0-GS-T007-3476d50492ca4dc1a8d3c8d24992fea8",
        "T": "GS-E003-T-GS-T007-709f825d8c5740f3b7bf452434f763d4",
    },
    "GS-T008": {
        "paper_label": "EXP3",
        "B0": "GS-E003-B0-GS-T008-779fa5a17f7a4dafaf5d45815babe399",
        "T": "GS-E003-T-GS-T008-0c5215d948a74fc4bb1b2796777f366c",
    },
    "GS-T017": {
        "paper_label": "EXP4",
        "B0": "GS-E003-B0-GS-T017-99a505227d02418aae23bea2909bbf5d",
        "T": "GS-E003-T-GS-T017-8aff75bedefc40f3a7f192786cfe806d",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reported_manifest_and_copied_evidence_are_canonical() -> None:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {record["task_id"] for record in records} == set(EXPECTED)
    assert len({record["task_id"] for record in records}) == 4

    run_ids: list[str] = []
    frozen_memory = ROOT / "research/evidence/reported/frozen_memory/memory_manifest.json"
    memory_ids = set(json.loads(frozen_memory.read_text(encoding="utf-8"))["pattern_ids"])
    assert len(memory_ids) == 5

    for record in records:
        task_id = record["task_id"]
        expected = EXPECTED[task_id]
        assert record["paper_label"] == expected["paper_label"]
        assert record["runs"] == {"B0": expected["B0"], "T": expected["T"]}
        assert record["frozen_memory_path"] == "research/evidence/reported/frozen_memory/"
        assert record["revision"]
        assert record["clone_url"].startswith("https://github.com/")
        assert record["objective_evaluator"]["name"] == "frozen_swesmith_fail_to_pass"

        run_ids.extend(record["runs"].values())
        case_path = ROOT / record["case_definition"]
        config_path = ROOT / record["config"]
        assert case_path.is_file()
        assert config_path.is_file()
        case = json.loads(case_path.read_text(encoding="utf-8"))
        assert case["task_id"] == task_id
        assert case["paper_label"] == expected["paper_label"]
        assert case["objective_evaluator"]["fail_to_pass"]
        assert case["frozen_memory_path"] == record["frozen_memory_path"]
        assert case["research_validity_class"] == "VALID_DEVELOPMENT"
        forbidden_case_data = {"dataset_patch", "test_patch", "gold_patch", "expected_solution"}
        assert not forbidden_case_data.intersection(case)

        evidence_case = task_id.removeprefix("GS-")
        case_manifest = json.loads(
            (ROOT / "research/evidence/reported" / evidence_case / "case_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert case_manifest["canonical_run_ids"] == {"B0": expected["B0"], "T": expected["T"]}
        assert case_manifest["frozen_memory_path"] == record["frozen_memory_path"]
        assert case_manifest["copied_config_sha256"] == _sha256(config_path)
        assert case_manifest["source_experiment_id"] == "GS-E003"
        source_config = ROOT / record["config_source"]
        assert case_manifest["config_source_sha256"] == _sha256(source_config)

        for key, expected_hash in case_manifest["copied_artifact_sha256"].items():
            condition, filename = key.split("/", maxsplit=1)
            copied = (
                ROOT / "research/evidence/reported" / evidence_case / condition / filename
            )
            assert copied.is_file()
            assert _sha256(copied) == expected_hash
            metadata_path = (
                ROOT
                / "research/evidence/reported"
                / evidence_case
                / condition
                / "run_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert metadata["experiment_id"] == case_manifest["source_experiment_id"]
            assert (
                metadata["execution_commit_sha"]
                == case_manifest["source_commit_sha_by_condition"][condition]
            )
            assert metadata["run_id"] == case_manifest["canonical_run_ids"][condition]

        assert case_manifest["source_commit_sha_by_condition"] == {
            condition: json.loads(
                (
                    ROOT
                    / "research/evidence/reported"
                    / evidence_case
                    / condition
                    / "run_metadata.json"
                ).read_text(encoding="utf-8")
            )["execution_commit_sha"]
            for condition in ("B0", "T")
        }

    assert len(run_ids) == len(set(run_ids))
