from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

import experiments.analysis.generate_results as results_generator
from experiments.analysis.generate_figure1 import generate_svg
from experiments.analysis.generate_results import build_results, write_outputs
from experiments.run_reported_evaluation import CASES, ROOT, load_cases, validate


def test_reported_runner_lists_all_cases() -> None:
    result = subprocess.run(
        [sys.executable, "experiments/run_reported_evaluation.py", "--list"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for case in CASES:
        assert case in result.stdout


def test_all_cases_validate_without_provider() -> None:
    for label in CASES:
        for condition in ("B0", "T"):
            assert validate(label, condition) == []
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/run_reported_evaluation.py",
                    "--case",
                    label,
                    "--condition",
                    condition,
                    "--validate-only",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            assert "VALID:" in result.stdout


def test_dry_run_does_not_initialize_provider() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_reported_evaluation.py",
            "--case",
            "EXP1",
            "--condition",
            "T",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"provider_called": false' in result.stdout


def test_manifest_pins_expected_urls_and_revisions() -> None:
    cases = load_cases()
    assert set(cases) == set(CASES)
    for case in cases.values():
        assert case["clone_url"].startswith("https://github.com/")
        assert len(case["revision"]) == 40
        assert case["task_id"] == CASES[case["paper_label"]]


def test_frozen_prompt_bytes_match_reported_run_evidence() -> None:
    prompt_path = ROOT / "configs/experiments/reported/system_prompt_r13b.txt"
    prompt_bytes = prompt_path.read_bytes()
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    assert prompt_hash == "f3bf4187cc9bfabcf2551aa3988e592a2d0f2d0e61ad2984d7d90ddf13d54502"
    raw: dict[str, Any] = json.loads(
        (ROOT / "research/evidence/reported/T006/B0/raw_evidence.json").read_text(encoding="utf-8")
    )
    evidence: Any = raw["messages"][0]["parts"][0]["content"]
    assert isinstance(evidence, str)
    assert evidence.encode("utf-8") == prompt_bytes


def test_condition_configs_keep_b0_off_and_t_on_exact_memory() -> None:
    cases = load_cases()
    for case in cases.values():
        config = yaml.safe_load((ROOT / case["config"]).read_text(encoding="utf-8"))
        assert config["condition_runtime"]["B0"]["advisory_service"] == "none"
        assert config["condition_runtime"]["T"]["advisory_service"] == "AdvisoryService"
        pattern_ids = [
            json.loads(line)["id"]
            for line in (ROOT / "research/evidence/reported/frozen_memory/recovery_patterns.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert pattern_ids == config["treatment_pattern_ids"]


def test_results_reproduce_recorded_values() -> None:
    results = build_results()
    exp1 = results["exp1_efficiency"]
    assert exp1["requests"] == {"B0": 20, "T": 13, "change_percent": -35.0}
    assert exp1["tool_calls"] == {"B0": 19, "T": 11, "change_percent": -42.1}
    assert exp1["total_tokens"] == {"B0": 182978, "T": 115102, "change_percent": -37.1}
    assert exp1["wall_clock_seconds"] == {"B0": 101.1, "T": 71.9, "change_percent": -28.9}
    assert results["recurrence"]["opportunity_count"] == 3
    assert results["recurrence"]["task_ids"] == ["GS-T006", "GS-T007", "GS-T008"]
    assert results["recurrence"]["B0"] == {
        "repeated": 2,
        "opportunities": 3,
        "rate": 2 / 3,
    }
    assert results["recurrence"]["T"] == {
        "repeated": 1,
        "opportunities": 3,
        "rate": 1 / 3,
    }
    assert results["recurrence_difference_percentage_points"] == 33.3
    assert results["task_success_exp1_exp3"] == {
        "B0": {"successes": 2, "cases": 3},
        "T": {"successes": 2, "cases": 3},
    }
    assert results["false_advice_rate"] == "NOT COMPUTABLE FROM RETAINED ARTIFACTS"
    assert results["retrieval"]["EXP1"]["pattern_id"] == "recovery-pattern-ad07a6a45718b848a30ad377"
    assert results["retrieval"]["EXP2"]["pattern_id"] is None
    assert results["retrieval"]["EXP3"]["pattern_id"] == "recovery-pattern-f490f62ab931191c6eac6db1"


def test_recurrence_opportunities_are_pre_treatment_annotated_transfers() -> None:
    opportunities = json.loads(
        (ROOT / "research/evidence/reported/recurrence_opportunities.json").read_text(
            encoding="utf-8"
        )
    )["opportunities"]
    validation = {
        row["review_id"]: row
        for row in csv.DictReader(
            (ROOT / "benchmark/annotations/recurrence_validation.csv").open(
                encoding="utf-8"
            )
        )
    }
    pilot = {
        record["task_id"]: record
        for record in (
            json.loads(line)
            for line in (ROOT / "benchmark/manifests/pilot.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    evidence = {
        record["review_id"]: record
        for record in (
            json.loads(line)
            for line in (ROOT / "benchmark/annotations/recurrence_evidence.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    assert [item["task_id"] for item in opportunities] == [
        "GS-T006",
        "GS-T007",
        "GS-T008",
    ]
    for item in opportunities:
        source = pilot[item["source_task_id"]]
        transfer = pilot[item["transfer_task_id"]]
        source_row = validation[source["review_id"]]
        transfer_row = validation[transfer["review_id"]]
        assert source["occurrence_index"] == item["source_occurrence"] == 1
        assert transfer["occurrence_index"] == item["transfer_occurrence"] == 2
        assert source["chronological_index"] < transfer["chronological_index"]
        assert source["family_id"] == transfer["family_id"] == item["recurrence_family_id"]
        assert transfer["review_id"] in item["relation_id"]
        assert source_row["keep"] == transfer_row["keep"] == "yes"
        assert source_row["transferable"] == transfer_row["transferable"] == "yes"
        assert evidence[source["review_id"]]["fail_to_pass"]
        assert evidence[transfer["review_id"]]["fail_to_pass"]
        assert item["transferable"] is True
        for path in item["provenance_paths"]:
            assert (ROOT / path).is_file()


def test_recurrence_adjudications_cover_all_cases_and_preserve_t007_miss() -> None:
    adjudications = json.loads(
        (ROOT / "research/evidence/reported/recurrence_adjudications.json").read_text(
            encoding="utf-8"
        )
    )["adjudications"]
    by_task = {item["task_id"]: item for item in adjudications}
    assert set(by_task) == {"GS-T006", "GS-T007", "GS-T008"}
    expected = {
        "GS-T006": {"B0": True, "T": False},
        "GS-T007": {"B0": True, "T": True},
        "GS-T008": {"B0": False, "T": False},
    }
    for task_id, conditions in expected.items():
        assert set(by_task[task_id]) == {"task_id", "B0", "T"}
        for condition, recurrence in conditions.items():
            assert by_task[task_id][condition]["recurrence"] is recurrence
            assert (ROOT / by_task[task_id][condition]["supporting_artifact_path"]).is_file()
    for condition in ("B0", "T"):
        record = by_task["GS-T007"][condition]
        assert record["recurrence"] is True
        assert record["matcher_result"] is False
        assert record["formal_matcher_agreed"] is False
        assert "Direct raw execution reproduces" in record["matcher_note"]
        metadata = json.loads(
            (ROOT / f"research/evidence/reported/T007/{condition}/run_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["recurrence_evaluator"]["result"] is False
        assert metadata["repeated_failure_result"] is False


def test_recurrence_aggregation_uses_adjudication_not_matcher_flag(monkeypatch: Any) -> None:
    original_run_metrics = results_generator.run_metrics

    def no_treatment_signals(task: str, condition: str) -> dict[str, Any]:
        metrics = original_run_metrics(task, condition)
        metrics["known_failure_repeated"] = False
        metrics["advice_issued"] = False
        metrics["recovery_pattern_id"] = None
        metrics["retrieved_incident_id"] = None
        return metrics

    monkeypatch.setattr(results_generator, "run_metrics", no_treatment_signals)
    results = build_results()
    for condition, expected_repeated in (("B0", 2), ("T", 1)):
        assert results["recurrence"][condition]["repeated"] == expected_repeated
    for condition in ("B0", "T"):
        artifact = json.loads(
            (ROOT / f"research/evidence/reported/T007/{condition}/artifact.json").read_text(
                encoding="utf-8"
            )
        )
        assert artifact["known_failure_repeated"] is False


def test_result_outputs_are_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    write_outputs(first)
    write_outputs(second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_figure_generator_produces_valid_svg() -> None:
    svg = generate_svg()
    assert len(svg) > 1000
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "Freeze before transfer" in "".join(root.itertext())


def test_reproducibility_guide_paths_exist() -> None:
    guide = (ROOT / "research/REPRODUCIBILITY.md").read_text(encoding="utf-8")
    for relative in (
        "research/evidence/reported/frozen_memory/recovery_patterns.jsonl",
        "configs/experiments/reported/system_prompt_r13b.txt",
        "benchmark/reported_cases/GS-T006.json",
        "experiments/analysis/generate_results.py",
        "experiments/analysis/generate_figure1.py",
    ):
        assert (ROOT / relative).exists()
        assert relative in guide
