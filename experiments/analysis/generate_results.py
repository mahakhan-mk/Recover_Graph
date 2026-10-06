"""Regenerate descriptive tables from canonical stored reported artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "research/evidence/reported"
CASE_TASKS = {"EXP1": "T006", "EXP2": "T007", "EXP3": "T008", "EXP4": "T017"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_metrics(task: str, condition: str) -> dict[str, Any]:
    base = EVIDENCE / task / condition
    artifact = read_json(base / "artifact.json")
    metadata = read_json(base / "run_metadata.json")
    usage: dict[str, Any] = metadata.get("usage") or {}
    retrieval_pattern: str | None = None
    advisory_path = base / "advisory_evidence.jsonl"
    if advisory_path.is_file():
        for line in advisory_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            if record.get("record_type") != "advisory_retrieval":
                continue
            retrieval: dict[str, Any] = record.get("retrieval") or {}
            eligible: list[dict[str, Any]] = retrieval.get("eligible_candidates") or []
            if eligible:
                retrieval_pattern = eligible[0].get("pattern_id")
                if retrieval_pattern:
                    break
    input_tokens = artifact.get("input_tokens")
    output_tokens = artifact.get("output_tokens")
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return {
        "task_id": f"GS-{task}",
        "condition": condition,
        "success": artifact.get("task_success"),
        "known_failure_repeated": artifact.get("known_failure_repeated"),
        "requests": usage.get("requests"),
        "tool_calls": artifact.get("tool_calls"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": artifact.get("latency_ms"),
        "advice_issued": (artifact.get("advice_count") or 0) > 0,
        "advice_accepted": artifact.get("advice_accepted"),
        "retrieved_incident_id": artifact.get("retrieved_incident_id"),
        "recovery_pattern_id": retrieval_pattern,
    }


def _change(before: float | int | None, after: float | int | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return round((after - before) / before * 100, 1)


def build_results() -> dict[str, Any]:
    rows = {
        label: {cond: run_metrics(task, cond) for cond in ("B0", "T")}
        for label, task in CASE_TASKS.items()
    }
    recurrence_meta = read_json(EVIDENCE / "recurrence_opportunities.json")
    adjudication_meta = read_json(EVIDENCE / "recurrence_adjudications.json")
    opportunities = recurrence_meta["opportunities"]
    opportunity_tasks = [item["task_id"] for item in opportunities]
    adjudications = {
        item["task_id"]: item for item in adjudication_meta["adjudications"]
    }
    if len(opportunity_tasks) != len(set(opportunity_tasks)):
        raise ValueError("recurrence opportunity task IDs must be unique")
    if set(adjudications) != set(opportunity_tasks):
        raise ValueError("recurrence adjudications must match opportunity task IDs")
    if any(set(adjudications[task]) != {"task_id", "B0", "T"} for task in opportunity_tasks):
        raise ValueError("each recurrence opportunity needs B0 and T adjudications")
    recurrence: dict[str, Any] = {
        "opportunity_count": len(opportunity_tasks),
        "task_ids": opportunity_tasks,
    }
    for cond in ("B0", "T"):
        repeated = sum(
            bool(adjudications[task][cond]["recurrence"])
            for task in opportunity_tasks
        )
        recurrence[cond] = {
            "repeated": repeated,
            "opportunities": len(opportunity_tasks),
            "rate": repeated / len(opportunity_tasks) if opportunity_tasks else None,
        }
    exp1 = rows["EXP1"]
    metrics = {}
    for key in ("requests", "tool_calls", "total_tokens"):
        metrics[key] = {
            "B0": exp1["B0"][key],
            "T": exp1["T"][key],
            "change_percent": _change(exp1["B0"][key], exp1["T"][key]),
        }
    for key in ("latency_ms",):
        metrics["wall_clock_seconds"] = {
            "B0": round(exp1["B0"][key] / 1000, 1),
            "T": round(exp1["T"][key] / 1000, 1),
            "change_percent": _change(exp1["B0"][key], exp1["T"][key]),
        }
    success = {
        cond: {
            "successes": sum(
                bool(rows[label][cond]["success"]) for label in ("EXP1", "EXP2", "EXP3")
            ),
            "cases": 3,
        }
        for cond in ("B0", "T")
    }
    advice = {label: rows[label]["T"]["advice_issued"] for label in CASE_TASKS}
    retrieval = {
        label: {
            "pattern_id": rows[label]["T"]["recovery_pattern_id"],
            "incident_id": rows[label]["T"]["retrieved_incident_id"],
            "advice_issued": advice[label],
        }
        for label in CASE_TASKS
    }
    return {
        "schema_version": 1,
        "case_metrics": rows,
        "exp1_efficiency": metrics,
        "task_success_exp1_exp3": success,
        "recurrence": recurrence,
        "recurrence_difference_percentage_points": round(
            (recurrence["B0"]["rate"] - recurrence["T"]["rate"]) * 100, 1
        ),
        "advice_issued": advice,
        "retrieval": retrieval,
        "false_advice_rate": "NOT COMPUTABLE FROM RETAINED ARTIFACTS",
        "interpretation": "Descriptive case-study results; not inferential statistical claims.",
    }


def write_outputs(output_dir: Path) -> None:
    result = build_results()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reported_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    rows = [
        result["case_metrics"][label][condition]
        for label in CASE_TASKS
        for condition in ("B0", "T")
    ]
    for filename, selected, fields in (
        (
            "case_summary.csv",
            rows,
            [
                "task_id",
                "condition",
                "success",
                "known_failure_repeated",
                "advice_issued",
                "advice_accepted",
                "recovery_pattern_id",
            ],
        ),
        (
            "exp1_metrics.csv",
            [row for row in rows if row["task_id"] == "GS-T006"],
            [
                "task_id",
                "condition",
                "requests",
                "tool_calls",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms",
            ],
        ),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/analysis/generated")
    args = parser.parse_args()
    write_outputs(args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
