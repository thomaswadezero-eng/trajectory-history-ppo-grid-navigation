#!/usr/bin/env python
"""Fail-closed descriptive aggregation for the complete 3 x 8 MovingAI matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from extension_utils import (
    ALL_SEEDS,
    METHODS,
    METRICS,
    NEW_RESULT_ROOT,
    IntegrityError,
    ProtocolError,
    load_all_24_records,
    read_episode_rows,
    read_json,
    sha256_file,
    validate_extension_identity,
    write_json_new,
)


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def validate_checkpoint_selection(record: Mapping[str, Any]) -> Dict[str, Any]:
    done = record["done"]
    run_dir = record["done_path"].parent
    best_path = run_dir / "models" / "best_checkpoint.json"
    best = read_json(best_path)
    if str(Path(str(best["checkpoint_path"])).resolve()) != str(Path(str(done["selected_checkpoint_path"])).resolve()):
        raise IntegrityError("selected checkpoint path mismatch: %s" % run_dir)
    if str(best["checkpoint_sha256"]).upper() != str(done["selected_checkpoint_sha256"]).upper():
        raise IntegrityError("selected checkpoint SHA mismatch: %s" % run_dir)
    if sha256_file(Path(str(best["checkpoint_path"]))) != str(best["checkpoint_sha256"]).upper():
        raise IntegrityError("selected checkpoint file mismatch: %s" % run_dir)
    with Path(str(done["authority_evaluation_csv_path"])).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validation = [row for row in rows if row.get("record_type") == "summary" and row.get("split") == "validation"]
    test_episodes = [row for row in rows if row.get("record_type") == "episode" and row.get("split") == "test"]
    test_summaries = [row for row in rows if row.get("record_type") == "summary" and row.get("split") == "test"]
    if len(validation) != 9 or len(test_episodes) != 750 or len(test_summaries) != 1:
        raise IntegrityError("validation/test record count mismatch: %s" % run_dir)
    def rank(row: Mapping[str, str]) -> Tuple[float, float, float, float, int]:
        return (
            -float(row["success_rate"]),
            -float(row["SPL"]),
            float(row["timeout_rate"]),
            float(row["average_steps"]),
            int(row["checkpoint_interaction"]),
        )
    selected = min(validation, key=rank)
    selected_interaction = int(selected["checkpoint_interaction"])
    if selected_interaction != int(best["interaction_count"]):
        raise IntegrityError("checkpoint was not selected by the frozen validation ranking: %s" % run_dir)
    return {
        "method": record["method"],
        "seed": record["seed"],
        "source": record["source"],
        "validation_summary_count": len(validation),
        "test_episode_count": len(test_episodes),
        "selected_checkpoint_interaction": selected_interaction,
        "selected_checkpoint_sha256": str(best["checkpoint_sha256"]).upper(),
        "test_used_for_checkpoint_selection": False,
        "validation_astar_runtime_calls": int(done["validation_astar_runtime_calls"]),
        "test_astar_runtime_calls": int(done["test_astar_runtime_calls"]),
        "verified": True,
    }


def summarize_subset(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    count = len(rows)
    successes = sum(int(row["success"]) for row in rows)
    collisions = sum(int(row["collision"]) for row in rows)
    timeouts = sum(int(row["timeout"]) for row in rows)
    successful_efficiencies = [float(row["path_efficiency"]) for row in rows if int(row["success"]) == 1]
    return {
        "episode_count": count,
        "success_count": successes,
        "collision_count": collisions,
        "timeout_count": timeouts,
        "success_rate": successes / float(count),
        "spl": sum(float(row["spl"]) for row in rows) / float(count),
        "path_efficiency": statistics.mean(successful_efficiencies) if successful_efficiencies else 0.0,
        "average_steps": sum(int(row["steps"]) for row in rows) / float(count),
        "average_path_length": sum(int(row["path_length"]) for row in rows) / float(count),
        "collision_rate": collisions / float(count),
        "timeout_rate": timeouts / float(count),
        "episode_reward": sum(float(row["episode_reward"]) for row in rows) / float(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the complete MovingAI64 eight-seed matrix")
    parser.add_argument("--aggregate", action="store_true", help="explicitly authorize result aggregation")
    args = parser.parse_args()
    if not args.aggregate:
        raise ProtocolError("--aggregate is required")

    identity = validate_extension_identity()
    records = load_all_24_records(identity)
    test_payload = read_json(Path(str(identity["config"]["inputs"]["test_tasks"]["path"])))
    test_tasks = {str(task["task_id"]): task for task in test_payload["tasks"]}
    if len(test_tasks) != 750:
        raise IntegrityError("frozen test task identity is not 750 unique tasks")

    per_seed: List[Dict[str, Any]] = []
    all_episodes: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    run_source_rows: List[Dict[str, Any]] = []
    for record in records:
        episode_rows, metrics = read_episode_rows(record, test_tasks)
        all_episodes.extend(episode_rows)
        checkpoint_rows.append(validate_checkpoint_selection(record))
        counts = summarize_subset(episode_rows)
        row = {
            "method": record["method"],
            "seed": record["seed"],
            "source": record["source"],
            "episode_count": 750,
            "success_count": int(counts["success_count"]),
            "collision_count": int(counts["collision_count"]),
            "timeout_count": int(counts["timeout_count"]),
            **metrics,
            "done_path": str(record["done_path"].resolve()),
            "episode_csv_path": str(Path(record["done"]["episode_csv_path"]).resolve()),
            "selected_checkpoint_sha256": record["done"]["selected_checkpoint_sha256"],
        }
        per_seed.append(row)
        run_source_rows.append(
            {
                "method": record["method"],
                "seed": record["seed"],
                "source": record["source"],
                "done_path": str(record["done_path"].resolve()),
                "done_sha256": sha256_file(record["done_path"]),
                "config_sha256": record["done"]["config_sha256"],
                "protocol_sha256": record["done"]["protocol_sha256"],
                "authority_program_sha256": record["done"]["authority_program_sha256"],
                "wrapper_program_sha256": record["done"]["wrapper_program_sha256"],
                "episode_csv_sha256": record["done"]["episode_csv_sha256"],
            }
        )

    per_seed.sort(key=lambda row: (METHODS.index(row["method"]), ALL_SEEDS.index(int(row["seed"]))))
    if len(all_episodes) != 24 * 750:
        raise IntegrityError("complete matrix must contain exactly 18,000 formal test rows")

    method_summary: List[Dict[str, Any]] = []
    for method in METHODS:
        chosen = [row for row in per_seed if row["method"] == method]
        summary: Dict[str, Any] = {"method": method, "n_seeds": 8, "episodes_per_seed": 750, "total_test_episodes": 6000}
        for metric in METRICS:
            average, sample_sd = mean_sd([float(row[metric]) for row in chosen])
            summary[metric + "_mean"] = average
            summary[metric + "_sample_sd"] = sample_sd
        method_summary.append(summary)

    indexed = {(row["method"], int(row["seed"])): row for row in per_seed}
    paired_rows: List[Dict[str, Any]] = []
    for left, right in (("PPO_Reward", "PPO"), ("FULL", "PPO_Reward"), ("FULL", "PPO")):
        for seed in ALL_SEEDS:
            for metric in METRICS:
                paired_rows.append(
                    {
                        "left_method": left,
                        "right_method": right,
                        "seed": seed,
                        "metric": metric,
                        "left_value": indexed[(left, seed)][metric],
                        "right_value": indexed[(right, seed)][metric],
                        "paired_difference_left_minus_right": float(indexed[(left, seed)][metric]) - float(indexed[(right, seed)][metric]),
                        "inferential_test": "NOT_AUTHORIZED",
                    }
                )

    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in all_episodes:
        grouped[(str(row["method"]), int(row["seed"]), str(row["subset"]))].append(row)
    family_per_seed: List[Dict[str, Any]] = []
    for method in METHODS:
        for seed in ALL_SEEDS:
            for subset, expected_count in (("DAO", 380), ("Room", 140), ("Maze", 230)):
                rows = grouped[(method, seed, subset)]
                if len(rows) != expected_count:
                    raise IntegrityError("family count mismatch: %s/%d/%s" % (method, seed, subset))
                family_per_seed.append({"method": method, "seed": seed, "subset": subset, **summarize_subset(rows)})
    family_aggregate: List[Dict[str, Any]] = []
    for method in METHODS:
        for subset in ("DAO", "Room", "Maze"):
            chosen = [row for row in family_per_seed if row["method"] == method and row["subset"] == subset]
            summary = {"method": method, "subset": subset, "n_seeds": 8, "episodes_per_seed": chosen[0]["episode_count"]}
            for metric in METRICS:
                average, sample_sd = mean_sd([float(row[metric]) for row in chosen])
                summary[metric + "_mean"] = average
                summary[metric + "_sample_sd"] = sample_sd
            family_aggregate.append(summary)

    output = NEW_RESULT_ROOT / "final_8seed_aggregate" / ("aggregate_%s" % datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "run_source_identity.csv", list(run_source_rows[0]), run_source_rows)
    write_csv(output / "checkpoint_selection_audit.csv", list(checkpoint_rows[0]), checkpoint_rows)
    write_csv(output / "per_seed_metrics.csv", list(per_seed[0]), per_seed)
    write_csv(output / "method_mean_sample_sd.csv", list(method_summary[0]), method_summary)
    write_csv(output / "paired_method_differences.csv", list(paired_rows[0]), paired_rows)
    write_csv(output / "map_family_per_seed.csv", list(family_per_seed[0]), family_per_seed)
    write_csv(output / "map_family_mean_sample_sd.csv", list(family_aggregate[0]), family_aggregate)
    table_rows = []
    labels = {"PPO": "PPO", "PPO_Reward": "PPO+Reward", "FULL": "FULL"}
    for row in method_summary:
        table_rows.append(
            {
                "method": labels[row["method"]],
                "n": 8,
                "success_rate_mean": row["success_rate_mean"],
                "success_rate_sample_sd": row["success_rate_sample_sd"],
                "spl_mean": row["spl_mean"],
                "spl_sample_sd": row["spl_sample_sd"],
                "path_efficiency_mean": row["path_efficiency_mean"],
                "path_efficiency_sample_sd": row["path_efficiency_sample_sd"],
                "average_steps_mean": row["average_steps_mean"],
                "average_steps_sample_sd": row["average_steps_sample_sd"],
                "timeout_rate_mean": row["timeout_rate_mean"],
                "timeout_rate_sample_sd": row["timeout_rate_sample_sd"],
                "interpretation": "DESCRIPTIVE_ONLY",
            }
        )
    write_csv(output / "table7_update_ready.csv", list(table_rows[0]), table_rows)

    interpretation = [
        "# MovingAI eight-seed descriptive interpretation draft",
        "",
        "This draft is generated only after all 24 method-by-seed combinations pass the integrity gates.",
        "It reports descriptive means and sample standard deviations across eight matched seeds. No formal",
        "significance test or significance claim is authorized by this extension.",
        "",
        "The final narrative must be written from `method_mean_sample_sd.csv` and the paired differences.",
        "In particular, it must not assume in advance that FULL outperforms PPO+Reward. Any ordering must",
        "be stated exactly as observed, with the descriptive-only boundary retained.",
    ]
    (output / "result_interpretation_draft.md").write_text("\n".join(interpretation) + "\n", encoding="utf-8")
    write_json_new(
        output / "aggregate_manifest.json",
        {
            "schema_version": "movingai_8seed_aggregate.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETE_24_OF_24",
            "methods": list(METHODS),
            "seeds": list(ALL_SEEDS),
            "legacy_seed_count": 3,
            "new_seed_count": 5,
            "method_seed_combination_count": 24,
            "formal_test_rows": len(all_episodes),
            "episodes_per_combination": 750,
            "input_sha256": identity["input_sha256"],
            "extension_config_sha256": identity["config_sha256"],
            "extension_protocol_sha256": identity["protocol_sha256"],
            "integrity_gates": {
                "canonical_legacy_runs": "PASS_9_OF_9",
                "new_runs": "PASS_15_OF_15",
                "complete_matrix": "PASS_24_OF_24",
                "test_task_identity": "PASS",
                "checkpoint_selection_validation_only": "PASS",
                "validation_astar_runtime_calls_zero": "PASS",
                "test_astar_runtime_calls_zero": "PASS",
            },
            "statistics": {
                "mean_and_sample_sd": True,
                "paired_differences": True,
                "formal_inferential_tests_executed": False,
                "inferential_status": "NOT_AUTHORIZED",
            },
        },
    )
    print(json.dumps({"status": "COMPLETE_24_OF_24", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

