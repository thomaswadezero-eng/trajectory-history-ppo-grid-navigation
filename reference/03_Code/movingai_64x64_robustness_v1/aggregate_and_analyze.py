#!/usr/bin/env python
"""Descriptive, result-agnostic aggregate with a fail-closed inference gate."""

from __future__ import print_function

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from protocol_utils import ALLOWED_METHODS, RESULT_ROOT, REVISION_DIR, IntegrityError, ProtocolError, read_json, validate_confirmed, write_json_new
from run_movingai64_robustness import completed_run


METRICS = ("success_rate", "spl", "path_efficiency", "average_steps", "average_path_length", "collision_rate", "timeout_rate", "episode_reward")


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) >= 2 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Aggregate complete MovingAI64 paired seeds")
    parser.add_argument("--config", type=Path, default=REVISION_DIR / "movingai64_config_confirmed.json")
    parser.add_argument("--protocol", type=Path, default=REVISION_DIR / "movingai64_protocol_confirmed.json")
    parser.add_argument("--confirm-aggregate", action="store_true")
    args = parser.parse_args()
    if not args.confirm_aggregate:
        raise ProtocolError("--confirm-aggregate is required")
    identity = validate_confirmed(args.config, args.protocol)
    seeds = list(identity["config"]["official_seeds"])
    records = []
    for method in ALLOWED_METHODS:
        for seed in seeds:
            done_path = completed_run(identity, method, seed)
            if done_path is None:
                raise IntegrityError("aggregate requires a complete paired matrix: %s/%d" % (method, seed))
            done = read_json(done_path)
            row = {"method": method, "seed": seed, "done_path": str(done_path.resolve())}
            row.update({metric: float(done["final_metrics"][metric]) for metric in METRICS})
            records.append(row)
    output = RESULT_ROOT / "analysis_ready" / ("aggregate_%s" % datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    seed_fields = ["method", "seed"] + list(METRICS) + ["done_path"]
    with (output / "seed_level_summary.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for method in ALLOWED_METHODS:
        chosen = [row for row in records if row["method"] == method]
        summary_row = {"method": method, "n": len(chosen)}
        for metric in METRICS:
            avg, sd = mean_sd([row[metric] for row in chosen])
            summary_row[metric + "_mean"] = avg
            summary_row[metric + "_sample_sd"] = sd
        summary.append(summary_row)
    with (output / "method_mean_sample_sd.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        fields = list(summary[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    differences = []
    pairs = (("PPO_Reward", "PPO"), ("FULL", "PPO_Reward"), ("FULL", "PPO"))
    indexed = {(row["method"], row["seed"]): row for row in records}
    for left, right in pairs:
        for seed in seeds:
            for metric in METRICS:
                differences.append({"left_method": left, "right_method": right, "seed": seed, "metric": metric, "paired_difference_left_minus_right": indexed[(left, seed)][metric] - indexed[(right, seed)][metric]})
    with (output / "paired_method_differences.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        fields = list(differences[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(differences)
    n = len(seeds)
    if n == 3:
        inference = {"status": "DESCRIPTIVE_ONLY", "reason": "The confirmed design has n=3 paired seeds; no formal significance test is authorized."}
    elif n >= 8 and identity["protocol"]["statistics"].get("inferential_status") == "CONFIRMED_BY_USER":
        inference = {"status": "READY_FOR_SEPARATE_CONFIRMED_INFERENCE", "reason": "No inferential test is automatically executed by this aggregate."}
    else:
        inference = {"status": "BLOCKED", "reason": "Inferential plan is not confirmed."}
    write_json_new(output / "aggregate_manifest.json", {
        "schema_version": "movingai64_aggregate.v1",
        "config_sha256": identity["config_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "methods": list(ALLOWED_METHODS),
        "seeds": seeds,
        "metrics": list(METRICS),
        "inference": inference,
        "formal_statistical_inference_executed": False,
    })
    print(json.dumps({"output": str(output), "inference": inference}, indent=2))


if __name__ == "__main__":
    main()
