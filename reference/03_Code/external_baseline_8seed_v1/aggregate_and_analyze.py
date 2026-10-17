#!/usr/bin/env python
"""Aggregate validated legacy plus revision runs and gate paired inference."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import audit_existing_runs
import common


PLAN_PATH = common.REVISION_ROOT / "external_baseline_hypothesis_plan_v1.json"
PLAN_SHA256 = "B732954C60FA1CB4D6CD2D8CB74186FBB6D09BB353686B5CEB444E21FA5FCA62"
AGGREGATE_ROOT = common.OUTPUT_ROOT / "aggregate"
FROZEN_STATUS = "FROZEN_BEFORE_NEW_FORMAL_RESULTS"
FAMILY_IDS = (
    "EXTERNAL_DDQN_3_METRICS",
    "EXTERNAL_D3QN_3_METRICS",
    "EXTERNAL_C51_3_METRICS",
    "EXTERNAL_MASKABLEPPO_3_METRICS",
)
PRIMARY_METRICS = ("success_rate", "spl", "path_efficiency")


def legacy_records(audit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in audit["legacy_runs"]:
        if item["status"] != "REUSABLE_LEGACY":
            continue
        records.append({
            "method": item["method"], "seed": int(item["seed"]), "provenance": "REUSABLE_LEGACY",
            "source_path": item["formal_result_path"], "source_sha256": item["formal_result_sha256"],
            **{metric: item.get(metric) for metric in common.METRICS},
            "success_count": item.get("success_count"), "collision_count": item.get("collision_count"),
            "timeout_count": item.get("timeout_count"), "episode_count": item.get("test_episode_count"),
        })
    return records


def revision_records() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []
    from run_external_baselines_8seeds import validate_done
    for method in common.METHODS:
        for seed in common.SEEDS:
            seed_dir = common.RUNS_ROOT / method / f"seed_{seed}"
            done_paths = sorted(seed_dir.glob("run_*/DONE.json")) if seed_dir.is_dir() else []
            valid = [path for path in done_paths if validate_done(path, method, seed)]
            invalid = [path for path in done_paths if path not in valid]
            if invalid or len(valid) > 1:
                statuses.append({"method": method, "seed": seed, "status": "INVALID_OR_AMBIGUOUS_DONE", "paths": json.dumps([str(path) for path in done_paths])})
                continue
            if not valid:
                statuses.append({"method": method, "seed": seed, "status": "MISSING_REVISION", "paths": ""})
                continue
            done = common.read_json(valid[0])
            result_path = Path(str(done["formal_result"]))
            result = common.read_json(result_path)
            test = result["test"]
            records.append({
                "method": method, "seed": seed, "provenance": "DONE_REVISION",
                "source_path": str(result_path), "source_sha256": common.sha256_file(result_path),
                "success_rate": test.get("success_rate"), "spl": test.get("spl"),
                "path_efficiency": test.get("path_efficiency"), "average_steps": test.get("average_steps"),
                "average_path_length": test.get("average_path_length"), "collision_rate": test.get("collision_rate"),
                "timeout_rate": test.get("timeout_rate"), "episode_reward": test.get("average_episode_reward"),
                "success_count": test.get("success_count"), "collision_count": test.get("collision_count"),
                "timeout_count": test.get("timeout_count"), "episode_count": test.get("episode_count"),
            })
            statuses.append({"method": method, "seed": seed, "status": "DONE_REVISION", "paths": str(valid[0])})
    return records, statuses


def full_records(audit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = []
    for item in audit["full_reference_runs"]:
        if item["status"] != "VALID_READ_ONLY_REFERENCE":
            continue
        test = item["test"]
        records.append({
            "method": "FULL", "seed": int(item["seed"]), "provenance": "READ_ONLY_FULL_REFERENCE",
            "source_path": item["formal_result_path"], "source_sha256": item["formal_result_sha256"],
            "success_rate": test.get("success_rate"), "spl": test.get("spl"),
            "path_efficiency": test.get("path_efficiency"), "average_steps": test.get("average_steps"),
            "average_path_length": test.get("average_path_length"), "collision_rate": test.get("collision_rate"),
            "timeout_rate": test.get("timeout_rate"), "episode_reward": test.get("average_episode_reward"),
            "success_count": test.get("success_count"), "collision_count": test.get("collision_count"),
            "timeout_count": test.get("timeout_count"), "episode_count": test.get("episode_count"),
        })
    return records


def summary_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for method in (*common.METHODS, "FULL"):
        subset = [record for record in records if record["method"] == method]
        for metric in common.METRICS:
            values = [float(record[metric]) for record in subset if record.get(metric) not in (None, "")]
            output.append({
                "method": method, "metric": metric, "n": len(values),
                "mean": statistics.fmean(values) if values else "",
                "sample_sd": statistics.stdev(values) if len(values) >= 2 else "",
                "seeds": ",".join(str(record["seed"]) for record in subset if record.get(metric) not in (None, "")),
            })
    return output


def paired_inference_complete(records: Sequence[Mapping[str, Any]]) -> bool:
    """Require one record for every exact method/seed pair and every primary metric."""
    expected = {
        (method, seed)
        for method in (*common.METHODS, "FULL")
        for seed in common.SEEDS
    }
    keys = [(str(record["method"]), int(record["seed"])) for record in records]
    if len(keys) != len(expected) or set(keys) != expected:
        return False
    return all(
        record.get(metric) not in (None, "")
        for record in records for metric in PRIMARY_METRICS
    )


def holm(p_values: Sequence[float]) -> List[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def validate_frozen_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("status") != FROZEN_STATUS or plan.get("frozen_at") != "2026-09-20":
        raise common.AuditError("statistical plan is not the frozen pre-result plan")
    if tuple(plan.get("paired_seeds", ())) != common.SEEDS:
        raise common.AuditError("frozen plan paired-seed identity mismatch")
    if tuple(plan.get("primary_metrics", ())) != PRIMARY_METRICS:
        raise common.AuditError("frozen plan primary-metric identity mismatch")
    reference = plan.get("reference", {})
    if reference.get("method") != "FULL" or reference.get("mode") != "READ_ONLY" or reference.get("retraining_permitted") is not False:
        raise common.AuditError("frozen plan FULL read-only identity mismatch")
    procedure = plan.get("procedure", {})
    expected_procedure = {
        "alternative": "two-sided", "alpha": 0.05,
        "pairing": "exact seed identity", "wilcoxon_zero_method": "wilcox",
        "wilcoxon_method": "auto", "global_12_test_holm": False,
    }
    if any(procedure.get(key) != value for key, value in expected_procedure.items()):
        raise common.AuditError("frozen statistical procedure mismatch")
    bootstrap = procedure.get("bootstrap", {})
    if bootstrap.get("replicates") != 100000 or bootstrap.get("analysis_seed") != 20260828:
        raise common.AuditError("frozen bootstrap identity mismatch")
    families = plan.get("families")
    if not isinstance(families, list) or tuple(family.get("family_id") for family in families) != FAMILY_IDS:
        raise common.AuditError("frozen family identity/order mismatch")
    expected_methods = common.METHODS
    for family, method in zip(families, expected_methods):
        hypotheses = family.get("hypotheses")
        metrics = tuple(item.get("metric") for item in hypotheses) if isinstance(hypotheses, list) else ()
        directions = tuple(item.get("direction") for item in hypotheses) if isinstance(hypotheses, list) else ()
        if family.get("family_size") != 3 or metrics != PRIMARY_METRICS:
            raise common.AuditError(f"invalid three-metric family: {family.get('family_id')}")
        if directions != ("higher_is_better",) * 3:
            raise common.AuditError(f"metric direction mismatch: {family.get('family_id')}")
        if family.get("comparison") != {"method_a": method, "method_b": "FULL"}:
            raise common.AuditError(f"comparison mismatch: {family.get('family_id')}")
    descriptive = tuple(plan.get("descriptive_metrics", ()))
    if "average_steps" not in descriptive or any(metric in PRIMARY_METRICS for metric in descriptive):
        raise common.AuditError("descriptive/primary metric boundary mismatch")


def load_frozen_plan() -> Dict[str, Any]:
    actual = common.sha256_file(PLAN_PATH)
    if actual != PLAN_SHA256:
        raise common.AuditError(f"frozen statistical-plan SHA mismatch: actual={actual}, expected={PLAN_SHA256}")
    plan = common.read_json(PLAN_PATH)
    validate_frozen_plan(plan)
    return plan


def apply_family_holm(rows: List[Dict[str, Any]]) -> None:
    """Apply exactly one Holm correction to each predefined three-row family."""
    observed = {str(row["family_id"]) for row in rows}
    if observed != set(FAMILY_IDS):
        raise common.AuditError("statistical rows do not contain the exact four frozen families")
    for family_id in FAMILY_IDS:
        family_rows = [row for row in rows if row["family_id"] == family_id]
        if len(family_rows) != 3:
            raise common.AuditError(f"Holm family must contain exactly three rows: {family_id}")
        adjusted = holm([float(row["raw_p_value"]) for row in family_rows])
        for row, value in zip(family_rows, adjusted):
            row["holm_adjusted_p_value"] = value
            row["holm_reject"] = value <= 0.05


def rank_biserial(benefits: Sequence[float]) -> float:
    nonzero = [value for value in benefits if value != 0.0]
    if not nonzero:
        return 0.0
    ordered = sorted(enumerate(nonzero), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(nonzero)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and abs(ordered[end][1]) == abs(ordered[cursor][1]):
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = average_rank
        cursor = end
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    denominator = positive + negative
    return 0.0 if denominator == 0 else (positive - negative) / denominator


def bootstrap_ci(values: Sequence[float], seed: int, replicates: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.fmean(sample))
    estimates.sort()
    return estimates[int(0.025 * replicates)], estimates[min(replicates - 1, int(0.975 * replicates))]


def statistical_rows(records: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    validate_frozen_plan(plan)
    from scipy.stats import wilcoxon  # imported only after both gates pass
    indexed = {(record["method"], int(record["seed"])): record for record in records}
    output: List[Dict[str, Any]] = []
    procedure = plan["procedure"]
    for family in plan["families"]:
        comparison = family["comparison"]
        for hypothesis in family["hypotheses"]:
            metric = hypothesis["metric"]
            left = [float(indexed[(comparison["method_a"], seed)][metric]) for seed in common.SEEDS]
            right = [float(indexed[(comparison["method_b"], seed)][metric]) for seed in common.SEEDS]
            direction = hypothesis["direction"]
            raw = [b - a for a, b in zip(left, right)]
            benefits = raw if direction == "higher_is_better" else [-value for value in raw]
            if all(value == 0.0 for value in benefits):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    benefits, alternative=procedure["alternative"],
                    zero_method=procedure["wilcoxon_zero_method"], method=procedure["wilcoxon_method"],
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            lower, upper = bootstrap_ci(
                benefits, int(procedure["bootstrap"]["analysis_seed"]),
                int(procedure["bootstrap"]["replicates"]),
            )
            output.append({
                "family_id": family["family_id"], "family_size": 3,
                "hypothesis_id": hypothesis["hypothesis_id"],
                "comparison": f"{comparison['method_a']} vs {comparison['method_b']}", "metric": metric,
                "n_pairs": 8, "seeds": ",".join(map(str, common.SEEDS)), "wilcoxon_statistic": statistic,
                "raw_p_value": p_value, "holm_adjusted_p_value": None, "holm_reject": None,
                "benefit_oriented_mean_difference": statistics.fmean(benefits),
                "rank_biserial_effect_size": rank_biserial(benefits),
                "bootstrap_mean_benefit_delta_lower": lower, "bootstrap_mean_benefit_delta_upper": upper,
            })
    apply_family_holm(output)
    return output


def aggregate() -> Path:
    audit = audit_existing_runs.run_audit()
    legacy = legacy_records(audit)
    revision, revision_status = revision_records()
    full = full_records(audit)
    records = sorted((*legacy, *revision, *full), key=lambda item: ((*common.METHODS, "FULL").index(item["method"]), int(item["seed"])))
    keys = [(record["method"], int(record["seed"])) for record in records]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    pairing_rows = []
    for method in common.METHODS:
        for seed in common.SEEDS:
            matching = [record for record in records if record["method"] == method and int(record["seed"]) == seed]
            pairing_rows.append({"method": method, "seed": seed, "count": len(matching), "status": "PAIRED_PRESENT" if len(matching) == 1 else "MISSING_OR_AMBIGUOUS"})
    complete = (
        not duplicates
        and all(row["status"] == "PAIRED_PRESENT" for row in pairing_rows)
        and paired_inference_complete(records)
    )
    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)
    seed_fields = ["method", "seed", "provenance", "source_path", "source_sha256", "episode_count", "success_count", "collision_count", "timeout_count", *common.METRICS]
    common.write_csv_atomic(AGGREGATE_ROOT / "external_baseline_seed_level.csv", seed_fields, records)
    summaries = summary_rows(records)
    common.write_csv_atomic(AGGREGATE_ROOT / "external_baseline_summary.csv", list(summaries[0]), summaries)
    common.write_csv_atomic(AGGREGATE_ROOT / "external_baseline_pairing_audit.csv", list(pairing_rows[0]), pairing_rows)
    plan = load_frozen_plan()
    if not complete:
        statistics_status = "INCOMPLETE_PAIRED_DATA"
        stats: List[Dict[str, Any]] = []
    else:
        statistics_status = "COMPLETED"
        stats = statistical_rows(records, plan)
    if stats:
        common.write_csv_atomic(AGGREGATE_ROOT / "external_baseline_statistics.csv", list(stats[0]), stats)
    else:
        common.write_csv_atomic(AGGREGATE_ROOT / "external_baseline_statistics.csv", ["status", "reason"], [{"status": statistics_status, "reason": statistics_status}])
    common.write_json_atomic(AGGREGATE_ROOT / "external_baseline_statistics.json", {"schema_version": "external_baseline_statistics.v1", "status": statistics_status, "plan": plan, "results": stats})
    common.write_json_atomic(AGGREGATE_ROOT / "external_baseline_provenance.json", {
        "schema_version": "external_baseline_8seed_provenance.v1", "generated_at": common.now_iso(),
        "legacy_audit_sha256": common.sha256_file(common.REVISION_ROOT / "legacy_run_audit.json"),
        "statistical_plan_sha256": PLAN_SHA256,
        "legacy_records": legacy, "revision_records": revision, "revision_status": revision_status,
        "full_reference_records": full, "duplicates": duplicates, "paired_complete": complete,
        "statistics_status": statistics_status, "full_is_read_only": True,
    })
    return AGGREGATE_ROOT


if __name__ == "__main__":
    path = aggregate()
    print(path)
