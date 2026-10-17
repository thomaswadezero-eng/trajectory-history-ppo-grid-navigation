#!/usr/bin/env python
"""Aggregate validated component runs and run the frozen paired analysis."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import component_ablation_8seed_v1 as impl


SCRIPT_PATH = Path(__file__).resolve()
PLAN_PATH = SCRIPT_PATH.parent / "component_hypothesis_plan_v1.json"
V7_ANALYSIS_PATH = impl.PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7_8seed" / "analyze_significance.py"
V7_OUTPUT_ROOT = Path(r"redacted_local_root\final_two_factor_experiment_v7")
METHODS = tuple(impl.CONFIGURATIONS)
SEEDS = tuple(impl.SEEDS)
REFERENCE = "PPO_Reward"
DISPLAY = {
    "PPO_Reward": "PPO+Reward",
    "REWARD_WO_PROGRESS": "w/o Progress",
    "REWARD_WO_REVISIT": "w/o Revisit",
    "REWARD_WO_OSCILLATION": "w/o Oscillation",
    "REWARD_WO_STAGNATION": "w/o Stagnation",
}
DESCRIPTIVE_METRICS = (
    "success_rate", "spl", "path_efficiency", "average_steps", "average_path_length",
    "collision_rate", "timeout_rate", "episode_reward",
)


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_frozen_plan() -> Dict[str, Any]:
    plan = read_json(PLAN_PATH)
    if plan.get("status") != "FROZEN_BEFORE_NEW_FORMAL_RESULTS":
        raise AnalysisError("component hypothesis plan is not frozen")
    source_plan = impl.PROJECT_ROOT / str(plan["provenance"]["source_plan"])
    if impl.sha256_file(source_plan) != str(plan["provenance"]["source_plan_sha256"]).upper():
        raise AnalysisError("source significance plan SHA mismatch")
    if tuple(plan.get("seeds", [])) != SEEDS or plan.get("reference_method") != REFERENCE:
        raise AnalysisError("hypothesis plan seed/reference mismatch")
    if plan.get("test") != "two-sided paired Wilcoxon signed-rank test":
        raise AnalysisError("hypothesis plan test mismatch")
    bootstrap = plan.get("bootstrap", {})
    if int(bootstrap.get("replicates", -1)) != 100000 or int(bootstrap.get("analysis_seed", -1)) != 20260828:
        raise AnalysisError("hypothesis plan bootstrap identity mismatch")
    if plan.get("family_id") != "COMPONENT_ABLATION_12_HYPOTHESES" or int(plan.get("hypothesis_count", -1)) != 12:
        raise AnalysisError("component hypothesis family identity/count mismatch")
    if plan.get("multiple_comparison_correction") != "One Holm correction jointly across all 12 raw two-sided paired Wilcoxon p-values in the predefined component-ablation family":
        raise AnalysisError("component Holm family definition mismatch")
    expected_methods = tuple(comparison["method_a"] for comparison in plan.get("comparisons", []))
    if expected_methods != METHODS:
        raise AnalysisError("hypothesis comparison order/method mismatch")
    for comparison in plan["comparisons"]:
        if comparison["method_b"] != REFERENCE or tuple(comparison["primary_metrics"]) != ("success_rate", "spl", "timeout_rate"):
            raise AnalysisError("component hypothesis metric set drift")
    if sum(len(comparison["primary_metrics"]) for comparison in plan["comparisons"]) != int(plan["hypothesis_count"]):
        raise AnalysisError("component hypothesis family cardinality mismatch")
    expected_hypothesis_order = tuple(
        f"{comparison['hypothesis']}__{metric.upper()}"
        for comparison in plan["comparisons"]
        for metric in comparison["primary_metrics"]
    )
    if tuple(plan.get("hypothesis_order", [])) != expected_hypothesis_order:
        raise AnalysisError("component hypothesis identity/order mismatch")
    return plan


def load_v7_analysis_module() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_v7_significance_validator", V7_ANALYSIS_PATH)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"cannot import V7 analysis validator: {V7_ANALYSIS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mean_episode_return(evaluation_path: Path) -> Optional[float]:
    with evaluation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["record_type"] == "episode" and row["split"] == "test"]
    if not rows or "episode_return" not in rows[0] or any(row["episode_return"] == "" for row in rows):
        return None
    return statistics.mean(float(row["episode_return"]) for row in rows)


def reference_records() -> List[Dict[str, Any]]:
    validator = load_v7_analysis_module()
    records: List[Dict[str, Any]] = []
    for seed in SEEDS:
        validated = validator.validate_run(V7_OUTPUT_ROOT, REFERENCE, seed)
        result_path = Path(str(validated["formal_result"]))
        result = read_json(result_path)
        test = dict(result["test"])
        records.append({
            "method": REFERENCE,
            "seed": seed,
            "status": "VALIDATED_REFERENCE",
            "provenance": "V7_PPO_Reward",
            "source_path": str(result_path),
            "source_sha256": impl.sha256_file(result_path),
            "test": test,
            "episode_reward": None,
        })
    return records


def component_records(output_root: Path, identity: Mapping[str, Any], legacy_records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    from run_component_ablation_8seeds import validate_done

    legacy_by_key = {(str(record["method"]), int(record["seed"])): record for record in legacy_records if record.get("status") == "REUSABLE"}
    records: List[Dict[str, Any]] = []
    statuses: Dict[str, str] = {}
    for method in METHODS:
        for seed in SEEDS:
            key = f"{method}/seed_{seed}"
            legacy_record = legacy_by_key.get((method, seed))
            if legacy_record is not None:
                result_path = Path(str(legacy_record["result_path"]))
                test = dict(legacy_record["test_metrics"])
                records.append({
                    "method": method,
                    "seed": seed,
                    "status": "REUSABLE_LEGACY",
                    "provenance": "internal_ablation_v1",
                    "source_path": str(result_path),
                    "source_sha256": str(legacy_record["result_sha256"]),
                    "test": test,
                    "episode_reward": mean_episode_return(result_path.parent / "evaluation_metrics.csv"),
                })
                statuses[key] = "REUSABLE_LEGACY"
                continue
            done_path = output_root / method / f"seed_{seed}" / "DONE.json"
            result = validate_done(done_path, method, seed, identity)
            if result is None:
                statuses[key] = "MISSING_OR_FAILED"
                continue
            source_path = Path(str(read_json(done_path)["result_path"]))
            records.append({
                "method": method,
                "seed": seed,
                "status": "DONE_REVISION",
                "provenance": "revision_component_ablation_8seed_v1",
                "source_path": str(source_path),
                "source_sha256": impl.sha256_file(source_path),
                "test": dict(result["test"]),
                "episode_reward": mean_episode_return(source_path.parent / "evaluation_metrics.csv"),
            })
            statuses[key] = "DONE_REVISION"
    return records, statuses


def values_by_seed(records: Sequence[Mapping[str, Any]], method: str, metric: str) -> Dict[int, float]:
    output: Dict[int, float] = {}
    for record in records:
        if record["method"] != method:
            continue
        value = record.get("episode_reward") if metric == "episode_reward" else record["test"].get(metric)
        if value is not None and value != "":
            output[int(record["seed"])] = float(value)
    return output


def rank_biserial(differences: Sequence[float]) -> float:
    from scipy.stats import rankdata

    nonzero = [float(value) for value in differences if float(value) != 0.0]
    if not nonzero:
        return 0.0
    ranks = rankdata(np.abs(np.asarray(nonzero, dtype=np.float64)), method="average")
    positive = float(sum(rank for rank, value in zip(ranks, nonzero) if value > 0.0))
    negative = float(sum(rank for rank, value in zip(ranks, nonzero) if value < 0.0))
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def apply_unified_family_holm(rows: List[Dict[str, Any]], alpha: float, family_id: str, expected_count: int) -> None:
    if len(rows) != expected_count:
        raise AnalysisError(f"unified Holm family requires {expected_count} hypotheses, received {len(rows)}")
    adjusted = holm_adjust([float(row["raw_p_value"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_family_id"] = family_id
        row["holm_family_size"] = expected_count
        row["holm_adjusted_p_value"] = adjusted_p
        row["holm_reject"] = bool(adjusted_p < alpha)


def bootstrap_ci(differences: Sequence[float], rng: np.random.Generator, replicates: int) -> Tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    chunk = 10000
    means: List[np.ndarray] = []
    remaining = replicates
    while remaining:
        count = min(chunk, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(means)
    return float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def sha_manifest(directory: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/"): impl.sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "aggregate_hash_manifest.json" and not path.name.endswith(".tmp")
    }


def aggregate_and_analyze(output_root: Path, identity: Mapping[str, Any], legacy_records: Sequence[Mapping[str, Any]], log: Callable[[str], None] = print) -> Path:
    plan = load_frozen_plan()
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    components, statuses = component_records(output_root, identity, legacy_records)
    references = reference_records()
    records = references + components
    per_seed_rows: List[Dict[str, Any]] = []
    for method in (REFERENCE,) + METHODS:
        by_seed = {int(record["seed"]): record for record in records if record["method"] == method}
        for seed in SEEDS:
            record = by_seed.get(seed)
            row: Dict[str, Any] = {"method": method, "display_name": DISPLAY[method], "seed": seed}
            if record is None:
                row.update({"status": "MISSING", "provenance": "", "source_path": "", "source_sha256": ""})
                row.update({metric: "" for metric in DESCRIPTIVE_METRICS})
            else:
                row.update({"status": record["status"], "provenance": record["provenance"], "source_path": record["source_path"], "source_sha256": record["source_sha256"]})
                for metric in DESCRIPTIVE_METRICS:
                    row[metric] = record.get("episode_reward") if metric == "episode_reward" else record["test"].get(metric, "")
                    if row[metric] is None:
                        row[metric] = ""
                test = record["test"]
                row["success_numerator"] = test.get("success_count", "")
                row["episode_denominator"] = test.get("episode_count", "")
            per_seed_rows.append(row)
    per_seed_fields = list(per_seed_rows[0])
    write_csv_atomic(aggregate_dir / "component_per_seed_metrics.csv", per_seed_fields, per_seed_rows)

    summary_rows: List[Dict[str, Any]] = []
    for method in (REFERENCE,) + METHODS:
        for metric in DESCRIPTIVE_METRICS:
            values = list(values_by_seed(records, method, metric).values())
            summary_rows.append({
                "method": method,
                "display_name": DISPLAY[method],
                "metric": metric,
                "actual_n": len(values),
                "planned_n": 8,
                "mean": statistics.mean(values) if values else "",
                "sample_sd": statistics.stdev(values) if len(values) >= 2 else "",
            })
    write_csv_atomic(aggregate_dir / "component_mean_sample_sd.csv", list(summary_rows[0]), summary_rows)

    paired_rows: List[Dict[str, Any]] = []
    for comparison in plan["comparisons"]:
        method = str(comparison["method_a"])
        for metric in DESCRIPTIVE_METRICS:
            left = values_by_seed(records, method, metric)
            right = values_by_seed(records, REFERENCE, metric)
            direction = plan["metric_directions"].get(metric, "descriptive_only")
            for seed in SEEDS:
                raw = ""
                benefit = ""
                if seed in left and seed in right:
                    raw = right[seed] - left[seed]
                    benefit = raw if direction == "higher_is_better" else (-raw if direction == "lower_is_better" else "")
                paired_rows.append({
                    "hypothesis": comparison["hypothesis"], "comparison": comparison["comparison"], "metric": metric,
                    "seed": seed, "method_without_component_value": left.get(seed, ""), "ppo_reward_reference_value": right.get(seed, ""),
                    "raw_difference_reference_minus_ablation": raw, "benefit_oriented_difference": benefit,
                    "direction": direction,
                })
    write_csv_atomic(aggregate_dir / "component_paired_differences.csv", list(paired_rows[0]), paired_rows)

    all_component_complete = len(components) == len(METHODS) * len(SEEDS)
    statistical_rows: List[Dict[str, Any]] = []
    statistical_status: Dict[str, Any] = {
        "schema_version": "component_ablation_statistical_status.v1",
        "status": "COMPLETE" if all_component_complete else "PENDING_INCOMPLETE_RUN_MATRIX",
        "required_pairs": 8,
        "component_records": len(components),
        "required_component_records": len(METHODS) * len(SEEDS),
        "hypothesis_plan": str(PLAN_PATH),
        "hypothesis_plan_sha256": impl.sha256_file(PLAN_PATH),
    }
    if all_component_complete:
        from scipy.stats import wilcoxon

        rng = np.random.default_rng(int(plan["bootstrap"]["analysis_seed"]))
        for comparison in plan["comparisons"]:
            for metric in comparison["primary_metrics"]:
                left_map = values_by_seed(records, comparison["method_a"], metric)
                right_map = values_by_seed(records, comparison["method_b"], metric)
                if tuple(left_map) != SEEDS or tuple(right_map) != SEEDS:
                    raise AnalysisError(f"paired seed mismatch for {comparison['hypothesis']}/{metric}")
                left = [left_map[seed] for seed in SEEDS]
                right = [right_map[seed] for seed in SEEDS]
                raw_differences = [b - a for a, b in zip(left, right)]
                higher = plan["metric_directions"][metric] == "higher_is_better"
                benefits = raw_differences if higher else [-value for value in raw_differences]
                if all(value == 0.0 for value in benefits):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = wilcoxon(benefits, zero_method="wilcox", correction=False, alternative="two-sided", method="auto")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                lower, upper = bootstrap_ci(benefits, rng, int(plan["bootstrap"]["replicates"]))
                row = {
                    "hypothesis_id": f"{comparison['hypothesis']}__{metric.upper()}",
                    "hypothesis": comparison["hypothesis"], "comparison": comparison["comparison"], "metric": metric,
                    "direction": plan["metric_directions"][metric], "n_pairs": 8, "seeds": ",".join(str(seed) for seed in SEEDS),
                    "method_a": comparison["method_a"], "method_b": comparison["method_b"],
                    "method_a_mean": statistics.mean(left), "method_a_sample_sd": statistics.stdev(left),
                    "method_b_mean": statistics.mean(right), "method_b_sample_sd": statistics.stdev(right),
                    "raw_mean_difference_b_minus_a": statistics.mean(raw_differences),
                    "benefit_oriented_mean_difference": statistics.mean(benefits),
                    "wilcoxon_statistic": statistic, "raw_p_value": p_value, "holm_adjusted_p_value": None,
                    "alpha": float(plan["alpha"]), "holm_reject": None,
                    "rank_biserial_effect_size": rank_biserial(benefits),
                    "bootstrap_ci_level": float(plan["bootstrap"]["confidence_level"]),
                    "bootstrap_mean_benefit_delta_lower": lower, "bootstrap_mean_benefit_delta_upper": upper,
                    "bootstrap_replicates": int(plan["bootstrap"]["replicates"]), "analysis_seed": int(plan["bootstrap"]["analysis_seed"]),
                    "method_a_values_by_seed": json.dumps(left), "method_b_values_by_seed": json.dumps(right),
                    "raw_deltas_by_seed": json.dumps(raw_differences), "benefit_deltas_by_seed": json.dumps(benefits),
                }
                statistical_rows.append(row)
        if tuple(row["hypothesis_id"] for row in statistical_rows) != tuple(plan["hypothesis_order"]):
            raise AnalysisError("generated component hypothesis identity/order mismatch")
        apply_unified_family_holm(
            statistical_rows,
            alpha=float(plan["alpha"]),
            family_id=str(plan["family_id"]),
            expected_count=int(plan["hypothesis_count"]),
        )
        write_csv_atomic(aggregate_dir / "component_statistical_tests.csv", list(statistical_rows[0]), statistical_rows)
        write_json_atomic(aggregate_dir / "component_statistical_tests.json", {"schema_version": "component_ablation_statistical_tests.v1", "plan": plan, "results": statistical_rows})
    else:
        write_csv_atomic(aggregate_dir / "component_statistical_tests.csv", ["status", "reason"], [{"status": "PENDING", "reason": "all 32 component runs are required"}])
        write_json_atomic(aggregate_dir / "component_statistical_tests.json", {"schema_version": "component_ablation_statistical_tests.v1", "status": "PENDING", "reason": "all 32 component runs are required", "results": []})
    write_json_atomic(aggregate_dir / "statistical_analysis_status.json", statistical_status)
    write_json_atomic(aggregate_dir / "completed_missing_failed_status.json", {
        "schema_version": "component_ablation_batch_status.v1",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "official_seeds": SEEDS,
        "methods": METHODS,
        "statuses": statuses,
        "legacy_reusable_count": sum(record.get("status") == "REUSABLE" for record in legacy_records),
        "revision_done_count": sum(value == "DONE_REVISION" for value in statuses.values()),
        "complete_count": sum(value in ("REUSABLE_LEGACY", "DONE_REVISION") for value in statuses.values()),
        "planned_count": len(METHODS) * len(SEEDS),
    })
    write_json_atomic(aggregate_dir / "audit_manifest.json", {
        "schema_version": "component_ablation_aggregate_audit.v1",
        "program_identity": dict(identity),
        "batch_runner_sha256": impl.sha256_file(SCRIPT_PATH.parent / "run_component_ablation_8seeds.py"),
        "aggregate_program_sha256": impl.sha256_file(SCRIPT_PATH),
        "hypothesis_plan_sha256": impl.sha256_file(PLAN_PATH),
        "reference_method": REFERENCE,
        "reference_records": [{key: record[key] for key in ("seed", "source_path", "source_sha256")} for record in references],
        "component_records": [{key: record[key] for key in ("method", "seed", "status", "source_path", "source_sha256")} for record in components],
    })
    write_json_atomic(aggregate_dir / "aggregate_hash_manifest.json", {"schema_version": "component_ablation_aggregate_hashes.v1", "files": sha_manifest(aggregate_dir)})
    log(f"[COMPONENT BATCH] aggregate generated: {aggregate_dir}; component records={len(components)}/32; statistics={statistical_status['status']}")
    return aggregate_dir


if __name__ == "__main__":
    raise SystemExit("This module is invoked by the validated batch runner after formal runs; do not run it as a training command.")
