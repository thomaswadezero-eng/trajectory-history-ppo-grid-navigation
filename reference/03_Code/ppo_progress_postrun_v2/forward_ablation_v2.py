#!/usr/bin/env python
"""Read-only paired forward-ablation analysis for PPO, PPO+Progress, PPO+Reward."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from postrun_common import (
    EXPECTED_INPUT_SHA, EXPECTED_SHA, METHOD_PROGRESS, PostrunError, SEEDS,
    V7_FORMAL_ROOT, assert_finite_tree, bootstrap_mean_ci, finite_number,
    holm_adjust, rank_biserial, read_csv, read_json, require, sha256_file,
    static_authority_checks, wilcoxon_two_sided, write_csv, write_json, write_text,
)


METRICS = ("success_rate", "spl", "timeout_rate")
METHODS = ("PPO", METHOD_PROGRESS, "PPO_Reward")
DISPLAY = {"PPO": "PPO", METHOD_PROGRESS: "PPO+Progress", "PPO_Reward": "PPO+Reward"}
V7_IDENTITIES = {
    (EXPECTED_SHA["v7_program"], EXPECTED_SHA["v7_config_identity"], EXPECTED_SHA["v7_protocol_identity"]),
    (EXPECTED_SHA["v7_8seed_program"], "A1B343EE2D5FA67D09D0D1D4119261DB045BD892A5A231F1E534399C2C86B5B0", "CDEE0BDB835466E8610E5952DFE6683480D25E78A13DD9BBCD822F09897421BA"),
}
FAMILIES = (
    {"family_id": "F1_PPO_TO_PPO_PROGRESS", "comparison": "PPO vs PPO+Progress", "earlier": "PPO", "later": METHOD_PROGRESS,
     "scientific_scope": "adding the Progress term to base PPO under the matched protocol"},
    {"family_id": "F2_PPO_PROGRESS_TO_PPO_REWARD", "comparison": "PPO+Progress vs PPO+Reward", "earlier": METHOD_PROGRESS, "later": "PPO_Reward",
     "scientific_scope": "incremental effect of Revisit, Oscillation, and Stagnation as a block on top of Progress"},
)
BOOTSTRAP_REPLICATES = 100000
ANALYSIS_SEED = 20260828


def resolve_v7_result(method: str, seed: int, root: Path = V7_FORMAL_ROOT) -> Tuple[Dict[str, Any], Path]:
    seed_dir = root / method / "seed_{}".format(seed)
    completed = []
    for run in sorted(path for path in seed_dir.glob("run_*") if path.is_dir()):
        manifest_path = run / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        results = list(run.glob("formal_result_*.json"))
        if len(results) != 1:
            continue
        result = read_json(results[0])
        if result.get("method") == method and int(result.get("seed", -1)) == seed:
            completed.append((result, results[0]))
    require(len(completed) == 1, "{}/seed {} requires exactly one completed canonical result; found {}".format(method, seed, len(completed)))
    result, path = completed[0]
    assert_finite_tree(result, "{}.seed{}".format(method, seed))
    identity = (str(result.get("program_sha256")), str(result.get("config_sha256")), str(result.get("protocol_sha256")))
    require(identity in V7_IDENTITIES, "unauthorized V7 identity for {}/seed {}".format(method, seed))
    require(result.get("input_sha256") == EXPECTED_INPUT_SHA, "input identity mismatch for {}/seed {}".format(method, seed))
    require(int(result.get("training_interactions", -1)) == 900000, "training budget mismatch")
    test = result.get("test", {})
    require(int(test.get("episode_count", -1)) == 1000, "test count mismatch")
    require(sum(int(test.get(key, -1)) for key in ("success_count", "collision_count", "timeout_count")) == 1000,
            "outcomes are not exhaustive")
    require(int(test.get("parameter_updates", -1)) == 0, "test parameter updates are nonzero")
    require(int(result.get("validation_astar_runtime_calls", -1)) == 0 and int(result.get("test_astar_runtime_calls", -1)) == 0,
            "runtime A* calls are nonzero")
    for metric in METRICS:
        finite_number(test.get(metric), "{}.seed{}.{}".format(method, seed, metric))
    return result, path


def load_progress_audit(output_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    audit_path = output_dir / "ppo_progress_final_audit.json"
    audit = read_json(audit_path)
    require(audit.get("status") == "PASS", "PPO+Progress final audit is not PASS")
    require(audit.get("seeds") == list(SEEDS), "PPO+Progress audit seed identity mismatch")
    metrics_path = Path(str(audit.get("per_seed_metrics_file", ""))).resolve()
    require(metrics_path == (output_dir / "ppo_progress_per_seed_metrics.csv").resolve(), "audit per-seed metrics path mismatch")
    rows = read_csv(metrics_path)
    require(len(rows) == 8, "PPO+Progress audit metrics must contain eight rows")
    by_seed = {}
    for row in rows:
        seed = int(row["seed"])
        require(seed in SEEDS and seed not in by_seed, "invalid/duplicate PPO+Progress seed")
        require(row.get("method") == METHOD_PROGRESS, "PPO+Progress method mismatch")
        values = {metric: finite_number(row.get(metric), "progress.{}.{}".format(seed, metric)) for metric in METRICS}
        values.update({"formal_result": row.get("formal_result"), "formal_result_sha256": row.get("formal_result_sha256")})
        result_path = Path(str(values["formal_result"]))
        require(result_path.is_file() and sha256_file(result_path) == values["formal_result_sha256"], "PPO+Progress source result SHA mismatch")
        by_seed[seed] = values
    require(set(by_seed) == set(SEEDS), "PPO+Progress audit seed set mismatch")
    return by_seed, audit


def build_seed_data(output_dir: Path, v7_root: Path = V7_FORMAL_ROOT) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, float]]]]:
    progress, _ = load_progress_audit(output_dir)
    values = {method: {} for method in METHODS}
    provenance = {method: {} for method in METHODS}
    for seed in SEEDS:
        for method in ("PPO", "PPO_Reward"):
            result, source = resolve_v7_result(method, seed, v7_root)
            values[method][seed] = {metric: float(result["test"][metric]) for metric in METRICS}
            provenance[method][seed] = str(source)
        values[METHOD_PROGRESS][seed] = {metric: float(progress[seed][metric]) for metric in METRICS}
        provenance[METHOD_PROGRESS][seed] = str(progress[seed]["formal_result"])

    rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for method in METHODS:
            prefix = method.lower()
            for metric in METRICS:
                row["{}_{}".format(prefix, metric)] = values[method][seed][metric]
            row["{}_source".format(prefix)] = provenance[method][seed]
        for family in FAMILIES:
            for metric in METRICS:
                earlier = values[family["earlier"]][seed][metric]
                later = values[family["later"]][seed][metric]
                raw = later - earlier
                benefit = earlier - later if metric == "timeout_rate" else raw
                row["{}_{}_raw_later_minus_earlier".format(family["family_id"].lower(), metric)] = raw
                row["{}_{}_benefit".format(family["family_id"].lower(), metric)] = benefit
        rows.append(row)
    return rows, values


def compute_statistics(values: Mapping[str, Mapping[int, Mapping[str, float]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import numpy as np
    rng = np.random.default_rng(ANALYSIS_SEED)
    statistics_rows = []
    holm_rows = []
    effect_rows = []
    for family in FAMILIES:
        family_rows = []
        for metric in METRICS:
            earlier = [float(values[family["earlier"]][seed][metric]) for seed in SEEDS]
            later = [float(values[family["later"]][seed][metric]) for seed in SEEDS]
            raw = [right - left for left, right in zip(earlier, later)]
            benefit = [left - right for left, right in zip(earlier, later)] if metric == "timeout_rate" else list(raw)
            statistic, raw_p, method_used = wilcoxon_two_sided(benefit)
            lower, upper = bootstrap_mean_ci(benefit, rng, BOOTSTRAP_REPLICATES)
            row = {
                "family_id": family["family_id"], "family_size": 3, "comparison": family["comparison"],
                "scientific_scope": family["scientific_scope"], "metric": metric,
                "better_direction": "lower_is_better" if metric == "timeout_rate" else "higher_is_better",
                "n_pairs": 8, "seeds": ",".join(str(seed) for seed in SEEDS),
                "earlier_method": family["earlier"], "later_method": family["later"],
                "earlier_mean": statistics.mean(earlier), "earlier_sample_sd": statistics.stdev(earlier),
                "later_mean": statistics.mean(later), "later_sample_sd": statistics.stdev(later),
                "raw_delta_definition": "later_minus_earlier", "raw_mean_difference": statistics.mean(raw),
                "raw_median_difference": statistics.median(raw),
                "benefit_delta_definition": "earlier_minus_later" if metric == "timeout_rate" else "later_minus_earlier",
                "benefit_mean_difference": statistics.mean(benefit), "benefit_median_difference": statistics.median(benefit),
                "wilcoxon_statistic": statistic, "wilcoxon_alternative": "two-sided",
                "wilcoxon_zero_method": "wilcox", "wilcoxon_method": method_used,
                "zero_difference_count": sum(value == 0.0 for value in benefit), "raw_p_value": raw_p,
                "holm_adjusted_p_value": None, "alpha": 0.05, "holm_reject": None,
                "rank_biserial_benefit_oriented": rank_biserial(benefit),
                "rank_biserial_orientation": "positive_means_later_method_is_better",
                "bootstrap_ci_level": 0.95, "bootstrap_mean_benefit_delta_lower": lower,
                "bootstrap_mean_benefit_delta_upper": upper, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "analysis_seed": ANALYSIS_SEED, "earlier_values_by_seed": json.dumps(earlier),
                "later_values_by_seed": json.dumps(later), "raw_deltas_by_seed": json.dumps(raw),
                "benefit_deltas_by_seed": json.dumps(benefit),
            }
            family_rows.append(row)
        adjusted = holm_adjust([float(row["raw_p_value"]) for row in family_rows])
        for row, adjusted_p in zip(family_rows, adjusted):
            row["holm_adjusted_p_value"] = adjusted_p
            row["holm_reject"] = adjusted_p < 0.05
            holm_rows.append({key: row[key] for key in ("family_id", "family_size", "comparison", "metric", "raw_p_value", "holm_adjusted_p_value", "alpha", "holm_reject")})
            effect_rows.append({key: row[key] for key in ("family_id", "comparison", "metric", "benefit_mean_difference", "rank_biserial_benefit_oriented", "rank_biserial_orientation", "bootstrap_ci_level", "bootstrap_mean_benefit_delta_lower", "bootstrap_mean_benefit_delta_upper", "bootstrap_replicates", "analysis_seed")})
        statistics_rows.extend(family_rows)
    require(len(statistics_rows) == 6 and len(holm_rows) == 6, "forward analysis must contain six tests in two separate families")
    return statistics_rows, holm_rows, effect_rows


def target_absence(output_dir: Path) -> None:
    names = (
        "forward_ablation_seed_level.csv", "forward_ablation_aggregate.csv", "forward_ablation_statistics.csv",
        "forward_ablation_holm_families.csv", "forward_ablation_effect_sizes.csv", "forward_ablation_summary.json",
        "forward_ablation_report.md", "manuscript_ready_facts.md",
    )
    require(output_dir.is_dir(), "output directory must already contain the PASS audit")
    for name in names:
        require(not (output_dir / name).exists(), "refusing to overwrite {}".format(output_dir / name))


def build_reports(stat_rows: Sequence[Mapping[str, Any]], aggregate_rows: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    report = ["# Forward Ablation Statistical Analysis", "", "Two independent comparison families were analyzed with n=8 paired seeds.", "",
              "- F1: PPO vs PPO+Progress; adding Progress to base PPO.",
              "- F2: PPO+Progress vs PPO+Reward; adding Revisit, Oscillation, and Stagnation jointly on top of Progress.", "",
              "Each family contains Success Rate, SPL, and Timeout Rate and receives its own three-test Holm correction. Positive benefit-oriented differences and rank-biserial effects mean the later method is better.", "",
              "| Family | Metric | Benefit mean | Raw p | Holm p | Rank-biserial | 95% paired bootstrap CI |", "|---|---|---:|---:|---:|---:|---|"]
    facts = ["# Forward Ablation Results", "", "The inferential unit was the independently trained seed (n=8 paired seeds).", "",
             "## Descriptive values", "", "| Method | Metric | Mean | Sample SD |", "|---|---|---:|---:|"]
    for aggregate in aggregate_rows:
        facts.append("| {} | {} | {:.12g} | {:.12g} |".format(
            aggregate["display_name"], aggregate["metric"], aggregate["mean"], aggregate["sample_sd"]))
    facts.extend(["", "## Paired comparisons", "", "| Family | Metric | Benefit mean | Raw p | Holm p | Rank-biserial | 95% paired bootstrap CI |", "|---|---|---:|---:|---:|---:|---|"])
    for row in stat_rows:
        report.append("| {} | {} | {:.12g} | {:.12g} | {:.12g} | {:.12g} | [{:.12g}, {:.12g}] |".format(
            row["family_id"], row["metric"], row["benefit_mean_difference"], row["raw_p_value"],
            row["holm_adjusted_p_value"], row["rank_biserial_benefit_oriented"],
            row["bootstrap_mean_benefit_delta_lower"], row["bootstrap_mean_benefit_delta_upper"]))
        facts.append("| {} | {} | {:.12g} | {:.12g} | {:.12g} | {:.12g} | [{:.12g}, {:.12g}] |".format(
            row["family_id"], row["metric"], row["benefit_mean_difference"], row["raw_p_value"],
            row["holm_adjusted_p_value"], row["rank_biserial_benefit_oriented"],
            row["bootstrap_mean_benefit_delta_lower"], row["bootstrap_mean_benefit_delta_upper"]))
    facts.extend([
        "",
        "F1 compares PPO with PPO+Progress and estimates the incremental effect of adding the Progress term under the matched protocol.",
        "",
        "F2 compares PPO+Progress with PPO+Reward and estimates the joint incremental effect of Revisit, Oscillation, and Stagnation on top of Progress. It does not identify the separate effect of any one of these three corrections.",
        "",
        "Success Rate and SPL use later minus earlier; Timeout Rate uses earlier minus later, so positive values consistently favor the later method.",
    ])
    return "\n".join(report), "\n".join(facts)


def run_formal(output_dir: Path, v7_root: Path = V7_FORMAL_ROOT) -> Dict[str, Any]:
    static_authority_checks()
    target_absence(output_dir)
    seed_rows, values = build_seed_data(output_dir, v7_root)
    stat_rows, holm_rows, effect_rows = compute_statistics(values)
    aggregate_rows = []
    for method in METHODS:
        for metric in METRICS:
            series = [float(values[method][seed][metric]) for seed in SEEDS]
            aggregate_rows.append({"method": method, "display_name": DISPLAY[method], "metric": metric,
                                   "n_seeds": 8, "mean": statistics.mean(series), "sample_sd": statistics.stdev(series)})
    seed_fields = list(seed_rows[0].keys())
    write_csv(output_dir / "forward_ablation_seed_level.csv", seed_rows, seed_fields)
    write_csv(output_dir / "forward_ablation_aggregate.csv", aggregate_rows,
              ("method", "display_name", "metric", "n_seeds", "mean", "sample_sd"))
    write_csv(output_dir / "forward_ablation_statistics.csv", stat_rows, list(stat_rows[0].keys()))
    write_csv(output_dir / "forward_ablation_holm_families.csv", holm_rows, list(holm_rows[0].keys()))
    write_csv(output_dir / "forward_ablation_effect_sizes.csv", effect_rows, list(effect_rows[0].keys()))
    summary = {"schema_version": "forward_ablation.v2", "status": "COMPLETED", "n_paired_seeds": 8,
               "seeds": list(SEEDS), "families": [{"family_id": family["family_id"], "comparison": family["comparison"], "metrics": list(METRICS), "holm_family_size": 3} for family in FAMILIES],
               "wilcoxon": {"paired": True, "alternative": "two-sided", "zero_method": "wilcox", "method": "exact_with_auto_zero_fallback_and_all_zero_shortcut"},
               "bootstrap": {"paired": True, "replicates": BOOTSTRAP_REPLICATES, "confidence_level": 0.95, "analysis_seed": ANALYSIS_SEED},
               "analysis_program_sha256": sha256_file(Path(__file__).resolve()),
               "frozen_plan_sha256": EXPECTED_SHA["forward_ablation_plan"],
               "episode_level_pseudoreplication": False, "training_executed": False, "inference_executed": False,
               "checkpoint_loaded": False, "environment_interaction": False, "cuda_initialized": False}
    write_json(output_dir / "forward_ablation_summary.json", summary)
    report, facts = build_reports(stat_rows, aggregate_rows)
    write_text(output_dir / "forward_ablation_report.md", report)
    write_text(output_dir / "manuscript_ready_facts.md", facts)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only forward ablation statistical analysis")
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--v7-root", type=Path, default=V7_FORMAL_ROOT)
    parser.add_argument("--confirm-progress-audit-pass", action="store_true")
    parser.add_argument("--confirm-forward-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        checks = static_authority_checks()
        print(json.dumps({"status": "PREFLIGHT_PASS", "formal_future_results_accessed": False,
                          "families": ["F1_PPO_TO_PPO_PROGRESS", "F2_PPO_PROGRESS_TO_PPO_REWARD"],
                          "family_size": 3, "metrics": list(METRICS), "n_paired_seeds": 8,
                          "training": False, "cuda": False, "inference": False,
                          "checkpoint_loading": False, "authority_checks": len(checks)}, indent=2))
        return
    require(args.confirm_progress_audit_pass, "formal mode requires --confirm-progress-audit-pass")
    require(args.confirm_forward_analysis, "formal mode requires --confirm-forward-analysis")
    require(args.output_dir is not None, "formal mode requires --output-dir")
    print(json.dumps(run_formal(args.output_dir.resolve(), args.v7_root.resolve()), indent=2))


if __name__ == "__main__":
    try:
        main()
    except PostrunError as exc:
        raise SystemExit("FAIL CLOSED: {}".format(exc))
