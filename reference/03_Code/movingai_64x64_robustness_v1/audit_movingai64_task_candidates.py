#!/usr/bin/env python
"""User-only candidate capacity and horizon audit; never creates task artifacts."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
from pathlib import Path

from movingai64_task_rules import CANDIDATE_HORIZONS, CROP_RULE_ID, SUBSETS, SPLITS, collect_candidates, simulate_source_map_split
from protocol_utils import RESULT_ROOT, REVISION_DIR, ProtocolError, read_json, write_json_new


PERCENTILES = (("median", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
SLACK_THRESHOLDS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
PERCENTILE_RULE = "LINEAR_INTERPOLATION_ON_SORTED_VALUES; index=(n-1)*q; value=x[floor(index)]+fraction*(x[ceil(index)]-x[floor(index)])"


def canonical_json_sha256(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def percentile(values, quantile):
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def distribution(values):
    ordered = sorted(int(value) for value in values)
    result = {
        "candidate_count": len(ordered), "minimum": min(ordered) if ordered else None,
        "mean": (sum(ordered) / float(len(ordered))) if ordered else None,
        "maximum": max(ordered) if ordered else None,
    }
    for name, quantile in PERCENTILES:
        result[name] = percentile(ordered, quantile)
    return result


def coverage(values, horizon):
    count = len(values)
    covered = sum(1 for value in values if int(value) <= int(horizon))
    excluded = count - covered
    return {
        "candidate_count": count, "covered_count": covered,
        "valid_candidate_count_with_shortest_path_le_H": covered,
        "coverage_ratio": covered / float(count) if count else None,
        "excluded_count": excluded, "excluded_candidate_count": excluded,
        "excluded_ratio": excluded / float(count) if count else None,
    }


def slack(values, horizon):
    count = len(values)
    result = {"candidate_count": count}
    for threshold in SLACK_THRESHOLDS:
        key = "slack_ratio_ge_%s" % str(threshold).replace(".", "_")
        qualifying = sum(1 for value in values if float(horizon) / float(value) >= threshold)
        result[key] = {"count": qualifying, "ratio": qualifying / float(count) if count else None}
    return result


def first_candidate_horizon(values, minimum_ratio):
    for horizon in CANDIDATE_HORIZONS:
        item = coverage(values, horizon)
        if item["coverage_ratio"] is not None and item["coverage_ratio"] >= minimum_ratio:
            return horizon
    return None


def filtered_candidates(by_subset_map, horizon):
    return {
        subset: {name: [row for row in rows if int(row["shortest_path_length"]) <= int(horizon)] for name, rows in by_subset_map[subset].items()}
        for subset in SUBSETS
    }


def capacity_at_horizon(by_subset_map, protocol, horizon):
    filtered = filtered_candidates(by_subset_map, horizon)
    assignment, shortages = simulate_source_map_split(filtered, protocol)
    targets = protocol["data"]["draft_task_targets"]
    cap = int(protocol["data"]["maximum_tasks_per_source_map"])
    rows = []
    all_feasible = not shortages
    all_margin_ok = not shortages
    for split in SPLITS:
        for subset in SUBSETS:
            names = assignment[split][subset]
            before = sum(len(by_subset_map[subset].get(name, [])) for name in names)
            after_counts = [len(filtered[subset].get(name, [])) for name in names]
            after = sum(after_counts)
            capped = sum(min(count, cap) for count in after_counts)
            target = int(targets[split][subset])
            feasible = len(names) == int(protocol["data"]["draft_source_map_quotas"][split][subset]) and capped >= target
            margin_ok = feasible and capped >= target * 1.10
            all_feasible = all_feasible and feasible
            all_margin_ok = all_margin_ok and margin_ok
            rows.append({
                "split": split, "subset": subset, "source_map_count": len(names),
                "candidate_count_before_horizon_filter": before,
                "candidate_count_after_horizon_filter": after,
                "capped_capacity_after_horizon_filter": capped, "draft_target": target,
                "surplus": capped - target,
                "utilization": target / float(capped) if capped else None,
                "cell_status": "QUOTA_FEASIBLE" if feasible else "QUOTA_NOT_FEASIBLE",
                "margin_status": "QUOTA_MARGIN_OK" if margin_ok else "QUOTA_MARGIN_TIGHT",
            })
    return {
        "horizon": horizon, "split_shortages": shortages, "split_subset_capacity": rows,
        "quota_feasibility_status": "QUOTA_FEASIBLE_AT_HORIZON" if all_feasible else "QUOTA_NOT_FEASIBLE_AT_HORIZON",
        "quota_margin_status": "QUOTA_MARGIN_OK" if all_margin_ok else "QUOTA_MARGIN_TIGHT",
    }


def refuse_forbidden_output(output, protocol):
    resolved = Path(output).resolve()
    forbidden = [Path(protocol["data"]["raw_root"]).resolve(), Path(protocol["data"]["processed_root"]).resolve(), RESULT_ROOT.resolve()]
    for root in forbidden:
        try:
            resolved.relative_to(root)
            raise ProtocolError("candidate audit report cannot be written under raw, processed, or formal result roots")
        except ValueError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Audit all reachable Moving AI 64x64 candidates without horizon truncation")
    parser.add_argument("--audit-candidates", action="store_true")
    parser.add_argument("--protocol", type=Path, default=REVISION_DIR / "movingai64_protocol_draft.json")
    parser.add_argument("--output", type=Path, default=REVISION_DIR / "candidate_audit_report.json")
    args = parser.parse_args()
    if not args.audit_candidates:
        raise ProtocolError("--audit-candidates is required")
    protocol = read_json(args.protocol)
    refuse_forbidden_output(args.output, protocol)
    by_subset_map, inventory = collect_candidates(Path(protocol["data"]["raw_root"]).resolve(), int(protocol["data"]["construction_seed"]), horizon=None)
    values_by_subset = {subset: [int(row["shortest_path_length"]) for rows in by_subset_map[subset].values() for row in rows] for subset in SUBSETS}
    all_values = [value for subset in SUBSETS for value in values_by_subset[subset]]
    distributions = {"ALL": distribution(all_values)}
    distributions.update({subset: distribution(values_by_subset[subset]) for subset in SUBSETS})
    coverage_by_horizon = {}
    capacity_by_horizon = {}
    slack_by_horizon = {}
    for horizon in CANDIDATE_HORIZONS:
        item = {"ALL": coverage(all_values, horizon)}
        item.update({subset: coverage(values_by_subset[subset], horizon) for subset in SUBSETS})
        coverage_by_horizon[str(horizon)] = item
        capacity_by_horizon[str(horizon)] = capacity_at_horizon(by_subset_map, protocol, horizon)
        slack_item = {"ALL": slack(all_values, horizon)}
        slack_item.update({subset: slack(values_by_subset[subset], horizon) for subset in SUBSETS})
        slack_by_horizon[str(horizon)] = slack_item
    recommendations = {}
    for label, values in [("ALL", all_values)] + [(subset, values_by_subset[subset]) for subset in SUBSETS]:
        recommendations[label] = {
            "smallest_horizon_covering_90_percent": first_candidate_horizon(values, 0.90),
            "smallest_horizon_covering_95_percent": first_candidate_horizon(values, 0.95),
            "smallest_horizon_covering_99_percent": first_candidate_horizon(values, 0.99),
        }
    every_subset = next((h for h in CANDIDATE_HORIZONS if all(coverage(values_by_subset[s], h)["coverage_ratio"] is not None and coverage(values_by_subset[s], h)["coverage_ratio"] >= 0.95 for s in SUBSETS)), None)
    per_map = []
    for subset in SUBSETS:
        for name in sorted(by_subset_map[subset]):
            count = len(by_subset_map[subset][name])
            if count:
                per_map.append({"subset": subset, "source_map": name, "valid_reachable_candidate_count": count, "capped_candidate_count": min(count, int(protocol["data"]["maximum_tasks_per_source_map"]))})
    target_identity = {
        "source_map_split_rule": protocol["data"]["source_map_split_rule"],
        "draft_source_map_quotas": protocol["data"]["draft_source_map_quotas"],
        "draft_task_targets": protocol["data"]["draft_task_targets"],
        "maximum_tasks_per_source_map": protocol["data"]["maximum_tasks_per_source_map"],
    }
    report = {
        "schema_version": "movingai64_candidate_audit.v2", "status": "PASS",
        "candidate_enumeration": {"map_size": 64, "horizon_filter_applied": False, "reachable_candidates_retained_before_horizon_simulation": True, "crop_rule_id": CROP_RULE_ID, "bfs": "four_neighbor_unit_cost"},
        "identity": {
            "raw_root": str(Path(protocol["data"]["raw_root"]).resolve()),
            "source_inventory_sha256": canonical_json_sha256(inventory), "source_inventory": inventory,
            "construction_seed": int(protocol["data"]["construction_seed"]), "crop_rule_id": CROP_RULE_ID,
            "source_map_split_rule": protocol["data"]["source_map_split_rule"],
            "task_target_identity_sha256": canonical_json_sha256(target_identity), "task_target_identity": target_identity,
        },
        "candidate_horizons": list(CANDIDATE_HORIZONS),
        "subset_summary": {subset: {"eligible_source_maps": sum(1 for rows in by_subset_map[subset].values() if rows), "total_valid_reachable_candidates": len(values_by_subset[subset])} for subset in SUBSETS},
        "per_source_map": per_map,
        "shortest_path_percentile_rule": PERCENTILE_RULE,
        "shortest_path_distribution": distributions,
        "horizon_coverage": coverage_by_horizon,
        "episode_slack": {"definition": "slack_ratio = horizon / shortest_path_length", "by_horizon": slack_by_horizon},
        "recommended_horizon_analysis": {
            "by_scope": recommendations,
            "smallest_candidate_horizon_covering_at_least_95_percent_in_every_subset": every_subset,
            "every_subset_95_percent_status": every_subset if every_subset is not None else "NO_CANDIDATE_HORIZON_REACHES_95_PERCENT_EVERY_SUBSET",
            "protocol_mutated": False,
        },
        "scale_reference": {
            "status": "REFERENCE_ONLY_NOT_PROTOCOL_DECISION", "old_map_side": 30, "old_max_manhattan": 58,
            "old_horizon": 180, "old_horizon_to_max_manhattan_ratio": 180.0 / 58.0,
            "new_map_side": 64, "new_max_manhattan": 126,
            "linear_scale_reference_horizon": 180.0 * 126.0 / 58.0, "rounded_reference_horizon": 400,
        },
        "quota_feasibility_by_horizon": capacity_by_horizon,
        "artifacts_created": {"candidate_audit_report_only": True, "processed_maps": False, "task_json": False, "map_split": False, "task_split": False, "formal_result_root": False},
        "torch_model_checkpoint_loaded": False,
    }
    write_json_new(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
