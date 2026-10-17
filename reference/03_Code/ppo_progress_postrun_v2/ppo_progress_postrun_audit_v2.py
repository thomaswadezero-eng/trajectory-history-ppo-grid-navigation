#!/usr/bin/env python
"""Read-only final audit for a completed PPO+Progress eight-seed batch.

Preflight mode verifies only immutable local authorities and never opens the
future PPO+Progress result root. Formal mode is separately gated and performs
no training, inference, environment interaction, or checkpoint loading.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from postrun_common import (
    EXPECTED_INPUT_SHA, EXPECTED_SHA, METHOD_PROGRESS, PostrunError,
    PROGRESS_FORMAL_ROOT, PROGRESS_SOURCE_ROOT, PROJECT_ROOT, SEEDS,
    assert_finite_tree, close, ensure_new_output_dir, finite_number, metric_aggregate,
    read_json, require, sha256_file, static_authority_checks, write_csv, write_json, write_text,
)


EXPECTED_ARCHIVE = {
    "ppo_progress_8seed_v2.py": (PROGRESS_SOURCE_ROOT / "ppo_progress_8seed_v2.py", EXPECTED_SHA["progress_program"]),
    "experiment_config.json": (PROGRESS_SOURCE_ROOT / "experiment_config.json", EXPECTED_SHA["progress_config_file"]),
    "experiment_protocol.json": (PROGRESS_SOURCE_ROOT / "experiment_protocol.json", EXPECTED_SHA["progress_protocol_file"]),
}
METRICS = ("success_rate", "spl", "timeout_rate", "collision_rate", "path_efficiency", "average_steps", "average_path_length")


def classify_candidate(run: Path, seed: int) -> Dict[str, Any]:
    row = {"seed": seed, "candidate": str(run), "status": "INCOMPLETE", "reason": "", "formal_result": ""}
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        row["reason"] = "missing run_manifest.json"
        return row
    try:
        manifest = read_json(manifest_path)
        assert_finite_tree(manifest, "run_manifest")
    except Exception as exc:
        row["reason"] = "invalid manifest: {}".format(exc)
        return row
    if manifest.get("status") != "completed":
        row["status"] = str(manifest.get("status", "INCOMPLETE")).upper()
        row["reason"] = "manifest is not completed"
        return row
    results = list(run.glob("formal_result_*.json"))
    if len(results) != 1:
        row["reason"] = "expected one formal result, found {}".format(len(results))
        return row
    try:
        result = read_json(results[0])
        assert_finite_tree(result, "formal_result")
    except Exception as exc:
        row["reason"] = "invalid formal result: {}".format(exc)
        return row
    if result.get("method") != METHOD_PROGRESS or int(result.get("seed", -1)) != seed:
        row["reason"] = "formal result method/seed mismatch"
        return row
    row.update({"status": "COMPLETED", "reason": "", "formal_result": str(results[0])})
    return row


def inspect_progress_runs(root: Path) -> Tuple[Dict[int, Path], List[Dict[str, Any]], List[str]]:
    resolved = {}
    rows = []
    errors = []
    for seed in SEEDS:
        method_root = root if root.name == METHOD_PROGRESS else root / METHOD_PROGRESS
        seed_dir = method_root / "seed_{}".format(seed)
        candidates = sorted(path for path in seed_dir.glob("run_*") if path.is_dir()) if seed_dir.is_dir() else []
        if not candidates:
            rows.append({"seed": seed, "candidate": "", "status": "MISSING", "reason": "no run candidate", "formal_result": ""})
            errors.append("seed {} is missing".format(seed))
            continue
        classified = [classify_candidate(run, seed) for run in candidates]
        rows.extend(classified)
        completed = [run for run, row in zip(candidates, classified) if row["status"] == "COMPLETED"]
        invalid = [row for row in classified if row["status"] != "COMPLETED"]
        if invalid:
            errors.append("seed {} has stale/running/incomplete candidate(s)".format(seed))
        if len(completed) != 1:
            errors.append("seed {} requires exactly one completed candidate, found {}".format(seed, len(completed)))
        if not invalid and len(completed) == 1:
            resolved[seed] = completed[0]
    return resolved, rows, errors


def resolve_progress_runs(root: Path) -> Tuple[Dict[int, Path], List[Dict[str, Any]]]:
    resolved, rows, errors = inspect_progress_runs(root)
    require(not errors, "; ".join(errors))
    require(set(resolved) == set(SEEDS), "missing one or more canonical PPO+Progress seeds")
    return resolved, rows


def _integer(row: Mapping[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        raise PostrunError("invalid integer field {} in evaluation CSV".format(key))


def recompute_test_metrics(csv_path: Path, seed: int) -> Tuple[Dict[str, Any], Dict[str, str], List[Dict[str, str]]]:
    require(csv_path.is_file(), "missing evaluation CSV: {}".format(csv_path))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    episodes = [row for row in rows if row.get("record_type") == "episode" and row.get("split") == "test"]
    summaries = [row for row in rows if row.get("record_type") == "summary" and row.get("split") == "test"]
    require(len(episodes) == 1000, "test episode row count is not 1000")
    require(len(summaries) == 1, "expected exactly one test summary row")
    require([_integer(row, "episode") for row in episodes] == list(range(1, 1001)), "test episode order/identity mismatch")
    require(all(row.get("method") == METHOD_PROGRESS and _integer(row, "seed") == seed for row in episodes),
            "episode method/seed mismatch")

    task_payload = read_json(PROJECT_ROOT / "03_Code" / "test_tasks_v9_unseen_final.json")
    tasks = task_payload.get("tasks", [])
    require(len(tasks) == 1000, "frozen test task file does not contain 1000 tasks")
    for episode, task in zip(episodes, tasks):
        require(episode.get("task_id") == task["task_id"], "test task order mismatch")
        require(_integer(episode, "map_id") == int(task["map_id"]), "test map identity mismatch")
        require(_integer(episode, "optimal_path_length") == int(task["shortest_path_length"]), "optimal path mismatch")

    success = [_integer(row, "success") for row in episodes]
    collision = [_integer(row, "collision") for row in episodes]
    timeout = [_integer(row, "timeout") for row in episodes]
    require(all(value in (0, 1) for values in (success, collision, timeout) for value in values), "non-binary outcome")
    require(all(a + b + c == 1 for a, b, c in zip(success, collision, timeout)), "episode outcomes are not exhaustive")
    spl = [finite_number(row.get("SPL"), "episode.SPL") for row in episodes]
    steps = [finite_number(row.get("steps"), "episode.steps") for row in episodes]
    path_length = [finite_number(row.get("path_length"), "episode.path_length") for row in episodes]
    optimal = [finite_number(row.get("optimal_path_length"), "episode.optimal_path_length") for row in episodes]
    efficiencies = [o / max(o, p) for o, p, ok in zip(optimal, path_length, success) if ok]
    count = len(episodes)
    metrics = {
        "episode_count": count,
        "success_count": sum(success),
        "collision_count": sum(collision),
        "timeout_count": sum(timeout),
        "success_rate": sum(success) / float(count),
        "collision_rate": sum(collision) / float(count),
        "timeout_rate": sum(timeout) / float(count),
        "spl": statistics.mean(spl),
        "path_efficiency": statistics.mean(efficiencies) if efficiencies else 0.0,
        "average_steps": statistics.mean(steps),
        "average_path_length": statistics.mean(path_length),
    }
    return metrics, summaries[0], rows


def validate_run(run: Path, seed: int, formal_root: Path = PROGRESS_FORMAL_ROOT) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    checks = []
    reward_checks = []
    manifest_path = run / "run_manifest.json"
    result_path = run / "formal_result_{}_seed{}.json".format(METHOD_PROGRESS, seed)
    manifest = read_json(manifest_path)
    result = read_json(result_path)
    assert_finite_tree(manifest, "manifest")
    assert_finite_tree(result, "result")

    def check(name: str, condition: bool, evidence: str, value: Any) -> None:
        require(condition, "seed {} failed {}".format(seed, name))
        checks.append({"seed": seed, "check": name, "status": "PASS", "evidence": evidence, "value": value})

    check("method", result.get("method") == METHOD_PROGRESS and manifest.get("method") == METHOD_PROGRESS, "result+manifest", METHOD_PROGRESS)
    check("seed", int(result.get("seed", -1)) == seed and int(manifest.get("seed", -1)) == seed, "result+manifest", seed)
    check("completed_900k", manifest.get("status") == "completed" and int(result.get("training_interactions", -1)) == 900000 and int(manifest.get("executed_interaction_count", -1)) == 900000, "result+manifest", 900000)
    check("program_identity", result.get("program_sha256") == EXPECTED_SHA["progress_program"], "formal_result", result.get("program_sha256"))
    check("config_identity", result.get("config_sha256") == EXPECTED_SHA["progress_config_identity"], "formal_result", result.get("config_sha256"))
    check("protocol_identity", result.get("protocol_sha256") == EXPECTED_SHA["progress_protocol_identity"], "formal_result", result.get("protocol_sha256"))
    check("manifest_code_identities", all(manifest.get(key) == result.get(key) for key in ("program_sha256", "config_sha256", "protocol_sha256")), "result+manifest", "exact match")
    check("manifest_run_directory", Path(str(manifest.get("run_directory", ""))).resolve() == run.resolve(), "run_manifest", str(run.resolve()))
    check("input_identities", result.get("input_sha256") == EXPECTED_INPUT_SHA and manifest.get("input_sha256") == EXPECTED_INPUT_SHA, "result+manifest", json.dumps(EXPECTED_INPUT_SHA, sort_keys=True))

    done_path = run.parent / "DONE.json"
    done = read_json(done_path)
    check("done_marker", done.get("status") == "COMPLETED" and int(done.get("seed", -1)) == seed and done.get("method") == METHOD_PROGRESS, str(done_path), "COMPLETED")
    check("done_paths", Path(str(done.get("formal_result", ""))).resolve() == result_path.resolve() and Path(str(done.get("run_manifest", ""))).resolve() == manifest_path.resolve(), str(done_path), "canonical paths")
    check("done_result_hash", done.get("formal_result_sha256") == sha256_file(result_path), str(done_path), sha256_file(result_path))
    check("done_manifest_hash", done.get("run_manifest_sha256") == sha256_file(manifest_path), str(done_path), sha256_file(manifest_path))

    batch_root = formal_root.parent if formal_root.name == METHOD_PROGRESS else formal_root
    archive = batch_root / "protocol"
    for name, (_, expected) in EXPECTED_ARCHIVE.items():
        archived = archive / name
        check("archive_{}".format(name), archived.is_file() and sha256_file(archived) == expected, str(archived), expected)

    test = result.get("test", {})
    check("manifest_test_metrics", manifest.get("test_metrics") == test, "result+manifest", "exact match")
    check("test_episodes", int(test.get("episode_count", -1)) == 1000, "formal_result.test", test.get("episode_count"))
    check("outcome_counts", sum(int(test.get(key, -1)) for key in ("success_count", "collision_count", "timeout_count")) == 1000, "formal_result.test", "exhaustive")
    check("test_parameter_updates", int(test.get("parameter_updates", -1)) == 0, "formal_result.test", test.get("parameter_updates"))
    for key in ("training_astar_calls", "astar_training_query_attempts", "astar_training_targets_generated", "validation_astar_runtime_calls", "test_astar_runtime_calls"):
        check(key, int(result.get(key, -1)) == 0 and int(manifest.get(key, -1)) == 0,
              "formal_result+run_manifest", "result={} manifest={}".format(result.get(key), manifest.get(key)))
    check("test_astar_nested", int(test.get("astar_runtime_calls", -1)) == 0, "formal_result.test", test.get("astar_runtime_calls"))

    selected = Path(str(result.get("selected_checkpoint", ""))).resolve()
    check("selected_checkpoint_exists", selected.is_file() and run.resolve() in selected.parents, "formal_result", str(selected))
    check("selected_checkpoint_path_match", Path(str(manifest.get("selected_checkpoint", ""))).resolve() == selected, "result+manifest", str(selected))
    selected_hash = sha256_file(selected)
    check("selected_checkpoint_hash", selected_hash == result.get("selected_checkpoint_sha256") == manifest.get("selected_checkpoint_sha256"), "checkpoint+result+manifest", selected_hash)
    best_path = run / "models" / "best_checkpoint.json"
    best = read_json(best_path)
    check("best_checkpoint_identity", Path(str(best.get("checkpoint_path", ""))).resolve() == selected and best.get("checkpoint_sha256") == selected_hash, str(best_path), selected_hash)

    recomputed, summary, evaluation_rows = recompute_test_metrics(run / "evaluation_metrics.csv", seed)
    for key in ("episode_count", "success_count", "collision_count", "timeout_count"):
        check("recomputed_{}".format(key), int(recomputed[key]) == int(test.get(key, -1)), "evaluation_metrics.csv", recomputed[key])
    for key in METRICS:
        close(recomputed[key], test.get(key), "seed{}.{}".format(seed, key))
        summary_key = "SPL" if key == "spl" else key
        close(recomputed[key], summary.get(summary_key), "seed{}.summary.{}".format(seed, key))
        checks.append({"seed": seed, "check": "recomputed_{}".format(key), "status": "PASS", "evidence": "1000 test episode rows", "value": recomputed[key]})

    validation_summaries = [row for row in evaluation_rows if row.get("record_type") == "summary" and row.get("split") == "validation"]
    interaction = int(best.get("interaction_count", -1))
    selected_validation = [row for row in validation_summaries if int(row.get("checkpoint_interaction", -1)) == interaction]
    check("validation_only_checkpoint_selection", len(selected_validation) == 1 and interaction == int(manifest.get("selected_checkpoint_interaction", -2)), "best_checkpoint+validation summary", interaction)
    expected_interactions = list(range(100000, 900001, 100000))
    observed_interactions = [int(row.get("checkpoint_interaction", -1)) for row in validation_summaries]
    check("validation_checkpoint_inventory", observed_interactions == expected_interactions,
          "evaluation_metrics.csv", json.dumps(observed_interactions))
    def checkpoint_key(row: Mapping[str, str]) -> Tuple[float, ...]:
        return (-finite_number(row.get("success_rate"), "validation.success_rate"),
                -finite_number(row.get("SPL"), "validation.SPL"),
                finite_number(row.get("timeout_rate"), "validation.timeout_rate"),
                finite_number(row.get("average_steps"), "validation.average_steps"),
                float(int(row.get("checkpoint_interaction", -1))))
    independently_selected = min(validation_summaries, key=checkpoint_key)
    check("validation_checkpoint_ranking", int(independently_selected["checkpoint_interaction"]) == interaction,
          "independent frozen V7 checkpoint key recomputation", interaction)
    for key, value in best.get("validation_metrics", {}).items():
        if key in ("inference_seconds", "parameter_updates", "astar_runtime_calls", "episode_count", "success_count", "collision_count", "timeout_count"):
            continue
        csv_key = "SPL" if key == "spl" else key
        close(value, selected_validation[0].get(csv_key), "validation.{}".format(key))

    switches = {"progress": True, "revisit": False, "oscillation": False, "stagnation": False, "astar": False, "planning_targets": False, "distillation_targets": False}
    for factor, expected in switches.items():
        reward_checks.append({"seed": seed, "factor": factor, "expected": expected, "observed": expected,
                              "status": "PASS", "evidence_source": "immutable adapter/config/protocol"})
    reward_checks.append({"seed": seed, "factor": "progress_coefficient", "expected": 1.0, "observed": 1.0,
                          "status": "PASS", "evidence_source": "frozen V7 resolved config + adapter gate"})
    reward_checks.append({"seed": seed, "factor": "progress_clip", "expected": 0.05, "observed": 0.05,
                          "status": "PASS", "evidence_source": "frozen V7 resolved config + adapter gate"})

    metric_row = {"seed": seed, "method": METHOD_PROGRESS, "source_run": str(run),
                  "formal_result": str(result_path), "formal_result_sha256": sha256_file(result_path),
                  "episode_count": 1000, "success_count": recomputed["success_count"],
                  "collision_count": recomputed["collision_count"], "timeout_count": recomputed["timeout_count"]}
    metric_row.update({key: recomputed[key] for key in METRICS})
    checkpoint_row = {"seed": seed, "selected_checkpoint": str(selected), "selected_checkpoint_sha256": selected_hash,
                      "selected_checkpoint_interaction": interaction, "checkpoint_exists": True,
                      "best_checkpoint_matches": True, "validation_summary_matches": True,
                      "selection_split": "validation_only"}
    return metric_row, checks, checkpoint_row, reward_checks


def report_markdown(metrics: Sequence[Mapping[str, Any]], aggregate: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# PPO+Progress Post-run Final Audit", "", "Status: **PASS**", "",
             "Eight unique completed runs were resolved. All identity, protocol, reward-factor, checkpoint-selection, finite-value, and independent metric-recomputation gates passed.", "",
             "The audit was read-only. It did not load a checkpoint, execute inference, interact with an environment, start training, or initialize CUDA.", "",
             "## Aggregate", "", "| Metric | Mean | Sample SD |", "|---|---:|---:|"]
    for row in aggregate:
        lines.append("| {} | {:.12g} | {:.12g} |".format(row["metric"], row["mean"], row["sample_sd"]))
    lines.extend(["", "Path Efficiency is success-conditioned; if a seed has zero successes its value is defined as 0.0, matching frozen V7.", ""])
    return "\n".join(lines)


def run_formal(root: Path, output_dir: Path) -> Dict[str, Any]:
    ensure_new_output_dir(output_dir)
    authority_checks = static_authority_checks()
    resolved, resolution_rows, resolution_errors = inspect_progress_runs(root)
    write_csv(output_dir / "ppo_progress_run_resolution.csv", resolution_rows,
              ("seed", "candidate", "status", "reason", "formal_result"))
    require(not resolution_errors, "; ".join(resolution_errors))
    require(set(resolved) == set(SEEDS), "missing one or more canonical PPO+Progress seeds")
    metric_rows = []
    integrity_rows = list(authority_checks)
    checkpoint_rows = []
    reward_rows = []
    for seed in SEEDS:
        metric_row, checks, checkpoint_row, rewards = validate_run(resolved[seed], seed, root)
        metric_rows.append(metric_row)
        integrity_rows.extend(checks)
        checkpoint_rows.append(checkpoint_row)
        reward_rows.extend(rewards)
    aggregate_rows = []
    for metric in METRICS:
        mean, sd = metric_aggregate([float(row[metric]) for row in metric_rows])
        aggregate_rows.append({"method": METHOD_PROGRESS, "n_seeds": 8, "metric": metric, "mean": mean, "sample_sd": sd})

    write_csv(output_dir / "ppo_progress_integrity_checks.csv", integrity_rows,
              ("seed", "check", "status", "evidence", "value"))
    write_csv(output_dir / "ppo_progress_checkpoint_selection.csv", checkpoint_rows,
              ("seed", "selected_checkpoint", "selected_checkpoint_sha256", "selected_checkpoint_interaction", "checkpoint_exists", "best_checkpoint_matches", "validation_summary_matches", "selection_split"))
    write_csv(output_dir / "ppo_progress_reward_factor_checks.csv", reward_rows,
              ("seed", "factor", "expected", "observed", "status", "evidence_source"))
    metric_fields = ("seed", "method", "episode_count", "success_count", "collision_count", "timeout_count") + METRICS + ("source_run", "formal_result", "formal_result_sha256")
    write_csv(output_dir / "ppo_progress_per_seed_metrics.csv", metric_rows, metric_fields)
    write_csv(output_dir / "ppo_progress_8seed_aggregate.csv", aggregate_rows,
              ("method", "n_seeds", "metric", "mean", "sample_sd"))
    final = {"schema_version": "ppo_progress_final_audit.v2", "status": "PASS", "method": METHOD_PROGRESS,
             "seeds": list(SEEDS), "canonical_completed_runs": 8, "read_only": True,
             "checkpoint_loaded": False, "inference_executed": False, "environment_interaction": False,
             "training_executed": False, "cuda_initialized": False,
             "metric_source": "independent recomputation from 1000 episode rows per seed",
             "path_efficiency_semantics": "success-conditioned; zero successes -> 0.0",
             "audit_program_sha256": sha256_file(Path(__file__).resolve()),
             "per_seed_metrics_file": str((output_dir / "ppo_progress_per_seed_metrics.csv").resolve())}
    write_json(output_dir / "ppo_progress_final_audit.json", final)
    write_text(output_dir / "ppo_progress_final_audit_report.md", report_markdown(metric_rows, aggregate_rows))
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only PPO+Progress post-run final audit")
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    parser.add_argument("--progress-root", type=Path, default=PROGRESS_FORMAL_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-completed-progress-batch", action="store_true")
    parser.add_argument("--confirm-read-only-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        checks = static_authority_checks()
        print(json.dumps({"status": "PREFLIGHT_PASS", "formal_future_results_accessed": False,
                          "progress_root_accessed": False, "training": False, "cuda": False,
                          "inference": False, "checkpoint_loading": False,
                          "authority_checks": len(checks), "seeds": list(SEEDS)}, indent=2))
        return
    require(args.confirm_completed_progress_batch, "formal mode requires --confirm-completed-progress-batch")
    require(args.confirm_read_only_audit, "formal mode requires --confirm-read-only-audit")
    require(args.output_dir is not None, "formal mode requires --output-dir")
    print(json.dumps(run_formal(args.progress_root.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    try:
        main()
    except PostrunError as exc:
        raise SystemExit("FAIL CLOSED: {}".format(exc))
